import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time
import copy
import torch.utils.data
import json
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchvision import datasets
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from tqdm import tqdm


# 하이퍼파라미터 설정
epochs = 10
batch_size = 8


data_dir = "Cycle_data_split"

#data_dir/
#│
#├── train/
#│   ├── rgb/안에 ca1 ca2 폴더
#│   │   ├── 이미지 파일들 (예: img1.jpg, img2.jpg, ...)
#│   └── labels/안에 ca1 ca2 폴더
#│       ├── 라벨 파일들 (예: img1.json, img2.json, ...)
#│
#├── val/
#│   ├── rgb/안에 ca1 ca2 폴더
#│   │   ├── 이미지 파일들 (예: img1.jpg, img2.jpg, ...)
#│   └── labels/안에 ca1 ca2 폴더
#│       ├── 라벨 파일들 (예: img1.json, img2.json, ...)
#│
#└── test/
#    ├── rgb/안에 ca1 ca2 폴더
#    │   ├── 이미지 파일들 (예: img1.jpg, img2.jpg, ...)
#    └── labels/안에 ca1 ca2 폴더
#        ├── 라벨 파일들 (예: img1.json, img2.json, ...)
 

# 데이터 전처리
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
cabbage_classes = ['ca1', 'ca2']
#CA1 : 대박나, CA2 : 꼬꼬마
tomato_classes = ['to1', 'to2']
pepper_classes = ['pe1', 'pe2']
use_classes = cabbage_classes

# 멀티모달 데이터셋 정의 (이미지 + 텍스트)
class FilteredImageFolderWithText(datasets.ImageFolder):
    def __init__(self, root, labels_dir, transform=None, target_transform=None, loader=datasets.folder.default_loader):
        super(FilteredImageFolderWithText, self).__init__(root, transform=transform, 
                                                         target_transform=target_transform, 
                                                         loader=loader)
        self.labels_dir = labels_dir
        
        # 기존의 초기화 코드는 그대로 유지...

    def __getitem__(self, index):
        # 최대 시도 횟수 설정
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # 이미지 데이터를 가져옵니다.
                path, target = self.samples[index]
                
                # 이미지를 로드합니다.
                sample = self.loader(path)
                
                if self.transform is not None:
                    sample = self.transform(sample)

                # JSON 파일 경로 생성
                json_filename = os.path.splitext(os.path.basename(path))[0] + '.json'
                json_path = os.path.join(self.labels_dir, 
                                        os.path.basename(os.path.dirname(path)), 
                                        json_filename)

                # JSON 파일에서 텍스트 데이터 로드
                with open(json_path, 'r', encoding='utf-8') as f:
                    text_features = json.load(f)
                
                # 모든 measurement 데이터를 텐서로 변환
                measurements = text_features['measurement'][0]
                measurement_values = [
                    measurements['fruit_length'],
                    measurements['fruit_width'],
                    measurements['leaf_length'],
                    measurements['leaf_width'],
                    measurements['joint_length'],
                    measurements['central_length']
                ]
                
                # '-'로 표시된 값을 0으로 변환
                measurement_values = [0.0 if v == '-' else float(v) for v in measurement_values]

                text_tensor = torch.tensor(measurement_values, dtype=torch.float32)
                
                return sample, text_tensor, target

            except Exception as e:
                print(f"Error loading sample at index {index}, path {path}: {e}")
                attempt += 1
                index = (index + 1) % len(self.samples)
                
                if attempt == max_attempts:
                    print(f"Failed to load any valid sample after {max_attempts} attempts")
                    # 마지막 시도로 더미 데이터를 반환
                    dummy_image = torch.zeros((3, 200, 200))  # 이미지 크기에 맞게 조정
                    dummy_text = torch.zeros(6)  # 텍스트 특성 수에 맞게 조정
                    return dummy_image, dummy_text, target


# 데이터셋 및 데이터로더 생성
image_datasets = {x: FilteredImageFolderWithText(os.path.join(data_dir, x, 'rgb'), 
                                                 os.path.join(data_dir, x, 'labels'), 
                                                 data_transforms[x]) 
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
print(class_names)

# 멀티모달 모델 정의
class MultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalModel, self).__init__()
        self.image_model = EfficientNet.from_pretrained('efficientnet-b2')
        
        # 텍스트 인코더 정의 (예: 간단한 MLP)
        self.text_encoder = nn.Sequential(
            nn.Linear(6, 128),  # 텍스트 입력 차원을 6으로 변경
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Global Average Pooling Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 이미지와 텍스트 임베딩을 결합하여 분류
        self.classifier = nn.Sequential(
            nn.Linear(64 + self.image_model._fc.in_features, 256),  # EfficientNet의 마지막 FC layer 크기
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, text):
        # EfficientNet을 통해 이미지 특징 추출
        image_features = self.image_model.extract_features(image)
        image_features = self.global_avg_pool(image_features)  # Global Average Pooling
        image_features = image_features.view(image_features.size(0), -1)  # 2차원 텐서로 변환
        
        # 텍스트 인코딩
        text_features = self.text_encoder(text)

        # 이미지와 텍스트 특징을 결합
        combined_features = torch.cat((image_features, text_features), dim=1)

        # 최종 분류
        outputs = self.classifier(combined_features)
        return outputs

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalModel(num_classes=len(use_classes))
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 학습 초기화
best_model_weights = copy.deepcopy(model.state_dict())
best_acc = 0.0

train_acc_list, train_prec_list, train_rec_list, train_f1_list = [], [], [], []
val_acc_list, val_prec_list, val_rec_list, val_f1_list = [], [], [], []
test_acc_list, test_prec_list, test_rec_list, test_f1_list = [], [], [], []
print("Class names and indices:", class_names)

# 학습 루프
for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch, epochs-1))
    print('-'*10)

    for phase in ['train', 'val', 'test']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        
        all_labels = []
        all_preds = []
        
        for inputs, texts, labels in tqdm(dataloaders[phase]):
            inputs = inputs.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs, texts)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        if phase == 'train':
            scheduler.step()
        #print(all_labels, all_preds, "하하")
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_prec = precision_score(all_labels, all_preds, average='weighted')
        epoch_rec = recall_score(all_labels, all_preds, average='weighted')
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

        if phase == 'train':
            train_acc_list.append(epoch_acc)
            train_prec_list.append(epoch_prec)
            train_rec_list.append(epoch_rec)
            train_f1_list.append(epoch_f1)
        elif phase == 'val':
            val_acc_list.append(epoch_acc)
            val_prec_list.append(epoch_prec)
            val_rec_list.append(epoch_rec)
            val_f1_list.append(epoch_f1)
        else:  # phase == 'test'
            test_acc_list.append(epoch_acc)
            test_prec_list.append(epoch_prec)
            test_rec_list.append(epoch_rec)
            test_f1_list.append(epoch_f1)

        print('{} Loss: {:.4f} Acc: {:.4f} Prec: {:.4f} Rec: {:.4f} F1: {:.4f}'.format(
            phase, epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1))


        #print(running_corrects, "ㅎㅇ")
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())

# metrics 딕셔너리 생성
metrics = {
    'train_acc': train_acc_list,
    'train_prec': train_prec_list,
    'train_rec': train_rec_list,
    'train_f1': train_f1_list,
    'val_acc': val_acc_list,
    'val_prec': val_prec_list,
    'val_rec': val_rec_list,
    'val_f1': val_f1_list,
    'test_acc': test_acc_list,
    'test_prec': test_prec_list,
    'test_rec': test_rec_list,
    'test_f1': test_f1_list
}

# 최적의 모델 가중치 저장
model.load_state_dict(best_model_weights)
torch.save(model.state_dict(), 'weights/2Cycle_CA.pth')
np.save('metrics/2Cycle_CA.npy', metrics)