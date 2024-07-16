import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time
import copy
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

data_dir = "yonsei_data"

data_transforms = {
    'train': {
        'rgb': transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'depth': transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 단일 채널이므로 평균과 표준편차는 스칼라 값으로 지정
        ])
    },
    'val': {
        'rgb': transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'depth': transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 단일 채널이므로 평균과 표준편차는 스칼라 값으로 지정
        ])
    }
}

class MultiModalDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.classes = sorted(os.listdir(rgb_dir))
        self.rgb_images = []
        self.depth_images = []
        self.labels = []
        
        for label, cls in enumerate(self.classes):
            rgb_cls_dir = os.path.join(rgb_dir, cls)
            depth_cls_dir = os.path.join(depth_dir, cls)
            rgb_images = sorted(os.listdir(rgb_cls_dir))
            depth_images = sorted(os.listdir(depth_cls_dir))
            
            self.rgb_images += [os.path.join(rgb_cls_dir, img) for img in rgb_images]
            self.depth_images += [os.path.join(depth_cls_dir, img) for img in depth_images]
            self.labels += [label] * len(rgb_images)
    
    def __len__(self):
        return len(self.rgb_images)
    
    def __getitem__(self, idx):
        rgb_path = self.rgb_images[idx]
        depth_path = self.depth_images[idx]
        
        rgb_image = Image.open(rgb_path).convert('RGB')
        depth_image = Image.open(depth_path).convert('L')  # 단일 채널
        
        if self.transform:
            rgb_image = self.transform['rgb'](rgb_image)
            depth_image = self.transform['depth'](depth_image)
        
        # Depth 이미지를 RGB 이미지와 결합
        depth_image = depth_image.expand(3, -1, -1)  # (1, H, W) -> (3, H, W)
        combined_image = torch.cat((rgb_image, depth_image), dim=0)  # (6, H, W)
        
        label = self.labels[idx]
        
        return combined_image, label

image_datasets = {
    x: MultiModalDataset(os.path.join(data_dir, x, 'rgb'),
                         os.path.join(data_dir, x, 'depth'),
                         data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=0)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = ["warehouse", "cafe", "classroom"]  # 실제 클래스 이름으로 대체

# 모델 정의
class EfficientNetMultiModal(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetMultiModal, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._conv_stem = nn.Conv2d(6, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        num_ftrs = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EfficientNetMultiModal(num_classes=len(class_names)).to(device)

# 초기화
train_acc_list, train_prec_list, train_rec_list, train_f1_list = [], [], [], []
val_acc_list, val_prec_list, val_rec_list, val_f1_list = [], [], [], []

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 학습 루프 정의
num_epochs = 300
best_acc = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)
    
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        
        all_labels = []
        all_preds = []

        for inputs, labels in tqdm(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        epoch_prec = precision_score(all_labels, all_preds, average='weighted')
        epoch_rec = recall_score(all_labels, all_preds, average='weighted')
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

        if phase == 'train':
            train_acc_list.append(epoch_acc.item())
            train_prec_list.append(epoch_prec)
            train_rec_list.append(epoch_rec)
            train_f1_list.append(epoch_f1)
        else:
            val_acc_list.append(epoch_acc.item())
            val_prec_list.append(epoch_prec)
            val_rec_list.append(epoch_rec)
            val_f1_list.append(epoch_f1)

        print('{} Loss: {:.4f} Acc: {:.4f} Prec: {:.4f} Rec: {:.4f} F1: {:.4f}'.format(
            phase, epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1))
        
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())
    
    print()

print('Training complete')

# 결과 저장
metrics = {
    'train_acc': train_acc_list,
    'train_prec': train_prec_list,
    'train_rec': train_rec_list,
    'train_f1': train_f1_list,
    'val_acc': val_acc_list,
    'val_prec': val_prec_list,
    'val_rec': val_rec_list,
    'val_f1': val_f1_list,
}

# 최고 성능 모델 저장
np.save('metrics/metrics_multi_300.npy', metrics)
torch.save(best_model_weights, 'weights/best_weights_b0_multi_300.pth')
