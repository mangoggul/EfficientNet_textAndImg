import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time
import copy
import torch.utils.data
import json

from torch.optim import lr_scheduler #optimizer
from torchvision import datasets 
from efficientnet_pytorch import EfficientNet
#from torch.utils.tensorboard import SummaryWriter #실시간 학습 현황
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from tqdm import tqdm

epochs = 300
data_dir = "yonsei_data"
#writer = SummaryWriter('runs/')

data_transforms = {"train" : transforms.Compose([
    transforms.Resize((200,200)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
    "val" : transforms.Compose([
    transforms.Resize((200,200)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
}


use_classes = ['warehouse', 'cafe', 'classroom']

# ImageFolder를 확장하여 원하는 클래스만 사용할 수 있도록 필터링
class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=datasets.folder.default_loader):
        super(FilteredImageFolder, self).__init__(root, transform=transform, target_transform=target_transform, loader=loader)
        
        # 사용할 클래스들만 필터링
        print("Before filtering:", len(self.samples))  # 필터링 전 샘플 수
        self.samples = [s for s in self.samples if os.path.basename(os.path.dirname(s[0])).lower() in [cls.lower() for cls in use_classes]]
        print("After filtering:", len(self.samples))  # 필터링 후 샘플 수

        if len(self.samples) == 0:
            raise ValueError("No samples found after filtering. Please check your class names and dataset.")

        self.targets = [s[1] for s in self.samples]
        
        # 클래스 인덱스 필터링
        self.class_to_idx = {cls_name: idx for cls_name, idx in self.class_to_idx.items() if cls_name.lower() in [cls.lower() for cls in use_classes]}
        self.classes = [cls_name for cls_name in use_classes if cls_name.lower() in [c.lower() for c in self.class_to_idx.keys()]]
        
        # 클래스 인덱스 재정렬
        self.class_to_idx = {cls_name: use_classes.index(cls_name) for cls_name in self.classes}
        
        # 샘플 재정렬
        self.samples = [(path, self.class_to_idx[os.path.basename(os.path.dirname(path)).lower()]) for path, _ in self.samples]
        self.targets = [self.class_to_idx[os.path.basename(os.path.dirname(s[0])).lower()] for s in self.samples]

# 데이터셋 및 데이터로더 생성
print(os.path.join(data_dir,'images'))

# 데이터셋 및 데이터로더 생성
image_datasets = {x: FilteredImageFolder(os.path.join(data_dir, x, 'rgb'), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)

# Load labels_map.txt
with open('labels_map.txt', 'r') as f:
    labels_map = json.load(f)
print(labels_map)
# Check if class_to_idx matches labels_map
is_labeling_correct = all(labels_map[str(idx)] == class_name for class_name, idx in image_datasets['train'].class_to_idx.items())
print(f"Labeling is correct: {is_labeling_correct}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_pretrained('efficientnet-b0' , num_classes=3)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_model_weights = copy.deepcopy(model.state_dict())
best_acc = 0.0

# 초기화
train_acc_list, train_prec_list, train_rec_list, train_f1_list = [], [], [], []
val_acc_list, val_prec_list, val_rec_list, val_f1_list = [], [], [], []

for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch, epochs-1))
    print('-'*10)

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

        if phase == 'train':
            scheduler.step()

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

np.save('metrics/metrics_single_300.npy', metrics)
torch.save(best_model_weights, 'weights/best_weights_b0_single_300.pth')