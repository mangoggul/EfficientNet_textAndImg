import cv2
import time
import os
import json
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets
from efficientnet_pytorch import EfficientNet
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader

# YOLO 모델 설정
yolo_model = YOLO('EX/TO12_TB_TB-012/train/yolov8n/weights/best.pt')

# 하이퍼파라미터 설정
epochs = 5
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# YOLO 탐지 결과를 포함한 커스텀 데이터셋
class CustomDatasetWithYOLO(Dataset):
    def __init__(self, rgb_folder, labels_dir, transform=None):
        self.rgb_folder = rgb_folder
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith(('.jpg', '.png'))])
        
        # YOLO로 모든 이미지에 대한 탐지 수행
        self.detection_results = self._perform_yolo_detection()

    def _perform_yolo_detection(self):
        detection_results = {}
        for idx, img_file in enumerate(self.image_files):
            img_path = os.path.join(self.rgb_folder, img_file)
            img = cv2.imread(img_path)
            results = yolo_model(img)
            
            detected = False
            for result in results:
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    detected = True
                    break
            
            detection_results[img_file] = detected
        return detection_results

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.rgb_folder, img_name)
        image = cv2.imread(img_path)
        
        if self.transform:
            image = self.transform(image)

        # JSON 레이블 로드
        json_filename = os.path.splitext(img_name)[0] + '.json'
        json_path = os.path.join(self.labels_dir, json_filename)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        
        # 측정값을 텐서로 변환
        measurements = label_data['measurement'][0]
        measurement_values = [
            measurements.get('fruit_length', 0),
            measurements.get('fruit_width', 0),
            measurements.get('leaf_length', 0),
            measurements.get('leaf_width', 0),
            measurements.get('joint_length', 0),
            measurements.get('central_length', 0)
        ]
        measurement_values = [0.0 if v == '-' else float(v) for v in measurement_values]
        
        # YOLO 탐지 결과 추가
        yolo_detection = float(self.detection_results[img_name])
        measurement_values.append(yolo_detection)
        
        text_tensor = torch.tensor(measurement_values, dtype=torch.float32)
        
        # 클래스 레이블 (이 부분은 실제 데이터에 맞게 수정 필요)
        label = label_data.get('class_label', 0)
        
        return image, text_tensor, label

# 멀티모달 모델 정의
class MultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalModel, self).__init__()
        self.image_model = EfficientNet.from_pretrained('efficientnet-b2')
        
        # 텍스트 인코더 (YOLO 탐지 결과를 포함하도록 입력 차원 증가)
        self.text_encoder = nn.Sequential(
            nn.Linear(7, 128),  # 6개 측정값 + 1개 YOLO 탐지 결과
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(64 + self.image_model._fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, text):
        image_features = self.image_model.extract_features(image)
        image_features = self.global_avg_pool(image_features)
        image_features = image_features.view(image_features.size(0), -1)
        
        text_features = self.text_encoder(text)
        
        combined_features = torch.cat((image_features, text_features), dim=1)
        outputs = self.classifier(combined_features)
        return outputs

# 데이터셋 및 데이터로더 생성
def create_dataloaders(rgb_folder, labels_dir):
    dataset = CustomDatasetWithYOLO(rgb_folder, labels_dir, transform=data_transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 학습 함수
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=5):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, texts, labels in dataloaders[phase]:
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

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        scheduler.step()

    model.load_state_dict(best_model_weights)
    return model

# 메인 실행 코드
def main():
    # 데이터 경로 설정
    train_rgb_folder = "CA/train/rgb"
    train_labels_dir = "CA/train/labels"
    val_rgb_folder = "CA/val/rgb"
    val_labels_dir = "CA/val/labels"

    # 데이터로더 생성
    dataloaders = {
        'train': create_dataloaders(train_rgb_folder, train_labels_dir),
        'val': create_dataloaders(val_rgb_folder, val_labels_dir)
    }

    # 모델 초기화
    model = MultimodalModel(num_classes=2)  # 클래스 수에 맞게 수정
    model = model.to(device)

    # 손실 함수, 옵티마이저, 스케줄러 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 모델 학습
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=epochs)

    # 모델 저장
    torch.save(model.state_dict(), 'weights/combined_yolo_efficientnet.pth')

if __name__ == "__main__":
    main()