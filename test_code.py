import torch
import torchvision.transforms as transforms
import json
import os
from efficientnet_pytorch import EfficientNet
from PIL import Image
import torch.nn as nn

# 모델 정의 (학습 코드와 동일)
class MultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalModel, self).__init__()
        self.image_model = EfficientNet.from_pretrained('efficientnet-b2')
        
        # 텍스트 인코더 정의 (예: 간단한 MLP)
        self.text_encoder = nn.Sequential(
            nn.Linear(6, 128),  #json 받을 텍스트가 6개라서 6개 차원, 출력차원은 128
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Global Average Pooling Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) #feature map 이 점점 늘어나는데 이러면 파라미터 수도 늘어남
        #따라서 차원을 감소 시킬 필요가 있음 -> 그래서 pooling layer 사용 여기서는 평균 풀링 , 맥스 풀링도 있음

        # 이미지와 텍스트 임베딩을 결합하여 분류
        self.classifier = nn.Sequential(
            # EfficientNet-B2 in_features : 1,408
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

# 하이퍼파라미터 설정
num_classes = 2  # 클래스 수 정의ㅇ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 초기화 및 학습된 가중치 로드
model = MultimodalModel(num_classes=num_classes)
model.load_state_dict(torch.load('weights/2Cycle_CA.pth'))
model.to(device)
model.eval()

# 이미지와 텍스트 전처리 정의
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 배치 차원을 추가 unsqueeze 함수는 차원을 행성해줌 파라미터로는 몇번쨰 index 차원생성할지
    return image

def preprocess_text(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        text_features = json.load(f)
    
    measurements = text_features['measurement'][0]
    measurement_values = [
        measurements['fruit_length'],
        measurements['fruit_width'],
        measurements['leaf_length'],
        measurements['leaf_width'],
        measurements['joint_length'],
        measurements['central_length']
    ]
    
    measurement_values = [0.0 if v == '-' else float(v) for v in measurement_values]
    text_tensor = torch.tensor(measurement_values, dtype=torch.float32).unsqueeze(0)  # 배치 차원을 추가
    return text_tensor

def get_true_label(json_path) :
    with open(json_path, 'r', encoding='utf-8') as f:
        text_features = json.load(f)
    
    image_name = text_features['images']["file_name"]
    return image_name #지금은 CA1 뱉을거야
    
    

# 폴더 단위로 inference 수행 및 metric 계산
def evaluate_folder(image_folder, json_folder):
    correct_predictions = 0
    total_predictions = 0
    
    to_class_names = ['to1_fruit', 'to2_fruit']
    ca_class_names = ['ca1_leaf', 'ca2_leaf']
    class_names = to_class_names  # 클래스 이름 정의 (예시)
    
    for json_file, image_file in zip(os.listdir(json_folder) , os.listdir(image_folder)):
        json_path = os.path.join(json_folder, json_file)
        image_path = os.path.join(image_folder, image_file)
        
        if not os.path.exists(image_path):
            print(f"Image file {image_path} does not exist, skipping.")
            continue

        # 예측 수행
        image = preprocess_image(image_path).to(device)
        text = preprocess_text(json_path).to(device)
        
        with torch.no_grad():
            outputs = model(image, text)
            probabilities = torch.softmax(outputs, dim=1)  # 소프트맥스 함수로 확률 계산
            _, predicted_class_idx = torch.max(outputs, 1)
        
        predicted_class_name = class_names[predicted_class_idx.item()]
        predicted_probability = probabilities[0][predicted_class_idx].item()
        
        # 예측 결과 출력
        print(f"Image: {image_path}, Predicted Class: {predicted_class_name}, Probability: {predicted_probability:.4f}")
        
        # JSON에서 images.file_name 을 읽어서 실제값 확인한다
        # 실제 정답 라벨과 비교하여 accuracy 계산

        true_label = get_true_label(json_path) #truelabel == CA1
        print(predicted_class_name[0:3], "예측값")
        print(true_label[4:7].lower(), "실제 값")
        if  predicted_class_name[0:3] == true_label[4:7].lower() : 
            correct_predictions += 1
        
        total_predictions += 1
    
    # Metric 출력 (accuracy 계산 예시)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.4f}")

# 테스트할 이미지 및 JSON 폴더 경로
test_image_folder = 'test_data/image'  
test_json_folder = 'test_data/json'  

# 폴더 단위로 예측 및 metric 계산
evaluate_folder(test_image_folder, test_json_folder)
