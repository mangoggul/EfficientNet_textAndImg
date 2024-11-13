import os
import json

# 폴더 경로를 설정합니다
folder_path = 'CA_01/004'

def extract_measurement_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data.get('measurement', [])

def process_json_files_in_folder(folder_path):
    measurement_data = {}
    
    # 폴더 내의 모든 파일을 순회합니다
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            measurements = extract_measurement_from_json(file_path)
            measurement_data[filename] = measurements
    
    return measurement_data

# 함수 호출 및 출력
all_measurement_data = process_json_files_in_folder(folder_path)
for filename, measurements in all_measurement_data.items():
    print(f"File: {filename}")
    print("Measurement Data:", measurements)
    print()
