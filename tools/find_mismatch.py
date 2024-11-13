
#이 코드는 rgb 와 text 폴더를 둘 넣고 둘이 mismatch 하는 파일을 확인하고 제거하는 코드 
#제거를 해야 모델을 돌릴수가 있음. 


import os

def find_and_delete_mismatched_files(jpg_dir, json_dir):
    # jpg 및 json 폴더의 파일 목록을 가져옵니다.
    jpg_files = sorted([f for f in os.listdir(jpg_dir) if f.endswith(('.jpg', '.jpeg'))])
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])

    jpg_basenames = set(os.path.splitext(f)[0] for f in jpg_files)
    json_basenames = set(os.path.splitext(f)[0] for f in json_files)

    # jpg와 json 파일 이름이 다른 것을 찾습니다.
    only_in_jpg = jpg_basenames - json_basenames
    only_in_json = json_basenames - jpg_basenames

    # 파일 삭제
    for basename in only_in_jpg:
        jpg_path = os.path.join(jpg_dir, basename + '.jpg')
        if os.path.exists(jpg_path):
            os.remove(jpg_path)
            print(f"{jpg_path} 파일이 삭제되었습니다.")
    
    for basename in only_in_json:
        json_path = os.path.join(json_dir, basename + '.json')
        if os.path.exists(json_path):
            os.remove(json_path)
            print(f"{json_path} 파일이 삭제되었습니다.")

# 이 폴더 안에 rgb 랑 label 을 넣어서 확인
lis_rgb = ['Cycle_data/CA/ca1_001', 'Cycle_data/CA/ca2_001', 'Cycle_data/TO/to1_001', 'Cycle_data/TO/to2_001']
lis_label = ['Cycle_data/CA_label/ca1_001', 'Cycle_data/CA_label/ca2_001', 'Cycle_data/TO_label/to1_001', 'Cycle_data/TO_label/to2_001']

# 두 리스트의 길이가 같은지 확인
if len(lis_rgb) == len(lis_label):
    for jpg_dir, json_dir in zip(lis_rgb, lis_label):
        # 서로 다른 파일을 찾고 삭제합니다.
        print(f"폴더 '{jpg_dir}'와 '{json_dir}' 파일 확인 및 삭제:")
        find_and_delete_mismatched_files(jpg_dir, json_dir)
        print("\n")
else:
    print("폴더 리스트의 길이가 일치하지 않습니다. 각 폴더의 경로 쌍을 확인하세요.")
