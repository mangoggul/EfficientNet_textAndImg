import os
import shutil
import random

def create_splits(rgb_dirs, json_dirs, dest_dir, split_ratio=(0.8, 0.1, 0.1)):
    sets = ['train', 'val', 'test']

    # 각 세트의 경로를 설정합니다.
    for set_name in sets:
        os.makedirs(os.path.join(dest_dir, set_name, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, set_name, 'labels'), exist_ok=True)

    # 각 rgb/json 폴더 쌍을 처리합니다.
    for rgb_dir, json_dir in zip(rgb_dirs, json_dirs):
        category = os.path.basename(rgb_dir)  # 폴더 이름을 카테고리로 사용

        # 카테고리 폴더 생성
        for set_name in sets:
            os.makedirs(os.path.join(dest_dir, set_name, 'rgb', category), exist_ok=True)
            os.makedirs(os.path.join(dest_dir, set_name, 'labels', category), exist_ok=True)

        rgb_files = sorted(os.listdir(rgb_dir))
        json_files = sorted(os.listdir(json_dir))

        data_pairs = list(zip(rgb_files, json_files))

        # 데이터를 섞습니다.
        random.shuffle(data_pairs)

        # 비율에 따라 데이터를 분할합니다.
        train_split = int(len(data_pairs) * split_ratio[0])
        val_split = train_split + int(len(data_pairs) * split_ratio[1])

        splits = {
            'train': data_pairs[:train_split],
            'val': data_pairs[train_split:val_split],
            'test': data_pairs[val_split:]
        }

        # 파일을 적절한 위치로 복사합니다.
        for set_name, files in splits.items():
            for rgb_file, json_file in files:
                shutil.copy(os.path.join(rgb_dir, rgb_file),
                            os.path.join(dest_dir, set_name, 'rgb', category, rgb_file))
                shutil.copy(os.path.join(json_dir, json_file),
                            os.path.join(dest_dir, set_name, 'labels', category, json_file))

    print("데이터 분류가 완료되었습니다잉~")

# 사용 방법
#'Cycle_data/TO/to1_001', 'Cycle_data/TO/to2_001'
#'Cycle_data/TO_label/to1_001', 'Cycle_data/TO_label/to2_001'
rgb_dirs = ['Cycle_data/CA/ca1_001', 'Cycle_data/CA/ca2_001']
json_dirs = ['Cycle_data/CA_label/ca1_001', 'Cycle_data/CA_label/ca2_001']
dest_dir = 'Cycle_data_split'  # 데이터를 분할할 대상 경로

create_splits(rgb_dirs, json_dirs, dest_dir)
