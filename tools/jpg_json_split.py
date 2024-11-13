import os
import shutil

# 분리할 폴더 경로 설정
source_folder = 'PE2_PB1/001'  # 여기에 분리할 폴더 경로를 입력하세요.
json_folder = os.path.join(source_folder, 'jsons')
jpg_folder = os.path.join(source_folder, 'jpgs')

# jsons 및 jpgs 폴더가 없으면 생성
os.makedirs(json_folder, exist_ok=True)
os.makedirs(jpg_folder, exist_ok=True)

# 폴더 내 파일을 분리
for filename in os.listdir(source_folder):
    # 경로 확인
    file_path = os.path.join(source_folder, filename)
    
    # 파일이 JSON일 경우 jsons 폴더로 이동
    if filename.endswith('.json'):
        shutil.move(file_path, os.path.join(json_folder, filename))
    
    # 파일이 JPG일 경우 jpgs 폴더로 이동
    elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
        shutil.move(file_path, os.path.join(jpg_folder, filename))

print('파일 분리가 완료되었습니다!')
