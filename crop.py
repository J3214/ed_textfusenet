import os
import cv2

# 경로 설정
input_dir = './input_images'
output_dir = './cropped_images'  # 크롭된 이미지 저장 폴더
bbox_dir = './output'            # bbox (좌표) 저장된 txt 파일 폴더

# 저장할 폴더 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# input_images 폴더에 있는 이미지 파일들 순회
for img_file in os.listdir(input_dir):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_name = os.path.splitext(img_file)[0]  # 확장자 제거한 파일 이름
        img_path = os.path.join(input_dir, img_file)
        bbox_path = os.path.join(bbox_dir, f'res_{img_name}.txt')

        # bbox 텍스트 파일이 존재할 때만 진행
        if os.path.exists(bbox_path):
            # 이미지 읽기
            img = cv2.imread(img_path)

            # bbox 파일 읽기
            with open(bbox_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    # 빈 항목 제거하고 숫자만 추출
                    parts = [p for p in line.split(',') if p.strip() != '']
                    if len(parts) >= 4:
                        coords = list(map(int, parts[:4]))  # 앞 4개만 사용
                        x1, y1, x2, y2 = coords

                        # 이미지 크롭
                        cropped = img[y1:y2, x1:x2]

                        # 저장 경로 설정
                        save_path = os.path.join(output_dir, f'{img_name}_crop.jpg')
                        cv2.imwrite(save_path, cropped)
                        print(f'저장 완료: {save_path}')
                    else:
                        print(f'경고: bbox 좌표가 부족합니다 - {bbox_path}')
                else:
                    print(f'경고: {bbox_path} 파일이 비어 있습니다')
        else:
            print(f'경고: bbox 파일이 존재하지 않습니다 - {bbox_path}')
