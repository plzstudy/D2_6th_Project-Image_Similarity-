# Data Preprocessing ( Team. NMSP )

import os
import re
import numpy as np

import skimage.io as io
import cv2



IMG_EXT = "jpg"  # 이미지 확장자
IMG_DIR = 'image' # 이미지 폴더

re_model = re.compile("^(\d+)_")
img_paths = os.listdir(IMG_DIR)

img_paths.sort()
labels_true = [re_model.match(img_path).group(1) for img_path in img_paths]


set_size=len(img_paths) # 전체 데이터로 설정.

X_set=np.empty((set_size,32,32,3),dtype=np.float32) # Data
y_set=np.empty((set_size,1),dtype=np.uint8) # Target

# Error 1. OverflowError : 타겟이 너무 길어서 에러 발생. 타겟을 임의로 설정해줘야 함.

labels=list(set(labels_true)) # 중복 제거한 라벨
labels.sort()


labels2=[] # 중복 제거한 라벨을 0~153까지의 값으로 만들기 위해 이용.

for x in range(len(labels)):
    labels2.append(x)


for i, filename in enumerate(img_paths):
    file_path=os.path.join(IMG_DIR,filename)
    img=io.imread(file_path)
    img=cv2.resize(img,(32,32)).astype(np.float32)
    X_set[i]=img
    
    #y_set[i]=labels_true[i] 이와 같이 사용했을 때, OverflowError 발생.
    
    for y in range(len(labels)):
        if labels_true[i]==labels[y]: # labels_true[i] 값이 중복 제거 된 라벨값에 포함이 될 경우
            y_set[i]=labels2[y]

# 전처리 한 데이터와 라벨을 '.npy' 파일로 저장.
np.save("entire_data.npy",X_set)
np.save("entire_label.npy",y_set)
