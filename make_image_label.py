# 이미지 및 라벨을 생성하기 위한 코드 ( Team. NMSP )

import numpy as np
import cv2
import os


DATA_DIR = "Group_Image" # 데이터(이미지) 생성 경로
LABEL_DIR="Group_Label" # 라벨 생성 경로

if os.path.exists(DATA_DIR) is False:
    os.makedirs(DATA_DIR) # 이미지 경로 폴더 생성

if os.path.exists(LABEL_DIR) is False:
    os.makedirs(LABEL_DIR) # 라벨 경로 폴더 생성

# 이미지 생성 및 label 파일을 생성하기 위해 Accuracy 95% 이상인 데이터 셋을 불러온다.
label_true=np.load("label_true.npy")
label_pred=np.load("label_pred.npy")
data_true=np.load("data_true.npy")
one_hot_label_true=np.load("one_hot_label_true.npy")


# 이미지 생성
for x in range(len(data_true)):
    cv2.imwrite(DATA_DIR+"/"+str(x)+".jpg",data_true[x])


label_t=[]
label_p=[]
for x in range(len(label_true)):
    label_t.append(str(label_true[x][0]))
    label_p.append(str(label_pred[x][0]))

# 라벨 생성
with open(os.path.join(LABEL_DIR,"label_true.txt"), 'w') as f:
    f.writelines([line + "\n" for line in label_t])

with open(os.path.join(LABEL_DIR,"label_pred.txt"), 'w') as f:
    f.writelines([line + "\n" for line in label_p])

