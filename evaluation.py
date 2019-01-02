# 생성한 두 true, pred txt 파일을 통한 정확도 측정 ( Team. NMSP )

import os
import numpy as np
from sklearn import metrics

def evaluation(labels_true_path, labels_pred_path):

    labels_true = np.loadtxt(labels_true_path, dtype=str)
    labels_pred = np.loadtxt(labels_pred_path, dtype=str)
    
    # compare labels
    return metrics.adjusted_rand_score(labels_true, labels_pred)


if __name__ == '__main__':
    
    LABEL_DIR="Group_Label" # 라벨 생성 경로
    
    
    score1 = evaluation(os.path.join(LABEL_DIR,"label_true.txt"), os.path.join(LABEL_DIR,"label_pred.txt"))
    
    print("Score for %s: %s" % ("label_pred.txt", score1))