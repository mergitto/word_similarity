import numpy as np

# コサイン類似度
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ユークリッド距離
def euclidean(v1, v2):
    return np.linalg.norm(v1 - v2)


# 性能評価：PCAで入力と対象のベクトル和の上位2軸のプロット
