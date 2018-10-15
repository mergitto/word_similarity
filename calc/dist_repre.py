import numpy as np
from scipy.spatial.distance import euclidean, minkowski, chebyshev, cityblock, hamming

def _validate_vector(u, dtype=None):
    # XXX Is order='c' really necessary?
    u = np.asarray(u, dtype=dtype, order='c').squeeze()
    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u

# コサイン類似度
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ユークリッド距離: 一般的な距離
#euclidean(v1, v2) # scipyを使う場合
def euclidean_dist(v1, v2):
    return np.linalg.norm(v1 - v2)

# ミンコフスキー距離: Euclid, Manhattan, Chebyshevを一般化したもの
#minkowski(v1, v2, 2) # scipyを使う場合
def minkowski_dist(v1, v2, p=2):
    v1 = _validate_vector(v1)
    v2 = _validate_vector(v2)
    if p < 1:
        raise ValueError("p must be at least 1")
    v1_v2 = v1 - v2
    dist = np.linalg.norm(v1_v2, ord=p)
    return dist

# マンハッタン距離: 外れ値の影響を受けにくい
#cityblock(v1, v2) # scipyを使う場合
def manhattan_dist(v1, v2):
    v1 = _validate_vector(v1)
    v2 = _validate_vector(v2)
    l1_diff = abs(v1 - v2)
    return l1_diff.sum()

# チェビシフ距離: 成分の差がもっとも大きい次元だけを抽出している
#chebyshev(v1, v2) # scipyを使う場合
def chebyshev_dist(v1, v2):
    v1 = _validate_vector(v1)
    v2 = _validate_vector(v2)
    return max(abs(v1 - v2))

# ハミング距離: ベクトルの要素中で一致していない要素数。カテゴリ変数に利用
#hamming(v1, v2) # scipyを使う場合
def hamming_dist(v1, v2):
    v1 = _validate_vector(v1)
    v2 = _validate_vector(v2)
    if v1.shape != v2.shape:
        raise ValueError('The 1d arrays must have equal lengths.')
    v1_ne_v2 = v1 != v2
    return np.average(v1_ne_v2)

# 性能評価：PCAで入力と対象のベクトル和の上位2軸のプロット
