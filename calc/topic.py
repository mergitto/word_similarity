import numpy as np

# コサイン類似度
def cos_sim(p, q):
    return np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))


# Jensen-Shannon-Divergense
def jsd(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    m = 0.5 * (p + q)
    return 0.5 * kld(p, m) + 0.5 * kld(q, m)

# KullbackLeibler距離
def kld(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(p * np.log1p(p / q), axis=(p.ndim - 1))

# jsdは非類似度を示し、値が大きいほど非類似度が大きい証明なので、
# その値をもっとも大きい値から引くことで類似度が大きい値を返すように
# する場合は以下の関数を用いる
def value_reverse(dictionary):
    listed = [dic for dic in dictionary.values()]
    for key in dictionary:
        dictionary[key] = max(listed) - dictionary[key]
    return dictionary


# Hellinger距離
def hellinger(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    _SQRT2 = np.sqrt(2)

    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2


# HistgramIntersection
def hist_intersection(p, q, bins=100):
    hist_1,_ = np.histogram(p, bins=bins)
    hist_2,_ = np.histogram(q, bins=bins)
    minimum = np.minimum(hist_1, hist_2)

    return np.true_divide(np.sum(minimum), np.sum(q))



# LDA以外のツール：LSI, pLSI, LSA, pLSA



# 性能評価：BIC情報量を基準としてk-means法を繰り返して最適化(x-means法)


