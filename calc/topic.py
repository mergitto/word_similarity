import numpy as np

def kld(p, q):
    p = np.array(p)
    q = np.array(q)
    return np.sum(p * np.log1p(p / q), axis=(p.ndim - 1))

def jsd(p, q):
    p = np.array(p)
    q = np.array(q)
    m = 0.5 * (p + q)
    return 0.5 * kld(p, m) + 0.5 * kld(q, m)

# jsdは非類似度を示し、値が大きいほど非類似度が大きい証明なので、
# その値をもっとも大きい値から引くことで類似度が大きい値を返すように
# する場合は以下の関数を用いる
def value_reverse(dictionary):
    listed = [dic for dic in dictionary.values()]
    for key in dictionary:
        dictionary[key] = max(listed) - dictionary[key]
    return dictionary


# LDA以外のツール：LSI, pLSI, LSA, pLSA

# 類似度指標：コサイン類似度、Jensen-Shannon-Divergense(jsd), Hellinger距離, Kullback-Leibler距離, Symmetric


# 性能評価：BIC情報量を基準としてk-means法を繰り返して最適化(x-means法)


