import numpy as np
# 各種類似手法の読み込み
import dist_repre
import set_similar
import topic
# 定数読み込み
from sample_vector import *


print("分散表現の類似指標（点と点の距離）")
print("===========適合文書=============")
print("コサイン類似度：", dist_repre.cos_sim(SHIKAKU, CORRECT_DOCUMENT))
print("ユークリッド距離", dist_repre.euclidean_dist(SHIKAKU, CORRECT_DOCUMENT))
print("ミンコフスキー距離", dist_repre.minkowski_dist(SHIKAKU, CORRECT_DOCUMENT))
print("マンハッタン距離", dist_repre.manhattan_dist(SHIKAKU, CORRECT_DOCUMENT))
print("チェビシフ距離", dist_repre.chebyshev_dist(SHIKAKU, CORRECT_DOCUMENT))
print("ハミング距離", dist_repre.hamming_dist(SHIKAKU, CORRECT_DOCUMENT))

print("===========非適合文書=============")
print("コサイン類似度：", dist_repre.cos_sim(SHIKAKU, INCORRECT_DOCUMENT))
print("ユークリッド距離", dist_repre.euclidean_dist(SHIKAKU, INCORRECT_DOCUMENT))
print("ミンコフスキー距離", dist_repre.minkowski_dist(SHIKAKU, INCORRECT_DOCUMENT))
print("マンハッタン距離", dist_repre.manhattan_dist(SHIKAKU, INCORRECT_DOCUMENT))
print("チェビシフ距離", dist_repre.chebyshev_dist(SHIKAKU, INCORRECT_DOCUMENT))
print("ハミング距離", dist_repre.hamming_dist(SHIKAKU, INCORRECT_DOCUMENT))

print("")

print("トピック分布の類似指標（トピックの類似度）")

print("===========適合文書=============")
print("コサイン類似度", topic.cos_sim(SHIKAKU_TOPIC, CORRECT_DOCUMENT_TOPIC))
print("KullbackLeibler距離", topic.kld(SHIKAKU_TOPIC, CORRECT_DOCUMENT_TOPIC))
print("Jensen-Shannon-Divergense", topic.jsd(SHIKAKU_TOPIC, CORRECT_DOCUMENT_TOPIC))
print("Hellinger距離", topic.hellinger(SHIKAKU_TOPIC, CORRECT_DOCUMENT_TOPIC))
print("HistgramIntersection", topic.hist_intersection(SHIKAKU_TOPIC, CORRECT_DOCUMENT_TOPIC))

print("===========非適合文書=============")
print("コサイン類似度", topic.cos_sim(SHIKAKU_TOPIC, INCORRECT_DOCUMENT_TOPIC))
print("KullbackLeibler距離", topic.kld(SHIKAKU_TOPIC, INCORRECT_DOCUMENT_TOPIC))
print("Jensen-Shannon-Divergense", topic.jsd(SHIKAKU_TOPIC, INCORRECT_DOCUMENT_TOPIC))
print("Hellinger距離", topic.hellinger(SHIKAKU_TOPIC, INCORRECT_DOCUMENT_TOPIC))
print("HistgramIntersection", topic.hist_intersection(SHIKAKU_TOPIC, INCORRECT_DOCUMENT_TOPIC))


