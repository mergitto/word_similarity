import numpy as np

#資格

SHIKAKU = np.asarray([
    0.42440349, 0.19072372, 0.099718586, -0.77085632, 0.16929626, 0.037787568, 0.29073009, -0.60986996, -0.053409051, 0.15223014,
    -0.4713468, -0.11732099, 0.1051048, -0.018203847, -0.22121894, 0.30620456, -0.52633685, -0.021409864, -0.22447515, -0.46648845,
    0.31003597, 0.11529651, -0.088379234, -0.15124038, 0.10020568, 0.71223253, 0.16819023, 0.3386583, -0.036085963, 0.52155113,
    0.16625008, 0.29046008, 1.0637453, 0.035186503, -0.27256778, -0.28778565, -0.07470046, -0.042177122, 0.60535085, 0.23258251,
    -0.11825229, -0.33874428, 0.4396432, -0.28831384, 0.49067006, -0.12783441, -0.21125489, -0.23436777, 0.24841367, -0.11601754,
    0.069074944, -0.1832916, -0.24743652, 0.6768468, 0.39374474, -0.35317862, 0.3815991, 0.5443055, 0.20078449, 0.20941895,
    -0.14501379, -0.033239149, -0.37616226, 0.098085329, 0.4157615, 0.17935899, -0.12494193, 0.49906084, -0.67993087, 0.11831599,
    -0.69798195, 0.025505401, -0.70000339, -0.3056469, 0.13950793, -0.18078308, -0.17298366, 0.038650803, -0.032814167, -0.52125651,
    0.11812772, 0.14842924, 0.0026944645, -0.031679679, 0.43843377, -0.57849944, 0.046856761, 0.17500384, 0.24977317, 0.21578297,
    0.23719518, 0.26466942, -0.066459574, -0.20302618, 0.33222714, -0.30841351, 0.14726992, 0.067696139, -0.40575948, -0.40840694], dtype=np.float64)

SHIKAKU_TOPIC = np.asarray([0.74756385, 0.25243615], dtype=np.float64)


# 適合文書
#['資格', '取得', 'する', 'いる', 'こと', '面接', 'アピール', 'できる', 'いる', '資格', '取得', '頑張る', '就職活動', '少し', '楽', 'なる', '思う', '面接', '自分', '思う', 'こと', '素直', '話す', 'こと', 'できる', 'うまく', '行ける', '思う']

CORRECT_DOCUMENT = np.asarray([
    -4.4284272, 1.2240541, 7.118248, -3.9558785, -5.1913099, -8.1822281, 7.5250463, -5.3542919, 1.1559004, -2.8991208,
    -2.5484195, 2.4881036, 2.3896508, 1.7197171, -7.1458025, -1.8726517, 2.2671256, -6.6040249, -0.63311458, 4.3721499,
    2.560966, -0.35184473, -0.037856475, -3.7068818, -8.6053762, 3.1089146, -0.74204147, 7.440958, -5.3968248, -8.1050463,
    -1.7976186, -3.207339, 0.29899788, -4.7033334, -2.4067235, 1.1728661, -4.2832026, -1.5049555, 3.9416747, -4.9118152,
    -1.8485922, -0.96148723, 0.47391108, 5.0042567, -0.82545954, -3.4346776, -4.0665016, -4.0721869, -3.633363, 2.7490971,
    6.3051023, -8.4679708, -7.6759415, 6.6509829, -0.65691805, 1.9770285, 3.2047822, -0.69657069, -0.84543812, 7.5510936,
    -2.5935223, 4.6235771, -7.991375, 2.8637707, 0.20224571, -4.6286092, 3.0013278, -0.11378777, -2.140002, -2.2823734,
    0.91549605, -0.50571305, -2.6691849, -3.4841473, -6.325624, 1.3585474, -3.0239174, -4.2619739, 3.9722695, 0.36152133,
    -7.1675467, -2.6194646, -1.330014, -2.8042595, -0.93511426, 4.9033017, 0.32092962, 3.0493414, 1.3588886, -5.4558673,
    -3.3017969, 1.2805009, 2.2047324, -1.2614541, 3.1165285, 3.797648, 5.243835, -0.087320089, 3.3242714, -0.8379913], dtype=np.float64)

CORRECT_DOCUMENT_TOPIC = np.asarray([0.51462873422426281, 0.48537126577573719], dtype=np.float64)