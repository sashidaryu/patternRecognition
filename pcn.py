#データの準備
from sklearn import datasets
dataset = datasets.load_iris()
 
#データの整理
target_names = dataset.target_names#ターゲット(花の種類)の名前リスト
targets = dataset.target#ターゲットに与えられた番号
feature_names = dataset.feature_names#特徴名リスト
features = dataset.data#各特徴のデータ


import pandas as pd
from pandas import DataFrame
df = DataFrame(features, columns = feature_names)
df['target'] = target_names[targets]
df.head()

#4次元を表すグラフ
import seaborn as sns
sns.pairplot(df, hue="target")

#tutorial_data=[[1, 2, 1], [2, 3, 1], [3, 5, 1], [2, 2, 1]]
#tutorial_sample=[[-1, -1, 0], [0, 0, 0], [1, 2, 0], [0, -1, 0]]

from sklearn.decomposition import PCA
#4次元のデータを2次元へ
pca2 = PCA(n_components=2)
#pca2.fit(features)
#transformed = pca2.fit_transform(features)#featuresを2次元に変換


test_features=[[7.0, 4.1, 2.1, 2.2], [6.8, 2.8, 1.4, 2.1], [4.9, 2.6, 6.1, 0.4], [7.8, 2.6, 6.8, 1.6], [4.7, 2.2, 5.1, 1.2]]

print(pca2.fit(features).transform(test_features))
