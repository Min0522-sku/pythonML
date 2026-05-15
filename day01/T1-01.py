
import pandas as pd

df = pd.read_csv('./day01/Fish.csv')
print(df.head())
df.info()
# 특정 물고기 추출, 도미/Bream
bream_df = df[df['Species'] == 'Bream']

# 도미의 길이, 무게 추출 df['열이름'].tolist()
bream_length = bream_df['Length2'].tolist()
bream_weight = bream_df['Weight'].tolist()
print(bream_length, bream_weight)

# 특정 물고기 추출, 빙어/Smelt
smelt_df = df[df['Species'] == 'Smelt'] 
smelt_length = smelt_df['Length2'].tolist()
smelt_weight = smelt_df['Weight'].tolist()

# 시각화
import matplotlib.pyplot as plt
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel("length(cm)")
plt.ylabel("weight(gram)")
plt.show()

# 도미와 빙어 자료 합치기
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 2차원 리스트
# zip(1차원리스트, 1차원리스트) : 두리스트를 요소 하나씩 반복
# 리스트 내포
fish_data = [[l,w]for l, w in zip(length, weight)]
print(fish_data)
# 도미 35마리 빙어 14마리

# target (정답지) 만들기, 1: 도미 의미하고 35개 만든다. 0 : 빙어 의미하고 14개 만든다
fish_target = [1]*35+[0]*14

# 알고리즘 모델 중: k-최근접 이웃(k-nn알고리즘) : 임의값을 넣었을 때 기존 값들 중에 가장 가까운 값 찾기
# 설치 : 사이킷런
# pip install scikit-leanr
# K-NN 모델 호출
from sklearn.neighbors import KNeighborsClassifier
# K-NN 모델 객체 생성
kn = KNeighborsClassifier()
# k-nn 학습하기, 문제와 답을 같이준다. -> 지도학습 : 문제와 정답을 알려주면
# 컴퓨터에게 미리 문제(자료) 제공 하고 그 문제에 따른 답(자료) 제공 하므로써 기억한다.
kn.fit(fish_data, fish_target) 

# 학습된 모델의 점수(정확도) 측정 kn.score(문제, 답), 0~1 사이값으로 반환, 1: 100점
# 컴퓨터에게 또 다른 문제(자료) 제공, 답(자료) 제공하여 체점하기
print(kn.score(fish_data, fish_target))

# 임의의 값 넣어서 예측 측정, kn.predict([임의값])
print(kn.predict([[30,600]])) # 임의의 물고기 길이와 무게 -> 도미인지 빙어인지 예측한다

# 임의의 값 시각화
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600)
plt.xlabel("length(cm)")
plt.ylabel("weight(gram)")
plt.show()

# 근접한 이웃 찾을 기준 정하기, 하이퍼파라미터(k값 조절)
# KNeighborsClassifier(n_neighbors= 참조할 이웃개수) 접근한 갯수 중에서 정답 찾기
kn = KNeighborsClassifier(n_neighbors= 49)
kn.fit(fish_data, fish_target)
print(kn.score(fish_data, fish_target))