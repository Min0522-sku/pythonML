import pandas as pd
df = pd.read_csv('./day01/Fish.csv')

# 특성 6개
fish_input = df[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]
# 어종 7개
fish_target = df['Species']

# 훈련 테스트 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target,  test_target = train_test_split(fish_input, fish_target, test_size= 0.3, random_state=42)
# 스케일링
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 확률적인 경사 하강법
# fit() 모델 학습에서는 정답(target)도 같이 학습중, 예측(y)값 과 실제 정답간의 오차(기울기/가중치 조절 오차 줄이기) 측정
# 산꼭대기에서 내려가는 방법중에 가장 최적의 경로로 내려오는 방법 = 경사하강법(수많은 경우의수 계산하여 판단)
# 전통 경사 하강법(정확도가 좋지만 학습속도가 느리다) vs 확률 경사 하강법(SGD: 정확도가 낮지만 학습속도가 빠르다 : 미니배치)

# 로그 로스 / 손실 함수, 손실(예측과 정답의 전체 차이)
# 로그 로스 함수는 0과 1의 확률 값이 아닌 오차 값을 측정

# SGDClassifier, 분류모델
from sklearn.linear_model import SGDClassifier
# loss = 'log_loss'
# random_state : SGD가 전체 데이터 학습이 아닌 일부 자료(미니배치) 가지고 학습 하는데 사용되는 분리 기준(난수값)
# max_iter : (반복)계산 횟수 # 미니배치 이므로 전체 데이터셋을 '10' 이면 10번 반복 학습하여 모델 성공 향상 / 최적의 정확도에서 멈춤 (에포크)
# tol=None : 최적의 정확도를 찾아도 계속 반복학습 설정
sc = SGDClassifier(loss='log_loss', random_state=42, max_iter=10, tol=None) 
sc.fit(train_scaled, train_target)
print(sc.score(test_scaled, test_target))
print(sc.predict(test_scaled[:3]))

# 점진적(부분) 학습 (중간에 일부 학습 가능하다.)
sc.partial_fit(train_scaled, train_target) # (위에서 이미 학습된 모델에 10번) + 1번 학습 = 11번 학습
print(sc.score(test_scaled, test_target))

# 최적의 학습횟수(에포크) 찾기
sc = SGDClassifier(loss='log_loss', random_state=42) # max_iter 생략시 1학습

train_score = [] # 학습용 정확도
test_score = [] # 테스트용 정확도

# 
import numpy as np
classes = np.unique(train_target)
for i in range(0, 150): # 150번 반복
    sc.partial_fit(train_scaled, train_target, classes= classes)

    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

# 정확도 시각화 # 과대적합 과 과소적합을 확인 
import matplotlib.pyplot as plt
plt.plot(train_score) 
plt.plot(test_score)
plt.show()
# 최적의 에포크(반복횟수)는 학습용과 테스트용이 고르게 오르는 시점


# hinge
sc =SGDClassifier(loss='hinge', max_iter=100, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# 로지스틱 회귀 : 확률을 이용한 분류
# SGD(확률 경사하강법/ 미니배치) : loss='log_loss' vs loss='hinge'
# 경사하강법 : 손실(예측과 정답 오차) 0 가깝게 처리 위한 반복 계산 (* 딥러닝)
# loss='log_loss' 
    # 기울기(가중치) 절편으로 수없이 조정하여 확률 100% 만드는 방법 (경사하강법)  
# loss='hinge'
    # 0인 지점인 애매/아슬 한 (경계선) 자료만 가지고 확률 조정 하는 방법  (경사하강법)