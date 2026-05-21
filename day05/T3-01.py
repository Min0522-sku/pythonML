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


# 로지스틱 회귀 = 이진분류 = 시그모이드 함수(공식)
# 선형 방정식의 출력값을 0과 1(확률/분류) 사이의 값으로 변환해주는 공식/함수
# 예시] 암 환자의 확률 / 스팸 메일 분류 등등 이진 분류 알고리즘 사용한다.
# 즉] 컴퓨터는 수치상의 150 또는 -82.3 (수치)값으로 확률화 하기 어렵다. 확률은 항상 0~1 사이이어야 하기 때문에
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1) # -5 부터 5까지 0.1 씩 증가하는 리스트 # 여기에 어느 시작값과 끝값을 넣더라도 y는 0과1사이
phi = 1 / (1+np.exp(-z)) # 시그모이드 공식
plt.plot(z, phi) # 시그모이드 시각화
plt.show()

# 이진 분류 # 로지스틱 회귀 모델
# 이진 분류는 0 또는 1 분류하는 방법

indexs = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[indexs]
target_bream_smelt = train_target[indexs]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:3])) # 3개만 예측 # ['Bream' 'Smelt' 'Bream']
print(lr.predict_proba(train_bream_smelt[:3])) # [[도미확률, 빙어확률]] # 총합의 확률은 1(100%)
# 임계값은 0.5 기준으로  0.5 이상이면 도미 예측 0.5 미만이면 빙어 예측
# [[0.99793611 0.00206389]
# [0.02391315 0.97608685]
# [0.99575505 0.00424495]]

# 다중 분류 # 로지스틱 회귀
# C : 규제를 완화하여 릿지/라쏘 모델 처럼 정확도 설정 가능 # 모델의 성능 향상 하기 위해서 가중치 값들을 자동 조절
# max_iter : 다중분류 계산 횟수 # 생략시 기본값 100으로 최적의 정확도를 찾을때 까지 계산 반복횟수 조정
lr = LogisticRegression(C=20, max_iter=100)
lr.fit(train_scaled, train_target) # 모든 어종 학습
print(lr.predict(test_scaled[:3])) # ['Perch' 'Smelt' 'Pike']
print(np.round(lr.predict_proba(test_scaled[:3]), 3)) # 분류 개수 만큼의 확률
print(lr.score(test_scaled, test_target)) # 선형 회귀 와 다르게 *결정계수* 라고 하지 않고 맞힌 비율(정확도)반환

# 소프트 맥스
from scipy.special import softmax
decision = lr.decision_function(test_scaled[:3])
print(decision)
print(softmax(decision)) # 소프트맥스 라는 함수로 결과값을 확인 했을때 predict 동일하게 출력된다.
print(np.round(softmax(decision), decimals=3)) # np.round(값, decimal=소수점)
# [[0.    0.001 0.042 0.    0.007 0.    0.   ]
#  [0.    0.002 0.036 0.    0.004 0.69  0.   ]
#  [0.    0.    0.007 0.203 0.002 0.007 0.   ]]
# 다중 분류의 확률 검증할때는 .classes_ 종속 변수들 순서 확인
print(lr.classes_) # 종속변수들 출력 : ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
