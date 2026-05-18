import pandas as pd

df = pd.read_csv('./day01/Fish.csv')

# Perch(농어) 만 추출
target_fish = df[df['Species'].isin(['Perch'])]
target_fish.info()

# 농어의 길이/무게 추출
perch_length = target_fish['Length2'].values
perch_weight = target_fish['Weight'].values

# 농어 길이에 따른 무게 예측
import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight)
plt.show()

# 학습 모델 만들기, 준비
from sklearn.model_selection import train_test_split
# train_test_split(학습자료, 정답자료, test_size=분리비율, random_state=분리기준난수)
# random_state = 분리할때 사용되는 난수값, 난수값에 따라 분리한다, 고정값 넣어주면 항상 동일한 분리 값 넣을 수 있다. 0~32억 사이
train_input, test_input, train_target,  test_target = train_test_split(perch_length, perch_weight, test_size=0.3, random_state=42)

# 준비 : 자료형식(모양) 구성, 대부분 2차원 사용
import numpy as np
print(train_input.shape) # (39,) 1차원 배열 -> 사이킷런 모델들은 1차원배열 학습이 불가능함
# reshape(행개수, 열개수): 행개수에는 -1 넣어서 자동(자료개수만큼), 열개수는 1개
train_input = train_input.reshape(-1, 1)
print(train_input.shape) # (39,1) 2차원 배열 
# train_target = train_target.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
# 모델 학습
from sklearn.neighbors import KNeighborsClassifier # k최근접이웃 모델 찾기
from sklearn.neighbors import KNeighborsRegressor # k최근접이웃 회귀 모델

knr = KNeighborsRegressor()
knr.fit(train_input, train_target) # 길이, 무게 길이에 따른 무게 학습
print(knr.score(test_input, test_target)) # 모델 평가 회귀모델에서는 결정계수 라고 한다
print(test_input) # 모델 예측할 값(길이)
print(knr.predict(test_input)) # 모델(무게) 예측

# k최근접이웃 회귀는 이웃의 평균으로 예측한다. 하이퍼라미터(k) 조절
# k = 이웃 개수 정하기
# 임의의 물고기 길이 5부터 45까지 생성
x = np.arange(5, 45).reshape(-1, 1)
for k in [1, 3, 5, 10]: # 이웃 개수를 3가지(1,3,5,10) 모델 학습
    knr.n_neighbors = k # 현재 모델의 이웃개수 대입
    knr.fit(train_input, train_target) # 총 4번 학습 예정
    print(knr.score(test_input, test_target)) # 총 4번 학습 평가
    
    pred = knr.predict(x) # 임의의 값으로 예측
    print(pred) # 총 45개의 물고기 길이의 몸무게 예측한다
    plt.scatter(train_input, train_target)
    plt.plot(x, pred) # 선차트 이면서 회귀(예측)선 x = 길이 pred = 몸무게(예측)
    plt.title(f'k = {k}' )
    plt.show()

# k 는 이웃개수, k최근접 회귀는 이웃의 평균으로 예측한다
#  k 가 1일때 0.9918926744767643        # 특정한 자료에 튀는 데이터(노이즈/이상치)까지 적용될 수 있으므로 예측이 망가질 수 있다. # 과대적합훈련
#  k 가 3일때 0.9766857219041255
#  k 가 5일때 0.9929281790592219
#  k 가 10일때 0.9742254836937329       # 많은 자료에 둔감하고 단순한 된 자료까지 적용될 수 있으므로 예측이 망가질수 있다. # 과소적합훈련

# k 가 5일때 가장 균형적인 추세 표현 회귀선이 너무 꺽이거나 완만한 일직선이 아니다.
# 결론] 머신러닝 에서는 가장 최적의 파리미터 찾는 과정을 튜닝이라고 한다 
