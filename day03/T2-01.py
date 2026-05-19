# 모델 : 데이터(자료)를 학습하는 프로그램/라이브러리(사이킷런) 
# 학습 : 데이터(자료)의 규칙 찾는 과정
# 예측 : 학습된 모델로 새로운 데이터(결과) 추론 과정
# 특성 : 학습에 입력되는 정보
# 타깃 : 학습에 정답되는 정보
# 표준화(스케일링) : 0~1 사이로 크기 맞춤
    # StandardScaler() # .transform()
# 과소적합 : 너무 단순한 경우 # 이웃이 너무 많아서 기준 애매 모호
# 과대적합/과적합 : 너무 암기된 경우 # 이웃이 너무 적어서 특정 이웃 학습
# -----------------------------------------------------------------------
# K-NN :  가까운 이웃 기준의 예측
    # KNeighborsClassifier() : k최근접이웃 분류
    # KNeighborsRegressor() : k최근접이웃 회귀
        # 하이퍼파라미터(K) : 이웃개수(K) 직접 설정하여 최적의 k찾기
        # 학습특성의 형태는 2차원 배열만 가능,
            # T1-01 : zip 활용 , T1-02 column_stack 활용, T1-03 reshape 활용
# -----------------------------------------------------------------------


import pandas as pd
df = pd.read_csv('./day01/Fish.csv')
# 숭어의 길이 무게
perch_df = df[df['Species'].isin(['Perch'])]
perch_length = perch_df['Length2'].values
perch_weight = perch_df['Weight'].values

# 훈련세트와 테스트세트 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target,  test_target = train_test_split(perch_length, perch_weight, test_size=0.2, random_state=42)

# 2차원 배열화
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# k-최근접 이웃 회귀 모델 훈련
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))

# 임의의 값으로 예측
print(knr.predict([[50]])) # [1010.]
print(knr.predict([[100]])) # [1010.]

# 문제점 : k-최근접 이웃의 문제점은 단순한 주변 이웃의 평균으로 예측하기 때문에 최댓값을 벗어나면 항상 동일한 값으로 예측함
# 즉] 소규모 또는 간단한 예측 프로그램 에서만 사용된다.

# 선형 회귀 모델
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.score(test_input, test_target))
print(lr.predict([[50]])) # [1238.3175398]
print(lr.predict([[100]])) # [3191.00026354]

# 직선공식(1차방적식) : y(예측) = w(가중치)x(특성) + b(절편)
# 즉] (물고기)무게 = 가중치 * (물고기)길이 + 절편
print(lr.coef_) # 기울기값 반환 [39.05365447] # 직선의 기울기(특성의 가중치)
    # 기울기(가중치) 공식 : x와 y의 편차 곱의 합 / x 의 편차 제곱합
print(lr.intercept_) # y절편 반환 -714.3651839448922 # 편향 # x(물고기길이) 가 0 일때 y의 값
    # y절편 공식 : y평균 - ( 기울기 * x의 평균 )

# x와 y 가 직선 관계 이며 실 자료들은 물고기가 길이 1씩 증가할 때 무게가 꼭 비례 증가 하지 않는다. < 애매하다 >
# 즉] 초반에는 길이에 따라 무게가 3배 증가 하다가 중/후 반에는 무게가 2/1배 증가 할 수 있다. # 사람 키( 어릴때 키가 자라고 나이들면 어느정도 고정 )

import matplotlib.pyplot as plt
plt.scatter(train_input, train_target)
plt.scatter(50, 1238) # 무게 50 일때는 길이 1238 일것이다
plt.scatter(100, 3191)  # 무게 100 일때는 3191 일것이다
plt.plot([15, 100], lr.predict([[15], [100]])) # 회귀선 그리기 # 길이의 시작점, 길이의 끝점
plt.show()
print(lr.score(test_input, test_target)) # 단순 선형 평가 0.8359630155975616

# [3] ( 다항 : 여러개 항 ) 선형회귀 모델 # 2차 방정식 
# 직선 공식( 1차 방정식) : Y(예측) = W(가중치) x X(특성) + B(Y절편)
# 곡선 공식( 2차 방정식 ) : Y(예측) = ( W(가중치) x X(특성)제곱 ) + ( W(가중치) X(특성) ) + B(Y절편)
# x(특성) 제곱 : 물고기 '길이' 에 제곱   * 최적의 제곱수 찾아서 정확도 최적화 한다. *
# x(특성) : 물고기 '길이'
# 가중치 : 기울기 
# 절편 : y절편/편향 
# 즉] x제곱 항목이 추가되면서 그래프가 U 또는 곡선 모양으로 나온다. 길이가 커질수록 무게는 뻥튀기되는 효과
import numpy as np
train_poly = np.column_stack((train_input**2, train_input)) # 길이제곱, 길이
print(train_poly) 

lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50**2, 50]])) # [1579.0440311]
#여러개 예측
point = np.arange(15, 50)
point_poly = np.column_stack((point**2, point))

plt.scatter(train_input, train_target) # 특성 자료
plt.plot(point, lr.predict(point_poly)) # 여러개 예측 시각화
plt.show()
test_poly = np.column_stack((test_input**2, test_input))
print(lr.score(test_poly, test_target)) # 다항 회귀 평가 0.9801885585527479