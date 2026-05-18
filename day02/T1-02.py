
import pandas as pd

df = pd.read_csv('./day01/Fish.csv')

# 필요한 어종 추출 : 조건식 대신에 .isin() 특정값만 추출, .isna() 결측치만 추출
target_fish = df[df['Species'].isin(['Bream', 'Smelt'])]
# 필요한 특성 추출 : Length2, Weight
import numpy as np
# zip 함수 대신에 2차원 리스트 구성
fish_data = np.column_stack((target_fish['Length2'], target_fish['Weight']))
# concatenate((리스트, 리스트)) : 두 리스트 연결
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

# 학습 모델 만들기 전 학습용, 테스트용 분리
from sklearn.model_selection import train_test_split
# train_test_split(학습자료, 정답지, test_size = 테스트자료 비율)
# 4개의 반환 타입을 갖는다.
train_input, test_input, train_target,  test_target = train_test_split(fish_data, fish_target, test_size=0.3) # 학습용 7 : 테스트용 3 비율로 분할
print(train_input.shape) # (34, 2) 49개 중 학습용 7에 해당하는 개수가 34개
print(test_input.shape) # (15, 2) 49개 중 테스트용 3에 해당하는 개수가 15개

# 학습 모델 : k-최근접 이웃 분류기 모델
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target) # 모델 학습
print(kn.score(test_input, test_target)) # 모델 평가

# 임의의 값으로 학습 모델 예측하기
print(kn.predict([[25,150]])) # 문제

# 예측값 시각화
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1]) # 학습용
plt.scatter(25, 150) # 예측값
plt.show()

# 예측하기 위한 이웃들 확인,  .kneighbors([예측값]):예측에 사용된 이웃(거리, 인덱스)들을 반환
dict, indexs = kn.kneighbors([[25, 150]])
plt.scatter(train_input[:,0], train_input[:,1]) # 학습용
plt.scatter(25, 100) # 예측값
plt.scatter(train_input[indexs, 0], train_input[indexs, 1]) # 문제 발견
plt.show()


# 포준화 필요성 : 공정하게 크기 단위 맞추는 작업 -> 길이와 무게 값의 차이가 커서 일관된 비교가 어렵다.
# 컴퓨터는 숫자가 더 큰 걸 더 중요하게 생각한다.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_input)
print(scaler.mean_) # 평균
print(scaler.scale_) # 표준편차
train_scaled = scaler.transform(train_input) # 표준화(스케일링)
print(train_scaled)

# 스케일링 이후 시각화, 차트 모양의 차이는 없지만 단위가 표준화 됨
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.show()

# 스케일링 이후 재 학습 모델 만들기
kn.fit(train_scaled, train_target) # 표준화된 자료로 재학습
# 임의의 예측값(스케일링된)
new_scaled = scaler.transform([[25, 150]])
print(kn.predict(new_scaled)) # 스케일링(표준화) 전에는 0, 이후에는 1 예측됨

# 예측에 사용된 이웃들 확인
dist, indexs = kn.kneighbors(new_scaled)
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new_scaled[:,0], new_scaled[:,1])
plt.scatter(train_scaled[indexs, 0], train_scaled[indexs, 1])
plt.show()

# 차트 비교하기