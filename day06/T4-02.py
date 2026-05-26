
import pandas as pd
df = pd.read_csv('./day06/wine.csv')
# alcohol,sugar,pH,class
data = df[['alcohol','sugar','pH']]
target = df['class'] # 1: 화이트와인, 0: 레드와인

from sklearn.model_selection import train_test_split
train_input, test_input, train_target,  test_target = train_test_split(data, target, random_state=42)

from sklearn.tree import DecisionTreeClassifier # 의사결정 트리 분류
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_input, train_target)
print(dt.score(test_input, test_target))

# 교차 검증
from sklearn.model_selection import cross_validate
# cross_validate(학습모델, 학습세트, 정답세트)
# 교차검증은 전체 데이터를  N등분(폴드)하여 돌아가면서 검증한다. 기본값은 5등분
# 데이터를 여러 조각으로 나누어 학습하는 방법
scores = cross_validate(dt, train_input, train_target)
# {'fit_time': array([0.01301837, 0.01252627, 0.01199341, 0.01199412, 0.01297736]), 
# 'score_time': array([0.0030055 , 0.00300717, 0.00200629, 0.00202298, 0.0020051 ]), 
# 'test_score': array([0.85128205, 0.84820513, 0.8788501 , 0.85112936, 0.84394251])}
import numpy as np
print(np.mean(scores['test_score'])) # 5등분 학습의 평균 검증 점수

#
from sklearn.model_selection import StratifiedKFold
# n_splits = N등분 # 데이터를 N등분으로 하여 교차 검중 수행
#
splits = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splits)
print(np.mean(scores['test_score'])) # 10등분 학습의 평균 검증 점수 조금 증가
# {'fit_time': array([0.01300216, 0.01399994, 0.01299977, 0.01299977, 0.01300573,
#       0.01251888, 0.01200008, 0.01300001, 0.01300025, 0.01300001]),
# 'score_time': array([0.00299621, 0.00200009, 0.00300002, 0.00199986, 0.00199986,
#       0.00199986, 0.00199986, 0.00200009, 0.00299978, 0.00200009]), 
# 'test_score': array([0.86680328, 0.84836066, 0.88090349, 0.83983573, 0.8788501 ,
#       0.85420945, 0.84804928, 0.85626283, 0.84804928, 0.86447639])}


# 그리드 서치, 최적의 파라미터(변수/학슴에필요한설정값) 찾기
from sklearn.model_selection import GridSearchCV
# 여러개 '최소불순도' 설정, 
# 임의의 최소 불순도 넣어서 리스트로 구성
params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

# GridSearchCV(트리모델, {파라미터들}, n_jobs= -1)
# n_jobs= -1 : 컴퓨터내 모든 CPU 코어 사용하여 병렬(쓰레드)연산
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print(dt.score(test_input, test_target)) # 0.8670769230769231
print(gs.best_params_) # {'min_impurity_decrease': 0.0003}
print(gs.best_score_) # 0.8731517927657558
# print(gs.cv_results_) # 기본값으로 교차검증 5가 적용된다.

# 다중 파라미터 
params = {
    # 최저 불순도
    'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001), # 0.0001~0.001 (미만까지) 0.0001씩 증가
    # 최대 깊이
    'max_depth' : range(5, 20, 1), # 5~20(미만까지) 1씩 증가
    # 노드 분할시 최저 샘플수 # 최저 샘플수 보다 작으면 노드 분할 안함
    'min_samples_split' : range(2, 100, 10),
    # 리프노드(나머지 뿌리/노드) 최저샘플수, 현재 리프노드가 최저 샘플수 보다 작으면 노드 분할 안함
    'min_samples_leaf' : range(1, 100 , 10)
}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, cv=5) # cv= 교차검증수 기본값 5

# 대략 학습 조합
# 최저불순도(9가지) * 깊이(15가지) * 최저분리샘플(10가지) * 최저리프샘플(10가지) = 대략 13,000 가지 조합 학습
# + 교차검증(N 등분) * 대략 13,000가지 조합 = 6만번의 학습모델
gs.fit(train_input, train_target)
print(gs.best_params_) # 최적의 파라미터 조합
# {'max_depth': 13, 'min_impurity_decrease': np.float64(0.0001), 'min_samples_leaf': 11, 'min_samples_split': 2}
print(gs.best_score_) # 0.8756162796819881

# 랜덤서치
# 조합 수가 많아지면 연산량이 많아져서 서버(컴퓨터)에 부하 발생할 수 있다.
# 고정된 값이 아니라 '확률 분포 함수'를 제공하여 무작위로 숫자를 뽑아 학습한다.
from sklearn.model_selection import RandomizedSearchCV
# n_iter = N # 정의된 조합수에서 무작위(랜덤)으로 N개의 조합만 추출하여 학습한다
# 대략 13,000개 조합에서 100개만 무작위로 추출 # 교차검증5 -> 500번 학습
rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=1, cv=5, random_state=42)
rs.fit(train_input, train_target)
print(rs.best_params_) # {'min_samples_split': 12, 'min_samples_leaf': 11, 'min_impurity_decrease': np.float64(0.0004), 'max_depth': 17}
print(rs.best_score_) # 0.8694571684304744