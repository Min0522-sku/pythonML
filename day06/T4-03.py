import pandas as pd
df = pd.read_csv('./day06/wine.csv')
# alcohol,sugar,pH,class
data = df[['alcohol','sugar','pH']]
target = df['class'] # 1: 화이트와인, 0: 레드와인

from sklearn.model_selection import train_test_split
train_input, test_input, train_target,  test_target = train_test_split(data, target, random_state=42)

# 트리의 앙상블 : 학습한 모델에서 오답들을 서로 상쇄하고 정답을 강화 하여 예측 정확도 높여 과대적합 방지하는 방법 # 여러가지 방법 존재
# 랜덤 포레스트
# 결정트리는 전체특성('alcohol','sugar','pH') 중에 가장 영향력 있는 특성으로 예측 결정하는 방법(한쪽 특성에만 과대적합*)
# 랜덤 포레스트는 모든 특성 사용한다.
    # - 부트스트랩 샘플링 : 전체 훈련 데이터 중에서 무작위로 샘플 선정한다
    # - 무작위 특성 : 전체 특성 중에서 무작위로 샘플 선정한다.
    # 모든 특성들을 사용하여 다양한 트리 구성한다

# obb(Out - of - Bag ) 무작위(중복허용) 선정시 1번도 선정 안된 자료들을 평가용으로 사용
from sklearn.ensemble import RandomForestClassifier
# oob_score=True 무작위 선출에 학습으로 한번도 선정 안된 샘플로 검증한다
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)

from sklearn.model_selection import cross_validate
scores = cross_validate(rf, train_input, train_target)
print(scores) # 'test_score': array([0.88      , 0.90051282, 0.90349076, 0.89014374, 0.88295688])
import numpy as np
print(np.mean(scores['test_score'])) # 0.8914208392565683 # T4-01, T4-02 보다 점수 높음

# 특성 중요도
rf.fit(train_input, train_target)
print(rf.feature_importances_) # [0.23155241 0.49706658 0.27138101] 결정트리보다 조금 더 골고루 분산 되었다

# 분류 모델중에서는 간단한모델은 로지스틱회귀모델
# vs 복잡한모델은 트리모델(+앙상블)

# 엑스트라 트리
# 랜덤포레스트 중복허용한 무작위 샘플/특성 선출
# 엑스트라 트리
    # - 모든 트리가 전체 샘플 자료를 학습한다
    # - 무작위 노드 분리 : 예] sugar 특성을 무작위로 1.4 기준으로 잘라서 분리한다 # 무작위라서 오답이 많이 발생한다
    # 예시] '나이' 특성에 20세~60세 가 존재한 경우 노드분할 예시
    #   Tree(1노드)에서 무작위로 나이 특성을 29세 이상 조건을 만든다 (수학적인 계산이 없어서 빠르다)
    #   Tree(2노드)에서 무작위로 나이 특성을 50세 이상 조건을 만든다
    #  즉] 노드 마다 서로 다른 기준점을 분할 하여 다양성 확보한다. 계산식이 업성서 허슬한 방법이지만 학습 수 와 방대한 양으로 오차 극복 
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, n_jobs=-1)
print(scores) # 'test_score': array([0.89128205, 0.89128205, 0.89938398, 0.88706366, 0.88295688])
print(np.mean(scores['test_score'])) # 0.8903937240035804
# 특성 중요도
et.fit(train_input, train_target)
print(et.feature_importances_) # [0.20702369 0.51313261 0.2798437 ]

# 그레이디언트 부스팅
# 랜덤포레스트 : 중복 허용한 무작위 샘플/특성 선정하여 학습
# 엑스트라트리 : 무작위로 (허슬한/계산식없이) 노드분할 기준 선정 학습
# 그레이디언트 부스팅 : 부모노드(트리)가 예측하고 오차를 자식노드(트리)에게 넘겨 학습
    # - 자식노드가 많아질수록 오차는 줄어든다(과대적합 주의)

    # 예시] Tree(1노드) 에서 실제 정답이 10을 목표로 하여 예측한 결과가 7이면 오차는 3 발생
    #       Tree(2노드) 에서 이제 정답이 10을 목표로 하여 7을 예측한다면 오차에서 1감소한 2추가해 8예측 하면 오차는 2발생
    #       ~~~ 반복해 오차는 0에 가깝게 도달하는 방법
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, n_jobs=-1)
print(scores) # 'test_score': array([0.86461538, 0.87794872, 0.88090349, 0.8613963 , 0.87268994])
print(np.mean(scores['test_score'])) # 0.8715107671247301
# 특성 중요도
gb.fit(train_input, train_target)
print(gb.feature_importances_) # [0.12517641 0.73300095 0.14182264]
# dt(결정트리)/rf(랜덤포레스트/et(엑스트라트리) 보다 뾰족하게 한쪽 특성에 집중된 결과


# 히스트그램 기반 그레이디언트 부스팅
# - 특성 정량화 : 연속적인 구간을 256개의 구간(정수)으로 나누어서 단순화 한다.
# - 분할 기준 : 자식노드를 만들때 256개 구간 기준으로 분할한다. <빠르다>
    # 예] 180, 180.8, 180.3 처럼 소수점 단위의 촘촘히 떨어져 있는 데이터 가정
    # 180~181까지 하나ㅡ이 구간으로 묶어서 계산한다.
    # 미세한 소수점 오차는 과감하게 버린다. 메모리 절약과 속도 향상

from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, n_jobs=-1)
print(scores) # 'test_score': array([0.87179487, 0.89333333, 0.8973306 , 0.85934292, 0.88090349])
print(np.mean(scores['test_score'])) # 0.8805410414363187
# hgb.fit(train_input, train_target)

# 분류 모델

# 사이킷런  앙상블(앞전 계산에 사용된 오차/결과를 다음/전체에 정확도 향상하는데 상쇄 방법)
# 1. 랜덤포레스트 : 샘플/특성 무작위로 선정하여 모델 학습, 튜닝 시간이 부족하거나 베이스 모델 사용
# 2. 엑스트라트리 : 노드분할기준을 무작위로 선정하여 모델 학습, 성능 변동이 있더라도 학습 속도 개선 사용
# 3. 그레이디언트부스팅 : 부모노드에서 오차를 자식노드에게 전달하는 모델 학습, 학습 속도 보다 정교한 모델 사용
# 4. 히스토그램기반 그레디언트부스팅 : 연속된 샘플들을 구간(256) 만들어서 모델 학습, 전처리 시간이 부족하거나 학습 속도 개선 사용 

# 외부 라이브러리 앙상블 (사용 빈도 큼)
# 1. pip install xgboost (캐글 대회에서 나온 알고리즘)
from xgboost import XGBClassifier
xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, n_jobs=-1)
print(np.mean(scores['test_score'])) # 0.883414731743273

# 2. pip install lightgbm (MS 회사에서 나온 알고리즘)
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, n_jobs=-1)
print(np.mean(scores['test_score'])) # 0.8846461327857632

# 3. pip install catboost (IT회사에서 나온 알고리즘)
from catboost import CatBoostClassifier
cat = CatBoostClassifier(random_state=42, verbose=0)
scores = cross_validate(cat, train_input, train_target, n_jobs=-1)
print(np.mean(scores['test_score'])) # 0.8809519296582952