# PythonML Practice 5: 다항 규제 회귀 기반 성적 예측
# 데이터 출처: https://www.kaggle.com/datasets/shambhurajejagadale/student-performance-prediction-dataset

# [1] 데이터 분할: 범주형 변수를 제외한 6개 특성을 독립변수로, `exam_score`를 타깃으로 설정하고 8:2 비율 로 학습 및 검증 세트를 분리하시오.
import pandas as pd
df = pd.read_csv('./day04/student.csv')
# study_hours,attendance,sleep_hours,internet_usage,assignments_completed,previous_score
# exam_score
student_full = df[['study_hours','attendance','sleep_hours','internet_usage','assignments_completed','previous_score']]
student_target = df['exam_score'].values

from sklearn.model_selection import train_test_split
train_input, test_input, train_target,  test_target = train_test_split(student_full, student_target, test_size=0.2, random_state=42)

# [2] 모델 전수 탐색: `LinearRegression`, `Ridge`, `Lasso` 모델과 다항 확장, 다양한 규제 강도 조합을 모두 학습시키시오.
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
# 최적 모델 찾기 위한 리스트
optimization = [ ]

for degree in [1,2,3,4,5]:
    poly = PolynomialFeatures(degree= degree, include_bias=False)
    poly.fit(train_input)
    train_poly = poly.transform(train_input)
    test_poly = poly.transform(test_input)
    # 선형 회귀
    lr = LinearRegression()
    lr.fit(train_poly, train_target)
    r2 = lr.score(test_poly, test_target)
    #print(f'{degree} 차수의 선형 회귀 결정계수 : {r2}') # 결정계수 : 해당 모델이 예측한 결과가 얼마나 잘 설명 되는지 수치화
    optimization.append({'r2':r2, 'model':lr, 'poly':poly, 'degree':degree, 'scaler':None, 'alpha':None})
    # 스캐일링
    ss = StandardScaler()
    ss.fit(train_poly)
    train_scaled = ss.transform( train_poly ) 
    test_scaled = ss.transform( test_poly )
    # 규제 강도, 릿지vs라쏘, 과적합된 자료들을 최적화
    for alpha in [0.01, 0.1, 1, 10, 100]:
        # 릿지 모델
        ridge = Ridge(alpha= alpha)
        ridge.fit(train_scaled, train_target)
        r2 = ridge.score(test_scaled, test_target)
        #print(f'{degree} 차수의 릿지 강도 : {alpha}의 결정계수 : {r2}')
        optimization.append({'r2':r2, 'model':ridge, 'poly':poly, 'degree':degree, 'scaler':ss, 'alpha':alpha})
        # 라쏘 모델
        lasso = Lasso( alpha= alpha)
        lasso.fit(train_scaled , train_target) 
        r2 = lasso.score(test_scaled, test_target)
        #print(f'{degree} 차수의 라쏘 강도 : {alpha}의 결정계수 : {r2}')
        optimization.append({'r2':r2, 'model':lasso, 'poly':poly, 'degree':degree, 'scaler':ss, 'alpha':alpha})



# [3] 최적 모델 선정: 테스트 데이터셋(`X_test`) 기준 최고의 결정계수를 달성하는 최적의 알고리즘, 차수, 알파 값을 자동 도출하고 추론 엔진에 매핑하시오.
# 리스트에서 결정계수 r2의 가장 큰 모델 찾기
# max(리스트, key= lambda x : x['기준열])
best_optimization = max(optimization, key=lambda x : x['r2'])
best_model = best_optimization['model']
best_poly = best_optimization['poly']
scaler = best_optimization['scaler']
print(f'최적의 모델 : {best_model}, 다항특성 : {best_poly}, 스케일링 : {scaler}')
# 최적의 모델 : Ridge(alpha=10), 다항특성 : PolynomialFeatures(degree=3, include_bias=False), 스케일링 : StandardScaler()

# [4] 추론 함수 구현: 새로운 학생의 6가지 특성 데이터를 인자로 받아 최적 모델의 다항 구조와 스케일링 기준을 거쳐 성적을 예측하는 함수를 구현하시오.
# 추론 함수 구현
def exam_score_predict(study_hours, attendance, sleep_hours, internet_usage, assignments_completed, previous_score): # 매개변수에 *(와일드카드) 한번에 여러 변수들을 튜플로 받을 수 있다. vs 키워드
    list_data = [[study_hours, attendance, sleep_hours, internet_usage, assignments_completed, previous_score]]
    # 특성 공학 적용
    list_poly = best_poly.transform(list_data)
    # 예측 모델이 스케일링 사용할지 여부 판단
    if scaler is not None: # 만약에 스케일링 사용한 모델이면
        list_poly = scaler.transform(list_poly)
    result = best_model.predict(list_poly)
    return result[0]

# [5] 샘플 데이터 검증: 구현된 함수에 두 가지 대조군 샘플을 대입하여 시험성적을 예측하시오.
    # study_hours=9, attendance=95, sleep_hours=7, internet_usage=2, assignments_completed=18, previous_score=85
    # study_hours=2, attendance=60, sleep_hours=5, internet_usage=9, assignments_completed=4, previous_score=50

result = exam_score_predict(9, 95, 7, 2, 18, 85)
print(f'result : {result}점') # result : 97.8779415640775점
result = exam_score_predict(2, 60, 5, 9, 4, 50)
print(f'result : {result}점') 