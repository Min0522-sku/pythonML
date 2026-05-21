# PythonML Practice 8: SGDClassifier 기반 취업 여부 분류 예측
# 데이터 출처: https://www.kaggle.com/datasets/shambhurajejagadale/student-performance-prediction-dataset
# [1] 데이터 전처리 및 독립/종속 변수 분할
# 제시된 학생 데이터셋에서 취업 여부를 나타내는 범주형 문자열 데이터인 placement_status (Placed, Not Placed)를 종속변수(타깃)로 설정하시오.
# 타깃을 제외한 study_hours, attendance 등 총 6개의 수치형 특성을 독립변수로 지정하고, 전체 데이터를 학습 세트와 검증 세트 비율 8:2로 정확하게 분리하시오.
import pandas as pd
df = pd.read_csv('./day04/student.csv')
student_full = df[['study_hours','attendance','sleep_hours','internet_usage','assignments_completed','previous_score']]
student_target = df['placement_status'].values
from sklearn.model_selection import train_test_split
train_input, test_input, train_target,  test_target = train_test_split(student_full, student_target, test_size=0.2, random_state=42)
# [2] 특성 다항 확장 및 경사하강법 필수 스케일링
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
# [3] 로그손실 또는 힌지손실 기반의 규제 하이퍼파라미터
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log_loss', random_state=42, max_iter=100, tol=None) 
sc.fit(train_scaled, train_target)

# [4] 최고 정확도(Score) 선정 , *0.90 이상 찾기* 
print(sc.score(test_scaled, test_target))
# [5] 신규 학생 데이터 취업 예측 (추론 함수 구현 및 검증)
# 구현된 모델에 아래 두 가지 샘플 데이터를 입력하여  취업 성공(Placed)과 실패(Not Placed)를 올바르게 분류해내는지 최종 검증하시오.
#  - 샘플 A : study_hours=9, attendance=95, sleep_hours=7, internet_usage=2, assignments_completed=18, previous_score=85
#  - 샘플 B : study_hours=2, attendance=60, sleep_hours=5, internet_usage=9, assignments_completed=4, previous_score=50
a = [[9,95,7,2,18,85]]
b = [[2,60,5,9,4,50]]
a_scaled = ss.transform(a)
b_scaled = ss.transform(b)
print(sc.predict(a_scaled))
print(sc.predict(b_scaled))
