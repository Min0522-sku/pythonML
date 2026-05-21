# PythonML Practice9: 패션 구매 카테고리 예측 시스템 구축 ( SGD )
# 데이터 출처: https://drive.google.com/file/d/1B5pVnqexkvh4D2l5c542HFaKmPjT5Iab/view?usp=sharing
# [상세 요구사항]
# 1. 고객의 인적 정보 및 구매 패턴 데이터를 데이터베이스(DB)에 적재 및 관리한다.
# 2. 관리자가 Spring Boot REST API를 호출하여, 데이터베이스에 저장된 전체 데이터를 기반으로 예측 모델을 (재)학습하고 최신화할 수 있어야 한다.
# 3. 일반 사용자가 Spring Boot REST API를 호출하여 4개 핵심 변수(나이, 성별, 유입 채널, 선호 스타일)를 입력하여 요청하면, 모델이 예측한 구매 옷 카테고리(0~7)를 실시간으로 반환한다.
# # 학습된 예측 모델의 예측 정밀도를 보장하기 위해 평가지표인 정확도가 최소 90% 이상(0.90 이상)을 달성해야 한다.
# # 프론트엔드 대신 Talend API를 사용한다.
# # REST API는 Spring API 2개와 FastAPI 2개로 구성한다.
# # 관리자와 일반 사용자는 FastAPI에 직접 접근하지 않으며, 모든 요청은 Spring API를 통해 처리한다.
# # 2인 ( 1:스프링 2:파이썬 )


import uvicorn
from fastapi import FastAPI

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True)

import controller
app.include_router(controller.router)