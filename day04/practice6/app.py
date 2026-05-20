# app.py : FastAPI 실행하는 파일
# controller.py : HTTP REST 파일
# service.py : 로직파일



# app.py
import uvicorn
from fastapi import FastAPI

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True)


# 라우터 : 다른 .py 파일에서 정의한 router객체를 합치기
# 라우터 연결
# .include_router(연결할 라우터)
import controller
app.include_router(controller.router)