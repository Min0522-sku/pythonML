
from fastapi import APIRouter,Request,Response
router = APIRouter(prefix="/api/model")

from service import service
# REST API 정의
# GET
@router.post("/train")
async def train(request : Request):
    list = await request.json()
    return service.train(list)

@router.post("/predict")
async def predict(car : dict):
    return service.predict(car)