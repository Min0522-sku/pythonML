from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
class Service:
    def __init__(self):
        self.model = None
        self.poly = None

    def train(self, carList):
        df = pd.DataFrame(carList)
        train_data = df[['fuelEfficiency', 'mileage', 'monthsSinceRelease', 'accidentCount', 'ownerChangeCount']]
        target_data = df['price'].values
        train_input, test_input, train_target,  test_target = train_test_split(train_data, target_data, test_size=0.2, random_state=42)
        poly = PolynomialFeatures(include_bias= False)
        poly.fit(train_input)
        lr = LinearRegression()
        train_poly = poly.transform(train_input)
        test_poly = poly.transform(test_input)
        lr.fit(train_poly, train_target)
        
        self.model = lr
        self.poly = poly
        return lr.score(test_poly, test_target)
    
    def predict(self, car):
        if self.model is None:
            return "학습 모델이 없습니다."
        keys = ['fuelEfficiency', 'mileage', 'monthsSinceRelease', 'accidentCount', 'ownerChangeCount']
        car_features = [[car[key] for key in keys]]
        car_features_poly = self.poly.transform(car_features)
        predict = self.model.predict(car_features_poly)
        return predict[0]
    
service = Service()