from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
class Service:
    def __init__(self):
        self.model = None
        self.poly = None
        self.scaled = None
        #self.df = pd.read_csv('./day05/practice9/user.csv')
    def train(self, userList):
        l = []
        #df = self.df
        df = pd.DataFrame(userList)
        train_data = df[['age', 'gender', 'inflow', 'style']]
        print(train_data)
        target_data = df['category'].values
        print(target_data)
        train_input, test_input, train_target,  test_target = train_test_split(train_data, target_data, test_size=0.2, random_state=42)
        for degree in [1,2,3,4,5]:
            poly = PolynomialFeatures(degree= degree, include_bias= False) # 차수 조정 필요 할 수도 있음
            poly.fit(train_input)
            train_poly = poly.transform(train_input)
            test_poly = poly.transform(test_input)

            ss = StandardScaler()
            ss.fit(train_poly)  
            train_scaled = ss.transform(train_poly)
            test_scaled = ss.transform(test_poly)
            
            for alpha in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
                for iter in [100, 1000, 100000, 1000000]:
                    sc = SGDClassifier(loss='hinge', random_state=42, max_iter=iter, alpha=alpha)
                    sc.fit(train_scaled, train_target)
                    accuracy = sc.score(test_scaled, test_target)
                    l.append( {'accuracy':accuracy, 'model':sc, 'poly':poly, 'degree':degree, 'scaler':ss, 'alpha':alpha})
        best_optimization = max(l, key=lambda x : x['accuracy'])
        self.model = best_optimization['model']
        self.poly = best_optimization['poly']
        self.scaled = best_optimization['scaler']
        print(f"🥇 [최적 설정] 정확도: {best_optimization['accuracy']:.4f}")
        return best_optimization['accuracy']
    
    def predict(self, user):
        if self.model is None:
            return "학습 모델이 없습니다."
        keys = ['age', 'gender', 'inflow', 'style']
        user_features = [[user[key] for key in keys]]
        user_features_poly = self.poly.transform(user_features)
        user_features_scaled = self.scaled.transform(user_features_poly)
        predict = self.model.predict(user_features_scaled)
        return int(predict[0])
    
service = Service()