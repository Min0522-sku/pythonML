
import pandas as pd

df = pd.read_csv('./day01/Fish.csv')
print(df.head())
df.info()
# 특정 물고기 추출, 도미/Bream
bream_df = df[df['Species'] == 'Bream']

# 도미의 길이, 무게 추출 df['열이름'].tolist()
bream_length = bream_df['Length2'].tolist()
bream_weight = bream_df['Weight'].tolist()
print(bream_length, bream_weight)

# 특정 물고기 추출, 빙어/Smelt
smelt_df = df[df['Species'] == 'Smelt'] 
smelt_length = smelt_df['Length2'].tolist()
smelt_weight = smelt_df['Weight'].tolist()

# 시각화
import matplotlib.pyplot as plt
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel("length(cm)")
plt.ylabel("weight(gram)")
plt.show()

