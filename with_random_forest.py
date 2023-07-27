import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 导入数据
df = pd.read_csv('metro_data.csv')

# 选择特征和目标变量
X = df[['passenger_flow', 'temp']] 
y = df['power_consumption']

# 拆分训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train) 

# 预测并评估
y_pred = rf_model.predict(X_test)
print(y_pred)
print('R-squared score:', rf_model.score(X_test, y_test))