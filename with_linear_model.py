import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
data = pd.read_csv('metro_data.csv')

# 提取特征和目标变量
X = data[['passenger_flow', 'temp']]
y = data['power_consumption']

# 构建线性回归模型
model = LinearRegression()
model.fit(X, y)

# 预测耗电量
y_pred = model.predict(X)

# 计算RMSE和R方
rmse = mean_squared_error(y, y_pred, squared=False)
r2 = r2_score(y, y_pred)

# 输出结果
print('RMSE:', rmse)
print('R方:', r2)