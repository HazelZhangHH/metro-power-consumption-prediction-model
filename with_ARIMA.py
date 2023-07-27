import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
# 读取数据
data = pd.read_csv('metro_data.csv')

# 提取目标变量
y = data['power_consumption']

# 拟合ARIMA模型
p = 1
d = 0
q = 1
model = ARIMA(y, order=(p, d, q))
model_fit = model.fit()
n=10
y_true = y
# 进行预测
y_pred = model_fit.predict(start=len(y),  end=len(y)+len(y_true)-1)

# 计算RMSE和R方
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

# 输出结果
print('RMSE:', rmse)
print('R方:', r2)