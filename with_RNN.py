import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 加载数据并处理
df = pd.read_csv('metro_data.csv')

scaler = MinMaxScaler()
df['power_consumption'] = scaler.fit_transform(df[['power_consumption']]) 

X = df[['passenger_flow', 'temp']].values
y = df['power_consumption'].values

# Reshape X 为 3D  
X = X.reshape(X.shape[0], 1, X.shape[1])

# 构建RNN模型
model = Sequential()
model.add(SimpleRNN(32, input_shape=(1, X.shape[2])))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam') 

# 训练模型
model.fit(X, y, epochs=100)  

# 预测
X_new = X[:1] 
y_pred = model.predict(X_new)

# 反归一化
y_pred = scaler.inverse_transform(y_pred)

print(y_pred)
# 计算RMSE
rmse_rnn = mean_squared_error(y_test, y_pred, squared=False) 

# 计算R方Score
r2_rnn = r2_score(y_test, y_pred)

print("RNN RMSE: ", rmse_rnn)

print("RNN R-squared: ", r2_rnn)  