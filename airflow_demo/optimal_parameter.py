import pandas as pd  # đọc dữ liệu
import numpy as np  # xử lý dữ liệu
import matplotlib.pyplot as plt  # vẽ biểu đồ
from sklearn.preprocessing import MinMaxScaler  # chuẩn hóa dữ liệu
from keras.api.callbacks import ModelCheckpoint
from keras.api.callbacks import EarlyStopping
from tensorflow.keras.models import * # tải mô hình

# các lớp để xây dựng mô hình
from keras.api.models import Sequential  # đầu vào
from keras.api.layers import LSTM  # học phụ thuộc
from keras.api.layers import Dropout  # tránh học tủ
from keras.api.layers import Dense  # đầu ra

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # tìm kiếm siêu tham số
from scikeras.wrappers import KerasRegressor  # tạo wrapper cho scikit-learn

# kiểm tra độ chính xác của mô hình
from sklearn.metrics import r2_score  # đo mức độ phù hợp
from sklearn.metrics import mean_absolute_error  # đo sai số tuyệt đối trung bình
from sklearn.metrics import mean_absolute_percentage_error  # đo % sai số tuyệt đối trung bình

file_path = 'stock_data.csv'
df = pd.read_csv(file_path)
df = df.drop(columns=["volume", "ticker"])
df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d")

df1 = pd.DataFrame(df, columns=['time', 'close'])
df1.index = df1.time
df1.drop('time', axis=1, inplace=True)

df1['close'] = df1['close'].astype(float)
print(df1.shape)

data = df1.values

# Chuan hoa du lieu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Hàm tạo dataset với time_step
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Hàm tạo mô hình LSTM với tham số truyền vào
def create_model(units=128, dropout=0.5):
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(None, 1), return_sequences=True))
    model.add(LSTM(units=units // 2))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

time_steps = list(range(5,50,1))

results = []

for time_step in time_steps:
    X, Y = create_dataset(scaled_data, time_step)

    # Reshape X để phù hợp với LSTM [samples, time_steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Chia dữ liệu thành tập train và test (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Tạo model KerasRegressor
    model = KerasRegressor(model=create_model, verbose=0)

    # Tạo lưới các tham số cần tìm kiếm
    param_grid = {
        'model__units': [64, 128, 256],
        'model__dropout': [0.2, 0.6],
        'batch_size': [16, 32, 64],
        'epochs': list(range(5, 100, 1))
    }

    # Tìm kiếm siêu tham số tốt nhất bằng GridSearchCV với TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)  # Chia dữ liệu theo thời gian
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=tscv, n_jobs=-1)
    grid_result = grid.fit(X_train, Y_train)

    # In ra kết quả
    print(f"Best result for time_step {time_step}: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Dự đoán và tính toán MAE trên tập test
    best_model = grid_result.best_estimator_
    Y_pred = best_model.predict(X_test)
    Y_test_scaled = scaler.inverse_transform([Y_test])
    Y_pred_scaled = scaler.inverse_transform(Y_pred.reshape(-1, 1))
    mae = mean_absolute_error(Y_test_scaled[0], Y_pred_scaled[:, 0])

    results.append((time_step, grid_result.best_params_['model__units'], grid_result.best_params_['model__dropout'],
                    grid_result.best_params_['batch_size'], grid_result.best_params_['epochs'], mae))

# Lưu lại kết quả vào DataFrame
results_df = pd.DataFrame(results,
                          columns=['time_steps', 'model__units', 'model__dropout', 'batch_size', 'epochs', 'MAE'])

# Lưu DataFrame vào file CSV
results_df.to_csv(f'Ket_Qua_Tim_Kiem_Sieu_Tham_So_VNM_1_{time_step}_.csv', index=False)
# In kết quả tổng hợp