from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import vnstock
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter, MonthLocator
from sklearn.preprocessing import MinMaxScaler
from keras.api.callbacks import ModelCheckpoint  # lưu lại huấn luyện tốt nhất
from keras.src.saving.saving_api import load_model

# các lớp để xây dựng mô hình
from keras.api.models import Sequential  # đầu vào
from keras.api.layers import LSTM  # học phụ thuộc
from keras.api.layers import Dropout  # tránh học tủ
from keras.api.layers import Dense  # đầu ra

# kiểm tra độ chính xác của mô hình
from sklearn.metrics import r2_score  # đo mức độ phù hợp
from sklearn.metrics import mean_absolute_error  # đo sai số tuyệt đối trung bình
from sklearn.metrics import mean_absolute_percentage_error  # đo % sai số tuyệt đối trung bình

from send_email import send_email_with_html_and_csv_attachments

import pendulum

local_tz = pendulum.timezone("Asia/Ho_Chi_Minh")

default_args = {
    'owner': 'Nguyen Khoa Doan',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'stock_prediction',
    default_args=default_args,
    description='Dự báo giá cổ phiếu',
    schedule_interval='0 17 * * *',  # Chạy mỗi ngày
    start_date=datetime(2024, 6, 13, tzinfo=local_tz),
    catchup=False,
)


def collect_data():
    file_path = '/opt/airflow/data/stock_data.csv'
    stock_code = 'VNM'
    start_date = '2014-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = vnstock.stock_historical_data(symbol=stock_code, start_date=start_date, end_date=end_date)
    data.to_csv(file_path, index=False)


def read_data(**kwargs):
    file_path = '/opt/airflow/data/stock_data.csv'
    df = pd.read_csv(file_path)
    df = df.drop(columns=["volume", "ticker"])
    print(df.tail())


def describe_data(**kwargs):
    # Lấy dataframe từ XCom (nếu cần)
    # ti = kwargs['ti']
    # df = ti.xcom_pull(task_ids='read_data_task', key='stock_dataframe')

    # Đọc dữ liệu từ file CSV
    file_path = '/opt/airflow/data/stock_data.csv'
    df = pd.read_csv(file_path)
    df = df.drop(columns=["volume", "ticker"])

    # Định dạng cấu trúc thời gian
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d")

    # Kích thước dữ liệu
    shape = df.shape
    print("Kích thước dữ liệu:")
    print(shape)

    # Thông tin về dữ liệu
    info = df.info()
    print("Thông tin dữ liệu:")
    print(info)

    # Mô tả bộ dữ liệu
    describe = df.describe()
    print("Mô tả bộ dữ liệu:")
    print(describe)


def visual_close_per_year_data(**kwargs):
    file_path = '/opt/airflow/data/stock_data.csv'
    df = pd.read_csv(file_path)
    df = df.drop(columns=["volume", "ticker"])
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d")

    # Sắp xếp lại dữ liệu theo thứ tự thời gian
    df = df.sort_values(by='time')

    # Chuyển đổi định dạng cột giá đóng thành số thực
    df['close'] = df['close'].astype(float)

    # Lấy thông tin năm từ cột time
    df['year'] = df['time'].dt.year

    # Tao bieu do gia dong cua qua cac nam
    plt.figure(figsize=(10, 5))
    plt.plot(df['time'], df['close'], label='Giá đóng cửa', color='red')
    plt.xlabel('Giá đóng cửa')
    plt.ylabel('Biểu đồ giá đóng cửa qua các năm')
    plt.legend(loc='best')

    # Định dạng biểu đồ hien thi
    years = YearLocator()
    yearsFmt = DateFormatter('%Y')
    months = MonthLocator()
    plt.gca().xaxis.set_major_locator(years)
    plt.gca().xaxis.set_major_formatter(yearsFmt)
    plt.gca().xaxis.set_minor_locator(months)

    plt.tight_layout()
    output_path = "/opt/airflow/data/visualize/close_per_year.png"
    plt.savefig(output_path)


def preprocessing_data(**kwargs):
    file_path = '/opt/airflow/data/stock_data.csv'
    df = pd.read_csv(file_path)
    df1 = pd.DataFrame(df, columns=['time', 'close'])

    # Chuyển đổi định dạng
    df1['close'] = df1['close'].astype(float)
    df1["time"] = pd.to_datetime(df1["time"], format="%Y-%m-%d")
    df1.index = df1.time
    df1.drop('time', axis=1, inplace=True)
    print(df1)
    df1.to_csv('/opt/airflow/data/preprocessing_data.csv', index=False)


def train_model(**kwargs):
    file_path = '/opt/airflow/data/preprocessing_data.csv'
    df = pd.read_csv(file_path)

    # Chia tập dữ liệu
    data = df.values
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    # print(data)

    # Chuẩn hoá dữ liệu
    sc = MinMaxScaler(feature_range=(0, 1))
    sc_train = sc.fit_transform(data)
    # print(sc_train)

    # Tạo tham số
    best_time_steps = 5
    best_model_units = 256
    best_model_dropout = 0.2
    best_batch_size = 16
    best_epochs = 37
    time_steps = best_time_steps
    model_units = best_model_units
    model_dropout = best_model_dropout
    batch_size = best_batch_size
    epochs = best_epochs

    # tạo vòng lặp các giá trị
    x_train, y_train = [], []

    for i in range(time_steps, len(train_data)):
        x_train.append(sc_train[i - time_steps:i, 0])  # lấy time_steps giá đóng cửa liên tục
        y_train.append(sc_train[i, 0])  # lấy ra giá đóng cửa ngày hôm sau

    # print(x_train)
    # print(y_train)

    # Xếp dữ liệu thành 1 mảng 2 chiều
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Xếp lại dữ liệu thành mảng 1 chiều
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))

    # Xây dựng mô hình
    model = Sequential()

    # Lớp LSTM
    model.add(LSTM(units=model_units, input_shape=(x_train.shape[1], 1), return_sequences=True))
    model.add(LSTM(units=int(model_units / 2)))
    model.add(Dropout(model_dropout))  # loại bỏ 1 số đơn vị tránh học tủ (overfitting)
    model.add(Dense(1))  # output đầu ra 1 chiều
    # đo sai số tuyệt đối trung bình có sử dụng trình tối ưu hóa adam
    model.compile(loss='mean_absolute_error', optimizer='adam')

    # Huấn luyện mô hình
    save_model_dir = '/opt/airflow/data/model/'
    save_model_path = os.path.join(save_model_dir, 'save_model.keras')
    os.makedirs(save_model_dir, exist_ok=True)

    # Tạo callback lưu model tốt nhất
    best_model = ModelCheckpoint(save_model_path, monitor='loss', verbose=2, save_best_only=True, mode='auto')

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[best_model])

    final_model = load_model('/opt/airflow/data/model/save_model.keras')

    return save_model_path


def test_model(**kwargs):
    # Load model
    save_model_path = kwargs['ti'].xcom_pull(task_ids='train_model_task')
    final_model = load_model(save_model_path)

    # Tham số
    best_time_steps = 5
    time_steps = best_time_steps

    df = pd.read_csv('/opt/airflow/data/stock_data.csv')
    df1 = pd.DataFrame(df, columns=['time', 'close'])

    # Chuyển đổi định dạng
    df1['close'] = df1['close'].astype(float)
    df1["time"] = pd.to_datetime(df1["time"], format="%Y-%m-%d")
    df1.index = df1.time
    df1.drop('time', axis=1, inplace=True)

    # Chia tập dữ liệu
    data = df1.values
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    sc = MinMaxScaler(feature_range=(0, 1))
    sc_train = sc.fit_transform(data)
    sc.fit(train_data)
    # TEST
    test = df1[len(train_data) - time_steps:].values
    test = test.reshape(-1, 1)
    sc_test = sc.transform(test)

    x_test = []
    for i in range(time_steps, test.shape[0]):
        x_test.append(sc_test[i - time_steps:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_test = data[train_size:]  # giá thực
    y_test_predict = final_model.predict(x_test)
    y_test_predict = sc.inverse_transform(y_test_predict)  # giá dự đoán
    print(len(y_test_predict))

    train_data1 = df1[time_steps:train_size]
    test_data1 = df1[train_size:]

    plt.figure(figsize=(24, 8))
    plt.plot(df1, label='Giá thực tế', color='red')
    test_data1['predict'] = y_test_predict  # thêm dữ liệu
    plt.plot(test_data1['predict'], label='Giá dự đoán test', color='blue')  # đường giá dự báo test
    plt.title('So sánh giá dự báo và giá thực tế')  # đặt tên biểu đồ
    plt.xlabel('Thời gian')  # đặt tên hàm x
    plt.ylabel('Giá đóng cửa (VNĐ)')  # đặt tên hàm y
    plt.legend()  # chú thích
    plt.savefig('/opt/airflow/data/visualize/visualize_train.png')

    '''
    Calculate
    '''
    print('Độ phù hợp tập test:', r2_score(y_test, y_test_predict))
    print('Sai số tuyệt đối trung bình trên tập test (VNĐ):', mean_absolute_error(y_test, y_test_predict))
    print('Phần trăm sai số tuyệt đối trung bình tập test:', mean_absolute_percentage_error(y_test, y_test_predict))

    result = []
    result.append((r2_score(y_test, y_test_predict),
                   mean_absolute_error(y_test, y_test_predict),
                   mean_absolute_percentage_error(y_test, y_test_predict)
                   ))
    result = pd.DataFrame(result, columns=['r2_score', 'MAE', 'MAPE'])
    result.to_csv('/opt/airflow/data/result/result.csv', mode='w', index=False)

    '''
    Predict_model
    '''
    # Lấy ngày kế tiếp sau ngày cuối cùng trong tập dữ liệu để dự đoán
    next_date = pd.to_datetime(df['time'].iloc[-1]) + pd.Timedelta(days=1)

    # Chuyển đổi ngày kế tiếp sang dạng datetime
    next_date = pd.to_datetime(next_date)

    # Lấy giá trị của ngày cuối cùng trong tập dữ liệu
    next_closing_price = np.array([df['close'].iloc[-1]])  # Lấy giá trị đóng cửa của ngày cuối cùng

    # Chuẩn hóa giá trị của ngày cuối cùng
    sc.transform(next_closing_price.reshape(-1, 1))  # Chuyển thành mảng 2D

    # Tạo dự đoán cho ngày kế tiếp bằng mô hình đã huấn luyện
    x_next = np.array([sc_train[-time_steps:, 0]])  # Lấy time_steps giá đóng cửa gần nhất
    x_next = np.reshape(x_next, (x_next.shape[0], x_next.shape[1], 1))
    y_next_predict = final_model.predict(x_next)
    y_next_predict = sc.inverse_transform(y_next_predict)

    actual_closing_price = df['close'].iloc[-1]

    # Tạo DataFrame so sánh giá dự đoán với giá ngày cuối trong tập dữ liệu
    comparison_df = pd.DataFrame(
        {'time': [next_date], 'predict': [y_next_predict[0][0]], 'pre_close': [actual_closing_price]})

    # In ra bảng so sánh
    comparison_df.to_csv('/opt/airflow/data/predict/predict.csv', mode='w', index=False)


def send_email():
    sender_email = "xxxxxx.vdt2024@gmail.com"
    receiver_email = "xxxxxxx1405@gmail.com"
    password = "xxxx xxxx xxxx xxxx"
    subject = "REPORT FROM AIRFLOW"

    # Nội dung HTML của email
    html_body = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DAG DONE</title>
    </head>
    <body>
        <h1>BÁO CÁO</h1>
        <p>1. File đính kèm giá dự đoán, kết quả train model.</p>
        <p>2. Ảnh biến động giá.</p>
    </body>
    </html>
    """
    # Danh sách đường dẫn đến các ảnh và file CSV cần đính kèm
    csv_paths = ['/opt/airflow/data/predict/predict.csv', '/opt/airflow/data/result/result.csv']
    image_paths = ['/opt/airflow/data/visualize/visualize_train.png', '/opt/airflow/data/visualize/close_per_year.png']

    # Gọi hàm để gửi email
    send_email_with_html_and_csv_attachments(sender_email, receiver_email,
                                             password, subject, html_body,
                                             image_paths, csv_paths)


start = PythonOperator(
    task_id='start',
    python_callable=lambda: print('START'),
    dag=dag
)

collect_data_task = PythonOperator(
    task_id='collect_data_task',
    python_callable=collect_data,
    dag=dag,
)

read_data_task = PythonOperator(
    task_id='read_data_task',
    python_callable=read_data,
    provide_context=True,
    dag=dag
)

describe_data_task = PythonOperator(
    task_id='describe_data_task',
    python_callable=describe_data,
    provide_context=True,
    dag=dag
)

visual_close_per_year_data_task = PythonOperator(
    task_id='visual_close_per_year_data_task',
    python_callable=visual_close_per_year_data,
    provide_context=True,
    dag=dag
)

preprocessing_data_task = PythonOperator(
    task_id='preprocessing_data_task',
    python_callable=preprocessing_data,
    provide_context=True,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model,
    provide_context=True,
    dag=dag
)

test_model_task = PythonOperator(
    task_id='test_model_task',
    python_callable=test_model,
    provide_context=True,
    dag=dag
)

send_email_task = PythonOperator(
    task_id='send_email_task',
    python_callable=send_email,
    dag=dag,
)

end = PythonOperator(
    task_id='end',
    python_callable=lambda: print('END'),
    dag=dag
)

start >> \
collect_data_task >> \
read_data_task >> \
describe_data_task >> \
visual_close_per_year_data_task >> \
preprocessing_data_task >> \
train_model_task >> \
test_model_task >> \
send_email_task >> end
