import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
import os

def send_email_with_html_and_csv_attachments(sender_email, receiver_email, password, subject, html_body, image_paths, csv_paths):
    # Tạo đối tượng MIMEMultipart
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Thêm nội dung HTML vào email
    msg.attach(MIMEText(html_body, 'html'))

    # Đính kèm các ảnh vào email
    for image_path in image_paths:
        with open(image_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
            msg.attach(part)

    # Đọc và đính kèm nội dung CSV dưới dạng HTML
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        csv_html = df.to_html(index=False)
        part = MIMEText(csv_html, 'html')
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(csv_path)}')
        msg.attach(part)

    # Thiết lập kết nối với máy chủ SMTP của Gmail và gửi email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
        print("Email đã được gửi thành công!")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi gửi email: {e}")

# Các thông tin cần thiết
sender_email = "airflow.vdt2024@gmail.com"
receiver_email = "doansvn1405@gmail.com"
password = "agoa emgo hblq liqq"  # Thay thế bằng mật khẩu ứng dụng của bạn
subject = "Email với nội dung HTML và đính kèm ảnh và CSV"

# Nội dung HTML của email
html_body = """
<!DOCTYPE html>
<html>
<head>
    <title>Email HTML với ảnh và CSV</title>
</head>
<body>
    <h1>Đây là tiêu đề HTML</h1>
    <p>Đây là nội dung của email ở định dạng HTML.</p>
    <p>Bạn có thể thêm các thẻ HTML khác vào đây.</p>
</body>
</html>
"""

# Danh sách đường dẫn đến các ảnh và file CSV cần đính kèm
image_paths = ['./close_per_year.png']  # Đường dẫn đến các ảnh
csv_paths = ['./stock_data.csv']  # Đường dẫn đến các file CSV

# Gọi hàm để gửi email
send_email_with_html_and_csv_attachments(sender_email, receiver_email, password, subject, html_body, image_paths, csv_paths)
