Cài đặt chạy ci-cd
Các file đã sửa bao gồm file docker-compose.yaml, .env và thêm file docker-compose.prod.yaml và thêm workflow airflow_cicd.yaml

A. Bước self-host
1. Cài đặt self host: Settings - Actions - Runners - New self-hostd runner và cài đặt theo hướng dẫn vào trong máy chủ.
2. Cấp quyền trong administrator dùng lệnh : Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
3. Kiểm tra trạng thái máy chủ

B. Cài đặt file .env và các secrets
1. File .env (gửi riêng):
   - Sửa Airflow_UID và API_KEY
   - Thông tin cơ sở dữ liệu Postgres
2. Các secrets
- DOCKERHUB_USERNAME
- DOCKERHUB_TOKEN
- ENV_FILE_CONTENT
3. Sửa ở file airflow_cicd
- thay IMAGE_NAME
