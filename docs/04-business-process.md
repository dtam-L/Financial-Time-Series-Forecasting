# Quy trình nghiệp vụ — Financial Time Series Forecasting

## 1. Quy trình thu thập dữ liệu thị trường

### 1.1 As-is (khi chưa có hệ thống)
- Nhà phân tích tải file CSV giá từ sàn hoặc website thủ công, mỗi lần một symbol / khung thời gian.
- Dữ liệu dễ **thiếu nến, trùng nến**, lệch múi giờ; không biết lần tải gần nhất là khi nào.
- Mỗi lần phân tích phải tự tính lại chỉ báo kỹ thuật.

### 1.2 To-be (với hệ thống)

```mermaid
flowchart TD
    T([⏰ Đến lịch: xx:02 mỗi giờ / 00:05 UTC mỗi ngày]) --> L{Lần chạy trước<br/>đã kết thúc?}
    L -- Chưa --> SK([Bỏ qua — không chạy song song])
    L -- Rồi --> Q[Lấy thời điểm nến mới nhất<br/>theo symbol × timeframe]
    Q --> E{Đã có dữ liệu?}
    E -- Chưa --> I[Chế độ initial<br/>since = START_DATE]
    E -- Có --> U[Chế độ incremental<br/>since = nến cuối + 1 giây]
    I --> LOG1[Ghi ingestion_log: running]
    U --> LOG1
    LOG1 --> F[Gọi Binance lấy OHLCV]
    F --> R{Thành công?}
    R -- Rate limit / lỗi mạng --> W[Chờ tăng dần rồi thử lại] --> F
    R -- Lỗi không khắc phục --> ERR[ingestion_log: error + thông báo lỗi]
    R -- Có --> C[Làm sạch dữ liệu<br/>BR-04]
    C --> UP[Upsert vào ohlcv_data<br/>đếm thêm mới / cập nhật]
    UP --> OK[ingestion_log: success + số dòng]
    OK --> D([Dữ liệu sẵn sàng cho Dashboard, API, huấn luyện])
```

**Giá trị mang lại:** dữ liệu luôn cập nhật ngay sau khi nến đóng, không trùng lặp, cùng múi giờ UTC, mỗi lần chạy đều có nhật ký để kiểm toán.

## 2. Vòng đời mô hình (Model lifecycle)

### 2.1 As-is (hiện trạng)
Data Scientist huấn luyện → kiểm tra kết quả bằng mắt → gọi reload thủ công. Không có bước kiểm tra chỉ số bắt buộc, nên mô hình kém hơn vẫn có thể được đưa vào sử dụng (xem G-01 trong SRS).

### 2.2 To-be (đề xuất — bổ sung cổng chất lượng)

```mermaid
flowchart TD
    subgraph DS[Data Scientist]
        A([Có dữ liệu mới / ý tưởng mô hình]) --> B[Chạy job huấn luyện<br/>trainer_gbm]
    end
    subgraph HT[Hệ thống]
        B --> C[Tạo đặc trưng<br/>tách train / calibration / test theo thời gian]
        C --> D[Optuna HPO → Stacking OOF → Conformal]
        D --> E[Đánh giá trên tập test<br/>MAE, RMSE, MAPE, coverage]
        E --> G{Cổng chất lượng<br/>MAPE &lt; 5% và coverage đạt?<br/><i>Planned</i>}
        G -- Không --> X[Lưu kết quả để phân tích<br/>giữ nguyên mô hình đang phục vụ]
        G -- Có --> S[Lưu gbm_forecaster.joblib<br/>+ metadata phiên bản]
        S --> R[POST /models/reload]
        R --> P[API phục vụ mô hình mới<br/>không gián đoạn]
    end
    P --> M[Theo dõi sai số thực tế<br/>khi giá mới về]
    M --> A
```

## 3. Luồng xử lý một yêu cầu dự báo

```mermaid
sequenceDiagram
    actor U as Người dùng
    participant D as Dashboard (Streamlit)
    participant A as Prediction API (FastAPI)
    participant R as ModelRegistry
    participant DB as PostgreSQL

    U->>D: Chọn symbol, timeframe, số bước
    D->>DB: Đọc nến lịch sử (cache 30s)
    DB-->>D: OHLCV
    D->>A: POST /predict/gbm {symbol, timeframe, steps, from_db: true}
    A->>R: Lấy mô hình GBM
    alt Mô hình chưa nạp
        A-->>D: 503
        D-->>U: Cảnh báo + hướng dẫn huấn luyện / reload
    else Mô hình sẵn sàng
        A->>DB: n_history nến gần nhất
        DB-->>A: OHLCV
        A->>A: Tính đặc trưng → dự báo đệ quy t+1…t+N<br/>+ khoảng conformal 80% / 90%
        A-->>D: 200 PredictResponse
        D-->>U: Đường dự báo + vùng tin cậy + bảng dự báo
    end
```

## 4. Sơ đồ trạng thái

### 4.1 Trạng thái một lần thu thập dữ liệu (`ingestion_log.status`)

```mermaid
stateDiagram-v2
    [*] --> running: Bắt đầu job
    running --> success: Lấy & upsert thành công
    running --> error: Lỗi sau khi thử lại
    success --> [*]
    error --> [*]
```

| Trạng thái | Ý nghĩa | Hành động của quản trị viên |
|-----------|---------|----------------------------|
| running | Đang chạy | — (nếu kéo dài bất thường: kiểm tra log scheduler) |
| success | Hoàn tất, có `rows_inserted` / `rows_updated` | — |
| error | Thất bại, có `error_message` | Kiểm tra khóa API, kết nối mạng, giới hạn tần suất; chạy lại `--mode update` |

### 4.2 Trạng thái sức khỏe API (`/health`)

```mermaid
stateDiagram-v2
    [*] --> error: Khởi động, chưa có CSDL & mô hình
    error --> degraded: CSDL kết nối hoặc 1 mô hình nạp
    degraded --> ok: CSDL kết nối và mọi mô hình đã nạp
    ok --> degraded: Mất CSDL hoặc 1 mô hình lỗi
    degraded --> error: Mất cả CSDL và mô hình
```

## 5. Lịch vận hành

| Thời điểm (UTC) | Job | Dữ liệu | Ghi chú |
|-----------------|-----|---------|---------|
| Khi khởi động scheduler | Cập nhật tăng dần | Mọi timeframe cấu hình | Đảm bảo dữ liệu mới ngay |
| Phút 02 mỗi giờ | `hourly_ohlcv` | Nến `1h` | Ân hạn 5 phút |
| 00:05 hằng ngày | `daily_ohlcv` | Nến `1d` | Ân hạn 10 phút |
| Theo nhu cầu | `trainer_gbm` / `trainer_tft` | Mô hình | Profile `training` |

## 6. Hành trình người dùng (User Journey) — nhà đầu tư mỗi sáng

| Bước | Người dùng làm gì | Màn hình / thành phần hỗ trợ |
|------|-------------------|-----------------------------|
| 1 | Xem giá BTC hiện tại, % thay đổi, RSI có quá mua / quá bán không | Hàng thẻ KPI |
| 2 | Xem xu hướng, Bollinger, khối lượng của 200 phiên gần nhất | Tab EDA Live |
| 3 | Xem dự báo 7 ngày tới và khoảng giá 80% / 90% | Tab Prediction |
| 4 | Đánh giá rủi ro: drawdown hiện tại, phân phối lợi nhuận có đuôi dày không | Tab Diagnostics |
| 5 | *(Planned)* Nhận cảnh báo khi giá biến động bất thường | Kênh thông báo |
