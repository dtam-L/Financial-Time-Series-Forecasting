# User Stories & Product Backlog — Financial Time Series Forecasting

Độ ưu tiên theo **MoSCoW**: **M**ust have · **S**hould have · **C**ould have · **W**on't have (this release).
Trạng thái: ✅ Implemented · 🕒 Planned.

## 1. Epic & User Stories

### Epic E1 — Dữ liệu thị trường tự động
| ID | User story | MoSCoW | Trạng thái | Yêu cầu |
|----|-----------|--------|-----------|---------|
| US-01 | Là **quản trị hệ thống**, tôi muốn nạp toàn bộ lịch sử giá từ một ngày bắt đầu bằng một lệnh, để có dữ liệu đủ dài cho phân tích và huấn luyện | M | ✅ | FR-01, FR-07 |
| US-02 | Là **nhà phân tích**, tôi muốn dữ liệu tự cập nhật mỗi giờ / mỗi ngày ngay sau khi nến đóng, để luôn phân tích trên số liệu mới nhất | M | ✅ | FR-02, FR-05 |
| US-03 | Là **quản trị hệ thống**, tôi muốn nạp lại dữ liệu mà không sinh bản ghi trùng, để an toàn khi chạy lại job | M | ✅ | FR-03 |
| US-04 | Là **quản trị hệ thống**, tôi muốn xem nhật ký từng lần thu thập (số bản ghi thêm / cập nhật, trạng thái, lỗi), để phát hiện và xử lý sự cố | M | ✅ | FR-04 |
| US-05 | Là **quản trị hệ thống**, tôi muốn hệ thống tự thử lại khi sàn giới hạn tần suất, để job không thất bại vì lỗi tạm thời | S | ✅ | FR-06 |

### Epic E2 — Chất lượng dữ liệu & phân tích
| ID | User story | MoSCoW | Trạng thái | Yêu cầu |
|----|-----------|--------|-----------|---------|
| US-06 | Là **data scientist**, tôi muốn nến lỗi (giá âm, high < low, trùng thời điểm…) bị loại tự động kèm báo cáo, để mô hình không học từ dữ liệu sai | M | ✅ | FR-08 |
| US-07 | Là **nhà phân tích**, tôi muốn các chỉ báo kỹ thuật được tính sẵn, để không phải tính tay mỗi lần | M | ✅ | FR-09 |
| US-08 | Là **nhà phân tích**, tôi muốn có báo cáo kiểm định tính dừng, phân phối và rủi ro đuôi (VaR/CVaR), để chọn cách mô hình hóa phù hợp | S | ✅ | FR-10, FR-11 |

### Epic E3 — Mô hình dự báo
| ID | User story | MoSCoW | Trạng thái | Yêu cầu |
|----|-----------|--------|-----------|---------|
| US-09 | Là **data scientist**, tôi muốn huấn luyện mô hình GBM bằng một lệnh và nhận kết quả đánh giá chuẩn (MAE, RMSE, MAPE, coverage), để so sánh các lần huấn luyện | M | ✅ | FR-13, FR-15 |
| US-10 | Là **nhà đầu tư**, tôi muốn mỗi dự báo đi kèm khoảng giá 80% / 90%, để biết mức độ không chắc chắn trước khi ra quyết định | M | ✅ | FR-14 |
| US-11 | Là **data scientist**, tôi muốn thử nghiệm mô hình TFT trên GPU (Colab), để đánh giá khả năng vượt GBM khi có nhiều dữ liệu | C | ✅ | FR-16 |
| US-12 | Là **data scientist**, tôi muốn mô hình chỉ được đưa vào sử dụng khi đạt ngưỡng MAPE và coverage, để tránh phát hành mô hình kém | S | 🕒 | G-01, G-02 |

### Epic E4 — Prediction API
| ID | User story | MoSCoW | Trạng thái | Yêu cầu |
|----|-----------|--------|-----------|---------|
| US-13 | Là **ứng dụng tích hợp**, tôi muốn gọi API để nhận dự báo N bước cho một symbol, để hiển thị trong hệ thống của mình | M | ✅ | FR-21, FR-23 |
| US-14 | Là **ứng dụng tích hợp**, tôi muốn tự gửi dữ liệu nến của mình lên để dự báo, để không phụ thuộc CSDL của hệ thống | S | ✅ | FR-21 |
| US-15 | Là **data scientist**, tôi muốn nạp mô hình mới vào API đang chạy, để cập nhật mô hình không gây gián đoạn | M | ✅ | FR-19 |
| US-16 | Là **quản trị hệ thống**, tôi muốn kiểm tra sức khỏe API (CSDL, mô hình), để giám sát dịch vụ | M | ✅ | FR-17, FR-18 |
| US-17 | Là **ứng dụng tích hợp**, tôi muốn lấy N nến gần nhất qua API | S | ✅ | FR-20 |
| US-18 | Là **ứng dụng tích hợp**, tôi muốn gọi dự báo TFT và nhận đúng kết quả của mô hình TFT | S | 🕒 | FR-22, G-03, G-07 |
| US-19 | Là **quản trị hệ thống**, tôi muốn các endpoint quản trị yêu cầu xác thực, để không ai khác thay được mô hình | S | 🕒 | G-05 |

### Epic E5 — Dashboard
| ID | User story | MoSCoW | Trạng thái | Yêu cầu |
|----|-----------|--------|-----------|---------|
| US-20 | Là **nhà đầu tư**, tôi muốn thấy ngay giá, % thay đổi, RSI, MACD trên một hàng thẻ KPI, để nắm tình hình trong vài giây | M | ✅ | FR-25 |
| US-21 | Là **nhà phân tích**, tôi muốn xem biểu đồ nến kèm MA, Bollinger, khối lượng, RSI, MACD | M | ✅ | FR-26 |
| US-22 | Là **nhà đầu tư**, tôi muốn xem đường dự báo và bảng dự báo từng ngày kèm % so với giá hiện tại và khoảng tin cậy | M | ✅ | FR-27 |
| US-23 | Là **nhà phân tích**, tôi muốn xem phân phối lợi nhuận, drawdown và kiểm định thống kê trên dashboard | S | ✅ | FR-28 |
| US-24 | Là **người dùng**, tôi muốn dashboard tự làm mới theo chu kỳ tôi chọn, để không phải tải lại trang | S | ✅ | FR-24 |
| US-25 | Là **người dùng**, tôi muốn chỉ thấy các symbol / timeframe thực sự có dữ liệu | S | 🕒 | G-04 |
| US-26 | Là **nhà đầu tư**, tôi muốn nhận cảnh báo khi giá biến động bất thường | C | 🕒 | G-08 |
| US-27 | Là **nhà đầu tư**, tôi muốn hệ thống tự đặt lệnh theo dự báo | W | — | Ngoài phạm vi |

## 2. Tiêu chí chấp nhận (Acceptance Criteria) — các story chính

### US-02 — Cập nhật dữ liệu tự động
- **AC1** — *Given* scheduler đang chạy, *when* đồng hồ đến phút 02 mỗi giờ, *then* hệ thống cập nhật nến `1h` mới nhất cho mọi symbol cấu hình.
- **AC2** — *Given* scheduler đang chạy, *when* đến 00:05 UTC, *then* hệ thống cập nhật nến `1d` của ngày vừa đóng.
- **AC3** — *Given* một cặp symbol × timeframe chưa có dữ liệu, *when* job cập nhật chạy, *then* hệ thống tự nạp lịch sử từ `START_DATE` (BR-05).
- **AC4** — *Given* lần chạy trước của job vẫn đang chạy, *then* job mới không được khởi động song song (BR-07).

### US-03 / US-04 — Không trùng lặp & nhật ký
- **AC1** — *Given* nến (BTC/USDT, 1d, 2024-01-01) đã tồn tại, *when* nạp lại nến này, *then* bản ghi được cập nhật, tổng số bản ghi không đổi, `rows_updated` tăng 1.
- **AC2** — *When* một lần chạy bắt đầu, *then* `ingestion_log` có bản ghi trạng thái `running`; khi kết thúc chuyển `success` kèm số dòng, hoặc `error` kèm `error_message`.

### US-06 — Làm sạch dữ liệu
- **AC1** — *Given* nến có `high < low` hoặc `close ≤ 0`, *then* nến bị loại và được đếm vào `dropped_invalid_ohlc` / `dropped_non_positive`.
- **AC2** — *Given* hai nến trùng `open_time`, *then* chỉ giữ một và tăng `dropped_duplicates`.
- **AC3** — Dữ liệu sau làm sạch được sắp xếp tăng dần theo thời gian.

### US-10 / US-13 — Dự báo kèm khoảng tin cậy qua API
- **AC1** — *Given* mô hình GBM đã nạp, *when* gọi `POST /predict/gbm` với `steps = 7`, *then* nhận `200` với đúng 7 phần tử `forecast`, mỗi phần tử có `y_pred`, `lower_80`, `upper_80`, `lower_90`, `upper_90`.
- **AC2** — *Given* `steps = 31` hoặc `n_history = 20`, *then* nhận `422` (BR-08).
- **AC3** — *Given* `from_db = false` và ít hơn 10 nến, *then* nhận `422`.
- **AC4** — *Given* mô hình chưa nạp, *then* nhận `503` kèm hướng dẫn huấn luyện và gọi `/models/reload` (BR-11).

### US-15 — Hot-reload mô hình
- **AC1** — *Given* file mô hình mới tồn tại, *when* gọi `POST /models/reload?model=gbm`, *then* phản hồi `{"reloaded": {"GBM": "loaded"}}` và các lần dự báo sau dùng mô hình mới, API không khởi động lại.
- **AC2** — *Given* file không tồn tại, *then* phản hồi `"failed (file not found)"`, mô hình cũ không bị ảnh hưởng.

### US-16 — Kiểm tra sức khỏe
- **AC1** — *Given* CSDL kết nối được và cả GBM, TFT đã nạp, *then* `status = "ok"`.
- **AC2** — *Given* chỉ GBM đã nạp, *then* `status = "degraded"`, `models_loaded = {"GBM": true, "TFT": false}`.
- **AC3** — *Given* CSDL lỗi và không có mô hình nào, *then* `status = "error"`.

### US-20 / US-22 — Dashboard
- **AC1** — Hàng KPI hiển thị 6 thẻ: Price, 24h Change, High, Low, RSI(14), MACD Hist; % thay đổi màu xanh khi ≥ 0, màu đỏ khi < 0; RSI > 70 tô đỏ, < 30 tô xanh.
- **AC2** — *When* đổi "Forecast steps" thành 14, *then* tab Prediction hiển thị 14 dòng t+1 … t+14, mỗi dòng có giá dự báo, chênh lệch và %, khoảng 80% / 90%.
- **AC3** — *Given* API không chạy, *then* tab Prediction hiển thị cảnh báo "Không kết nối được API" và hướng dẫn, các tab khác vẫn hoạt động.
- **AC4** — *Given* CSDL không có dữ liệu cho lựa chọn hiện tại, *then* hiển thị "No data found" và dừng render.

### US-12 — Cổng chất lượng mô hình (Planned)
- **AC1** — *Given* mô hình mới có MAPE ≥ 5% hoặc coverage 90% < 85% trên tập test, *then* mô hình không được đưa vào API và lý do được ghi lại.
- **AC2** — `/models/status` hiển thị phiên bản, ngày huấn luyện và chỉ số đánh giá của mô hình đang phục vụ.

## 3. Ma trận truy vết (RTM)

| Mục tiêu | User story | Yêu cầu | Use case | Thành phần / API |
|----------|-----------|---------|----------|------------------|
| BO-01 | US-01 – US-05 | FR-01 – FR-07 | UC-01, UC-02 | `data_collection_api/` · `ohlcv_data`, `ingestion_log` |
| BO-05 | US-06 – US-08 | FR-08 – FR-12 | UC-07 | `clean_feature_engineering_data/`, `eda/` |
| BO-02 | US-09 – US-12 | FR-13 – FR-16 | UC-03 | `models/`, `scripts/`, `gbm_output/` |
| BO-03 | US-13 – US-19 | FR-17 – FR-23 | UC-04, UC-06, UC-08, UC-09 | `api/` · `/health`, `/models/*`, `/data/latest`, `/predict/*` |
| BO-04 | US-20 – US-26 | FR-24 – FR-29 | UC-05 | `dashboard/` · `http://localhost:8501` |
