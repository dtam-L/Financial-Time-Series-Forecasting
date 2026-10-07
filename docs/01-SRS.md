# Đặc tả Yêu cầu Phần mềm (SRS) — Financial Time Series Forecasting

| Mục | Thông tin |
|-----|-----------|
| Dự án | Financial Time Series Forecasting — Nền tảng thu thập dữ liệu & dự báo giá tài sản số |
| Phiên bản tài liệu | 1.0 |
| Tác giả | Đặng Văn Tâm |
| Trạng thái | Baseline theo hiện trạng mã nguồn |

---

## 1. Giới thiệu

### 1.1 Bối cảnh & vấn đề
Thị trường tài sản số (BTC, ETH…) giao dịch 24/7, biến động mạnh. Nhà đầu tư cá nhân và nhà phân tích thường gặp các vấn đề:

- Dữ liệu giá phải **tải thủ công** từ sàn, không đồng bộ, dễ thiếu nến hoặc trùng lặp.
- Các chỉ báo kỹ thuật (RSI, MACD, Bollinger…) và kiểm định thống kê **phải tính lại bằng tay** mỗi lần phân tích.
- Dự báo thường chỉ là **một con số**, không cho biết mức độ không chắc chắn, nên khó dùng để ra quyết định hoặc quản trị rủi ro.
- Kết quả mô hình nằm trong notebook, **không có kênh chuẩn** (API / dashboard) để người dùng không chuyên kỹ thuật hoặc hệ thống khác sử dụng.

### 1.2 Mục tiêu nghiệp vụ
| Mã | Mục tiêu | Chỉ số đo (KPI) |
|----|----------|-----------------|
| BO-01 | Tự động thu thập dữ liệu thị trường đầy đủ, không trùng lặp | Lịch sử OHLCV từ `START_DATE` (mặc định 2022-01-01) cho mọi cặp symbol × timeframe; 0 bản ghi trùng khóa; tỷ lệ lần chạy `success` trong `ingestion_log`; dữ liệu trễ không quá 1 chu kỳ nến + 5 phút |
| BO-02 | Dự báo giá đóng cửa nhiều bước kèm khoảng tin cậy | Dự báo 1–30 bước; MAPE < 5% trên tập test (theo thời gian); độ phủ (coverage) khoảng 80% / 90% đạt gần mức danh nghĩa |
| BO-03 | Cung cấp dự báo qua kênh chuẩn cho hệ thống khác | REST API có tài liệu OpenAPI, có endpoint kiểm tra sức khỏe; cập nhật mô hình mới **không cần khởi động lại** |
| BO-04 | Giúp người dùng không chuyên kỹ thuật theo dõi thị trường & dự báo trên một màn hình | Dashboard 3 tab, 6 thẻ KPI, bảng dự báo có % thay đổi và khoảng tin cậy; tự làm mới theo chu kỳ |
| BO-05 | Hiểu đặc tính thống kê của chuỗi giá trước khi chọn mô hình | Báo cáo chẩn đoán: thống kê mô tả, VaR/CVaR 5%, ADF, KPSS, Ljung-Box, Jarque-Bera, Hurst |

### 1.3 Phạm vi
**Trong phạm vi (đã triển khai):**
1. Thu thập dữ liệu OHLCV từ Binance (lần đầu + cập nhật tăng dần theo lịch) — module *Data Ingestion*.
2. Làm sạch dữ liệu và tạo đặc trưng kỹ thuật — module *Clean & Feature Engineering*.
3. Phân tích khám phá (EDA) và kiểm định thống kê — module *EDA & Diagnostics*.
4. Huấn luyện, đánh giá mô hình GBM Stacking Ensemble (XGBoost + LightGBM + Ridge) với khoảng dự báo Conformal; mô hình nghiên cứu TFT huấn luyện trên Colab — module *Modelling*.
5. REST API phục vụ dự báo GBM, dữ liệu nến mới nhất, trạng thái & hot-reload mô hình — module *Prediction API*.
6. Dashboard Streamlit thời gian thực — module *Dashboard*.
7. Đóng gói & vận hành bằng Docker Compose (PostgreSQL, scheduler, API, dashboard, job huấn luyện).

**Ngoài phạm vi phiên bản hiện tại (Planned / Won't):**
- Phục vụ dự báo TFT qua API (*Planned* — mô hình đã huấn luyện được, chưa có serving).
- Xác thực / phân quyền cho API và dashboard (*Planned*).
- Cảnh báo bất thường giá theo thời gian thực (email / Telegram) (*Planned* — hiện chỉ có biểu đồ tín hiệu bất thường Z-score ±3σ trong EDA).
- Quản lý phiên bản mô hình và cổng chất lượng tự động trước khi đưa mô hình vào sử dụng (*Planned*).
- Đặt lệnh giao dịch tự động / khuyến nghị đầu tư (*Won't* — ngoài mục đích hệ thống).

### 1.4 Thuật ngữ
| Thuật ngữ | Giải thích |
|-----------|-----------|
| OHLCV | Một nến giá: Open, High, Low, Close, Volume trong một khung thời gian |
| Symbol | Cặp tài sản giao dịch, ví dụ `BTC/USDT` |
| Timeframe | Khung thời gian của nến: `1h`, `1d`… |
| Initial load / Incremental update | Nạp toàn bộ lịch sử / chỉ nạp các nến mới hơn nến cuối cùng trong CSDL |
| Upsert | Ghi mới nếu chưa có, cập nhật nếu đã tồn tại (theo khóa nến) |
| Horizon / Step | Số bước dự báo phía trước (t+1, t+2, …) |
| Prediction interval (Conformal) | Khoảng giá dự kiến chứa giá thực tế với xác suất danh nghĩa 80% / 90% |
| Coverage | Tỷ lệ giá thực tế thực sự nằm trong khoảng dự báo |
| MAE / RMSE / MAPE | Sai số tuyệt đối trung bình / căn bậc hai sai số bình phương / sai số phần trăm tuyệt đối trung bình |
| Hot-reload | Nạp lại mô hình mới vào API đang chạy mà không khởi động lại dịch vụ |

---

## 2. Stakeholder & người dùng

| Nhóm (persona) | Vai trò / Nhu cầu chính | Mức ảnh hưởng | Mức quan tâm |
|----------------|------------------------|---------------|--------------|
| **Nhà đầu tư cá nhân** | Xem giá, chỉ báo, dự báo vài ngày tới kèm khoảng rủi ro trên dashboard dễ hiểu | Trung bình | Cao |
| **Nhà phân tích thị trường / Data Analyst** | Phân tích chỉ báo kỹ thuật, phân phối lợi nhuận, drawdown, kiểm định tính dừng | Trung bình | Cao |
| **Data Scientist / ML Engineer** | Huấn luyện, đánh giá, so sánh mô hình; đưa mô hình mới vào sử dụng | Cao | Cao |
| **Ứng dụng tích hợp** (bot, BI, hệ thống nội bộ) | Lấy dự báo & dữ liệu qua API có hợp đồng dữ liệu ổn định | Trung bình | Trung bình |
| **Quản trị hệ thống** | Vận hành Docker, theo dõi lịch thu thập dữ liệu, log và sức khỏe hệ thống | Cao | Trung bình |

**Hệ thống bên ngoài:** Binance (qua thư viện `ccxt`) — nguồn dữ liệu OHLCV; Google Colab — môi trường GPU để huấn luyện TFT.

---

## 3. Yêu cầu chức năng

### 3.1 Module M1 — Thu thập dữ liệu (Data Ingestion)
| Mã | Yêu cầu | Ưu tiên | Trạng thái |
|----|---------|---------|-----------|
| FR-01 | Nạp toàn bộ lịch sử OHLCV từ `START_DATE` đến hiện tại cho mọi cặp symbol × timeframe cấu hình (`SYMBOLS`, `TIMEFRAMES`) | Must | Implemented |
| FR-02 | Cập nhật tăng dần: chỉ lấy các nến mới hơn nến cuối cùng trong CSDL; nếu chưa có dữ liệu thì tự chuyển sang nạp lần đầu | Must | Implemented |
| FR-03 | Ghi dữ liệu theo cơ chế upsert, đếm số bản ghi thêm mới / cập nhật | Must | Implemented |
| FR-04 | Ghi nhật ký mỗi lần chạy: symbol, timeframe, chế độ, số bản ghi, thời điểm bắt đầu/kết thúc, trạng thái, lỗi | Must | Implemented |
| FR-05 | Lập lịch tự động: nến 1h lúc phút 02 mỗi giờ, nến 1d lúc 00:05 UTC mỗi ngày; chạy cập nhật ngay khi khởi động | Must | Implemented |
| FR-06 | Tự thử lại khi bị giới hạn tần suất (rate limit) hoặc lỗi mạng, thời gian chờ tăng theo cấp số nhân | Should | Implemented |
| FR-07 | Chạy thủ công qua dòng lệnh với các chế độ `initial`, `update`, `schedule` | Should | Implemented |

### 3.2 Module M2 — Làm sạch & tạo đặc trưng
| Mã | Yêu cầu | Ưu tiên | Trạng thái |
|----|---------|---------|-----------|
| FR-08 | Làm sạch dữ liệu: loại bỏ bản ghi thiếu, trùng thời điểm, giá trị vô hạn, giá ≤ 0, khối lượng âm, nến OHLC không hợp lệ (BR-04); trả về báo cáo số dòng bị loại theo từng lý do | Must | Implemented |
| FR-09 | Tạo 25+ đặc trưng: lợi nhuận, MA/EMA, MACD, RSI(14), Bollinger, ATR(14), tỷ lệ khối lượng, biến động, Z-score, drawdown, chế độ thị trường, đặc trưng lịch dạng sin/cos | Must | Implemented |

### 3.3 Module M3 — EDA & kiểm định thống kê
| Mã | Yêu cầu | Ưu tiên | Trạng thái |
|----|---------|---------|-----------|
| FR-10 | Sinh báo cáo chẩn đoán cho chuỗi giá / log return: thống kê mô tả, VaR 5%, CVaR 5%, ADF, KPSS, Ljung-Box, Jarque-Bera, Hurst | Must | Implemented |
| FR-11 | Xuất bộ biểu đồ EDA (PNG) gồm biểu đồ tín hiệu bất thường theo Z-score ±3σ | Should | Implemented |
| FR-12 | So sánh mô hình cơ sở / phân cụm chế độ thị trường (KMeans, Hierarchical, DBSCAN) | Could | Implemented |

### 3.4 Module M4 — Mô hình dự báo
| Mã | Yêu cầu | Ưu tiên | Trạng thái |
|----|---------|---------|-----------|
| FR-13 | Huấn luyện GBM Stacking: tạo lag / rolling / EWMA, tối ưu siêu tham số bằng Optuna (30 trials), stacking OOF 5-fold với meta-learner Ridge | Must | Implemented |
| FR-14 | Tạo khoảng dự báo Conformal 80% và 90% từ tập hiệu chỉnh (15% cuối tập train) | Must | Implemented |
| FR-15 | Đánh giá trên tập test tách theo thời gian: MAE, RMSE, MAPE, sMAPE, Winkler score, coverage; lưu kết quả JSON, feature importance và file mô hình `.joblib` | Must | Implemented |
| FR-16 | Huấn luyện mô hình nghiên cứu TFT (quantile output) trên Colab từ dữ liệu xuất ra `colab_data/` | Could | Implemented (research) |

### 3.5 Module M5 — Prediction API
| Mã | Yêu cầu | Ưu tiên | Trạng thái |
|----|---------|---------|-----------|
| FR-17 | `GET /health`: trả trạng thái `ok` / `degraded` / `error`, mô hình nào đã nạp, kết nối CSDL (BR-10) | Must | Implemented |
| FR-18 | `GET /models/status`: thông tin mô hình đã nạp (đường dẫn, target, số đặc trưng, ngưỡng conformal) | Should | Implemented |
| FR-19 | `POST /models/reload`: nạp lại mô hình GBM / TFT / cả hai mà không khởi động lại API | Must | Implemented |
| FR-20 | `GET /data/latest`: trả N nến gần nhất (1–500) theo symbol, timeframe | Should | Implemented |
| FR-21 | `POST /predict/gbm`: dự báo 1–30 bước từ dữ liệu trong CSDL hoặc nến do client gửi lên, mỗi bước kèm khoảng 80% / 90% | Must | Implemented |
| FR-22 | `POST /predict/tft`: dự báo bằng mô hình TFT | Should | Planned (xem G-03, G-07) |
| FR-23 | Tài liệu API tự động (Swagger / OpenAPI tại `/docs`) | Should | Implemented |

### 3.6 Module M6 — Dashboard
| Mã | Yêu cầu | Ưu tiên | Trạng thái |
|----|---------|---------|-----------|
| FR-24 | Thanh cấu hình: chọn symbol, timeframe, số nến lịch sử (50–500), số bước dự báo (1–30), bật/tắt tự làm mới và chu kỳ (10–300 giây), nút làm mới ngay | Must | Implemented |
| FR-25 | 6 thẻ KPI: Giá, % thay đổi so với nến trước, High, Low, RSI(14), MACD Histogram — tô màu tăng/giảm, quá mua/quá bán | Must | Implemented |
| FR-26 | Tab **EDA Live**: biểu đồ nến + MA7/MA25 + Bollinger, khối lượng, RSI, MACD, bảng thống kê mô tả | Must | Implemented |
| FR-27 | Tab **Prediction**: đường dự báo GBM với vùng khoảng tin cậy 80% / 90%; bảng dự báo từng bước gồm giá dự báo, chênh lệch và % so với giá hiện tại, khoảng 80% / 90% | Must | Implemented |
| FR-28 | Tab **Diagnostics**: phân phối log return, drawdown, Z-score giá, kết quả ADF / KPSS / Hurst | Should | Implemented |
| FR-29 | Thông báo lỗi thân thiện khi chưa có dữ liệu, API không kết nối được hoặc mô hình chưa được nạp, kèm hướng dẫn khắc phục | Should | Implemented |

---

## 4. Quy tắc nghiệp vụ (Business Rules)

| Mã | Quy tắc | Nơi áp dụng |
|----|---------|-------------|
| BR-01 | Mỗi nến được định danh duy nhất bởi **(symbol, timeframe, open_time)**; nạp lại cùng nến thì cập nhật giá trị, không tạo bản ghi trùng | M1 |
| BR-02 | Mọi thời điểm lưu theo **UTC** (`TIMESTAMPTZ`) | M1, M5 |
| BR-03 | Chỉ lấy nến đã đóng: job 1h chạy lúc phút 02, job 1d chạy lúc 00:05 UTC (nến ngày đóng lúc 00:00 UTC) | M1 |
| BR-04 | Nến hợp lệ khi: open, high, low, close > 0; volume ≥ 0; high ≥ max(open, close, low); low ≤ min(open, close, high). Nến không hợp lệ bị loại và được đếm trong báo cáo làm sạch | M2 |
| BR-05 | Cập nhật tăng dần bắt đầu từ nến cuối cùng + 1 giây; nếu cặp symbol × timeframe chưa có dữ liệu thì nạp từ `START_DATE` | M1 |
| BR-06 | Mỗi lần chạy thu thập có trạng thái `running` → `success` hoặc `error` (kèm thông báo lỗi) | M1 |
| BR-07 | Mỗi job lịch chỉ chạy **1 phiên tại một thời điểm**; các lần chạy bị lỡ được gộp lại; thời gian ân hạn 5 phút (job giờ) / 10 phút (job ngày) | M1 |
| BR-08 | Tham số dự báo: `steps` 1–30; `n_history` 30–1000 (khi lấy từ CSDL); khi client tự gửi nến (`from_db = false`) phải có ít nhất 10 nến | M5 |
| BR-09 | Đánh giá mô hình dùng **tách dữ liệu theo thời gian** (không xáo trộn) để tránh rò rỉ dữ liệu tương lai; cross-validation dùng TimeSeriesSplit | M4 |
| BR-10 | Trạng thái sức khỏe: `ok` khi CSDL kết nối được **và** mọi mô hình đã nạp; `degraded` khi có CSDL **hoặc** ít nhất một mô hình; `error` khi không có gì | M5 |
| BR-11 | API trả `503` khi mô hình chưa nạp hoặc CSDL không khả dụng; dashboard hiển thị hướng dẫn huấn luyện và reload mô hình | M5, M6 |
| BR-12 | `GET /data/latest` trả tối đa 500 nến mỗi lần | M5 |
| BR-13 | Dữ liệu và kết quả gọi API trên dashboard được cache **30 giây** để giảm tải CSDL | M6 |

---

## 5. Yêu cầu phi chức năng

| Mã | Nhóm | Yêu cầu |
|----|------|---------|
| NFR-01 | Toàn vẹn dữ liệu | Upsert trong một transaction; khóa chính (symbol, timeframe, open_time); trigger tự cập nhật `updated_at` |
| NFR-02 | Độ tin cậy | Thử lại khi lỗi mạng / rate limit (exponential back-off); PostgreSQL có health check; các service tự khởi động lại (`restart: unless-stopped`) |
| NFR-03 | Hiệu năng | Truy vấn nến mới nhất dùng index (symbol, timeframe, open_time DESC); dự báo (tác vụ nặng CPU) chạy trong thread pool để không chặn API; API chạy 2 worker Uvicorn |
| NFR-04 | Bảo mật | Khóa API Binance và mật khẩu CSDL lưu trong `.env` (không commit); chỉ cần quyền đọc dữ liệu thị trường |
| NFR-05 | Khả năng mở rộng | Thêm symbol / timeframe bằng cấu hình, không sửa mã; mô hình dữ liệu dùng chung cho mọi cặp tài sản |
| NFR-06 | Khả năng quan sát | Log xoay vòng (Loguru) theo ngày cho ingestion, scheduler, EDA; bảng `ingestion_log` để kiểm toán |
| NFR-07 | Khả năng triển khai | Toàn bộ hệ thống chạy bằng Docker Compose; job huấn luyện và công cụ quản trị tách thành profile riêng (`training`, `tools`) |
| NFR-08 | Khả năng bảo trì | Mã tách module theo tầng (ingestion / features / models / api / dashboard); có bộ kiểm thử `pytest` cho client, CSDL, làm sạch, đặc trưng |
| NFR-09 | Khả dụng (UX) | Dashboard giao diện tối kiểu TradingView, có trạng thái đang tải và thông báo lỗi rõ ràng |

---

## 6. Giả định & ràng buộc
- Dữ liệu phụ thuộc Binance API (giới hạn tần suất, thời gian hoạt động).
- Bộ dữ liệu huấn luyện nhỏ (≈ 710 nến ngày BTC/USDT, test ≈ 86 mẫu) nên kết quả đánh giá **dao động mạnh giữa các lần huấn luyện**.
- TFT cần GPU nên được huấn luyện trên Google Colab, sau đó tải về.
- Hệ thống chưa có đăng nhập; giả định chạy trong môi trường cá nhân / mạng nội bộ.

---

## 7. Đánh giá KPI mô hình (BO-02)

Đánh giá trên BTC/USDT, khung 1d, dự báo 7 bước, tập test tách theo thời gian.

| Chỉ số | Mục tiêu | Lần chạy được báo cáo trong README | Artifact hiện tại (`gbm_output/gbm_results.json`) |
|--------|----------|-----------------------------------|------------------------------------------------|
| MAPE | < 5% | 2,17% ✅ | 9,07% ❌ |
| MAE | — | 1.385 USD | 5.865 USD |
| RMSE | — | 1.777 USD | 6.173 USD |
| Coverage 80% | ≈ 80% | 63,95% ❌ | 45,35% ❌ |
| Coverage 90% | ≈ 90% | 68,60% ❌ | 50,00% ❌ |

**Nhận định:** độ chính xác điểm của GBM từng đạt mục tiêu, nhưng **khoảng tin cậy chưa đạt độ phủ danh nghĩa** ở cả hai lần chạy, và kết quả của artifact mới nhất kém hơn số liệu công bố trong README. Xem G-01, G-02.

---

## 8. Gap analysis — hiện trạng so với yêu cầu

| # | Khoảng trống | Ảnh hưởng nghiệp vụ | Đề xuất |
|---|--------------|---------------------|---------|
| G-01 | Số liệu hiệu năng trong README khác với artifact mô hình mới nhất (MAPE 2,17% so với 9,07%) | Người dùng có thể tin vào độ chính xác không đúng với mô hình đang chạy | Lưu metadata phiên bản (ngày huấn luyện, dữ liệu, chỉ số) cùng mô hình; hiển thị trong `/models/status`; chỉ cập nhật README từ kết quả của mô hình đang dùng |
| G-02 | Coverage của khoảng 80% / 90% thấp hơn mức danh nghĩa | Rủi ro bị đánh giá thấp — người dùng nghĩ khoảng giá "an toàn" hơn thực tế | Hiệu chỉnh lại conformal (tập hiệu chỉnh lớn hơn — hiện chỉ 69 mẫu, conformal theo từng horizon hoặc adaptive conformal); đặt coverage làm tiêu chí chấp nhận mô hình |
| G-03 | `POST /predict/tft` được khai báo 2 lần trong `api/main.py`; handler đăng ký trước kiểm tra và gọi **dự báo GBM** | Client gọi TFT nhưng nhận kết quả GBM (`model = "GBM"`) — sai hợp đồng API | Xóa handler trùng; trả `501 Not Implemented` cho đến khi có TFT serving |
| G-04 | Dashboard cho chọn BNB/USDT, SOL/USDT và timeframe 4h, 1w, trong khi cấu hình thu thập mặc định chỉ có BTC/ETH (và `.env.example` chỉ có `1d`) | Người dùng chọn các lựa chọn này sẽ gặp "No data found" | Lấy danh sách symbol / timeframe từ view `v_data_summary` thay vì danh sách cố định |
| G-05 | API chưa xác thực; `POST /models/reload` nhận tham số đường dẫn file tùy ý | Bất kỳ ai truy cập được API đều có thể thay mô hình đang phục vụ | Thêm API key / vai trò quản trị cho các endpoint quản trị; giới hạn đường dẫn trong thư mục output |
| G-06 | Mô tả trường `candles` ghi "cần ít nhất 30 rows" nhưng kiểm tra thực tế chỉ yêu cầu 10; khi CSDL có ít hơn 10 nến, `/predict/gbm` trả `500` thay vì `422` | Hợp đồng dữ liệu không nhất quán, client khó biết yêu cầu đúng và không phân biệt được lỗi dữ liệu với lỗi hệ thống | Thống nhất một ngưỡng (khuyến nghị ≥ 30 để đủ tính lag 21 bước), trả `422` cho lỗi dữ liệu không đủ và cập nhật tài liệu |
| G-07 | Chưa có TFT serving (`predict_tft` trả `NotImplementedError`) | Chưa khai thác được mô hình nghiên cứu cho người dùng cuối | Hoàn thiện serialize TFT, chỉ đưa vào sử dụng khi vượt GBM trên cùng tập test |
| G-08 | Chưa có cảnh báo bất thường thời gian thực (chỉ có biểu đồ Z-score trong EDA) | Người dùng phải tự theo dõi dashboard | Job phát hiện bất thường (Z-score / biến động) + kênh thông báo |
| G-09 | Chưa có CI chạy tự động bộ kiểm thử | Lỗi hồi quy (như G-03) không được phát hiện sớm | Thiết lập GitHub Actions chạy `pytest` và lint cho mỗi pull request |
