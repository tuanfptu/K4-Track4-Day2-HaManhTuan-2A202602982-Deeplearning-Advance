# DeepWeeds — Lab Day 2 — 2A202602982 Hà Mạnh Tuấn

## Kết quả đã đo

Bài dùng **DeepWeeds fold 0** (17.509 ảnh, 9 lớp), chạy trên **NVIDIA RTX 3090 24 GB**. Backbone được chọn là **ViT Tiny** (`B05`), công thức cuối thêm **RandAugment**, suy luận bằng trung bình xác suất của ảnh gốc và ảnh lật (`I01`). Trên test, 3 seed của `F01` đạt **macro-F1 0,9327 ± 0,0029**, **top-1 94,84% ± 0,31 điểm phần trăm**. Macro-F1 tăng **0,0140** so với mốc `F00` cùng backbone. Bộ chấm `eval.py` đề xuất **14/20 điểm phần I** theo ngưỡng tạm thời; đây không phải điểm toàn bài.

Sáu backbone `B01`–`B06`, bảy thí nghiệm công thức `T00`–`T06`, năm phương pháp suy luận `I00`–`I04` và sáu lượt train cuối (`F00`/`F01` × 3 seed) đã có số liệu thật. `B06` (ResNeXt-50) được thêm sau khi đã khóa lựa chọn chung kết; test predictions không đổi.

## Sản phẩm nộp

| Mục | Nội dung |
|---|---|
| [results.xlsx](results.xlsx) | Backbones, Training, Inference, Final, PerClass, Latency, BackboneLatency, Summary |
| [report.md](report.md) | Thiết lập, kết quả, đồ thị, phân tích lỗi, giới hạn |
| [curves/](curves/) | 19 biểu đồ riêng cho 6 B, 7 T và 6 lượt F |
| [predictions/](predictions/) | Dự đoán validation/test; F00 và F01 có đủ 3 seed |
| [code/](code/) | Mã nguồn, script chạy server, notebook Kaggle tham khảo và kiểm tra |
| [evidence/](evidence/) | Chia dữ liệu, phiên bản, log, hình EDA, phép đo độ trễ, tự chấm |
| [run_metadata/](run_metadata/) | Config, history, summary và kiểm tra split của 19 lượt train |

Checkpoint lớn và ảnh dataset **không có trong Git**. Notebook [deepweeds_kaggle_t4x2.ipynb](code/deepweeds_kaggle_t4x2.ipynb) là đường chạy Kaggle chuẩn bị trước đó; kết quả nộp này được tạo bằng **script server** bên dưới, rồi bổ sung B06 và bằng chứng audit. Notebook không phải nguồn của các con số RTX 3090.

## Chạy lại trên server NVIDIA

Repo: <https://github.com/tuanfptu/K4-Track4-Day2-HaManhTuan-2A202602982-Deeplearning-Advance>. Cần Python 3.10, driver NVIDIA tương thích, Internet để tải dữ liệu và trọng số ImageNet. Lượt đã đo dùng Python **3.10.12**, PyTorch **2.6.0+cu124**, torchvision **0.21.0+cu124**, timm **1.0.30**; các phiên bản khác xem [software.json](evidence/software.json). Từ thư mục gốc repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r submissions/2A202602982_HaManhTuan/requirements-server.txt
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
set -o pipefail
bash submissions/2A202602982_HaManhTuan/run_server.sh 2>&1 | tee server.log
```

`run_server.sh` tải Zenodo `images.zip`, kiểm tra MD5 và 17.509 ảnh, lấy CSV fold 0 gốc, chạy unit test và smoke test, rồi chạy `B01`–`B05`, `T00`–`T06`, `F00`/`F01` (3 seed), so sánh `I00`–`I04` trên validation và đánh giá test. Mặc định: **10 epoch** cho B/T, **12 epoch** cho F, batch 64, AMP, 4 worker; chọn checkpoint tốt nhất trên validation, **không early stopping**. Output ở `lab_output/full/`. Chạy lại cùng lệnh và cùng thư mục output sẽ bỏ qua các run đã hoàn tất với config khớp. Có thể đặt `DEEPWEEDS_DATA`, `DEEPWEEDS_OUTPUT`, `DEEPWEEDS_EPOCHS`, `DEEPWEEDS_FINAL_EPOCHS`, `DEEPWEEDS_WORKERS`; nếu đổi config sau khi đã chạy, dùng output mới.

Sau khi `lab_output/full/evidence/completed.json` xuất hiện, chạy bổ sung đúng thứ tự:

```bash
python submissions/2A202602982_HaManhTuan/code/supplement_backbone.py --output lab_output/full 2>&1 | tee supplement.log
python submissions/2A202602982_HaManhTuan/code/audit_evidence.py --output lab_output/full 2>&1 | tee audit.log
cp server.log supplement.log audit.log lab_output/full/evidence/
python submissions/2A202602982_HaManhTuan/code/package_submission.py \
  --output lab_output/full --submission submissions/2A202602982_HaManhTuan
python submissions/2A202602982_HaManhTuan/code/assemble_workbook.py \
  --submission submissions/2A202602982_HaManhTuan
python submissions/2A202602982_HaManhTuan/code/verify_submission.py \
  --submission submissions/2A202602982_HaManhTuan --data data
```

`supplement_backbone.py` train thêm `B06 = resnext50_32x4d` với config B01, cập nhật bảng Backbones và lưu kết quả validation. `audit_evidence.py` đo p50/p95/p99 của cả sáu backbone, lưu ảnh sau augmentation và kiểm tra học một batch nhỏ. **Kiểm tra một batch trong lần nộp này được làm sau lượt train chính**; không mô tả nó như bước kiểm tra trước train. Script đóng gói giữ config/history nhỏ, loại checkpoint.

`assemble_workbook.py` điền các cột yêu cầu trong GUIDE từ config và predictions đã lưu; `verify_submission.py` tính lại metric từ predictions và kiểm tra số liệu khớp workbook. Hai script này chỉ xử lý kết quả đã đo, không train model và không chọn lại cấu hình từ test.

Để tự kiểm lại điểm phần I, chạy `python eval.py score` và `python eval.py grade` với các file trong `predictions/` và CSV fold 0 theo ví dụ trong [README gốc](../../README.md). Kết quả đã lưu tại [grade_I.json](evidence/grade/grade_I.json). `eval.py` của đề được giữ nguyên.

## Giới hạn

Điểm screening B/T dùng một seed và một fold; các backbone dùng các bộ pretrained khác nhau. Phân chia sẵn có thể lạc quan khi triển khai tại địa điểm hoặc mùa khác. Phần I4a được 0 điểm vì phương pháp I01 được chọn không dùng temperature scaling. Độ trễ báo cáo là thời gian GPU, chưa gồm đọc và tiền xử lý ảnh. B06 được bổ sung sau khi test đã chốt và **không dùng test để chọn lại mô hình**. Xem [báo cáo](report.md) để biết phân tích chi tiết và các con số đo được.
