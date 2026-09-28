# Biomedical Image Registration

So sánh 4 phương pháp đăng ký ảnh y khoa 3D trên cùng một pipeline dữ liệu, huấn luyện và đánh giá thống nhất: **Diffeomorphic Demons**, **PSO (affine)**, **VoxelMorph-style CNN**, và **TransMorph-style Transformer**.

Capstone Project — Computer Vision (IT4343E), Hanoi University of Science and Technology.

## Bài toán

Cho ảnh cố định F và ảnh di động M, ước lượng phép biến đổi φ sao cho ảnh đã warp M∘φ khớp với F về cả cường độ và giải phẫu:

```
φ* = argmin_φ  D(F, M∘φ) + λ·R(φ)
```

## Phương pháp

| Phương pháp | Loại | Cần huấn luyện | Đầu ra |
|---|---|---|---|
| Diffeomorphic Demons | Dense, cổ điển | Không | Trường dịch chuyển (displacement field) |
| PSO | Affine, metaheuristic | Không | 12 tham số affine |
| VoxelMorph-style CNN | Dense, học sâu | Có | Trường dịch chuyển |
| TransMorph-style Transformer | Dense, học sâu | Có | Trường dịch chuyển |

## Dữ liệu

OASIS-1 (T1-weighted brain MRI, xử lý qua FreeSurfer): 425 volume → 424 cặp ảnh liền kề → chia train/val/test = 297/42/85 (seed cố định, xem `splits.json`).

## Kết quả (85 cặp test)

| Phương pháp | MSE ↓ | NCC ↑ | Dice ↑ | Thời gian chạy (s) ↓ | Tỷ lệ gấp mô % |
|---|---|---|---|---|---|
| Classical Demons | 0.0137 | 0.732 | 0.164 | 0.51 | 0.000 |
| PSO affine | 0.0109 | 0.793 | **0.331** | 9.36 | N/A |
| VoxelMorph | **0.0089** | **0.827** | 0.165 | **0.36** | 0.158 |
| TransMorph | 0.0099 | 0.807 | 0.219 | 0.39 | 0.035 |

Chi tiết phân tích: xem [báo cáo đầy đủ](docs/Group_15_Report.pdf).

## Cấu trúc thư mục

```
src/
├── data/              # load & tạo dữ liệu synthetic
├── methods/
│   ├── classical/     # Diffeomorphic Demons (SimpleITK)
│   ├── metaheuristic/ # PSO (from-scratch)
│   ├── voxelmorph/    # CNN model, train, inference
│   └── transmorph/    # Transformer model, train, inference
└── utils/             # metrics, warping, I/O
configs/                # cấu hình train (.yaml)
scripts/                # tiền xử lý dữ liệu, benchmark
tests/                  # unit test
```



## Công nghệ sử dụng

Python, PyTorch, SimpleITK, NiBabel, CUDA

