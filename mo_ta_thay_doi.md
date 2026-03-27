# Mô Tả Thay Đổi — Chuẩn Bị Huấn Luyện Lại ABSA

> Ngày: 2026-03-27 | Tổng: **8 files** modified/created | Mục đích: Bổ sung đầy đủ artifacts cho luận văn

---

## Tổng quan

Sửa đổi 4 training scripts và 4 analysis scripts để bổ sung:
1. **Predictions export** — lưu per-sample predictions với exact-match columns cho McNemar's test
2. **McNemar's test** — kiểm định thống kê sự khác biệt giữa 6 cặp model
3. **LaTeX tables** — 3 bảng cho chương 4 luận văn
4. **Error analysis** — phân tích lỗi chi tiết, top-20 misclassified
5. **Inference benchmark** — đo latency cho các backbone

---

## Chi tiết thay đổi

### Training Scripts (4 files)

| File | Thay đổi |
|------|----------|
| `VisoBERT-MTL/train_visobert_mtl.py` | +47 dòng: thêm `save_mtl_predictions()` + gọi trong `main()` |
| `phoBERT-MTL/train_phobert_mtl.py` | +47 dòng: thêm `save_mtl_predictions()` + gọi trong `main()` |
| `BILSTM-MTL/train_bilstm_mtl.py` | +47 dòng: thêm `save_mtl_predictions()` + gọi trong `main()` |
| `BILSTM-STL/train_two_stage_bilstm.py` | +48 dòng: thêm prediction saving cho cả AD stage và SC stage |

**Hàm `save_mtl_predictions()`** lưu CSV với format:
- Per-aspect: `{Aspect}_ad_pred`, `{Aspect}_ad_true`, `{Aspect}_ad_correct`
- Per-aspect: `{Aspect}_sc_pred`, `{Aspect}_sc_true`, `{Aspect}_sc_correct`  
- Per-sample: `ad_exact_match`, `sc_exact_match` (dùng cho McNemar)

### Analysis Scripts (4 files)

| File | Thay đổi |
|------|----------|
| `scripts/run_mcnemar_test.py` | **Viết lại hoàn toàn** (~280 dòng) — sửa path tìm predictions, thêm Bonferroni correction, hỗ trợ cả MTL/STL format |
| `scripts/generate_thesis_tables.py` | **Viết lại hoàn toàn** (~230 dòng) — 3 bảng LaTeX (overall, per-aspect, McNemar) |
| `scripts/run_error_analysis_all.py` | **Viết lại hoàn toàn** (~200 dòng) — top-20 misclassified, error pattern analysis |
| `scripts/run_inference_benchmark.py` | **Tạo mới** (~170 dòng) — benchmark backbone latency (ViSoBERT, PhoBERT) |

---

## Vấn đề kỹ thuật đã sửa

| # | Vấn đề | Nguyên nhân | Cách sửa |
|:-:|--------|-------------|----------|
| 1 | MTL scripts không lưu predictions | `evaluate_mtl()` trả về numpy nhưng không ghi CSV | Thêm `save_mtl_predictions()` |
| 2 | McNemar tìm sai path | Script cũ tìm `results/**/predictions_detailed_ad.csv` | Rewrite với paths đúng cho cả MTL/STL |
| 3 | BiLSTM-STL không lưu gì | Không có code save predictions cho AD lẫn SC | Thêm inline prediction saving cho cả 2 stages |
| 4 | STL thiếu exact-match | CSV có per-aspect nhưng thiếu cột tổng hợp | McNemar script tự compute từ per-aspect columns |
| 5 | W&B run bị trùng tên | Hardcoded `name` trong `wandb.init()` | Thêm dynamic timestamp vào tên run trên cả 6 scripts |

---

## Cách chạy trên Kaggle

Notebook `absa-kaggle-training.ipynb` đã có sẵn section 8 (McNemar + LaTeX) và section 9 (Save results). Chỉ cần:

```bash
# Section 8 cells — chạy McNemar + LaTeX + Error Analysis
python scripts/run_mcnemar_test.py --results_dir results/ABSA-results
python scripts/generate_thesis_tables.py --results_dir results/ABSA-results
python scripts/run_error_analysis_all.py --results_dir results/ABSA-results

# Section 8.5 — Inference Benchmark (nên thêm cell mới)
python scripts/run_inference_benchmark.py --results_dir results/ABSA-results
```

## Verification

```bash
# Syntax check — tất cả 8 files đều pass ✓
python -c "import ast; [ast.parse(open(f).read()) for f in ['file1.py', ...]]"
```

## Outputs dự kiến

Sau khi chạy notebook xong, thư mục `ABSA-results/` sẽ có thêm:
- 6 × `test_predictions_detailed.csv` — predictions cho mỗi model
- `error_analysis_results/mcnemar_results.json` + `.txt`
- `error_analysis_results/thesis_tables.tex`
- `error_analysis_results/error_analysis_detailed.json` + `_report.txt`
- `error_analysis_results/inference_benchmark.json` + `.txt`
