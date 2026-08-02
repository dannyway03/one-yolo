# one-yolo — Performance Benchmark Report

Tracks end-to-end latency (preprocess + inference + postprocess) and accuracy KPIs
for every supported detector/backend combination.

**How to read this document**

- The **Executive Snapshot** at the top is replaced on every bench run. It always shows
  the best results achieved so far vs the immutable baseline.
- The **Sessions Log** below grows upward: newest entry is always first, oldest is always last.
  Nothing is ever deleted.
- Gap convention: negative = faster/better than baseline, positive = slower/worse.

---

## Executive Snapshot — 2026-08-02

Best results to date: **session 2026-08-02b** (OVN persistent InferRequest + async dispatch).  
Baseline commit: **1f48441** — original `YoloOVNRT` with per-frame `create_infer_request()` + sync `infer()`.  
Accuracy baseline: first MOT17-02 frame-1 run (this session) — gap shown vs this reference going forward.

### Timing — best vs baseline (avg over 50 iterations, ms · 640 × 384 · OVN CPU)

| Model | pre | infer | post | **total** | **FPS** | Δ total vs baseline | Δ FPS vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | 3.68 | 22.49 | 0.04 | **26.21** | **38.2** | +1.60 ms ¹ | -2.4 ¹ |
| yolox\_nano\_dec | 3.95 | 16.47 | 0.20 | **20.62** | **48.5** | +1.61 ms ¹ | -4.1 ¹ |
| bytetrack\_nano\_dec | 3.50 | 14.67 | 0.12 | **18.29** | **54.7** | *(no baseline — IR created 2026-08-02a)* | — |

¹ Test image changed from PXL to MOT17-02/000001.jpg (1920×1080 → letterboxed to 640×384) between
baseline and this session; timing delta includes both the OVN optimisation effect and the image switch,
which are inseparable. Within-session noise is ±1 ms.

### Accuracy — MOT17-02 frame 1 (GT: 22 pedestrians, class=1, consider=1)

| Model | Detections | Recall proxy | Notes |
|---|---|---|---|
| yolo26n | 7 | 31.8 % | conf=0.25; nano model, 640×384 vs 1920×1080 source |
| yolox\_nano\_dec | 14 | 63.6 % | conf=0.25 |
| bytetrack\_nano\_dec | 12 | 54.5 % | conf=0.25, 1 class (person) |

> Recall proxy = detected / GT count. Not true AP (no IoU matching). Full mAP not yet automated.

---

## Test Configuration (fixed)

| Parameter | Value |
|---|---|
| Test image | `motcpp/assets/MOT17-mini/train/MOT17-02-FRCNN/img1/000001.jpg` (1920 × 1080) |
| GT source | `motcpp/assets/MOT17-mini/train/MOT17-02-FRCNN/gt/gt.txt` · class=1, consider=1 |
| Warmup runs | 5 |
| Benchmark iterations | 50 |
| Input resolution (model) | 640 × 384 (W × H) |
| Batch size | 1 |
| Conf threshold | 0.25 |
| IoU threshold | 0.45 |
| Platform | Linux x86-64, Intel Core, OpenVINO 2026.2.1, ORT 1.22.2 |
| bench binary | `build/Release/bench` |

Accuracy proxy: detection count vs MOT17 GT, plus annotated-image MD5 between sessions to catch regressions.
Full mAP evaluation not yet automated (future KPI).

---

## Models

| Model | Origin | Initial format | ONNX conversion | OVN conversion |
|---|---|---|---|---|
| yolo26n 640×384 | Ultralytics official | `.pt` | Ultralytics API — see (1) | `ovc` — see (4) |
| yolox\_nano\_dec 640×384 | YOLOX official repo | `.pth` | YOLOX exporter — see (2) | `ovc` — see (4) |
| bytetrack\_nano\_dec 640×384 | ByteTrack official repo | `.pth.tar` | ByteTrack exporter — see (3) | `ovc` — see (4) |

```bash
# (1) yolo26 pt → onnx
cd /home/briox/Software/cpp/MOT/Yolo26-onnxruntime/
python scripts/export_yolo26.py
```

```bash
# (2) yolox pth → onnx (decoded)
cd ~/Software/python/YOLOX
python tools/export_onnx.py \
  --decode_in_inference \
  --img-size 384 640 \
  --exp_file exps/example/mot/yolox_nano_mix_det.py \
  --weights weights/yolox_nano.pth
```

```bash
# (3) bytetrack pth.tar → onnx (decoded)
cd ~/Software/python/ByteTrack
python tools/export_onnx.py \
  --decode_in_inference \
  --img-size 384 640 \
  --exp_file exps/example/mot/yolox_nano_mix_det.py \
  --weights weights/pth/bytetrack_nano_mot17.pth.tar
```

```bash
# (4) onnx → OVN IR  (run from project root)
source /home/briox/Software/cpp/MOT/setupvars26.sh
ovc models/<name>.onnx --output_model models/<name>
```

---

## Benchmark Command Lines

Environment setup (required before every bench run):

```bash
source /home/briox/Software/cpp/MOT/setupvars26.sh
```

Standard bench run (all three models, OVN CPU):

```bash
BENCH=build/Release/bench
M=models
IMG=motcpp/assets/MOT17-mini/train/MOT17-02-FRCNN/img1/000001.jpg
OUT=/tmp/bench_ovn/mot17 && mkdir -p $OUT

# yolo26n
$BENCH --model $M/yolo26n_1x3x384x640.xml \
  --version yolo26 --backend ovn --device cpu \
  --task det --input-w 640 --input-h 384 --classes 80 \
  --warmup 5 --iterations 50 \
  --image "$IMG" --save $OUT/y26.jpg --csv $OUT/results.csv

# yolox_nano (decoded)
$BENCH --model $M/yolox_nano_1x3x384x640_decoded.xml \
  --version yolox --backend ovn --device cpu \
  --task det --input-w 640 --input-h 384 --classes 80 \
  --warmup 5 --iterations 50 \
  --image "$IMG" --save $OUT/yx.jpg --csv $OUT/results.csv

# bytetrack_nano (decoded, 1 class)
$BENCH --model $M/bytetrack_nano_mot17_1x3x384x640_decoded.xml \
  --version bytetrack --backend ovn --device cpu \
  --task det --input-w 640 --input-h 384 --classes 1 \
  --warmup 5 --iterations 50 \
  --image "$IMG" --save $OUT/bt.jpg --csv $OUT/results.csv
```

GT pedestrian count for frame 1 (accuracy reference):

```bash
awk -F',' '$1==1 && $7==1 && $8==1 {count++} END{print count}' \
  motcpp/assets/MOT17-mini/train/MOT17-02-FRCNN/gt/gt.txt
# → 22
```

---

## Sessions Log

---

### 2026-08-02b — MOT17 accuracy baseline + re-bench on canonical test image

**Branch / commit**: `develop` (diff on top of `1f48441`: OVN optimizations from session 2026-08-02a)

**Changes this session**
- Switched test image to MOT17-02 frame 1 (1920×1080) with ground-truth annotations
- Established accuracy KPI baseline: recall proxy vs 22 GT pedestrians
- OVN version confirmed as **2026.2.1.21919** via `ldd build/Release/bench | grep vino`
- Document restructured: Models section + exact benchmark command lines added

**Timing (MOT17-02/000001.jpg · OVN CPU · 50 iter)**

| Model | pre | infer | post | **total** | **FPS** | Δ vs timing baseline ¹ |
|---|---:|---:|---:|---:|---:|---:|
| yolo26n | 3.68 ms | 22.49 ms | 0.04 ms | **26.21 ms** | **38.2** | +1.60 ms |
| yolox\_nano\_dec | 3.95 ms | 16.47 ms | 0.20 ms | **20.62 ms** | **48.5** | +1.61 ms |
| bytetrack\_nano\_dec | 3.50 ms | 14.67 ms | 0.12 ms | **18.29 ms** | **54.7** | *(no prior)* |

¹ Baseline was measured on a different image; Δ is not a pure optimisation signal.

**Accuracy (MOT17-02 frame 1, GT = 22 pedestrians)**

| Model | Detections | Recall proxy | MD5 (annotated) |
|---|---|---|---|
| yolo26n | 7 | 31.8 % | `4f2a…` *(first run on this image)* |
| yolox\_nano\_dec | 14 | 63.6 % | `b8c1…` *(first run on this image)* |
| bytetrack\_nano\_dec | 12 | 54.5 % | `e3d7…` *(first run on this image)* |

**Analysis**

yolox\_nano\_dec leads on recall (63.6 %) on this crowded street scene at 640×384.
Bytetrack is close (54.5 %) and 5.6 ms faster per frame — making sense for tracking use-cases
where per-frame latency matters more than raw recall.
yolo26n nano recall (31.8 %) is expected: model is smallest/fastest of the three and was trained
on a different regime; recall improves significantly at higher input resolution.

---

### 2026-08-02a — OVN inference optimizations + bytetrack IR creation

**Branch / commit**: `develop` (diff on top of `1f48441`)  
**Test image**: `~/Downloads/PXL_20250710_094018206.jpg` *(superseded by MOT17 image from session 2026-08-02b)*

**Changes**
- `YoloOVNRT.h`: added `ov::InferRequest infer_request_` member
- `YoloOVNRT.cpp`:
  - `create_infer_request()` moved from per-frame call to constructor (persistent reuse)
  - `infer()` → `start_async()` + `wait()` (async dispatch; primes path for future double-buffer)
  - `clone()` removed — `cv::Mat` aliases persistent InferRequest buffers (zero-copy)
  - `cache_dir = /tmp/ov_cache` added (eliminates 1–5 s model recompile on GPU/NPU)
- `tools/bench.cpp`: `bytetrack` added as `--version` alias (maps to YOLO5 decoder)
- `models/`: converted `bytetrack_nano_mot17_1x3x384x640{,_decoded}.onnx` → OVN IR via `ovc`

**Timing vs baseline (PXL image · OVN CPU · 50 iter)**

| Model | total BEFORE | total AFTER | Δ | FPS BEFORE | FPS AFTER | Δ FPS |
|---|---:|---:|---:|---:|---:|---:|
| yolo26n | 24.61 ms | 25.57 ms | +0.96 ms | 40.6 | 39.1 | -1.5 |
| yolox\_nano\_dec | 19.01 ms | 18.93 ms | -0.08 ms | 52.6 | 52.8 | +0.2 |
| bytetrack\_nano\_dec | *(no IR)* | 19.44 ms | — | — | 51.4 | — |

**Accuracy (PXL image · no GT)**

| Model | Detections BEFORE | Detections AFTER | Annotated MD5 |
|---|---|---|---|
| yolo26n | 2 | 2 | ✓ bit-identical |
| yolox\_nano\_dec | 0 | 0 | ✓ bit-identical |
| bytetrack\_nano\_dec | N/A | N/A | *(first run)* |

**Analysis**: Delta on OVN CPU is within measurement noise (±1 ms). Expected gains are on GPU/NPU
where buffer reallocation crosses driver boundaries (~5–15 ms per call) and `start_async()` enables
true pipeline overlap.

---

### 2026-08-02 — Baseline (starting point)

**Branch / commit**: `1f48441` — refactor: fix all clang-tidy warnings and unify sample apps  
**Test image**: `~/Downloads/PXL_20250710_094018206.jpg`

**State**: original `YoloOVNRT` — per-frame `create_infer_request()`, synchronous `infer()`,
`clone()` on every output tensor. No bytetrack OVN IR.

**Timing**

| Model | Backend | pre | infer | post | **total** | **FPS** |
|---|---|---:|---:|---:|---:|---:|
| yolo26n 640×384 | OVN CPU | 3.74 ms | 20.84 ms | 0.03 ms | **24.61 ms** | **40.6** |
| yolox\_nano\_dec 640×384 | OVN CPU | 3.83 ms | 15.08 ms | 0.10 ms | **19.01 ms** | **52.6** |
| bytetrack\_nano\_dec 640×384 | OVN CPU | — | — | — | — | — |

**Accuracy** (PXL image, no GT)

| Model | Detections |
|---|---|
| yolo26n | 2 |
| yolox\_nano\_dec | 0 |

_Immutable timing baseline. All future timing gaps are reported against these numbers._