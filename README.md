<p style="" align="center">
  <img src="./docs/logo.png" alt="Logo" width="85%">
</p>
<p style="margin:0px;color:gray" align="center">
[🚀🚀🚀All Yolo Tasks · All Yolo Versions · All Yolo Runtimes🚀🚀🚀]
</p>
<p style="margin:0px" align="center">
  <a href='./README_CN.md'>中文README</a> | <a href='https://github.com/sherlockchou86/VideoPipe'>VideoPipe </a>
</p>

# one-yolo

A unified C++ toolkit for YOLO `v5/v8/v11/v26/...`, covering `classification/detection/segmentation/pose/obb` tasks with
easy python-like APIs from `ultralytics/ultralytics`. Support `All Yolo Tasks, All Yolo Versions, All Yolo Runtimes`,
it's time to make all in one.
<p style="" align="center">
  <img src="./docs/showcase.gif" alt="Logo" width="85%">
</p>

## ✨ highlight

1. support all `Yolo` tasks including `classification`/`detection`/`segmentation`/`pose`/`obb`.
2. support all `Yolo` versions including `yolov5(anchor-based)`/`yolov5u(anchor-free)`/`yolov8`/`yolov11`/
   `yolov26(nms-free)`/`more in the future`, sub versions like `n/s/m/l/x` are also supported.
3. support all `Yolo` inference backends(runtime) such as `OpenCV::DNN`/`ONNXRuntime`/`TensorRT`/`OpenVINO`/`RKNN`/
   `CoreML`/`CANN`/`PaddlePaddle`...
4. easy APIs to use and integrate, as simple as python APIs from `ultralytics/ultralytics` library.
5. toolkit works out of box, provide the model and set up the config parameters, go predict!

## 🚀 quick start

### requirements

1. C++ >= 17, clang++ >= 20 or GCC >= 12
2. CMake >= 3.21 (for preset support)
3. OpenCV >= 4.10
4. ONNXRuntime / OpenVINO / TensorRT / RKNN — optional, enable via build flags

### build with CMakePresets (recommended)

```bash
git clone https://github.com/sherlockchou86/one-yolo.git
cd one-yolo

# configure (choose Debug / Release / RelWithDebInfo / rr)
cmake --preset Release \
  -DBUILD_WITH_ORT=ON \   # ONNXRuntime
  -DBUILD_WITH_OVN=ON     # OpenVINO

# build everything
cmake --build --preset Release

# or build specific targets
cmake --build --preset Release --target det cls seg pose obb bench
```

Available `-DBUILD_WITH_*` flags:

| flag                | backend                      |
|---------------------|------------------------------|
| `BUILD_WITH_ORT=ON` | ONNXRuntime (CPU / CUDA)     |
| `BUILD_WITH_OVN=ON` | OpenVINO (CPU / iGPU / AUTO) |
| `BUILD_WITH_TRT=ON` | TensorRT (NVIDIA GPU)        |
| `BUILD_WITH_RKN=ON` | RKNN (RockChip NPU)          |
| `BUILD_WITH_CML=ON` | CoreML (Apple)               |
| `BUILD_WITH_PDL=ON` | PaddlePaddle                 |
| `BUILD_WITH_CAN=ON` | CANN (HuaWei NPU)            |

Without any flag, OpenCV::DNN is used as the default backend.

### legacy cmake (no presets)

```bash
mkdir build && cd build
cmake .. -DBUILD_WITH_ORT=ON -DBUILD_WITH_OVN=ON
make -j8
```

### run samples

Samples live in `build/<preset>/samples/`. All share the same flags:

```
--model      <path>            .onnx or .xml model file
--version    yolox|yolo26|yolo11|yolo8|yolo5|yolo5u
--backend    ort|ovn|dnn       inference backend
--device     cpu|gpu|auto|cuda
--source     <path|0|1|...>   image, video file, or webcam index
--input-w    <int>             model input width  (default: 640)
--input-h    <int>             model input height (default: 640)
--classes    <int>             number of classes  (default: 80)
--names      <a,b,c,...>       comma-separated class names
--conf       <float>           confidence threshold (default: 0.25)
--iou        <float>           NMS IoU threshold   (default: 0.45)
--scale      <float>           display scale       (default: 1.0)
--no-json                      suppress per-frame JSON output
--no-csv                       suppress per-frame CSV output
--no-track                     disable SORT tracker (det/seg/pose)
```

**Detection — YOLOX-Nano on OpenVINO CPU**

```bash
./det --model models/yolox_nano_1x3x384x640_decoded.xml \
      --version yolox --backend ovn --input-w 640 --input-h 384 \
      --source media/crowd_mall.mp4
```

**Detection — YOLO11n on ONNXRuntime, webcam**

```bash
./det --model models/yolo11n.onnx --version yolo11 --backend ort \
      --source 0 --no-json --no-csv
```

**Detection — custom 3-class model**

```bash
./det --model models/fire_smoke_yolo8s.onnx \
      --version yolo8 --backend ovn --device cpu \
      --classes 3 --names "fire,smoke,light" \
      --source media/fire.mp4
```

**Classification — YOLO26n-cls on ONNXRuntime**

```bash
./cls --model models/yolo26n-cls.onnx --version yolo26 --backend ort \
      --input-w 224 --input-h 224 --classes 1000 \
      --source test_images/cat.jpg
```

**Segmentation — YOLO8n-seg on OpenVINO AUTO**

```bash
./seg --model models/yolo8n-seg.onnx --version yolo8 \
      --backend ovn --device auto \
      --source media/crowd_mall.mp4
```

**Pose estimation — YOLO11n-pose on OpenVINO CPU**

```bash
./pose --model models/yolo11n-pose.onnx --version yolo11 --backend ovn \
       --classes 1 --names person \
       --source media/people.mp4
```

**OBB — YOLO8n-obb on ONNXRuntime**

```bash
./obb --model models/yolo8n-obb.onnx --version yolo8 --backend ort \
      --input-w 1024 --input-h 1024 \
      --classes 15 --names "plane,ship,storage tank,..." \
      --source test_images/satellite.png
```

### bench — pipeline latency benchmark

Measures the full pipeline (preprocess + inference + postprocess) with percentile stats and optional CSV logging:

```bash
# YOLOX-Nano: OVN vs ORT side by side
./bench --model models/yolox_nano_1x3x384x640_decoded.xml \
        --version yolox --backend ovn --device cpu \
        --input-w 640 --input-h 384 \
        --warmup 10 --iterations 100 \
        --image media/frame.jpg --csv results.csv

./bench --model models/yolox_nano_1x3x384x640_decoded.onnx \
        --version yolox --backend ort \
        --input-w 640 --input-h 384 \
        --warmup 10 --iterations 100 \
        --image media/frame.jpg --csv results.csv

# omit --image to use random noise (pure throughput, no I/O)
./bench --model models/yolo11n.onnx --version yolo11 --backend ort --iterations 200
```

`results.csv` accumulates one row per run — compare OVN vs ORT vs DNN across versions in a single file.

### hello one-yolo

vehicle detection & tracking task using `yolov8s`:

```c++
#include "Yolo.h"
#include "track/YoloTracker.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc_        = "vehicle detection task using yolov8s(custom model)";
    cfg.version_     = YoloVersion::YOLO8;
    cfg.task_        = YoloTaskType::DET;
    cfg.target_rt_   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path_  = "./vp_data/models/det_cls/vehicel_v8s-det_c6_20260205.onnx";
    cfg.input_w_     = 640;
    cfg.input_h_     = 384;
    cfg.batch_size_  = 1;
    cfg.num_classes_ = 6;
    cfg.names_       = {"person", "car", "bus", "truck", "2wheel", "other"};

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* 3. construct YoloTrackConfig */
    YoloTrackConfig t_cfg;
    t_cfg.algo = YoloTrackAlgo::SORT;
    t_cfg.iou_thresh = 0.6f;

    /* 4. create YoloTracker using YoloTrackConfig */
    auto tracker = YoloTracker(t_cfg);
    tracker.info();

    /* 5. open video and predict frames in a loop */
    cv::VideoCapture cap("./vp_data/test_video/rgb.mp4");
    while (cap.isOpened()) {
        // collect frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // resize original image
        if (frame.cols > 720) {
            cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
        }
        
        // predict with batch mode (batch size == 1)
        auto results = model(std::vector<cv::Mat>{frame});

        // track result
        tracker(results[0]);

        // show and print
        results[0].info();           // print summary
        results[0].to_json(true);    // convert structured result to json and print
        results[0].toCsv(true);     // convert structured result to csv and print
        if (results[0].show(
            false, 1.0f, DrawParam(), // show annotated image & input image(640*384) & original image with unblock mode
            true, true) == 27) {      // exit loop if user has pressed ESC
            break;
        }

        /*
         * you can also get structured results like below:
         * auto boxes        = results[0].boxes();          // get bounding boxes in detection task
         * auto cls_ids      = results[0].cls_ids();        // get class ids in detection task
         * auto confs        = results[0].confs();          // get confidences in detection task
         * auto labels       = results[0].labels();         // get labels in detection task
         * auto track_ids    = results[0].track_ids();      // get track ids in detection task
         * auto track_points = results[0].track_points();   // get track points in detection task
        */
    }
}
```

### demo video

video result of vehicle detection & tracking using yolov8s:

https://github.com/user-attachments/assets/d8b0b711-8922-41f8-8ec7-d1cea1f48afc

### demo output

json/csv output result of vechile detection & tracking using yolov8s:

```
json output:
[
    {
        "box": {
            "height": 76,
            "width": 33,
            "x": 368,
            "y": 378
        },
        "cls_id": 4,
        "conf": 0.8655326962471008,
        "label": "2wheel",
        "track_id": 1
    },
    {
        "box": {
            "height": 21,
            "width": 10,
            "x": 647,
            "y": 145
        },
        "cls_id": 4,
        "conf": 0.8104556202888489,
        "label": "2wheel",
        "track_id": 37
    },
    {
        "box": {
            "height": 15,
            "width": 9,
            "x": 676,
            "y": 137
        },
        "cls_id": 4,
        "conf": 0.7772445678710938,
        "label": "2wheel",
        "track_id": 23
    },
    {
        "box": {
            "height": 14,
            "width": 7,
            "x": 710,
            "y": 118
        },
        "cls_id": 4,
        "conf": 0.523908257484436,
        "label": "2wheel",
        "track_id": 41
    },
    {
        "box": {
            "height": 14,
            "width": 12,
            "x": 793,
            "y": 93
        },
        "cls_id": 3,
        "conf": 0.5332302451133728,
        "label": "truck",
        "track_id": 44
    },
    {
        "box": {
            "height": 128,
            "width": 113,
            "x": 494,
            "y": 369
        },
        "cls_id": 1,
        "conf": 0.9514954090118408,
        "label": "car",
        "track_id": 5
    },
    {
        "box": {
            "height": 9,
            "width": 13,
            "x": 721,
            "y": 117
        },
        "cls_id": 1,
        "conf": 0.7941694259643555,
        "label": "car",
        "track_id": 25
    },
    {
        "box": {
            "height": 9,
            "width": 14,
            "x": 753,
            "y": 116
        },
        "cls_id": 1,
        "conf": 0.7911720871925354,
        "label": "car",
        "track_id": 13
    },
    {
        "box": {
            "height": 11,
            "width": 13,
            "x": 770,
            "y": 107
        },
        "cls_id": 1,
        "conf": 0.5813544988632202,
        "label": "car",
        "track_id": 42
    }
]
csv output:
id,cls_id,conf,label,track_id
1,4,0.865533,2wheel,1
2,4,0.810456,2wheel,37
3,4,0.777245,2wheel,23
4,4,0.523908,2wheel,41
5,3,0.53323,truck,44
6,1,0.951495,car,5
7,1,0.794169,car,25
8,1,0.791172,car,13
9,1,0.581354,car,42
```

## 🆒 architecture diagram

See [docs/BENCHMARKS.md — System Architecture & Dataflow](./docs/BENCHMARKS.md#system-architecture--dataflow) for class
hierarchy, sequential and pipeline dataflow diagrams (Mermaid, rendered on GitHub).

## 📚 references

1. [Samples](./samples/) — `det` / `cls` / `seg` / `pose` / `obb` unified CLI apps + `common.hpp`
2. [Tools](./tools/) — `bench` latency benchmark, `test_yolox_img` headless image test
3. [VideoPipe](https://github.com/sherlockchou86/VideoPipe) for integrating Yolo
