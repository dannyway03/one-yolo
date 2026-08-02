# Plan: Slim one-yolo — ORT + OVN, DET + CLS, JSON config

**Goal**: Two independent workstreams executed in order.
1. **Phase 1** — delete dead backends (OpenCV-DNN, TRT, RKNN), dead tasks (SEG, POSE, OBB).  
   Tracker subsystem is **kept**.
2. **Phase 2** — replace the wall of CLI flags with a single JSON config file;
   add `YoloConfig::from_json()`; shrink common.hpp; update bench.

**Executor rules**: follow every step exactly. Do not reorganise, rename, or improve anything not in
the list. Do not touch files not mentioned. Run the gate check before moving to the next step.
If a gate fails, stop and report — do not attempt to fix silently.

---

## Phase 0 — Backup (ALREADY DONE — skip)

Backup branch `backup/pre-slim-20260802` exists on remote. `develop` is up to date.

---

## Phase 1 — Slim backends and tasks

### 1-A  Delete removed backend files

```bash
git rm include/YoloOpenCVRT.h src/YoloOpenCVRT.cpp
git rm include/trt/YoloTRT.h  src/trt/YoloTRT.cpp
git rm include/rkn/YoloRKNNRT.h src/rkn/YoloRKNNRT.cpp
```

> **Do NOT touch** `include/track/` or `src/track/` — the tracker is kept.

**Gate 1-A**: all three commands must exit 0; `ls include/trt include/rkn` must fail.

---

### 1-B  Delete removed task files

```bash
git rm include/YoloSegTask.h  src/YoloSegTask.cpp
git rm include/YoloPoseTask.h src/YoloPoseTask.cpp
git rm include/YoloObbTask.h  src/YoloObbTask.cpp
```

**Gate 1-B**: `ls include/YoloSeg* include/YoloPose* include/YoloObb*` must fail.

---

### 1-C  Delete removed sample apps

```bash
git rm samples/seg.cpp samples/pose.cpp samples/obb.cpp
```

**Gate 1-C**: `ls samples/` shows only `cls.cpp`, `det.cpp`, `CMakeLists.txt`.

---

### 1-D  Edit `include/YoloConfig.h` — trim enums

**`YoloTaskType`** — replace body with:
```cpp
enum class YoloTaskType
{
  CLS,
  DET,
};
```

**`YoloTargetRT`** — replace body with:
```cpp
enum class YoloTargetRT
{
  ORT_CPU,     // use onnxruntime(cpu) as inference backend for Yolo
  ORT_CUDA,    // use onnxruntime(cuda) as inference backend for Yolo
  OVN_AUTO,    // use openvino(auto) as inference backend for Yolo
  OVN_CPU,     // use openvino(cpu) as inference backend for Yolo
  OVN_GPU,     // use openvino(integrated gpu) as inference backend for Yolo
};
```

**Gate 1-D**: `grep -n "OPENCV\|TRT\|RKNN\|SEG\|POSE\|OBB" include/YoloConfig.h` → 0 lines.

---

### 1-E  Edit `src/Yolo.cpp` — remove dead task includes and switch cases

Remove these three includes (top of file):
```cpp
#include "YoloObbTask.h"
#include "YoloPoseTask.h"
#include "YoloSegTask.h"
```

Remove these three switch cases:
```cpp
    case YoloTaskType::SEG:
      task_ = std::make_shared<yolo::YoloSegTask>(cfg);
      break;
    case YoloTaskType::POSE:
      task_ = std::make_shared<yolo::YoloPoseTask>(cfg);
      break;
    case YoloTaskType::OBB:
      task_ = std::make_shared<yolo::YoloObbTask>(cfg);
      break;
```

**Gate 1-E**: `grep -n "Seg\|Pose\|Obb" src/Yolo.cpp` → 0 lines.

---

### 1-F  Edit `src/YoloTask.cpp` — remove dead backend includes and switch cases

Remove (unconditional include, line 8):
```cpp
#include "YoloOpenCVRT.h"
```

Remove (guarded blocks):
```cpp
#ifdef BUILD_WITH_TRT
  #include "trt/YoloTRT.h"
#endif
#ifdef BUILD_WITH_RKN
  #include "rkn/YoloRKNNRT.h"
#endif
```

Remove from the `switch (_cfg.target_rt_)` statement:
```cpp
    case YoloTargetRT::OPENCV_CPU:
      _rt = std::make_shared<yolo::YoloOpenCVRT>(_cfg.model_path_, false);
      break;
    case YoloTargetRT::OPENCV_CUDA:
      _rt = std::make_shared<yolo::YoloOpenCVRT>(_cfg.model_path_, true);
      break;
```
```cpp
#ifdef BUILD_WITH_TRT
    case YoloTargetRT::TRT:
      _rt = std::make_shared<yolo::YoloTRT>(_cfg.model_path_);
      break;
#endif
```
```cpp
#ifdef BUILD_WITH_RKN
    case YoloTargetRT::RKNN:
      _rt = std::make_shared<yolo::YoloRKNNRT>(_cfg.model_path_);
      break;
#endif
```

**Gate 1-F**: `grep -n "OpenCV\|TRT\|RKNN\|trt/\|rkn/" src/YoloTask.cpp` → 0 lines.

---

### 1-G  Edit `CMakeLists.txt` — remove dead globs and option blocks

> OpenCV **library** is still required (`cv::Mat`, `blobFromImages`). Do NOT remove
> `find_package(OpenCV REQUIRED)` or its `include_directories` / `link_libraries` lines.

Remove glob lines:
```cmake
file(GLOB_RECURSE TRACK_SRCS "src/track/*.cpp")
file(GLOB_RECURSE TRT_SRCS   "src/trt/*.cpp")
file(GLOB_RECURSE RKN_SRCS   "src/rkn/*.cpp")
```

> TRACK_SRCS is being removed from the glob list because we will add the track sources explicitly
> in step 1-G-addback below — do not lose them.

After removing the glob, add the track sources explicitly:
```cmake
file(GLOB_RECURSE TRACK_SRCS "src/track/*.cpp")
list(APPEND ONE_YOLO_DEPEND_SRCS ${TRACK_SRCS})
```
Place this immediately after the OVN `endif()` block (line ~67).

Remove the commented TRT option line:
```cmake
#option(BUILD_WITH_TRT "enable TensorRT(Nvidia/CUDA Platform)?" OFF)
```

Remove the entire TRT `if` block:
```cmake
if(BUILD_WITH_TRT)
    ...
    list(APPEND ONE_YOLO_DEPEND_SRCS ${TRT_SRCS})
endif()
```

Remove the entire RKN `if` block (including its `link_directories` and `include_directories`):
```cmake
if(BUILD_WITH_RKN)
    include_directories("/sd/z/rknnrt/include")
    link_directories("/sd/z/rknnrt/libs")
    add_definitions(-DBUILD_WITH_RKN)
    list(APPEND ONE_YOLO_DEPEND_SRCS ${RKN_SRCS})
    list(APPEND ONE_YOLO_DEPEND_LIBS rknnrt)
endif()
```

**Gate 1-G**: `grep -n "TRT\|RKNN\|RKN_SRCS\|TRT_SRCS" CMakeLists.txt` → 0 lines.  
Also verify: `grep -n "TRACK_SRCS" CMakeLists.txt` → exactly 2 lines (the glob and the append).

---

### 1-H  Edit `samples/CMakeLists.txt` — remove dead sample targets

Replace entire file content with:
```cmake
add_executable(det "det.cpp")
target_link_libraries(det ${PROJECT_NAME})

add_executable(cls "cls.cpp")
target_link_libraries(cls ${PROJECT_NAME})
```

**Gate 1-H**: `grep -n "seg\|pose\|obb" samples/CMakeLists.txt` → 0 lines.

---

### 1-I  Edit `samples/common.hpp` — remove dead backend resolver

In `resolveRuntime()`, remove the `dnn` branch:
```cpp
  if (backend == "dnn")
    return (device == "cuda") ? yolo::YoloTargetRT::OPENCV_CUDA : yolo::YoloTargetRT::OPENCV_CPU;
```

**Gate 1-I**: `grep -n "dnn\|OPENCV\|OpenCV" samples/common.hpp` → 0 lines  
(grep is case-sensitive; "OpenCV" in a comment about the library is acceptable — only inference references must go).

---

### 1-J  Purge dead enum references in remaining sources (APPROVED EXTENSION)

> **Why this step exists**: after 1-D trims the enums, the build (gate 1-K) fails because
> `src/YoloConfig.cpp`, `src/YoloResult.cpp`, `src/track/YoloTracker.cpp`, and `tools/bench.cpp`
> still reference the removed enumerators (`SEG/POSE/OBB`, `OPENCV_CPU/OPENCV_CUDA`, `TRT`, `RKNN`).
> These files are not in the original Phase 1 edit list, so this step was added with explicit
> user approval. It is a **minimal dead-reference purge only** — no reorganisation, no renaming,
> no behaviour change. `tools/bench.cpp` keeps its model flags until Phase 2.

**1-J-1  `src/YoloConfig.cpp`** — remove dead switch cases:
- In `toString(YoloTaskType)`: remove the `SEG`, `POSE`, `OBB` cases.
- In `toString(YoloTargetRT)`: remove the `OPENCV_CPU`, `OPENCV_CUDA`, `TRT`, `RKNN` cases.

**1-J-2  `src/YoloResult.cpp`** — remove dead task branches and guards:
- In `plot()`: remove the `SEG`, `POSE`, `OBB` cases.
- In `to_csv()`: remove the `SEG`, `POSE`, `OBB` cases.
- In `to_json()`: remove the `SEG`, `POSE`, `OBB` cases.
- In `info()`: remove the `else if (task == YoloTaskType::OBB)` branch.
- In `boxes()`, `cls_ids()`, `confs()`, `labels()`, `track_ids()`, `track_points()`: trim the
  guard condition to `task != YoloTaskType::DET` only.
- In `rboxes()`, `masks()`, `contours()`, `kpts()`: change the guard to `task != YoloTaskType::DET`
  (these accessors are now unreachable for their original tasks but must still compile).

**1-J-3  `src/track/YoloTracker.cpp`** — remove dead task branches (SEG/POSE).

**1-J-4  `tools/bench.cpp`** — remove dead enum references only:
- Remove `OPENCV_CPU`/`OPENCV_CUDA` from the runtime resolver and `SEG`/`POSE`/`OBB` from the
  task resolver. Keep all model flags (`--model`, `--version`, etc.) — those are removed in Phase 2.

**Gate 1-J**: `grep -rn "YoloTaskType::SEG\|YoloTaskType::POSE\|YoloTaskType::OBB\|YoloTargetRT::OPENCV\|YoloTargetRT::TRT\|YoloTargetRT::RKNN" src/` → 0 lines.

---

### 1-K  Build check (Phase 1)

> The project uses CMake presets (`CMakePresets.json`). The plan's literal `cmake -B build` fails
> because it omits the preset cache variables (`OpenCV_DIR`, `onnxruntime_DIR`, `OpenVINO_DIR`).
> Use the `Release` preset; binaries land under `build/Release/`.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc) 2>&1 | tail -40
```

**Gate 1-J (hard stop — do NOT proceed if this fails)**:
- 0 compiler errors
- `build/libone-yolo.so` exists
- `build/bench` exists
- `build/samples/det` and `build/samples/cls` exist

---

### 1-K  Commit Phase 1

```bash
git add -u
git commit -m "$(cat <<'EOF'
refactor: slim to ORT+OVN backends, DET+CLS tasks

Remove: OpenCV-DNN backend, TRT, RKNN, SEG/POSE/OBB tasks.
Keep: YoloONNXRT (ORT), YoloOVNRT (OVN), tracker subsystem,
YoloDetTask, YoloClsTask. OpenCV library retained (cv::Mat still used).
EOF
)"
```

---

## Phase 2 — JSON config system

### Background — what the JSON replaces

Current CLI has 12+ flags per invocation. All model-specific params move to a JSON file.
The new CLI contract is:

```
[binary]  <source>  <config.json>  [--backend ort|ovn]  [--device cpu|gpu|auto|cuda]
```

For `bench` specifically:
```
bench  <config.json>  --img <path>  [--backend ort|ovn]  [--device cpu|gpu|auto|cuda]
       [--iter N]  [--pipeline]
```

`--no-track` stays in `det` (runtime decision, not model config).  
`--scale` stays (display parameter, not model parameter).  
`--no-json` and `--no-csv` are **dropped** (per-frame file output removed entirely).  
`--names`, `--classes`, `--version`, `--input-w`, `--input-h`, `--conf`, `--iou`, `--model` are **dropped**.

---

### 2-A  Delete bundled nlohmann/json — use system package

```bash
git rm -r include/nlohmann/
```

In `CMakeLists.txt`, add after the existing `find_package` blocks:
```cmake
find_package(nlohmann_json REQUIRED)
```

Add to `target_link_libraries` for the main library target:
```cmake
target_link_libraries(${PROJECT_NAME} ${ONE_YOLO_DEPEND_LIBS} nlohmann_json::nlohmann_json)
```

**Gate 2-A**: `ls include/nlohmann` must fail; `grep "nlohmann_json" CMakeLists.txt` → 2 lines.

---

### 2-B  JSON schema — reference

Every model ships with a `.json` sidecar. Required fields:

```json
{
  "model_path":  "/absolute/or/relative/to/json/model.xml",
  "task":        "det",
  "version":     "yolo11",
  "batch":       1,
  "input_w":     640,
  "input_h":     384,
  "num_classes": 80,
  "conf_thresh": 0.25,
  "iou_thresh":  0.45,
  "scale_f":     0.00392156862,
  "nchw":        true,
  "rgb":         true,
  "mean":        [],
  "std":         []
}
```

Notes:
- `model_path` relative → resolved relative to the JSON file's directory.
- `version` maps to `YoloVersion`: `"yolo5"`, `"yolo5u"`, `"yolo8"`, `"yolo11"`, `"yolo26"`,
  `"yolox"` (alias for `yolo5` decoder + `scale_f: 1.0`).
- `mean` / `std` can be `[]` (empty = skip normalisation).
- No `names` field — COCO names auto-selected when `num_classes == 80`,
  ImageNet when `num_classes == 1000`, `"obj"` otherwise.

This schema is documentation only — no validation code beyond what nlohmann throws on missing keys.

---

### 2-C  Add `YoloConfig::from_json()` static factory

**Edit `include/YoloConfig.h`**:

Add `#include <nlohmann/json.hpp>` at the top of the file (after existing includes).

Add this declaration inside `struct YoloConfig` (after the existing member declarations):
```cpp
  [[nodiscard]] static auto
  from_json(const std::string& json_path) -> YoloConfig;
```

**Edit `src/YoloConfig.cpp`**:

Add the implementation (find a suitable place after existing functions):
```cpp
auto
YoloConfig::from_json(const std::string& json_path) -> YoloConfig
{
  std::ifstream f(json_path);
  if (!f)
    throw std::runtime_error("cannot open config: " + json_path);

  const auto j = nlohmann::json::parse(f);

  YoloConfig cfg;

  // resolve model_path relative to the JSON file's directory
  const std::filesystem::path base = std::filesystem::path(json_path).parent_path();
  const std::string raw_path = j.at("model_path").get<std::string>();
  cfg.model_path_ = std::filesystem::path(raw_path).is_absolute()
                      ? raw_path
                      : (base / raw_path).string();

  const std::string ver = j.at("version").get<std::string>();
  if      (ver == "yolo5")  cfg.version_ = YoloVersion::YOLO5;
  else if (ver == "yolo5u") cfg.version_ = YoloVersion::YOLO5U;
  else if (ver == "yolo8")  cfg.version_ = YoloVersion::YOLO8;
  else if (ver == "yolo11") cfg.version_ = YoloVersion::YOLO11;
  else if (ver == "yolo26") cfg.version_ = YoloVersion::YOLO26;
  else if (ver == "yolox")  cfg.version_ = YoloVersion::YOLO5;  // same decoder
  else throw std::runtime_error("unknown version in config: " + ver);

  const std::string task = j.at("task").get<std::string>();
  if      (task == "det") cfg.task_ = YoloTaskType::DET;
  else if (task == "cls") cfg.task_ = YoloTaskType::CLS;
  else throw std::runtime_error("unknown task in config: " + task);

  cfg.batch_size_  = j.at("batch").get<int>();
  cfg.input_w_     = j.at("input_w").get<int>();
  cfg.input_h_     = j.at("input_h").get<int>();
  cfg.num_classes_ = j.at("num_classes").get<int>();
  cfg.conf_thresh_ = j.at("conf_thresh").get<float>();
  cfg.iou_thresh_  = j.at("iou_thresh").get<float>();
  cfg.scale_f_     = j.at("scale_f").get<float>();
  cfg.nchw_        = j.at("nchw").get<bool>();
  cfg.rgb_         = j.at("rgb").get<bool>();

  if (j.contains("mean") && !j["mean"].empty())
    cfg.mean_ = j["mean"].get<std::vector<float>>();
  if (j.contains("std") && !j["std"].empty())
    cfg.std_  = j["std"].get<std::vector<float>>();

  // auto-select class names
  if      (cfg.num_classes_ == 80)   cfg.names_ = std::vector<std::string>{COCO_NAMES};
  else if (cfg.num_classes_ == 1000) cfg.names_ = std::vector<std::string>{IMAGENET_NAMES};
  else cfg.names_ = std::vector<std::string>(static_cast<size_t>(cfg.num_classes_), "obj");

  cfg.desc_ = ver + "/" + task;
  return cfg;
}
```

Also add these includes at the top of `src/YoloConfig.cpp` (if not already present):
```cpp
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
```

**Gate 2-C**:
- `grep "from_json" include/YoloConfig.h` → 1 line
- `grep "from_json" src/YoloConfig.cpp` → at least 1 line

---

### 2-D  Rewrite `samples/common.hpp`

The new `CliArgs` struct, `parseArgs`, `buildYoloConfig`, `printUsage` must implement
the contract: `[binary] <source> <config.json> [--backend ort|ovn] [--device cpu|gpu|auto|cuda] [--no-track] [--scale N]`.

Replace the entire `CliArgs` struct with:
```cpp
struct CliArgs
{
  std::string source_;
  std::string config_;
  std::string backend_ = "ort";
  std::string device_  = "cpu";
  float       scale_   = kDefaultScale;
  bool        no_track_ = false;
};
```

Remove from `CliArgs`: `model_`, `version_`, `input_w_`, `input_h_`, `classes_`,
`conf_`, `iou_`, `no_json_`, `no_csv_`, `names_`.

Replace `buildYoloConfig()` with:
```cpp
[[nodiscard]] inline auto
buildYoloConfig(const CliArgs& a, yolo::YoloTaskType /*task*/) -> yolo::YoloConfig
{
  auto cfg = yolo::YoloConfig::from_json(a.config_);
  cfg.target_rt_ = resolveRuntime(a.backend_, a.device_);
  return cfg;
}
```

(The `task` parameter is kept in the signature so sample `main()` functions don't need changing,
but `from_json` already sets the task from the JSON — the parameter is ignored.)

Replace `parseArgs()` — new version accepts positional args `source` and `config` plus
optional flags `--backend`, `--device`, `--no-track`, `--scale`:
```cpp
[[nodiscard]] inline auto
parseArgs(int argc, char** argv) -> CliArgs
{
  CliArgs a;
  int positional = 0;
  for (int i = 1; i < argc; ++i)
  {
    const std::string k = argv[i];
    auto next = [&]() -> std::string
    {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--backend")       { a.backend_  = next(); }
    else if (k == "--device")   { a.device_   = next(); }
    else if (k == "--scale")    { a.scale_    = std::stof(next()); }
    else if (k == "--no-track") { a.no_track_ = true; }
    else if (k == "--help" || k == "-h") { /* caller handles */ }
    else if (k.starts_with("--")) { throw std::runtime_error("unknown argument: " + k); }
    else
    {
      if (positional == 0)      a.source_ = k;
      else if (positional == 1) a.config_ = k;
      else throw std::runtime_error("unexpected positional argument: " + k);
      ++positional;
    }
  }
  return a;
}
```

Replace `printUsage()`:
```cpp
inline void
printUsage(const char* app_name, const char* task_desc)
{
  std::cout << "usage: " << app_name << " <source> <config.json> [options]\n"
            << "  task: " << task_desc << "\n\n"
            << "  <source>         image file, video file, or webcam index\n"
            << "  <config.json>    model config (model path + blob params)\n"
            << "  --backend        ort|ovn                     (default: ort)\n"
            << "  --device         cpu|gpu|auto|cuda           (default: cpu)\n"
            << "  --scale          <float>  display scale      (default: 1.0)\n"
            << "  --no-track       disable SORT tracker (det only)\n";
}
```

Update `runLoop()` — change guard from `a.source_.empty()` check (unchanged),
but remove all `no_json_` / `no_csv_` references inside the loop body.  
The `runLoop` function itself doesn't reference those — they are in the sample `main()` bodies.

**Gate 2-D**: `grep -n "no_json\|no_csv\|model_\|version_\|input_w_\|input_h_\|classes_\|conf_\|iou_\|names_" samples/common.hpp` → 0 lines.

---

### 2-E  Edit sample `main()` functions

**`samples/det.cpp`** — remove `no_json_` and `no_csv_` usage:

Remove these lines from inside the lambda:
```cpp
                   if (!a.no_json_)
                     result.to_json(/*print=*/true);
                   if (!a.no_csv_)
                     result.to_csv(/*print=*/true);
```

**`samples/cls.cpp`** — remove `no_json_` and `no_csv_` usage:

Remove these lines from inside the lambda:
```cpp
                   if (!a.no_json_)
                     result.to_json(/*print=*/true);
                   if (!a.no_csv_)
                     result.to_csv(/*print=*/true);
```

Also update the `argc < 2` guard and `a.model_.empty()` guard in both files:

In `det.cpp` replace:
```cpp
    if (a.model_.empty())
    {
      app::printUsage(argv[0], "object detection (det) with optional SORT tracking");
      return 1;
    }
```
with:
```cpp
    if (a.config_.empty())
    {
      app::printUsage(argv[0], "object detection (det) with optional SORT tracking");
      return 1;
    }
```

Same pattern in `cls.cpp` (`a.model_` → `a.config_`).

**Gate 2-E**: `grep -n "no_json\|no_csv\|a\.model_" samples/det.cpp samples/cls.cpp` → 0 lines.

---

### 2-F  Update `tools/bench.cpp` — replace model/version/etc flags with config.json

`bench` current flags related to model params: `--model`, `--backend`, `--device`, `--iter`,
`--pipeline`, `--img`, `--precision` (if present), `--classes`, `--input-w`, `--input-h`, `--conf`.

New contract:
```
bench <config.json> --img <path> [--backend ort|ovn] [--device cpu|gpu|auto|cuda]
      [--iter N] [--pipeline]
```

In `bench.cpp`:
- Change `struct BenchArgs` (or equivalent): replace `model_`, `version_`, `classes_`,
  `input_w_`, `input_h_`, `conf_` with a single `config_` field.
- In `parseArgs` (or inline arg loop): first positional becomes `config_`; remove flags
  `--model`, `--version`, `--classes`, `--input-w`, `--input-h`, `--conf`.
- Where `YoloConfig` is built (look for `YoloConfig cfg;` or `buildYoloConfig`):
  replace the manual field assignments with:
  ```cpp
  auto cfg = yolo::YoloConfig::from_json(args.config_);
  cfg.target_rt_ = resolveRuntime(args.backend_, args.device_);
  ```
- Update `printUsage` / help text to match new contract.

> `resolveRuntime` is defined in `samples/common.hpp`. Since bench does not include that header,
> either (a) duplicate the small function inline in bench.cpp, or (b) move `resolveRuntime` to a
> new shared header `tools/app_common.hpp` included by both `common.hpp` and `bench.cpp`.
> **Prefer option (b)** to avoid duplication.

**Gate 2-F**: `grep -n "\-\-model\|\-\-version\|\-\-classes\|\-\-input-w\|\-\-input-h" tools/bench.cpp` → 0 lines.

---

### 2-G  Build check (Phase 2)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc) 2>&1 | tail -40
```

**Gate 2-G (hard stop)**:
- 0 compiler errors
- All four binaries exist: `libone-yolo.so`, `bench`, `samples/det`, `samples/cls`

---

### 2-H  Smoke test

Create a minimal test config `/tmp/test.json`:
```json
{
  "model_path":  "/path/to/yolox_nano_dec.xml",
  "task":        "det",
  "version":     "yolox",
  "batch":       1,
  "input_w":     640,
  "input_h":     384,
  "num_classes": 80,
  "conf_thresh": 0.25,
  "iou_thresh":  0.45,
  "scale_f":     1.0,
  "nchw":        true,
  "rgb":         true,
  "mean":        [],
  "std":         []
}
```

Run:
```bash
./build/bench /tmp/test.json --img /path/to/mot17_frame.jpg --backend ovn --device GPU --iter 5
```

**Gate 2-H**: exit 0, detection count printed, no crash.

---

### 2-I  Commit Phase 2

```bash
git add -u
git commit -m "$(cat <<'EOF'
feat: JSON config system — replace CLI flag wall with config.json

YoloConfig::from_json() loads all model params from JSON sidecar.
CLI reduced to: [binary] source config.json [--backend] [--device].
Drop: --model, --version, --classes, --input-w/h, --conf, --iou,
      --names, --no-json, --no-csv (per-frame output removed).
Keep: --no-track, --scale (runtime/display decisions).
nlohmann/json: bundled copy removed, system package used instead.
EOF
)"
```

---

## Acceptance gates — full summary

| # | Phase | Check | Command | Pass |
|---|-------|-------|---------|------|
| 1-A | 1 | TRT/RKNN dirs gone | `ls include/trt include/rkn` | both fail |
| 1-B | 1 | Task headers gone | `ls include/YoloSeg* include/YoloPose* include/YoloObb*` | all fail |
| 1-C | 1 | Samples cleaned | `ls samples/` | only cls.cpp, det.cpp, CMakeLists.txt |
| 1-D | 1 | Enums clean | `grep -n "OPENCV\|TRT\|RKNN\|SEG\|POSE\|OBB" include/YoloConfig.h` | 0 lines |
| 1-E | 1 | Yolo.cpp clean | `grep -n "Seg\|Pose\|Obb" src/Yolo.cpp` | 0 lines |
| 1-F | 1 | YoloTask.cpp clean | `grep -n "OpenCV\|TRT\|RKNN\|trt/\|rkn/" src/YoloTask.cpp` | 0 lines |
| 1-G | 1 | CMakeLists clean | `grep -n "TRT\|RKNN\|RKN_SRCS\|TRT_SRCS" CMakeLists.txt` | 0 lines |
| 1-H | 1 | samples/CMakeLists | `grep -n "seg\|pose\|obb" samples/CMakeLists.txt` | 0 lines |
| 1-I | 1 | No dnn resolver | `grep -n "\"dnn\"\|OPENCV_CPU\|OPENCV_CUDA" samples/common.hpp` | 0 lines |
| 1-J | 1 | **Build** | `cmake --build build` | 0 errors |
| 2-A | 2 | nlohmann removed | `ls include/nlohmann` | fails |
| 2-C | 2 | from_json exists | `grep "from_json" include/YoloConfig.h` | 1 line |
| 2-D | 2 | common.hpp clean | `grep -n "no_json\|no_csv\|model_\|version_\|classes_" samples/common.hpp` | 0 lines |
| 2-E | 2 | samples clean | `grep -n "no_json\|no_csv\|a\.model_" samples/det.cpp samples/cls.cpp` | 0 lines |
| 2-F | 2 | bench clean | `grep -n "\-\-model\|\-\-version\|\-\-classes" tools/bench.cpp` | 0 lines |
| 2-G | 2 | **Build** | `cmake --build build` | 0 errors |
| 2-H | 2 | **Smoke test** | `bench config.json --img frame.jpg --backend ovn --device GPU --iter 5` | exit 0 |

---

## Files touched — complete list

| Action | Path | Phase |
|--------|------|-------|
| DELETE | `include/YoloOpenCVRT.h` | 1 |
| DELETE | `src/YoloOpenCVRT.cpp` | 1 |
| DELETE | `include/trt/YoloTRT.h` | 1 |
| DELETE | `src/trt/YoloTRT.cpp` | 1 |
| DELETE | `include/rkn/YoloRKNNRT.h` | 1 |
| DELETE | `src/rkn/YoloRKNNRT.cpp` | 1 |
| DELETE | `include/YoloSegTask.h` | 1 |
| DELETE | `src/YoloSegTask.cpp` | 1 |
| DELETE | `include/YoloPoseTask.h` | 1 |
| DELETE | `src/YoloPoseTask.cpp` | 1 |
| DELETE | `include/YoloObbTask.h` | 1 |
| DELETE | `src/YoloObbTask.cpp` | 1 |
| DELETE | `samples/seg.cpp` | 1 |
| DELETE | `samples/pose.cpp` | 1 |
| DELETE | `samples/obb.cpp` | 1 |
| DELETE | `include/nlohmann/json.hpp` | 2 |
| EDIT | `include/YoloConfig.h` | 1+2 |
| EDIT | `src/Yolo.cpp` | 1 |
| EDIT | `src/YoloTask.cpp` | 1 |
| EDIT | `src/YoloConfig.cpp` | 2 |
| EDIT | `CMakeLists.txt` | 1+2 |
| EDIT | `samples/CMakeLists.txt` | 1 |
| EDIT | `samples/common.hpp` | 2 |
| EDIT | `samples/det.cpp` | 2 |
| EDIT | `samples/cls.cpp` | 2 |
| EDIT | `tools/bench.cpp` | 2 |
| CREATE | `tools/app_common.hpp` | 2 |

**Do not touch any other file.**
