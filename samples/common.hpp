#pragma once

#include <cstring>

#include <algorithm>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/opencv.hpp>

#include "track/YoloTracker.h"
#include "Yolo.h"

namespace app {

constexpr int kDefaultW = 640;
constexpr int kDefaultH = 640;
constexpr int kDefaultClasses = 80;
constexpr float kDefaultConf = 0.25f;
constexpr float kDefaultIou = 0.45f;
constexpr float kDefaultScale = 1.0f;

// ── CLI args ──────────────────────────────────────────────────────────────────

struct CliArgs
{
  std::string model_;
  std::string version_ = "yolo11";
  std::string backend_ = "ort";
  std::string device_ = "cpu";
  std::string source_;
  int input_w_ = kDefaultW;
  int input_h_ = kDefaultH;
  int classes_ = kDefaultClasses;
  float conf_ = kDefaultConf;
  float iou_ = kDefaultIou;
  float scale_ = kDefaultScale;
  bool no_json_ = false;
  bool no_csv_ = false;
  bool no_track_ = false;
  std::optional<std::string> names_;
};

// ── source classification ─────────────────────────────────────────────────────

[[nodiscard]] inline auto
isImageSource(std::string_view path) -> bool
{
  for (auto* ext : {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})
  {
    const auto len = std::strlen(ext);
    if (path.size() >= len && path.compare(path.size() - len, len, ext) == 0)
      return true;
  }
  return false;
}

[[nodiscard]] inline auto
isCameraIndex(std::string_view s) -> bool
{
  return !s.empty() && std::all_of(s.begin(), s.end(),
                                   [](char c)
                                   {
                                     return std::isdigit(c) != 0;
                                   });
}

// ── config builders ───────────────────────────────────────────────────────────

[[nodiscard]] inline auto
resolveVersion(const std::string& s) -> yolo::YoloVersion
{
  if (s == "yolo5")
    return yolo::YoloVersion::YOLO5;
  if (s == "yolo5u")
    return yolo::YoloVersion::YOLO5U;
  if (s == "yolo8")
    return yolo::YoloVersion::YOLO8;
  if (s == "yolo11")
    return yolo::YoloVersion::YOLO11;
  if (s == "yolo26")
    return yolo::YoloVersion::YOLO26;
  // decoded YOLOX output matches the YOLO5 decoder layout [cx,cy,w,h,obj_conf,cls...]
  if (s == "yolox")
    return yolo::YoloVersion::YOLO5;
  throw std::runtime_error("unknown --version: " + s);
}

[[nodiscard]] inline auto
resolveRuntime(const std::string& backend, const std::string& device) -> yolo::YoloTargetRT
{
  if (backend == "ort")
    return (device == "cuda") ? yolo::YoloTargetRT::ORT_CUDA : yolo::YoloTargetRT::ORT_CPU;
  if (backend == "ovn")
  {
    if (device == "gpu")
      return yolo::YoloTargetRT::OVN_GPU;
    if (device == "auto")
      return yolo::YoloTargetRT::OVN_AUTO;
    return yolo::YoloTargetRT::OVN_CPU;
  }
  if (backend == "dnn")
    return (device == "cuda") ? yolo::YoloTargetRT::OPENCV_CUDA : yolo::YoloTargetRT::OPENCV_CPU;
  throw std::runtime_error("unknown --backend: " + backend);
}

[[nodiscard]] inline auto
buildYoloConfig(const CliArgs& a, yolo::YoloTaskType task) -> yolo::YoloConfig
{
  yolo::YoloConfig cfg;
  cfg.desc_ = a.version_ + " [" + a.backend_ + "/" + a.device_ + "]";
  cfg.model_path_ = a.model_;
  cfg.version_ = resolveVersion(a.version_);
  cfg.target_rt_ = resolveRuntime(a.backend_, a.device_);
  cfg.task_ = task;
  cfg.input_w_ = a.input_w_;
  cfg.input_h_ = a.input_h_;
  cfg.batch_size_ = 1;
  cfg.num_classes_ = a.classes_;
  cfg.conf_thresh_ = a.conf_;
  cfg.iou_thresh_ = a.iou_;
  cfg.rgb_ = true;
  // YOLOX decoded models expect raw [0, 255] pixel values; all other versions use [0, 1]
  cfg.scale_f_ = (a.version_ == "yolox") ? 1.0f : 1.0f / 255.0f;

  if (a.names_.has_value())
  {
    std::vector<std::string> name_list;
    std::istringstream ss(*a.names_);
    std::string token;
    while (std::getline(ss, token, ','))
      if (!token.empty())
        name_list.push_back(token);
    cfg.names_ = name_list;
  }
  else if (a.classes_ == 80)
  {
    cfg.names_ = std::vector<std::string>{COCO_NAMES};
  }
  else if (a.classes_ == 1000)
  {
    cfg.names_ = std::vector<std::string>{IMAGENET_NAMES};
  }
  else
  {
    cfg.names_ = std::vector<std::string>(static_cast<size_t>(a.classes_), "obj");
  }
  return cfg;
}

[[nodiscard]] inline auto
buildTrackerConfig() -> yolo::YoloTrackConfig
{
  yolo::YoloTrackConfig t;
  t.algo = yolo::YoloTrackAlgo::SORT;
  t.iou_thresh = 0.6f;
  return t;
}

// ── usage ─────────────────────────────────────────────────────────────────────

inline void
printUsage(const char* app_name, const char* task_desc)
{
  std::cout << "usage: " << app_name << " --model <path> [options]\n"
            << "  task:       " << task_desc << "\n\n"
            << "  --model      <path>                model file (.onnx / .xml)\n"
            << "  --version    yolox|yolo26|yolo11|yolo8|yolo5|yolo5u  (default: yolo11)\n"
            << "  --backend    ort|ovn|dnn           inference backend (default: ort)\n"
            << "  --device     cpu|gpu|auto|cuda     device for backend (default: cpu)\n"
            << "  --source     <path|0|1|...>        image, video file, or webcam index\n"
            << "  --input-w    <int>                 model input width  (default: 640)\n"
            << "  --input-h    <int>                 model input height (default: 640)\n"
            << "  --classes    <int>                 number of classes  (default: 80)\n"
            << "  --names      <a,b,c,...>           comma-separated class names\n"
            << "  --conf       <float>               confidence threshold (default: 0.25)\n"
            << "  --iou        <float>               NMS IoU threshold   (default: 0.45)\n"
            << "  --scale      <float>               display scale       (default: 1.0)\n"
            << "  --no-json                          suppress per-frame JSON output\n"
            << "  --no-csv                           suppress per-frame CSV output\n"
            << "  --no-track                         disable SORT tracker\n";
}

// ── arg parser ────────────────────────────────────────────────────────────────

[[nodiscard]] inline auto
parseArgs(int argc, char** argv) -> CliArgs // NOLINT(modernize-avoid-c-arrays)
{
  CliArgs a;
  for (int i = 1; i < argc; ++i)
  {
    const std::string k = argv[i];
    auto next = [&]() -> std::string
    {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + k);
      return argv[++i]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    };
    if (k == "--model")
    {
      a.model_ = next();
    }
    else if (k == "--version")
    {
      a.version_ = next();
    }
    else if (k == "--backend")
    {
      a.backend_ = next();
    }
    else if (k == "--device")
    {
      a.device_ = next();
    }
    else if (k == "--source")
    {
      a.source_ = next();
    }
    else if (k == "--input-w")
    {
      a.input_w_ = std::stoi(next());
    }
    else if (k == "--input-h")
    {
      a.input_h_ = std::stoi(next());
    }
    else if (k == "--classes")
    {
      a.classes_ = std::stoi(next());
    }
    else if (k == "--names")
    {
      a.names_ = next();
    }
    else if (k == "--conf")
    {
      a.conf_ = std::stof(next());
    }
    else if (k == "--iou")
    {
      a.iou_ = std::stof(next());
    }
    else if (k == "--scale")
    {
      a.scale_ = std::stof(next());
    }
    else if (k == "--no-json")
    {
      a.no_json_ = true;
    }
    else if (k == "--no-csv")
    {
      a.no_csv_ = true;
    }
    else if (k == "--no-track")
    {
      a.no_track_ = true;
    }
    else if (k == "--help" || k == "-h")
    { /* handled by caller */
    }
    else
    {
      throw std::runtime_error("unknown argument: " + k);
    }
  }
  return a;
}

// ── source loop ───────────────────────────────────────────────────────────────

// handler(frame, single_shot) → bool: return false to break the loop
template<typename Handler>
inline void
runLoop(const CliArgs& a, Handler&& handler)
{
  if (a.source_.empty())
    throw std::runtime_error("--source is required");

  if (isImageSource(a.source_))
  {
    const cv::Mat img = cv::imread(a.source_);
    if (img.empty())
      throw std::runtime_error("cannot read image: " + a.source_);
    handler(img, /*single_shot=*/true);
    return;
  }

  cv::VideoCapture cap;
  const bool is_camera = isCameraIndex(a.source_);
  if (is_camera)
    cap.open(std::stoi(a.source_));
  else
    cap.open(a.source_);
  if (!cap.isOpened())
    throw std::runtime_error("cannot open source: " + a.source_);

  while (cap.isOpened())
  {
    cv::Mat frame;
    if (!cap.read(frame))
    {
      if (!is_camera)
      {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        continue;
      }
      break;
    }
    if (!handler(frame, /*single_shot=*/false))
      break;
  }
}

} // namespace app
