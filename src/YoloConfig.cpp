#include "YoloConfig.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>
#include "YoloUtils.h"
using json = nlohmann::json;

namespace yolo {

auto
toString(YoloTaskType task) -> std::string
{
  switch (task)
  {
    case YoloTaskType::CLS:
      return "classification";
    case YoloTaskType::DET:
      return "detection";
    default:
      return "unknown";
  }
}

auto
toString(YoloVersion version) -> std::string
{
  switch (version)
  {
    case YoloVersion::YOLO5:
      return "yolov5";
    case YoloVersion::YOLO5U:
      return "yolov5u(anchor-free)";
    case YoloVersion::YOLO8:
      return "yolov8";
    case YoloVersion::YOLO11:
      return "yolov11";
    case YoloVersion::YOLO26:
      return "yolov26";
    default:
      return "unknown";
  }
}

auto
toString(YoloTargetRT target_rt) -> std::string
{
  switch (target_rt)
  {
    case YoloTargetRT::ORT_CPU:
      return "onnxruntime(cpu)";
    case YoloTargetRT::ORT_CUDA:
      return "onnxruntime(cuda)";
    case YoloTargetRT::OVN_AUTO:
      return "openvino(auto)";
    case YoloTargetRT::OVN_CPU:
      return "openvino(cpu)";
    case YoloTargetRT::OVN_GPU:
      return "openvino(integrated gpu)";
    default:
      return "unknown";
  }
}

auto
toString(const YoloConfig& cfg) -> std::string
{
  std::ostringstream oss;
  oss << "########### YoloConfig ###########" << '\n';
  oss << "description       : " << cfg.desc_ << '\n';
  oss << "model path        : " << cfg.model_path_ << '\n';
  oss << "task              : " << toString(cfg.task_) << '\n';
  oss << "yolo version      : " << toString(cfg.version_) << '\n';
  oss << "yolo runtime      : " << toString(cfg.target_rt_) << '\n';
  oss << "batch size        : " << cfg.batch_size_ << '\n';
  oss << "input width       : " << cfg.input_w_ << '\n';
  oss << "input height      : " << cfg.input_h_ << '\n';
  oss << "num classes       : " << cfg.num_classes_ << '\n';
  oss << "num kpts          : " << cfg.num_kpts_ << '\n';
  oss << "num channels      : " << cfg.num_channels_ << '\n';
  oss << "conf threshold    : " << cfg.conf_thresh_ << '\n';
  oss << "iou threshold     : " << cfg.iou_thresh_ << '\n';
  oss << "scale factor      : " << cfg.scale_f_ << '\n';
  oss << "nchw              : " << std::string(cfg.nchw_ ? "yes" : "no") << '\n';
  oss << "rgb               : " << std::string(cfg.rgb_ ? "yes" : "no") << '\n';

  auto n = std::min(cfg.names_.size(), static_cast<size_t>(5));
  auto cfg_top5_names = std::vector<std::string>(cfg.names_.begin(), cfg.names_.begin() + n);
  oss << "names(top5)       : " << cfg_top5_names;

  return oss.str();
}

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
  cfg.model_path_ = std::filesystem::path(raw_path).is_absolute() ? raw_path : (base / raw_path).string();

  const std::string ver = j.at("version").get<std::string>();
  if (ver == "yolo5" || ver == "yolox") // yolox uses the same decoder as yolo5
    cfg.version_ = YoloVersion::YOLO5;
  else if (ver == "yolo5u")
    cfg.version_ = YoloVersion::YOLO5U;
  else if (ver == "yolo8")
    cfg.version_ = YoloVersion::YOLO8;
  else if (ver == "yolo11")
    cfg.version_ = YoloVersion::YOLO11;
  else if (ver == "yolo26")
    cfg.version_ = YoloVersion::YOLO26;
  else
    throw std::runtime_error("unknown version in config: " + ver);

  const std::string task = j.at("task").get<std::string>();
  if (task == "det")
    cfg.task_ = YoloTaskType::DET;
  else if (task == "cls")
    cfg.task_ = YoloTaskType::CLS;
  else
    throw std::runtime_error("unknown task in config: " + task);

  cfg.batch_size_ = j.at("batch").get<int>();
  cfg.input_w_ = j.at("input_w").get<int>();
  cfg.input_h_ = j.at("input_h").get<int>();
  cfg.num_classes_ = j.at("num_classes").get<int>();
  cfg.conf_thresh_ = j.at("conf_thresh").get<float>();
  cfg.iou_thresh_ = j.at("iou_thresh").get<float>();
  cfg.scale_f_ = j.at("scale_f").get<float>();
  cfg.nchw_ = j.at("nchw").get<bool>();
  cfg.rgb_ = j.at("rgb").get<bool>();

  if (j.contains("mean") && !j["mean"].empty())
    cfg.mean_ = j["mean"].get<std::vector<float>>();
  if (j.contains("std") && !j["std"].empty())
    cfg.std_ = j["std"].get<std::vector<float>>();

  // auto-select class names
  if (cfg.num_classes_ == 80)
    cfg.names_ = std::vector<std::string>{COCO_NAMES};
  else if (cfg.num_classes_ == 1000)
    cfg.names_ = std::vector<std::string>{IMAGENET_NAMES};
  else
    cfg.names_ = std::vector<std::string>(static_cast<size_t>(cfg.num_classes_), "obj");

  cfg.desc_ = ver + "/" + task;
  return cfg;
}

} // namespace yolo
