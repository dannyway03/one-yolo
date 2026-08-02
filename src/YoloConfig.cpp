#include "YoloConfig.h"

#include <sstream>

#include "nlohmann/json.hpp"
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
    case YoloTaskType::SEG:
      return "segmentation";
    case YoloTaskType::POSE:
      return "pose";
    case YoloTaskType::OBB:
      return "obb";
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
    case YoloTargetRT::OPENCV_CPU:
      return "opencv::dnn(cpu)";
    case YoloTargetRT::OPENCV_CUDA:
      return "opencv::dnn(cuda)";
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
    case YoloTargetRT::TRT:
      return "tensorrt";
    case YoloTargetRT::RKNN:
      return "rknn";
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

} // namespace yolo