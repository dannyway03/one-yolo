#include "Yolo.h"

#include <stdexcept>

#include "YoloClsTask.h"
#include "YoloDetTask.h"
#include "YoloObbTask.h"
#include "YoloPoseTask.h"
#include "YoloSegTask.h"

namespace yolo {

Yolo::Yolo(const YoloConfig& cfg) : cfg_(cfg)
{
  if (cfg_.num_classes_ != cfg_.names_.size())
    throw std::invalid_argument("num_classes != labels.size() in YoloConfig!");

  switch (cfg_.task_)
  {
    case YoloTaskType::CLS:
      task_ = std::make_shared<yolo::YoloClsTask>(cfg);
      break;
    case YoloTaskType::DET:
      task_ = std::make_shared<yolo::YoloDetTask>(cfg);
      break;
    case YoloTaskType::SEG:
      task_ = std::make_shared<yolo::YoloSegTask>(cfg);
      break;
    case YoloTaskType::POSE:
      task_ = std::make_shared<yolo::YoloPoseTask>(cfg);
      break;
    case YoloTaskType::OBB:
      task_ = std::make_shared<yolo::YoloObbTask>(cfg);
      break;
    default:
      throw std::invalid_argument("invalid YoloTaskType parameter when initializing Yolo!");
      break;
  }
}

Yolo::~Yolo() = default;

auto
Yolo::predict(const cv::Mat& image) -> YoloResult
{
  return predict(std::vector<cv::Mat>{image})[0];
}

auto
Yolo::operator()(const cv::Mat& image) -> YoloResult
{
  return (*this)(std::vector<cv::Mat>{image})[0];
}

auto
Yolo::predict(const std::vector<cv::Mat>& images) -> std::vector<YoloResult>
{
  if (cfg_.batch_size_ && cfg_.batch_size_ != images.size())
    throw std::runtime_error("got invalid batch size when calling Yolo::predict()!");

  return (*task_)(images);
}

auto
Yolo::operator()(const std::vector<cv::Mat>& images) -> std::vector<YoloResult>
{
  return predict(images);
}

auto
Yolo::info(bool print) -> std::string
{
  auto cfg_summary = toString(cfg_);

  if (print)
    std::cout << cfg_summary << '\n';
  return cfg_summary;
}

} // namespace yolo