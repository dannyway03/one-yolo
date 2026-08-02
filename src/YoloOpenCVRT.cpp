#include "YoloOpenCVRT.h"

namespace yolo {

YoloOpenCVRT::YoloOpenCVRT(const std::string& model_path, bool use_cuda) : YoloRuntime("OpenCV::DNN")
{
  net_ = cv::dnn::readNet(model_path);
  if (use_cuda)
  {
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  }
}

YoloOpenCVRT::~YoloOpenCVRT() = default;

auto
YoloOpenCVRT::inference(const cv::Mat& blob) -> std::vector<cv::Mat>
{
  std::vector<cv::Mat> outputs;
  net_.setInput(blob);
  net_.forward(outputs, net_.getUnconnectedOutLayersNames());
  return outputs;
}

} // namespace yolo