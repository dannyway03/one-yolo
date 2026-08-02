#include "YoloRuntime.h"

namespace yolo {

class YoloOpenCVRT : public YoloRuntime
{
private:
  cv::dnn::Net net_;

public:
  YoloOpenCVRT(const std::string& model_path, bool use_cuda = true);
  ~YoloOpenCVRT();
  auto
  inference(const cv::Mat& blob) -> std::vector<cv::Mat> override;
};

} // namespace yolo