#pragma once
#include "YoloTask.h"

namespace yolo {

class YoloClsTask : public YoloTask
{
private:
  /* data */
  auto
  softmax(const cv::Mat& logits) -> cv::Mat;
  auto
  isProbDistribution(const cv::Mat& out, double eps = 1e-3) -> bool;

protected:
  void
  postprocessOne(const std::vector<cv::Mat>& raw_outputs, int batch_id, cv::Size orig_size, LetterBoxInfo lb_info,
                 YoloResult& result) override;

public:
  YoloClsTask(const YoloConfig& cfg);
  ~YoloClsTask();
};

} // namespace yolo