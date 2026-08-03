#pragma once
#include "YoloTask.h"

namespace yolo {

class YoloDetTask : public YoloTask
{
private:
  /* data */
  void
  collectBoxesYolo5(const cv::Mat& output, int batch_id, const cv::Size& orig_size, const LetterBoxInfo& lb,
                    std::vector<cv::Rect>& boxes, std::vector<float>& scores, std::vector<int>& cls_ids);
  void
  collectBoxesYolo5u811(const cv::Mat& output, int batch_id, const cv::Size& orig_size, const LetterBoxInfo& lb,
                        std::vector<cv::Rect>& boxes, std::vector<float>& scores, std::vector<int>& cls_ids);
  void
  collectBoxesYolo26(const cv::Mat& output, int batch_id, const cv::Size& orig_size, const LetterBoxInfo& lb,
                     std::vector<cv::Rect>& boxes, std::vector<float>& scores, std::vector<int>& cls_ids);

protected:
  void
  postprocessOne(const std::vector<cv::Mat>& raw_outputs, int batch_id, cv::Size orig_size, LetterBoxInfo lb_info,
                 YoloResult& result) override;

public:
  YoloDetTask(const YoloConfig& cfg);
  ~YoloDetTask();
};

} // namespace yolo