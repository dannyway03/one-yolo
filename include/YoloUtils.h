#pragma once
#include <sstream>

#include <opencv2/opencv.hpp>

#include "YoloObjs.h"

namespace yolo {

/**
 * @brief
 * convert float value to string with precision.
 */
auto
toString(const float f, const int precision = 2) -> std::string;

/**
 * @brief
 * convert std::vector<T> to string with array format in json.
 */
template<typename T>
auto
toString(const std::vector<T>& v) -> std::string
{
  std::ostringstream oss;
  oss << v;
  return oss.str();
}

/**
 * @brief
 * create 48 kind of colors.
 */
auto
getColors48() -> std::vector<cv::Scalar>;

/**
 * @brief
 * make std::vector<T> serializable with array format in json.
 */
template<typename T>
auto
operator<<(std::ostream& os, const std::vector<T>& v) -> std::ostream&
{
  os << json(v).dump();
  return os;
}

/**
 * @brief
 * toolkit for drawing YoloResult.
 */
auto
drawResults(const cv::Mat& image, const DrawParam& param, const std::vector<int>& top5,
            const std::vector<float>& top5_confs, const std::vector<std::string>& top5_labels,
            const std::vector<int>& cls_ids, const std::vector<float>& confs, const std::vector<std::string>& labels,
            const std::vector<cv::Rect>& boxes, const std::vector<int>& track_ids,
            const std::vector<std::vector<cv::Point>>& tracks) -> cv::Mat;

class YoloUtils
{
private:
  /* data */

public:
  YoloUtils(/* args */);
  ~YoloUtils() = default;
  auto
  letterbox(const cv::Mat& img, int new_w, int new_h, LetterBoxInfo& info,
            const cv::Scalar& color = cv::Scalar(114, 114, 114)) -> cv::Mat;
  void
  classAwareNms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, const std::vector<int>& cls_ids,
                float conf_thresh, float nms_thresh, std::vector<int>& keep_indices);
  void
  classAwareNms(const std::vector<cv::RotatedRect>& rboxes, const std::vector<float>& scores,
                const std::vector<int>& cls_ids, float conf_thresh, float nms_thresh, std::vector<int>& keep_indices);
  auto
  decodeBox(float cx, float cy, float w, float h, const LetterBoxInfo& lb, const cv::Size& orig_size) -> cv::Rect;
};

} // namespace yolo