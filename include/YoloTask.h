#pragma once
#include <memory>

#include "YoloConfig.h"
#include "YoloResult.h"
#include "YoloRuntime.h"
#include "YoloUtils.h"

namespace yolo {

class YoloTask
{
private:

protected:
  YoloConfig cfg_;
  std::vector<cv::Size> orig_sizes_;
  std::vector<LetterBoxInfo> letterbox_infos_;
  std::vector<cv::Mat> input_images_;
  std::shared_ptr<YoloRuntime> rt_ = nullptr;

  /**
   * @brief
   * preprocess for input images and return a 4D matrix to be used later.
   *
   * @param images input images with batch mode.
   * @return a 4D matrix.
   */
  virtual auto
  inference(const cv::Mat& blob) -> std::vector<cv::Mat>;

  /**
   * @brief
   * preprocess for input images one by one.
   *
   * @param image single image.
   * @return preprocessed image (scale/padding).
   */
  virtual auto
  preprocessOne(const cv::Mat& image) -> cv::Mat;

  /**
   * @brief
   * postprocess for raw outputs one by one, extract structured result and fill `YoloResult`.
   *
   * @param raw_outputs raw output matrixs from Yolo network.
   * @param batch_id bacth index to be processed.
   * @param orig_size original image size.
   * @param lb_info letterbox info (scale/padding).
   * @param result `YoloResult` to be filled.
   *
   * @note
   * MUST override in child classes.
   */
  virtual void
  postprocessOne(const std::vector<cv::Mat>& raw_outputs, int batch_id, cv::Size orig_size, LetterBoxInfo lb_info,
                 YoloResult& result) = 0;

public:
  YoloTask(YoloConfig cfg);
  ~YoloTask();

  [[nodiscard]] auto
  runtime() const -> std::shared_ptr<YoloRuntime>
  {
    return rt_;
  }

  auto
  preprocess(const std::vector<cv::Mat>& images) -> cv::Mat;
  auto
  postprocess(const std::vector<cv::Mat>& raw_outputs, int batch_size) -> std::vector<yolo::YoloResult>;

  /**
   * @brief
   * run task with input images.
   *
   * @param images input images with batch mode.
   * @return a list of `YoloResult`, has the same size of input images.
   */
  auto
  run(const std::vector<cv::Mat>& images) -> std::vector<yolo::YoloResult>;

  /**
   * @brief
   * make `YoloTask` callable, act as same as `run(...)`.
   *
   * @param images input images with batch mode.
   * @return a list of `YoloResult`, has the same size of input images.
   *
   */
  auto
  operator()(const std::vector<cv::Mat>& images) -> std::vector<yolo::YoloResult>;
};

} // namespace yolo