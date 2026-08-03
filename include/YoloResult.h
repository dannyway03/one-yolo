#pragma once
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "YoloConfig.h"
#include "YoloObjs.h"

namespace yolo {

/**
 * @brief
 * wrapper structure of prediction result for **SINGLE** image in Yolo.
 * supports classification/detection/segmentation/pose/obb 5 tasks.
 *
 * include:
 * 1. structured result(boxes/masks/keypoints/confidences/labels/...) output by Yolo.
 * 2. debug information(cost time/original image/input image...) when calling Yolo.
 * 3. tookit such as serializing result to file, show result on GUI.
 * 4. easy api for accessing structured results.
 *
 * @note
 * `YoloResult` works for ONLY 1 task at the same time.
 * and MUST be initialized by Yolo (for example, returned by `Yolo::predict(...)`), you'd better never create by
 * yourself.
 */
struct YoloResult
{
  /* structured results */
  std::vector<YoloClsObj> classes_;    // YoloClsObj list for classification task
  std::vector<YoloDetObj> detections_; // YoloDetObj list for detection task

  /* debug information */
  int id_ = -1; // -1 means not initialized by Yolo. set as batch index if Yolo works with batch mode, or 0 forever.
  YoloTaskType task_;
  YoloVersion version_;
  YoloTargetRT target_rt_;
  int batch_size_{};
  int input_w_{};
  int input_h_{};
  cv::Mat input_image_;
  LetterBoxInfo letterbox_info_{};
  cv::Mat orig_image_;
  cv::Size orig_size_;
  std::vector<std::string> names_;
  std::vector<float> speed_; // time for preprocess, inference, postprocess

  /* tookit */
  /* ****** */
  /**
   * @brief
   * get annotated image.
   *
   * @param param parameter for drawing.
   */
  [[nodiscard]] auto
  plot(const DrawParam& param = DrawParam()) const -> cv::Mat;

  /**
   * @brief
   * show annotated image on GUI(block thread or not).
   *
   * @param block block the calling thread or not.
   * @param scale_f scale factor when showing annotated image.
   * @param param parameter for drawing.
   * @param show_orig_img show original image or not.
   * @param show_input_img show input image (after preprocess and before sending network) or not.
   *
   * @return
   * key value user has pressed, just like `cv::waitKey(delay)`.
   *
   * @note
   * use block mode if show single annotated image, it will block thread until user press any key such as ESC/Enter/...
   * use un-block mode if show multi annotated images from video sequences in a loop,
   * check the return value to determine if it's time to break the loop.
   */
  [[nodiscard]] auto
  show(bool block = true, float scale_f = 1.0f, const DrawParam& param = DrawParam(), bool show_orig_img = false,
       bool show_input_img = false) const -> int;

  /**
   * @brief
   * get structured results as json format.
   *
   * @param print print json to console or not.
   * @param indent indent (4 spaces and multi lines) or not (only 1 line).
   *
   * @return string of json.
   */
  auto
  toJson(bool print = false, bool indent = true) -> std::string;

  /**
   * @brief
   * get structured results as csv format.
   *
   * @param print print csv to console or not.
   *
   * @return string of csv.
   */
  auto
  toCsv(bool print = false) -> std::string;

  /**
   * @brief
   * get summary of `YoloResult` (debug information, config data).
   *
   * @param print print summary to console or not.
   *
   * @return summary for `YoloResult`.
   */
  auto
  info(bool print = true) -> std::string;

  /* easy api for classification task. */
  /************************************/
  /**
   * @brief
   * get class id of top 1 from classification task.
   *
   * @return
   * class id of top1.
   *
   * @note
   * throw error if it's not a classification task.
   */
  [[nodiscard]] auto
  top1() const -> int;

  /**
   * @brief
   * get confidence of top 1 from classification task.
   *
   * @return
   * confidence of top1.
   *
   * @note
   * throw error if it's not a classification task.
   */
  [[nodiscard]] auto
  top1Conf() const -> float;

  /**
   * @brief
   * get label of top 1 from classification task.
   *
   * @return
   * label of top1.
   *
   * @note
   * throw error if it's not a classification task.
   */
  [[nodiscard]] auto
  top1Label() const -> std::string;

  /**
   * @brief
   * get class ids list of top 5 from classification task.
   *
   * @return
   * class ids list of top5.
   *
   * @note
   * throw error if it's not a classification task.
   */
  [[nodiscard]] auto
  top5() const -> std::vector<int>;

  /**
   * @brief
   * get confidences list of top 5 from classification task.
   *
   * @return
   * confidences list of top5.
   *
   * @note
   * throw error if it's not a classification task.
   */
  [[nodiscard]] auto
  top5Confs() const -> std::vector<float>;

  /**
   * @brief
   * get labels list of top 5 from classification task.
   *
   * @return
   * labels list of top5.
   *
   * @note
   * throw error if it's not a classification task.
   */
  [[nodiscard]] auto
  top5Labels() const -> std::vector<std::string>;

  /* easy api for detection task. */
  /********************************/
  [[nodiscard]] auto
  boxes() const -> std::vector<cv::Rect>;
  [[nodiscard]] auto
  clsIds() const -> std::vector<int>;
  [[nodiscard]] auto
  confs() const -> std::vector<float>;
  [[nodiscard]] auto
  labels() const -> std::vector<std::string>;
  [[nodiscard]] auto
  trackIds() const -> std::vector<int>;
  [[nodiscard]] auto
  trackPoints() const -> std::vector<std::vector<cv::Point>>;
};

} // namespace yolo