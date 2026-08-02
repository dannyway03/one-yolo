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
  std::vector<YoloClsObj> classes;    // YoloClsObj list for classification task
  std::vector<YoloDetObj> detections; // YoloDetObj list for detection task

  /* debug information */
  int id = -1; // -1 means not initialized by Yolo. set as batch index if Yolo works with batch mode, or 0 forever.
  YoloTaskType task;
  YoloVersion version;
  YoloTargetRT target_rt;
  int batch_size;
  int input_w;
  int input_h;
  cv::Mat input_image;
  LetterBoxInfo letterbox_info;
  cv::Mat orig_image;
  cv::Size orig_size;
  std::vector<std::string> names;
  std::vector<float> speed; // time for preprocess, inference, postprocess

  /* tookit */
  /* ****** */
  /**
   * @brief
   * get annotated image.
   *
   * @param param parameter for drawing.
   */
  auto
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
  int
  show(bool block = true, float scale_f = 1.0f, const DrawParam& param = DrawParam(), bool show_orig_img = false,
       bool show_input_img = false);

  /**
   * @brief
   * save annotated image to file and return filename.
   *
   * @return filename of annotated image.
   */
  std::string
  save(const DrawParam& param = DrawParam());

  /**
   * @brief
   * get structured results as json format.
   *
   * @param print print json to console or not.
   * @param indent indent (4 spaces and multi lines) or not (only 1 line).
   *
   * @return string of json.
   */
  std::string
  to_json(bool print = false, bool indent = true);

  /**
   * @brief
   * get structured results as csv format.
   *
   * @param print print csv to console or not.
   *
   * @return string of csv.
   */
  std::string
  to_csv(bool print = false);

  /**
   * @brief
   * get summary of `YoloResult` (debug information, config data).
   *
   * @param print print summary to console or not.
   *
   * @return summary for `YoloResult`.
   */
  std::string
  info(bool print = true);

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
  int
  top1() const;

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
  float
  top1_conf() const;

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
  std::string
  top1_label() const;

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
  std::vector<int>
  top5() const;

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
  std::vector<float>
  top5_confs() const;

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
  std::vector<std::string>
  top5_labels() const;

  /* easy api for detection task. */
  /********************************/
  std::vector<cv::Rect>       boxes() const;
  std::vector<int>            cls_ids() const;
  std::vector<float>          confs() const;
  std::vector<std::string>    labels() const;
  std::vector<int>            track_ids() const;
  std::vector<std::vector<cv::Point>> track_points() const;
};

} // namespace yolo