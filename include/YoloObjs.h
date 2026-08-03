#pragma once
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

using json = nlohmann::json; // NOLINT

namespace yolo {

struct YoloClsObj
{
  int cls_id_;        // class id for classification task
  float conf_;        // confidence for classification task
  std::string label_; // label for classification task
};

struct YoloDetObj
{
  cv::Rect box_;      // bounding box for detected object in detection task
  int cls_id_;        // class id for detected object in detection task
  float conf_;        // confidence for detected object in detection task
  std::string label_; // label for detected object in detection task
  int track_id_ = -1; // track id for detected object in detection task (-1 means no tracked yet)
  std::vector<cv::Point>
    track_points_; // track points for detected object in detection task (empty means no tracked yet)
};

struct LetterBoxInfo
{
  float scale_;
  int pad_w_;
  int pad_h_;
};

struct DrawParam
{
  bool top1_only_ = false;     // draw top1 only or not for classification task
  bool color_by_class_ = true; // choose background color by class or by instance

  bool cls_ids_ = false;  // draw class ids or not
  bool confs_ = true;     // draw confidences or not
  bool labels_ = true;    // draw labels or not
  bool boxes_ = true;     // draw boxes or not
  bool track_ids_ = true; // draw track ids or not
  bool tracks_ = true;    // draw tracks or not

  int box_line_width_ = 2;                            // line width of box, <=0 means no drawing
  int track_line_width_ = 1;                          // line width of track, <=0 means no drawing
  int loc_radius_ = 4;                                // radius of located point, <=0 means no drawing
  cv::Scalar font_color_ = cv::Scalar(255, 255, 255); // color for drawing text(BGR)
  float font_scale_ = 0.5f;                           // font scale for drawing text
  int font_face_ = cv::FONT_HERSHEY_SIMPLEX;          // font face for drawing text
  int font_thickness_ = 1;                            // font thickness for drawing text
  float scale_ = 1.0f;
};

[[maybe_unused]] void
to_json(json& j, const YoloClsObj& obj); // NOLINT(*-identifier-naming)
[[maybe_unused]] void
to_json(json& j, const YoloDetObj& obj); // NOLINT(*-identifier-naming)

} // namespace yolo
