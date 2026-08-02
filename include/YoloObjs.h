#pragma once
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace yolo {

struct YoloClsObj
{
  int cls_id;        // class id for classification task
  float conf;        // confidence for classification task
  std::string label; // label for classification task
};

struct YoloDetObj
{
  cv::Rect box;      // bounding box for detected object in detection task
  int cls_id;        // class id for detected object in detection task
  float conf;        // confidence for detected object in detection task
  std::string label; // label for detected object in detection task
  int track_id = -1; // track id for detected object in detection task (-1 means no tracked yet)
  std::vector<cv::Point>
    track_points; // track points for detected object in detection task (empty means no tracked yet)
};

struct LetterBoxInfo
{
  float scale;
  int pad_w;
  int pad_h;
};

struct DrawParam
{
  bool top1_only = false;     // draw top1 only or not for classification task
  bool color_by_class = true; // choose background color by class or by instance

  bool cls_ids = false;  // draw class ids or not
  bool confs = true;     // draw confidences or not
  bool labels = true;    // draw labels or not
  bool boxes = true;     // draw boxes or not
  bool track_ids = true; // draw track ids or not
  bool tracks = true;    // draw tracks or not

  int box_line_width = 2;                            // line width of box, <=0 means no drawing
  int track_line_width = 1;                          // line width of track, <=0 means no drawing
  int loc_radius = 4;                                // radius of located point, <=0 means no drawing
  cv::Scalar font_color = cv::Scalar(255, 255, 255); // color for drawing text(BGR)
  float font_scale = 0.5f;                           // font scale for drawing text
  int font_face = cv::FONT_HERSHEY_SIMPLEX;          // font face for drawing text
  int font_thickness = 1;                            // font thickness for drawing text
  [[maybe_unused]] float scale = 1.0f;               // resize canvas or not
};

[[maybe_unused]] void
to_json(json& j, const YoloClsObj& obj);
[[maybe_unused]] void
to_json(json& j, const YoloDetObj& obj);

} // namespace yolo
