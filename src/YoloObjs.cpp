#include "YoloObjs.h"

// cv::to_json must be declared before yolo::to_json(YoloDetObj) uses cv::Rect/cv::Point
namespace cv {

void
to_json(json& j, const cv::Point& obj) // NOLINT
{
  j = json{
    {"x", obj.x},
    {"y", obj.y}
  };
}

void
to_json(json& j, const cv::Rect& obj) // NOLINT
{
  j = json{
    {     "x",      obj.x},
    {     "y",      obj.y},
    { "width",  obj.width},
    {"height", obj.height}
  };
}

} // namespace cv

namespace yolo {

void
to_json(json& j, const YoloClsObj& obj)
{
  j = json{
    {"cls_id_", obj.cls_id_},
    {   "conf",   obj.conf_},
    {  "label",  obj.label_}
  };
}

void
to_json(json& j, const YoloDetObj& obj)
{
  j = json{
    {     "box",      obj.box_},
    { "cls_id_",   obj.cls_id_},
    {    "conf",     obj.conf_},
    {   "label",    obj.label_},
    {"track_id", obj.track_id_}
  };
}

} // namespace yolo
