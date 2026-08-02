#include "YoloObjs.h"

// cv::to_json must be declared before yolo::to_json(YoloDetObj) uses cv::Rect/cv::Point
namespace cv {

void
to_json(json& j, const cv::Point& obj)
{
  j = json{
    {"x", obj.x},
    {"y", obj.y}
  };
}

void
to_json(json& j, const cv::Rect& obj)
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

[[maybe_unused]] void
to_json(json& j, const YoloClsObj& obj)
{
  j = json{
    {"cls_id", obj.cls_id},
    {  "conf",   obj.conf},
    { "label",  obj.label}
  };
}

[[maybe_unused]] void
to_json(json& j, const YoloDetObj& obj)
{
  j = json{
    {     "box",      obj.box},
    {  "cls_id",   obj.cls_id},
    {    "conf",     obj.conf},
    {   "label",    obj.label},
    {"track_id", obj.track_id}
  };
}

} // namespace yolo
