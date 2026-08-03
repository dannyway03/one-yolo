#pragma once
#include <string>

namespace yolo {

enum class YoloTrackAlgo
{
  SORT,
  BYTE_TRACK
};
enum class YoloTrackLoc
{
  CENTER,
  BOTTOM_CENTER,
  BOTTOM_CUSTOM
};

struct YoloTrackConfig
{
  YoloTrackAlgo algo_ = YoloTrackAlgo::SORT;
  YoloTrackLoc loc_ = YoloTrackLoc::BOTTOM_CENTER;

  float loc_f_ = 0.5f; // valid only if loc == YoloTrackLoc::BOTTOM_CUSTOM
  int max_miss_ = 1;
  int min_hits_ = 3;
  float iou_thresh_ = 0.8f;
};

auto
toString(YoloTrackAlgo algo) -> std::string;
auto
toString(YoloTrackLoc loc) -> std::string;

} // namespace yolo