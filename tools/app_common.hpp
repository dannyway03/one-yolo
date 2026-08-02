#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>

#include "YoloConfig.h"

namespace app {

[[nodiscard]] inline auto
resolveRuntime(const std::string& backend, const std::string& device) -> yolo::YoloTargetRT
{
  auto lower = [](std::string s) { std::transform(s.begin(), s.end(), s.begin(), ::tolower); return s; };
  const auto b = lower(backend);
  const auto d = lower(device);
  if (b == "ort")
    return (d == "cuda") ? yolo::YoloTargetRT::ORT_CUDA : yolo::YoloTargetRT::ORT_CPU;
  if (b == "ovn")
  {
    if (d == "gpu")
      return yolo::YoloTargetRT::OVN_GPU;
    if (d == "auto")
      return yolo::YoloTargetRT::OVN_AUTO;
    return yolo::YoloTargetRT::OVN_CPU;
  }
  throw std::runtime_error("unknown --backend: " + backend);
}

} // namespace app