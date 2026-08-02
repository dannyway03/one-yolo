#pragma once

#include <stdexcept>
#include <string>

#include "YoloConfig.h"

namespace app {

[[nodiscard]] inline auto
resolveRuntime(const std::string& backend, const std::string& device) -> yolo::YoloTargetRT
{
  if (backend == "ort")
    return (device == "cuda") ? yolo::YoloTargetRT::ORT_CUDA : yolo::YoloTargetRT::ORT_CPU;
  if (backend == "ovn")
  {
    if (device == "gpu")
      return yolo::YoloTargetRT::OVN_GPU;
    if (device == "auto")
      return yolo::YoloTargetRT::OVN_AUTO;
    return yolo::YoloTargetRT::OVN_CPU;
  }
  throw std::runtime_error("unknown --backend: " + backend);
}

} // namespace app