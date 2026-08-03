#pragma once

#include <cstring>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../tools/app_common.hpp"
#include "track/YoloTracker.h"
#include "Yolo.h"

namespace app {

constexpr float kDefaultScale = 1.0f;

// ── CLI args ──────────────────────────────────────────────────────────────────

struct CliArgs
{
  std::string source_;
  std::string config_;
  std::string backend_ = "ort";
  std::string device_ = "cpu";
  float scale_ = kDefaultScale;
  bool no_track_ = false;
};

// ── source classification ─────────────────────────────────────────────────────

[[nodiscard]] inline auto
isImageSource(std::string_view path) -> bool
{
  for (auto* ext : {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})
  {
    const auto len = std::strlen(ext);
    if (path.size() >= len && path.compare(path.size() - len, len, ext) == 0)
      return true;
  }
  return false;
}

[[nodiscard]] inline auto
isCameraIndex(std::string_view s) -> bool
{
  return !s.empty() && std::all_of(s.begin(), s.end(),
                                   [](char c) -> bool
                                   {
                                     return std::isdigit(c) != 0;
                                   });
}

// ── config builders ───────────────────────────────────────────────────────────

[[nodiscard]] inline auto
buildYoloConfig(const CliArgs& a, yolo::YoloTaskType /*task*/) -> yolo::YoloConfig
{
  auto cfg = yolo::YoloConfig::from_json(a.config_);
  cfg.target_rt_ = resolveRuntime(a.backend_, a.device_);
  return cfg;
}

[[nodiscard]] inline auto
buildTrackerConfig() -> yolo::YoloTrackConfig
{
  yolo::YoloTrackConfig t;
  t.algo_ = yolo::YoloTrackAlgo::SORT;
  t.iou_thresh_ = 0.6f;
  return t;
}

// ── usage ─────────────────────────────────────────────────────────────────────

inline void
printUsage(const char* app_name, const char* task_desc)
{
  std::cout << "usage: " << app_name << " <source> <config.json> [options]\n"
            << "  task: " << task_desc << "\n\n"
            << "  <source>         image file, video file, or webcam index\n"
            << "  <config.json>    model config (model path + blob params)\n"
            << "  --backend        ort|ovn                     (default: ort)\n"
            << "  --device         cpu|gpu|auto|cuda           (default: cpu)\n"
            << "  --scale          <float>  display scale      (default: 1.0)\n"
            << "  --no-track       disable SORT tracker (det only)\n";
}

// ── arg parser ────────────────────────────────────────────────────────────────

[[nodiscard]] inline auto
parseArgs(int argc, char** argv) -> CliArgs
{
  CliArgs a;
  int positional = 0;
  for (int i = 1; i < argc; ++i)
  {
    const std::string k = argv[i];
    auto next = [&]() -> std::string
    {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--backend")
    {
      a.backend_ = next();
    }
    else if (k == "--device")
    {
      a.device_ = next();
    }
    else if (k == "--scale")
    {
      a.scale_ = std::stof(next());
    }
    else if (k == "--no-track")
    {
      a.no_track_ = true;
    }
    else if (k == "--help" || k == "-h")
    { /* caller handles */
    }
    else if (k.rfind("--", 0) == 0)
    {
      throw std::runtime_error("unknown argument: " + k);
    }
    else
    {
      if (positional == 0)
        a.source_ = k;
      else if (positional == 1)
        a.config_ = k;
      else
        throw std::runtime_error("unexpected positional argument: " + k);
      ++positional;
    }
  }
  return a;
}

// ── source loop ───────────────────────────────────────────────────────────────

// handler(frame, single_shot) → bool: return false to break the loop
template<typename Handler>
inline void
runLoop(const CliArgs& a, Handler&& handler)
{
  if (a.source_.empty())
    throw std::runtime_error("--source is required");

  if (isImageSource(a.source_))
  {
    const cv::Mat img = cv::imread(a.source_);
    if (img.empty())
      throw std::runtime_error("cannot read image: " + a.source_);
    handler(img, /*single_shot=*/true);
    return;
  }

  cv::VideoCapture cap;
  const bool is_camera = isCameraIndex(a.source_);
  if (is_camera)
    cap.open(std::stoi(a.source_));
  else
    cap.open(a.source_);
  if (!cap.isOpened())
    throw std::runtime_error("cannot open source: " + a.source_);

  while (cap.isOpened())
  {
    cv::Mat frame;
    if (!cap.read(frame))
    {
      if (!is_camera)
      {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        continue;
      }
      break;
    }
    if (!handler(frame, /*single_shot=*/false))
      break;
  }
}

} // namespace app
