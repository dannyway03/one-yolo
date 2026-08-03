#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>

#include "common.hpp"

using namespace yolo;

static auto
fmtMs(float ms) -> std::string
{
  std::ostringstream s;
  s << std::fixed << std::setprecision(1) << ms;
  return s.str();
}

static void
drawOverlay(cv::Mat& img, const YoloResult& result, bool tracking)
{
  const float pre = !result.speed_.empty() ? result.speed_[0] : 0.f;
  const float inf = result.speed_.size() > 1 ? result.speed_[1] : 0.f;
  const float post = result.speed_.size() > 2 ? result.speed_[2] : 0.f;
  const float total = pre + inf + post;
  const int fps = total > 0.f ? static_cast<int>(1000.f / total) : 0;

  const std::string line1 = toString(result.target_rt_) + "  |  " + (tracking ? "SORT" : "no-track");
  const std::string line2 = "pre " + fmtMs(pre) + "  inf " + fmtMs(inf) + "  post " + fmtMs(post) + " ms  |  " +
                            std::to_string(fps) + " FPS  |  " +
                            std::to_string(static_cast<int>(result.detections_.size())) + " det";

  constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
  constexpr double kFscale = 0.55;
  constexpr int kThick = 1;
  constexpr int kPad = 6;
  constexpr int kLineGap = 4;

  int base = 0;
  const auto sz1 = cv::getTextSize(line1, kFont, kFscale, kThick, &base);
  const auto sz2 = cv::getTextSize(line2, kFont, kFscale, kThick, &base);

  const int bar_w = std::min(std::max(sz1.width, sz2.width) + kPad * 2, img.cols);
  const int bar_h = std::min(sz1.height + sz2.height + kLineGap + kPad * 2, img.rows);

  cv::Mat roi = img(cv::Rect(0, 0, bar_w, bar_h));
  cv::Mat dark = cv::Mat::zeros(roi.size(), roi.type());
  cv::addWeighted(roi, 0.35, dark, 0.65, 0, roi);

  auto put = [&](const std::string& text, cv::Point pt) -> void
  {
    cv::putText(img, text, pt + cv::Point(1, 1), kFont, kFscale, {0, 0, 0}, kThick + 1, cv::LINE_AA);
    cv::putText(img, text, pt, kFont, kFscale, {255, 255, 255}, kThick, cv::LINE_AA);
  };

  put(line1, {kPad, kPad + sz1.height});
  put(line2, {kPad, kPad + sz1.height + kLineGap + sz2.height});
}

auto
main(int argc, char* argv[]) -> int
{
  try
  {
    if (argc < 2)
    {
      app::printUsage(argv[0], "object detection (det) with optional SORT tracking");
      return 0;
    }
    const app::CliArgs a = app::parseArgs(argc, argv);
    if (a.config_.empty())
    {
      app::printUsage(argv[0], "object detection (det) with optional SORT tracking");
      return 1;
    }

    auto model = Yolo(app::buildYoloConfig(a, YoloTaskType::DET));
    model.info();

    std::optional<YoloTracker> tracker;
    if (!a.no_track_)
    {
      tracker.emplace(app::buildTrackerConfig());
      tracker->info();
    }

    const bool tracking = tracker.has_value();

    app::runLoop(a,
                 [&](const cv::Mat& frame, bool single_shot) -> bool
                 {
                   auto result = model(frame);
                   if (tracking && !single_shot)
                     (*tracker)(result);

                   if (single_shot)
                     result.info();

                   auto vis = result.plot();
                   if (a.scale_ != 1.0f)
                     cv::resize(vis, vis, cv::Size(), a.scale_, a.scale_);
                   drawOverlay(vis, result, tracking && !single_shot);
                   cv::imshow("det", vis);
                   const int key = cv::waitKey(single_shot ? 0 : 1);
                   return !single_shot && (key != 27);
                 });
  }
  catch (const std::exception& e)
  {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
