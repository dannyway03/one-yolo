#include <iostream>
#include <optional>
#include <stdexcept>

#include "common.hpp"

using namespace yolo;

auto
main(int argc, char* argv[]) -> int
{
  try
  {
    if (argc < 2)
    {
      app::printUsage(argv[0], "pose estimation (pose) with optional SORT tracking");
      return 0;
    }
    const app::CliArgs a = app::parseArgs(argc, argv);
    if (a.model_.empty())
    {
      app::printUsage(argv[0], "pose estimation (pose) with optional SORT tracking");
      return 1;
    }

    auto model = Yolo(app::buildYoloConfig(a, YoloTaskType::POSE));
    model.info();

    std::optional<YoloTracker> tracker;
    if (!a.no_track_)
    {
      tracker.emplace(app::buildTrackerConfig());
      tracker->info();
    }

    app::runLoop(a,
                 [&](const cv::Mat& frame, bool single_shot) -> bool
                 {
                   auto result = model(frame);
                   if (tracker.has_value() && !single_shot)
                     (*tracker)(result);
                   result.info();
                   if (!a.no_json_)
                     result.to_json(/*print=*/true);
                   if (!a.no_csv_)
                     result.to_csv(/*print=*/true);
                   const int key = result.show(/*block=*/single_shot, a.scale_, DrawParam{}, false, !single_shot);
                   return !single_shot && (key != 27);
                   /*
                    * auto boxes        = result.boxes();
                    * auto cls_ids      = result.cls_ids();
                    * auto confs        = result.confs();
                    * auto labels       = result.labels();
                    * auto kpts         = result.kpts();
                    * auto track_ids    = result.track_ids();
                    * auto track_points = result.track_points();
                    *
                    * serialise to string for downstream consumers (kafka, udp, etc.):
                    * auto str_boxes        = yolo::to_string(boxes);
                    * auto str_kpts         = yolo::to_string(kpts);
                    * auto str_track_points = yolo::to_string(track_points);
                    */
                 });
  }
  catch (const std::exception& e)
  {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
