#include <iostream>
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
      app::printUsage(argv[0], "oriented bounding box detection (obb)");
      return 0;
    }
    const app::CliArgs a = app::parseArgs(argc, argv);
    if (a.model_.empty())
    {
      app::printUsage(argv[0], "oriented bounding box detection (obb)");
      return 1;
    }

    auto model = Yolo(app::buildYoloConfig(a, YoloTaskType::OBB));
    model.info();

    app::runLoop(a,
                 [&](const cv::Mat& frame, bool single_shot) -> bool
                 {
                   auto result = model(frame);
                   result.info();
                   if (!a.no_json_)
                     result.to_json(/*print=*/true);
                   if (!a.no_csv_)
                     result.to_csv(/*print=*/true);
                   const int key = result.show(/*block=*/single_shot, a.scale_, DrawParam{}, false, !single_shot);
                   return !single_shot && (key != 27);
                   /*
                    * auto rboxes  = result.rboxes();
                    * auto cls_ids = result.cls_ids();
                    * auto confs   = result.confs();
                    * auto labels  = result.labels();
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
