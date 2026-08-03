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
      app::printUsage(argv[0], "image classification (cls) — top1/top5 output");
      return 0;
    }
    const app::CliArgs a = app::parseArgs(argc, argv);
    if (a.config_.empty())
    {
      app::printUsage(argv[0], "image classification (cls) — top1/top5 output");
      return 1;
    }

    auto model = Yolo(app::buildYoloConfig(a, YoloTaskType::CLS));
    model.info();

    app::runLoop(a,
                 [&](const cv::Mat& frame, bool single_shot) -> bool
                 {
                   auto result = model(frame);
                   result.info();
                   const int key = result.show(/*block=*/single_shot, a.scale_, DrawParam{}, false, false);
                   return !single_shot && (key != 27);
                   /*
                    * auto top1       = result.top1();
                    * auto top1Conf  = result.top1Conf();
                    * auto top1_label = result.top1_label();
                    * auto top5       = result.top5();
                    * auto top5_confs = result.top5_confs();
                    * auto top5_labels = result.top5_labels();
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
