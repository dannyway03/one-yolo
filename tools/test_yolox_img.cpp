#include <iostream>

#include "Yolo.h"
using namespace yolo;

auto
main(int argc, char** argv) -> int // NOLINT(modernize-avoid-c-arrays)
{
  if (argc < 3)
  {
    std::cerr << "usage: " << argv[0] << " <ort|ovn> <image_path>\n";
    return 1;
  }

  const std::string backend = argv[1];
  const std::string img_path = argv[2];

  YoloConfig cfg;
  cfg.version_ = YoloVersion::YOLO5;
  cfg.task_ = YoloTaskType::DET;
  cfg.input_w_ = 640;
  cfg.input_h_ = 384;
  cfg.batch_size_ = 1;
  cfg.num_classes_ = 80;
  cfg.scale_f_ = 1.0f;
  cfg.rgb_ = true;
  cfg.names_ = COCO_NAMES;

  if (backend == "ort")
  {
    cfg.desc_ = "YOLOX-Nano ORT test";
    cfg.target_rt_ = YoloTargetRT::ORT_CPU;
    cfg.model_path_ = "models/yolox_nano_1x3x384x640_decoded.onnx";
  }
  else if (backend == "ovn")
  {
    cfg.desc_ = "YOLOX-Nano OVN test";
    cfg.target_rt_ = YoloTargetRT::OVN_CPU;
    cfg.model_path_ = "models/yolox_nano_1x3x384x640_decoded.xml";
  }
  else
  {
    std::cerr << "unknown backend: " << backend << "\n";
    return 1;
  }

  cv::Mat image = cv::imread(img_path);
  if (image.empty())
  {
    std::cerr << "failed to read image: " << img_path << "\n";
    return 1;
  }

  auto model = Yolo(cfg);
  model.info();

  auto results = model(std::vector<cv::Mat>{image});
  auto& r = results[0];

  r.info();
  r.toJson(true);
  r.toCsv(true);

  auto annotated = r.plot();
  const std::string out_path = "/tmp/yolox_result_" + backend + ".jpg";
  cv::imwrite(out_path, annotated);
  std::cout << "\nannotated image saved to: " << out_path << "\n";
  return 0;
}