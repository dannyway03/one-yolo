#pragma once
#include <openvino/openvino.hpp>

#include "YoloRuntime.h"

namespace yolo {

/**
 * @brief
 * Yolo runtime based on openvino library.
 */
class YoloOVNRT : public YoloRuntime
{
private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  std::array<ov::InferRequest, 2> infer_req_;
  bool pipeline_ready_ = false;
  int submit_idx_ = 0;
  int collect_idx_ = 0;

public:
  YoloOVNRT(const std::string& model_path, const std::string& device = "CPU");
  ~YoloOVNRT();

  // Blocking single-frame inference. Always uses infer_req_[0]; zero overhead
  // from the second request. Do not mix with submit/collect in the same loop.
  auto
  inference(const cv::Mat& blob) -> std::vector<cv::Mat> override;

  // Pipeline API for GPU/NPU: submit() dispatches async, collect() waits.
  // Call submit(blob_N), do CPU preprocessing of blob_N+1, then collect().
  // infer_req_[1] is created on the first submit() call.
  // Returned cv::Mat aliases infer_req_ buffers — do not hold past the next
  // submit() to the same slot (every other call).
  void
  submit(const cv::Mat& blob);
  auto
  collect() -> std::vector<cv::Mat>;
};

} // namespace yolo