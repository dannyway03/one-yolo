#include "ovn/YoloOVNRT.h"

/**
 * tested for:
 * 1. OpenVINO==2024
*/
namespace yolo {
    // docs.openvino.ai/2025/openvino-workflow/running-inference.html
    YoloOVNRT::YoloOVNRT(
        const std::string& model_path, 
        const std::string& device):
        YoloRuntime("OpenVINO") {
        auto model = core_.read_model(model_path);

        ov::AnyMap config;
        // LATENCY mode: minimise per-request latency for single-stream real-time use.
        // Switch to THROUGHPUT + ov::num_streams > 1 for batch/multi-stream pipelines.
        config[ov::hint::performance_mode.name()] =
            ov::hint::PerformanceMode::LATENCY;
        // Skip IR recompilation on subsequent launches (saves 1-5 s on GPU/NPU).
        config[ov::cache_dir.name()] = "/tmp/ov_cache";

        compiled_model_ = core_.compile_model(model, device, config);
        infer_request_ = compiled_model_.create_infer_request();
    }

    YoloOVNRT::~YoloOVNRT() = default;

    auto
    YoloOVNRT::inference(const cv::Mat& blob) -> std::vector<cv::Mat>
    {
      // [batch, 3, input_h, input_w] or [batch, input_h, input_w, 3]
      assert(blob.isContinuous());
      assert(blob.type() == CV_32F);
      assert(blob.dims == 4);

      int d0 = blob.size[0];
      int d1 = blob.size[1];
      int d2 = blob.size[2];
      int d3 = blob.size[3];

      // zero copy: wrap blob data directly into OV tensor (no allocation)
      ov::Tensor input_tensor(ov::element::f32, {(size_t)d0, (size_t)d1, (size_t)d2, (size_t)d3},
                              const_cast<float*>(reinterpret_cast<const float*>(blob.data)));

      infer_request_.set_input_tensor(input_tensor);
      // start_async + wait: device runs async, ready for double-buffer upgrade later.
      // Caller must consume returned cv::Mat before the next inference() call —
      // output data aliases infer_request_'s internal buffers (no copy).
      infer_request_.start_async();
      infer_request_.wait();

      std::vector<cv::Mat> outputs;
      const auto& output_ports = compiled_model_.outputs();

      for (size_t i = 0; i < output_ports.size(); ++i)
      {
        ov::Tensor out_tensor = infer_request_.get_output_tensor(i);
        const ov::Shape& shape = out_tensor.get_shape();

        auto* out_data = out_tensor.data<float>();

        int mat_dims = static_cast<int>(shape.size());
        std::vector<int> mat_sizes(mat_dims);
        for (int d = 0; d < mat_dims; ++d)
          mat_sizes[d] = static_cast<int>(shape[d]);

        // No clone: cv::Mat aliases infer_request_'s internal buffer directly.
        outputs.emplace_back(mat_dims, mat_sizes.data(), CV_32F, out_data);
      }

      return outputs;
    }

}