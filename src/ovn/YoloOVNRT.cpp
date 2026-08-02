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
        // TODO(perf): for throughput-oriented video pipelines (batch > 1 or async),
        // switch to PerformanceMode::THROUGHPUT and set ov::num_streams to a value
        // > 1 so the runtime can schedule across cores. For real-time single-stream
        // use, LATENCY is correct.
        config[ov::hint::performance_mode.name()] =
            ov::hint::PerformanceMode::LATENCY;
        // TODO(perf): add `config[ov::cache_dir.name()] = "/tmp/ov_cache"` to skip
        // model recompilation on subsequent launches (saves 1-5 s on GPU/NPU).

        compiled_model_ = core_.compile_model(model, device, config);
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

      // TODO(perf): create_infer_request() allocates internal buffers on every call
      // (~2-5 ms overhead). Pre-allocate one (or a pool for async) in the constructor
      // and reuse it across frames. For a single-threaded video loop this is trivial:
      //   store `ov::InferRequest __infer_request` as a member and call only once.
      auto infer_request = compiled_model_.create_infer_request();

      // zero copy
      ov::Tensor input_tensor(ov::element::f32, {(size_t)d0, (size_t)d1, (size_t)d2, (size_t)d3},
                              const_cast<float*>(reinterpret_cast<const float*>(blob.data)));

      infer_request.set_input_tensor(input_tensor);
      // TODO(perf): infer() is synchronous. For video pipelines, replace with
      // start_async() + wait() and double-buffer two InferRequests to overlap
      // CPU preprocessing of frame N+1 with GPU/NPU inference on frame N.
      infer_request.infer();

      std::vector<cv::Mat> outputs;
      const auto& output_ports = compiled_model_.outputs();

      for (size_t i = 0; i < output_ports.size(); ++i)
      {
        ov::Tensor out_tensor = infer_request.get_output_tensor(i);
        const ov::Shape& shape = out_tensor.get_shape();

        auto* out_data = out_tensor.data<float>();

        int mat_dims = static_cast<int>(shape.size());
        std::vector<int> mat_sizes(mat_dims);
        for (int d = 0; d < mat_dims; ++d)
          mat_sizes[d] = static_cast<int>(shape[d]);

        cv::Mat out_mat(mat_dims, mat_sizes.data(), CV_32F, out_data);
        // TODO(perf): clone() copies the full output tensor on every frame.
        // If the InferRequest is reused (see above), the tensor lifetime is
        // tied to it, so clone is necessary — but with a persistent request
        // the postprocessor can read directly from out_data without copying.
        outputs.push_back(out_mat.clone());
      }

      return outputs;
    }

}