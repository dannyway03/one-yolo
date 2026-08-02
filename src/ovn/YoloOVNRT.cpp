#include "ovn/YoloOVNRT.h"

/**
 * tested for:
 * 1. OpenVINO==2026
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
        // Only req[0] is created here. req[1] is created lazily on the first
        // submit() call so that callers using only inference() pay no L3 penalty.
        infer_req_[0] = compiled_model_.create_infer_request();
    }

    YoloOVNRT::~YoloOVNRT() = default;

    // ── blocking single-frame path ────────────────────────────────────────────

    auto YoloOVNRT::inference(const cv::Mat& blob) -> std::vector<cv::Mat>
    {
        // Always uses infer_req_[0]. Do not mix with submit/collect in the same loop.
        assert(blob.isContinuous());
        assert(blob.type() == CV_32F);
        assert(blob.dims == 4);

        ov::Tensor input_tensor(
            ov::element::f32,
            {(size_t)blob.size[0], (size_t)blob.size[1],
             (size_t)blob.size[2], (size_t)blob.size[3]},
            const_cast<float*>(reinterpret_cast<const float*>(blob.data)));

        infer_req_[0].set_input_tensor(input_tensor);
        infer_req_[0].start_async();
        infer_req_[0].wait();

        std::vector<cv::Mat> outputs;
        const auto& output_ports = compiled_model_.outputs();
        outputs.reserve(output_ports.size());

        for (size_t i = 0; i < output_ports.size(); ++i)
        {
            ov::Tensor out_tensor = infer_req_[0].get_output_tensor(i);
            const ov::Shape& shape = out_tensor.get_shape();
            auto* out_data = out_tensor.data<float>();

            int mat_dims = static_cast<int>(shape.size());
            std::vector<int> mat_sizes(mat_dims);
            for (int d = 0; d < mat_dims; ++d)
                mat_sizes[d] = static_cast<int>(shape[d]);

            outputs.emplace_back(mat_dims, mat_sizes.data(), CV_32F, out_data);
        }

        return outputs;
    }

    // ── pipeline API (GPU/NPU double-buffer) ──────────────────────────────────

    void YoloOVNRT::submit(const cv::Mat& blob)
    {
        if (!pipeline_ready_)
        {
            infer_req_[1] = compiled_model_.create_infer_request();
            pipeline_ready_ = true;
        }

        assert(blob.isContinuous());
        assert(blob.type() == CV_32F);
        assert(blob.dims == 4);

        // Zero-copy: blob must stay valid and unmodified until the matching collect().
        ov::Tensor input_tensor(
            ov::element::f32,
            {(size_t)blob.size[0], (size_t)blob.size[1],
             (size_t)blob.size[2], (size_t)blob.size[3]},
            const_cast<float*>(reinterpret_cast<const float*>(blob.data)));

        infer_req_[submit_idx_].set_input_tensor(input_tensor);
        infer_req_[submit_idx_].start_async();
        submit_idx_ ^= 1;
    }

    auto YoloOVNRT::collect() -> std::vector<cv::Mat>
    {
        infer_req_[collect_idx_].wait();

        std::vector<cv::Mat> outputs;
        const auto& output_ports = compiled_model_.outputs();
        outputs.reserve(output_ports.size());

        for (size_t i = 0; i < output_ports.size(); ++i)
        {
            ov::Tensor out_tensor = infer_req_[collect_idx_].get_output_tensor(i);
            const ov::Shape& shape = out_tensor.get_shape();
            auto* out_data = out_tensor.data<float>();

            int mat_dims = static_cast<int>(shape.size());
            std::vector<int> mat_sizes(mat_dims);
            for (int d = 0; d < mat_dims; ++d)
                mat_sizes[d] = static_cast<int>(shape[d]);

            outputs.emplace_back(mat_dims, mat_sizes.data(), CV_32F, out_data);
        }

        collect_idx_ ^= 1;
        return outputs;
    }

}
