#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "YoloRuntime.h"

namespace yolo {
    /**
     * @brief
     * Yolo runtime based on tensorrt library.
    */
    class YoloTRT: public YoloRuntime
    {
    private:
        nvinfer1::IRuntime* runtime_ = nullptr;
        nvinfer1::ICudaEngine* engine_ = nullptr;
        nvinfer1::IExecutionContext* context_ = nullptr;

        std::vector<void*> device_buffers_{};
        std::vector<std::vector<int64_t>> output_shapes_{};

        void
        allocateBuffers();

      public:
        YoloTRT(const std::string& model_path);
        ~YoloTRT();
        auto inference(const cv::Mat& blob) -> std::vector<cv::Mat> override;
    };
}