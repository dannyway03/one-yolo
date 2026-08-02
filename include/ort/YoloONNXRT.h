#pragma once
#include <onnxruntime_cxx_api.h>
#include "YoloRuntime.h"

namespace yolo {
    /**
     * @brief
     * Yolo runtime based on onnxruntime library.
    */
    class YoloONNXRT: public YoloRuntime
    {
    private:
        Ort::Env env_;
        Ort::Session session_ {nullptr};
        Ort::SessionOptions session_options_;
        void
        ortForward(const cv::Mat& input_4d, std::vector<cv::Mat>& outputs);

      public:
        YoloONNXRT(
            const std::string& model_path, 
            bool use_cuda = true);
        ~YoloONNXRT();
        auto
        inference(const cv::Mat& blob) -> std::vector<cv::Mat> override;
    };
}