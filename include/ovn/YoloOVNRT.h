#pragma once
#include <openvino/openvino.hpp>
#include "YoloRuntime.h"

namespace yolo {
    /**
     * @brief
     * Yolo runtime based on openvino library.
    */
    class YoloOVNRT: public YoloRuntime
    {
    private:
        ov::Core core_;
        ov::CompiledModel compiled_model_;
    public:
        YoloOVNRT(
            const std::string& model_path, 
            const std::string& device = "CPU");
        ~YoloOVNRT();
        auto
        inference(const cv::Mat& blob) -> std::vector<cv::Mat> override;
    };
}