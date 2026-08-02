#pragma once

#include <rknn_api.h>
#include "YoloRuntime.h"

namespace yolo {
    /**
     * @brief
     * Yolo runtime based on rknn library.
    */
    class YoloRKNNRT: public YoloRuntime
    {
    private:
        rknn_context ctx_ = 0;
        rknn_input_output_num io_num_{};
        rknn_tensor_attr input_attr_{};
        std::vector<rknn_tensor_attr> output_attrs_{};

        auto
        queryIO() -> bool;

      public:
        YoloRKNNRT(const std::string& model_path);
        ~YoloRKNNRT();
        auto inference(const cv::Mat& blob) -> std::vector<cv::Mat> override;
    };
}