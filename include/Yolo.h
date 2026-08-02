
#pragma once
#include "YoloConfig.h"
#include "YoloResult.h"
#include "YoloTask.h"

namespace yolo {
    /**
     * @brief
     * wrapper class for unified interface accessing Yolo powered by ultralytics (ultralytics/ultralytics repo from github). 
     * 
     * support:
     * 1. yolo:  yolov5/yolov5u/yolov8/yolov11/yolov26
     * 2. tasks: classification/detection/segmentation/pose/obb
    */
    class Yolo final {
    private:
        YoloConfig                cfg_;
        std::shared_ptr<YoloTask> task_ = nullptr;
    public:
        Yolo(const YoloConfig& cfg);
        ~Yolo();

        /**
         * @brief
         * predict with single image mode.
         * 
         * @param image image to be predicted.
         * @return structured result, single `YoloResult` object.
        */
        auto
        predict(const cv::Mat& image) -> YoloResult;

        /**
         * @brief
         * predict with batch mode.
         * 
         * @param images a list of images to be predicted with batch mode.
         * @return structured results, a list of `YoloResult` objects.
        */
        auto
        predict(const std::vector<cv::Mat>& images) -> std::vector<YoloResult>;

        /**
         * @brief
         * make `Yolo` callable, act as same as predict(...) with single image mode.
         * 
         * @param image image to be predicted.
         * @return structured result, single `YoloResult` object.
        */
        auto
        operator()(const cv::Mat& image) -> YoloResult;

        /**
         * @brief
         * make `Yolo` callable, act as same as predict(...) with batch mode.
         * 
         * @param images a list of images to be predicted with batch mode.
         * @return structured results, a list of `YoloResult` objects.
        */
        auto
        operator()(const std::vector<cv::Mat>& images) -> std::vector<YoloResult>;

        /**
         * @brief
         * get summary for `Yolo`(such as config data initialized using `YoloConfig`).
         * 
         * @param print print summary to console or not.
         * @return summary for `Yolo`.
        */
        auto
        info(bool print = true) -> std::string;
    };
}