#pragma once
#include "track/BaseTrackAlgo.h"
#include "track/sort/KalmanTracker.h"
#include "track/sort/Hungarian.h"

namespace yolo {
    class SortTrackAlgo: public BaseTrackAlgo {
    private:
      using TrackingBox = struct TrackingBox
      {
        int id_;
        Rect_<float> box_;
      };

      std::vector<KalmanTracker> trackers_;
      std::vector<cv::Rect_<float>> predicted_boxes_;
      std::vector<vector<double>> iou_matrix_;
      std::vector<int> assignment_;
      std::set<int> unmatched_detections_;
      std::set<int> unmatched_trajectories_;
      std::set<int> all_items_;
      std::set<int> matched_items_;
      std::vector<cv::Point> matched_pairs_;
      std::vector<TrackingBox> frame_tracking_result_;
      auto
      getIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt) -> double;

    public:
        SortTrackAlgo(const YoloTrackConfig& cfg);
        ~SortTrackAlgo();
        void
        run(const std::vector<cv::Rect>& boxes, const std::vector<std::vector<float>>& embeddings,
            std::vector<int>& track_ids) override;
    };
}