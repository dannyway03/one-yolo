#include "track/sort/SortTrackAlgo.h"

namespace yolo {

SortTrackAlgo::SortTrackAlgo(const YoloTrackConfig& cfg) : BaseTrackAlgo(cfg) {}

SortTrackAlgo::~SortTrackAlgo() = default;

auto
SortTrackAlgo::getIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt) -> double
{
  float in = (bb_test & bb_gt).area();
  float un = bb_test.area() + bb_gt.area() - in;

  if (un < DBL_EPSILON)
    return 0;

  return (double)(in / un);
}

void
SortTrackAlgo::run(const std::vector<cv::Rect>& boxes, const std::vector<std::vector<float>>& embeddings,
                   std::vector<int>& track_ids)
{
  // fill track_ids with default value(-1) according to boxes' size (embeddings ignored)
  track_ids.resize(boxes.size());
  for (auto& track_id : track_ids)
    track_id = -1;

  // first time to initialize KalmanTracker
  if (trackers_.empty())
  {
    for (auto boxe : boxes)
    {
      auto trk = KalmanTracker(cv::Rect_<float>(boxe.x, boxe.y, boxe.width, boxe.height));
      trackers_.emplace_back(trk);
    }
    return;
  }

  // 3.1. get predicted locations from existing trackers_.
  predicted_boxes_.clear();
  for (auto it = trackers_.begin(); it != trackers_.end();)
  {
    auto p_box = (*it).predict();
    if (p_box.x >= 0 && p_box.y >= 0)
    {
      predicted_boxes_.emplace_back(p_box);
      it++;
    }
    else
    {
      it = trackers_.erase(it);
    }
  }

  // 3.2. associate detections to tracked object (both represented as bounding boxes)
  // dets : detFrameData[fi]
  auto trk_num = predicted_boxes_.size();
  auto det_num = boxes.size();

  iou_matrix_.clear();
  iou_matrix_.resize(trk_num, vector<double>(det_num, 0));

  // compute iou matrix as a distance matrix
  for (unsigned int i = 0; i < trk_num; i++)
  {
    for (unsigned int j = 0; j < det_num; j++)
    {
      // use 1-iou because the hungarian algorithm computes a minimum-cost assignment_.
      iou_matrix_[i][j] =
        1 - getIOU(predicted_boxes_[i], cv::Rect_<float>(boxes[j].x, boxes[j].y, boxes[j].width, boxes[j].height));
    }
  }

  // solve the assignment_ problem using hungarian algorithm.
  // the resulting assignment_ is [track(prediction) : detection], with len=preNum
  HungarianAlgorithm hung_algo;
  assignment_.clear();
  hung_algo.Solve(iou_matrix_, assignment_);

  // find matches, unmatched_detections and unmatched_predictions
  unmatched_trajectories_.clear();
  unmatched_detections_.clear();
  all_items_.clear();
  matched_items_.clear();

  // there are unmatched detections
  if (det_num > trk_num)
  {
    for (unsigned int n = 0; n < det_num; n++)
      all_items_.insert(n);

    for (unsigned int i = 0; i < trk_num; ++i)
      matched_items_.insert(assignment_[i]);

    set_difference(all_items_.begin(), all_items_.end(), matched_items_.begin(), matched_items_.end(),
                   insert_iterator<set<int>>(unmatched_detections_, unmatched_detections_.begin()));
  }
  // there are unmatched trajectory/predictions
  else if (det_num < trk_num)
  {
    for (unsigned int i = 0; i < trk_num; ++i)
      if (assignment_[i] == -1) // unassigned label will be set as -1 in the assignment_ algorithm
        unmatched_trajectories_.insert(i);
  }
  else
  {
  }

  // filter out matched with low IOU
  matched_pairs_.clear();
  for (unsigned int i = 0; i < trk_num; ++i)
  {
    if (assignment_[i] == -1) // pass over invalid values
      continue;
    if (1 - iou_matrix_[i][assignment_[i]] < _cfg.iou_thresh)
    {
      unmatched_trajectories_.insert(i);
      unmatched_detections_.insert(assignment_[i]);
    }
    else
    {
      matched_pairs_.emplace_back(i, assignment_[i]);
    }
  }

  // 3.3. updating trackers_
  // update matched trackers_ with assigned detections.
  // each prediction is corresponding to a tracker_
  int det_idx, trk_idx;
  for (auto& matchedPair : matched_pairs_)
  {
    trk_idx = matchedPair.x;
    det_idx = matchedPair.y;
    trackers_[trk_idx].update(
      cv::Rect_<float>(boxes[det_idx].x, boxes[det_idx].y, boxes[det_idx].width, boxes[det_idx].height));
  }

  // create and initialise new trackers_ for unmatched detections
  for (auto& umd : unmatched_detections_)
  {
    auto tracker_ = KalmanTracker(cv::Rect_<float>(boxes[umd].x, boxes[umd].y, boxes[umd].width, boxes[umd].height));
    trackers_.emplace_back(tracker_);
  }

  // get trackers_' output
  frame_tracking_result_.clear();
  for (auto it = trackers_.begin(); it != trackers_.end();)
  {
    if (((*it).m_time_since_update < 1) && ((*it).m_hit_streak >= _cfg.min_hits))
    {
      TrackingBox res;
      res.box_ = (*it).get_state();
      res.id_ = (*it).m_id + 1;
      frame_tracking_result_.emplace_back(res);
      it++;
    }
    else
      it++;

    // remove dead tracker_
    if (it != trackers_.end() && (*it).m_time_since_update > _cfg.max_miss)
      it = trackers_.erase(it);
  }

  for (const auto& tb : frame_tracking_result_)
  {
    // id and box need to correspond
    for (int i = 0; i < boxes.size(); ++i)
    {
      if (getIOU(cv::Rect_<float>(boxes[i].x, boxes[i].y, boxes[i].width, boxes[i].height),
                 cv::Rect_<float>(tb.box_.x, tb.box_.y, tb.box_.width, tb.box_.height)) > 0.8)
      {
        track_ids[i] = tb.id_;
      }
    }
  }
}

} // namespace yolo