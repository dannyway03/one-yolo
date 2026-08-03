///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.h: KalmanTracker Class Declaration

#ifndef KALMAN_H
#define KALMAN_H

#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;

#define StateType Rect_<float>

namespace yolo {

// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker
{
public:
  KalmanTracker()
  {
    initKf(StateType());
    m_time_since_update_ = 0;
    m_hits_ = 0;
    m_hit_streak_ = 0;
    m_age_ = 0;
    m_id_ = kf_count;
    // kf_count++;
  }

  KalmanTracker(StateType init_rect)
  {
    initKf(init_rect);
    m_time_since_update_ = 0;
    m_hits_ = 0;
    m_hit_streak_ = 0;
    m_age_ = 0;
    m_id_ = kf_count;
    kf_count++;
  }

  ~KalmanTracker() { m_history_.clear(); }

  auto
  predict() -> StateType;
  void
  update(StateType state_mat);

  auto
  getState() -> StateType;
  auto
  getRectXysr(float cx, float cy, float s, float r) -> StateType;

  static int kf_count;

  int m_time_since_update_;
  int m_hits_;
  int m_hit_streak_;
  int m_age_;
  int m_id_;

private:
  void
  initKf(StateType state_mat);
  cv::KalmanFilter kf_;
  cv::Mat measurement_;
  std::vector<StateType> m_history_;
};

} // namespace yolo
#endif