///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.cpp: KalmanTracker Class Implementation Declaration

#include "track/sort/KalmanTracker.h"

namespace yolo {

int KalmanTracker::kf_count = 0;

// initialize Kalman filter
void
KalmanTracker::initKf(StateType state_mat)
{
  int state_num = 7;
  int measure_num = 4;
  kf_ = KalmanFilter(state_num, measure_num, 0);

  measurement_ = cv::Mat::zeros(measure_num, 1, CV_32F);

  kf_.transitionMatrix = (cv::Mat_<float>(state_num, state_num) << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                          0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1);

  setIdentity(kf_.measurementMatrix);
  setIdentity(kf_.processNoiseCov, Scalar::all(1e-2));
  setIdentity(kf_.measurementNoiseCov, Scalar::all(1e-1));
  setIdentity(kf_.errorCovPost, Scalar::all(1));

  // initialize state vector with bounding box in [cx,cy,s,r] style
  kf_.statePost.at<float>(0, 0) = state_mat.x + state_mat.width / 2;
  kf_.statePost.at<float>(1, 0) = state_mat.y + state_mat.height / 2;
  kf_.statePost.at<float>(2, 0) = state_mat.area();
  kf_.statePost.at<float>(3, 0) = state_mat.width / state_mat.height;
}

// Predict the estimated bounding box.
auto
KalmanTracker::predict() -> StateType
{
  // predict
  Mat p = kf_.predict();
  m_age_ += 1;

  if (m_time_since_update_ > 0)
    m_hit_streak_ = 0;
  m_time_since_update_ += 1;

  StateType predict_box = getRectXysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));

  m_history_.push_back(predict_box);
  return m_history_.back();
}

// Update the state vector with observed bounding box.
void
KalmanTracker::update(StateType state_mat)
{
  m_time_since_update_ = 0;
  m_history_.clear();
  m_hits_ += 1;
  m_hit_streak_ += 1;

  // measurement
  measurement_.at<float>(0, 0) = state_mat.x + state_mat.width / 2;
  measurement_.at<float>(1, 0) = state_mat.y + state_mat.height / 2;
  measurement_.at<float>(2, 0) = state_mat.area();
  measurement_.at<float>(3, 0) = state_mat.width / state_mat.height;

  // update
  kf_.correct(measurement_);
}

// Return the current state vector
auto
KalmanTracker::getState() -> StateType
{
  Mat s = kf_.statePost;
  return getRectXysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
}

// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
auto
KalmanTracker::getRectXysr(float cx, float cy, float s, float r) -> StateType
{
  float w = sqrt(s * r);
  float h = s / w;
  float x = (cx - w / 2);
  float y = (cy - h / 2);

  if (x < 0 && cx > 0)
    x = 0;
  if (y < 0 && cy > 0)
    y = 0;

  return StateType(x, y, w, h);
}

} // namespace yolo
