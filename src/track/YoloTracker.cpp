#include "track/YoloTracker.h"

#include "track/sort/SortTrackAlgo.h"

namespace yolo {

YoloTracker::YoloTracker(const YoloTrackConfig& cfg) : cfg_(cfg)
{
  init();
}

YoloTracker::~YoloTracker() = default;

void
YoloTracker::init()
{
  // choose track algorithm
  switch (cfg_.algo)
  {
    case YoloTrackAlgo::SORT:
    {
      tracker_ = std::make_shared<SortTrackAlgo>(cfg_);
      break;
    }
    case YoloTrackAlgo::BYTE_TRACK:
    {
      throw std::invalid_argument("invalid YoloTrackAlgo parameter when initializing YoloTracker!");
      break;
    }
    default:
      throw std::invalid_argument("invalid YoloTrackAlgo parameter when initializing YoloTracker!");
      break;
  }

  // initialize status
  tracking_points_.clear();
  tracking_miss_times_.clear();
}

void
YoloTracker::reset()
{
  init();
}

void
YoloTracker::preprocess(const YoloResult& res, std::vector<cv::Rect>& boxes,
                        std::vector<std::vector<float>>& embeddings)
{
  auto res_boxes = res.boxes();
  boxes.insert(boxes.end(), res_boxes.begin(), res_boxes.end());

  /* embeddings reserved because no embeddings in YoloResult now */
  // auto res_embeddings = res.embeddings();
  // embeddings.insert(embeddings.end(), res_embeddings.begin(), res_embeddings.end());
}

void
YoloTracker::run(const std::vector<cv::Rect>& boxes, const std::vector<std::vector<float>>& embeddings,
                 std::vector<int>& track_ids)
{
  (*tracker_).run(boxes, embeddings, track_ids);
}

void
YoloTracker::postprocess(const std::vector<cv::Rect>& boxes, const std::vector<std::vector<float>>& embeddings,
                         const std::vector<int>& track_ids, YoloResult& res)
{
  assert(boxes.size() == track_ids.size());
  // assert(embeddings.size() == track_ids.size());

  for (size_t i = 0; i < track_ids.size(); i++)
  {
    auto& box = boxes[i];
    auto& track_id = track_ids[i];

    if (track_id < 0)
      continue;

    cv::Point track_point;

    switch (cfg_.loc)
    {
      case YoloTrackLoc::CENTER:
      {
        track_point.x = box.x + box.width / 2;
        track_point.y = box.y + box.height / 2;
        break;
      }
      case YoloTrackLoc::BOTTOM_CENTER:
      {
        track_point.x = box.x + box.width / 2;
        track_point.y = box.y + box.height;
        break;
      }
      case YoloTrackLoc::BOTTOM_CUSTOM:
      {
        track_point.x = box.x + int(box.width * cfg_.loc_f);
        track_point.y = box.y + box.height;
        break;
      }
      default:
        throw std::runtime_error("got unsupported YoloTrackLoc when calling YoloTracker::postprocess()!");
        break;
    }

    tracking_points_[track_id].emplace_back(track_point);
    // reset to 0 since it got hit
    tracking_miss_times_[track_id] = 0;

    /* update track id & track points for YoloResult via indice directly
       important: they have the same indice order
    */
    switch (res.task)
    {
      case YoloTaskType::DET:
      {
        res.detections[i].track_id = track_id;
        res.detections[i].track_points = tracking_points_[track_id];
        break;
      }
      default:
        throw std::runtime_error("got unsupported YoloTaskType when calling YoloTracker::postprocess()!");
        break;
    }
  }

  // check miss times & clear garbage data
  for (auto i = tracking_miss_times_.begin(); i != tracking_miss_times_.end();)
  {
    // not got hit, increase by 1
    if (i->second)
      i->second++;

    if (i->second > cfg_.max_miss)
    {
      tracking_points_.erase(i->first);
      i = tracking_miss_times_.erase(i);
    }
    else
    {
      i++;
    }

    // assert(tracking_miss_times_.size() == tracking_points_.size());
  }
}

void
YoloTracker::track(YoloResult& res)
{
  if (res.task != YoloTaskType::DET)
  {
    throw std::runtime_error("got unsupported task type when calling YoloTracker::track()!");
    return;
  }
  // support bbox & embedding as input
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<float>> embeddings;
  std::vector<int> track_ids;
  // step1. preprocess, collect boxes & embeddings(Reserved)
  preprocess(res, boxes, embeddings);

  // step2. run track algorithm
  run(boxes, embeddings, track_ids);

  // step3. postprocess, update YoloResult
  postprocess(boxes, embeddings, track_ids, res);
}

auto
YoloTracker::trackCopy(const YoloResult& res) -> YoloResult
{
  auto copy = res;
  track(copy);

  return copy;
}

void
YoloTracker::operator()(YoloResult& res)
{
  track(res);
}

auto
YoloTracker::info(bool print) -> std::string
{
  return "";
}

} // namespace yolo