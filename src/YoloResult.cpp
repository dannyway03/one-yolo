#include "YoloResult.h"

#include <sstream>

#include "YoloUtils.h"

namespace yolo {

auto
YoloResult::plot(const DrawParam& param) const -> cv::Mat
{
  switch (task_)
  {
    case YoloTaskType::CLS:
    {
      return drawResults(orig_image_, param, top5(), top5Confs(), top5Labels(), {}, {}, {}, {}, {}, {});
    }
    case YoloTaskType::DET:
    {
      return drawResults(orig_image_, param, {}, {}, {}, clsIds(), confs(), labels(), boxes(), trackIds(),
                         trackPoints());
    }
    default:
      throw std::runtime_error("invalid task type in YoloResult!");
  }
}

auto
YoloResult::show(bool block, float scale_f, const DrawParam& param, bool show_orig_img, bool show_input_img) const
  -> int
{
  auto plot_img = plot(param);
  scale_f = std::max(0.0f, std::min(1.0f, scale_f));
  if (std::abs(scale_f - 1.0f) > FLT_EPSILON)
    cv::resize(plot_img, plot_img, cv::Size(), scale_f, scale_f);

  cv::imshow("plot-img-" + std::to_string(id_) + "(" + std::to_string(plot_img.cols) + "*" +
               std::to_string(plot_img.rows) + ")",
             plot_img);
  if (show_orig_img)
  {
    cv::imshow("orig-img-" + std::to_string(id_) + "(" + std::to_string(orig_image_.cols) + "*" +
                 std::to_string(orig_image_.rows) + ")",
               orig_image_);
  }
  if (show_input_img)
  {
    cv::imshow("input-img-" + std::to_string(id_) + "(" + std::to_string(input_image_.cols) + "*" +
                 std::to_string(input_image_.rows) + ")",
               input_image_);
  }

  auto delay = block ? 0 : 1;
  return cv::waitKey(delay);
}

auto
YoloResult::toCsv(bool print) -> std::string
{
  std::ostringstream oss;
  switch (task_)
  {
    case YoloTaskType::CLS:
    {
      oss << "rank,cls_id_,conf,label\n";
      for (size_t i = 0; i < classes_.size(); i++)
      {
        auto& obj = classes_[i];
        oss << (i + 1) << "," << obj.cls_id_ << "," << obj.conf_ << "," << obj.label_;
        if (i + 1 != classes_.size())
          oss << "\n";
      }
      break;
    }
    case YoloTaskType::DET:
    {
      oss << "id,cls_id_,conf,label,track_id\n";
      for (size_t i = 0; i < detections_.size(); i++)
      {
        auto& obj = detections_[i];
        oss << (i + 1) << "," << obj.cls_id_ << "," << obj.conf_ << "," << obj.label_ << "," << obj.track_id_;
        if (i + 1 != detections_.size())
          oss << "\n";
      }
      break;
    }
    default:
      throw std::runtime_error("invalid task type in YoloResult!");
  }
  auto c_str = oss.str();
  if (print)
    std::cout << c_str << "\n";

  return c_str;
}

std::string
YoloResult::toJson(bool print, bool indent) // NOLINT
{
  std::string j_str;
  auto indent_num = indent ? 4 : -1;
  switch (task_)
  {
    case YoloTaskType::CLS:
    {
      json j = classes_;
      j_str = j.dump(indent_num);
      break;
    }
    case YoloTaskType::DET:
    {
      json j = detections_;
      j_str = j.dump(indent_num);
      break;
    }
    default:
      throw std::runtime_error("invalid task type in YoloResult!");
  }
  if (print)
    std::cout << j_str << "\n";

  return j_str;
}

auto
YoloResult::info(bool print) -> std::string
{
  std::ostringstream oss;
  oss << "########### YoloResult ###########\n";
  oss << "id                : " << id_ << "\n";
  oss << "task              : " << toString(task_) << "\n";
  oss << "yolo version      : " << toString(version_) << "\n";
  oss << "yolo runtime      : " << toString(target_rt_) << "\n";
  oss << "batch size        : " << batch_size_ << "\n";
  oss << "input width       : " << input_w_ << "\n";
  oss << "input height      : " << input_h_ << "\n";
  oss << "original size     : " << orig_size_.width << " * " << orig_size_.height << "\n";
  oss << "letterbox         : " << "scale: " << letterbox_info_.scale_ << ", pad_w: " << letterbox_info_.pad_w_
      << ", pad_h: " << letterbox_info_.pad_h_ << "\n";
  oss << "speed             : " << "pre: " << speed_[0] << "ms, infer: " << speed_[1] << "ms, post: " << speed_[2]
      << "ms"
      << "\n";
  auto n = std::min(names_.size(), static_cast<size_t>(5));
  auto top5_names = std::vector<std::string>(names_.begin(), names_.begin() + n);
  oss << "names(top5)       : " << top5_names << "\n";
  if (task_ == YoloTaskType::CLS)
  {
    auto t5_labels = top5Labels();
    auto t5_confs = top5Confs();
    auto t5_out = t5_labels[0] + "(" + std::to_string(t5_confs[0]) + ")";
    for (size_t i = 1; i < t5_labels.size(); i++)
      t5_out += ", " + t5_labels[i] + "(" + std::to_string(t5_confs[i]) + ")";
    oss << "top5              : " << t5_out;
  }
  else
  {
    oss << "objects count     : " << boxes().size();
  }

  auto summary = oss.str();
  if (print)
    std::cout << summary << "\n";
  return summary;
}

auto
YoloResult::top1() const -> int
{
  if (task_ != YoloTaskType::CLS || classes_.empty())
  {
    throw std::runtime_error("could not get top1 from YoloResult, "
                             "it's not a classification task.");
  }
  return classes_[0].cls_id_;
}

auto
YoloResult::top1Conf() const -> float
{
  if (task_ != YoloTaskType::CLS || classes_.empty())
  {
    throw std::runtime_error("could not get top1 conf from YoloResult, "
                             "it's not a classification task.");
  }
  return classes_[0].conf_;
}

auto
YoloResult::top1Label() const -> std::string
{
  if (task_ != YoloTaskType::CLS || classes_.empty())
  {
    throw std::runtime_error("could not get top1 label from YoloResult, "
                             "it's not a classification task.");
  }
  return classes_[0].label_;
}

auto
YoloResult::top5() const -> std::vector<int>
{
  if (task_ != YoloTaskType::CLS || classes_.empty())
  {
    throw std::runtime_error("could not get top5 from YoloResult, "
                             "it's not a classification task.");
  }
  auto n = std::min(classes_.size(), static_cast<size_t>(5));
  std::vector<int> cls_ids;
  cls_ids.reserve(n);
  for (size_t i = 0; i < n; ++i)
    cls_ids.push_back(classes_[i].cls_id_);
  return cls_ids;
}

auto
YoloResult::top5Confs() const -> std::vector<float>
{
  if (task_ != YoloTaskType::CLS || classes_.empty())
  {
    throw std::runtime_error("could not get top5 confs from YoloResult, "
                             "it's not a classification task.");
  }
  auto n = std::min(classes_.size(), static_cast<size_t>(5));
  std::vector<float> confs;
  confs.reserve(n);
  for (size_t i = 0; i < n; ++i)
    confs.push_back(classes_[i].conf_);
  return confs;
}

auto
YoloResult::top5Labels() const -> std::vector<std::string>
{
  if (task_ != YoloTaskType::CLS || classes_.empty())
  {
    throw std::runtime_error("could not get top5 labels from YoloResult, "
                             "it's not a classification task.");
  }
  auto n = std::min(classes_.size(), static_cast<size_t>(5));
  std::vector<std::string> labels;
  labels.reserve(n);
  for (size_t i = 0; i < n; ++i)
    labels.push_back(classes_[i].label_);
  return labels;
}

auto
YoloResult::boxes() const -> std::vector<cv::Rect>
{
  if (task_ != YoloTaskType::DET)
  {
    throw std::runtime_error("could not get boxes from YoloResult, "
                             "it's not a detection|segmentation|pose task.");
  }
  std::vector<cv::Rect> boxes;

  if (!detections_.empty())
  {
    boxes.reserve(detections_.size());
    for (const auto& detection : detections_)
      boxes.emplace_back(detection.box_);
  }
  return boxes;
}

auto
YoloResult::clsIds() const -> std::vector<int>
{
  if (task_ != YoloTaskType::DET)
    throw std::runtime_error("could not get cls_ids from YoloResult: not a detection task.");
  std::vector<int> cls_ids;
  cls_ids.reserve(detections_.size());
  for (const auto& d : detections_)
    cls_ids.emplace_back(d.cls_id_);
  return cls_ids;
}

auto
YoloResult::confs() const -> std::vector<float>
{
  if (task_ != YoloTaskType::DET)
    throw std::runtime_error("could not get confs from YoloResult: not a detection task.");
  std::vector<float> confs;
  confs.reserve(detections_.size());
  for (const auto& d : detections_)
    confs.emplace_back(d.conf_);
  return confs;
}

auto
YoloResult::labels() const -> std::vector<std::string>
{
  if (task_ != YoloTaskType::DET)
    throw std::runtime_error("could not get labels from YoloResult: not a detection task.");
  std::vector<std::string> labels;
  labels.reserve(detections_.size());
  for (const auto& d : detections_)
    labels.emplace_back(d.label_);
  return labels;
}

auto
YoloResult::trackIds() const -> std::vector<int>
{
  if (task_ != YoloTaskType::DET)
    throw std::runtime_error("could not get track ids from YoloResult: not a detection task.");
  std::vector<int> track_ids;
  track_ids.reserve(detections_.size());
  for (const auto& d : detections_)
    track_ids.emplace_back(d.track_id_);
  return track_ids;
}

auto
YoloResult::trackPoints() const -> std::vector<std::vector<cv::Point>>
{
  if (task_ != YoloTaskType::DET)
    throw std::runtime_error("could not get track points from YoloResult: not a detection task.");
  std::vector<std::vector<cv::Point>> track_points;
  track_points.reserve(detections_.size());
  for (const auto& d : detections_)
    track_points.emplace_back(d.track_points_);
  return track_points;
}

} // namespace yolo