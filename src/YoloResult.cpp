#include "YoloResult.h"

#include <sstream>

#include "YoloUtils.h"

namespace yolo {

auto
YoloResult::plot(const DrawParam& param) const -> cv::Mat
{
  switch (task)
  {
    case YoloTaskType::CLS:
    {
      return draw_results(orig_image, param, top5(), top5_confs(), top5_labels(),
                          {}, {}, {}, {}, {}, {});
    }
    case YoloTaskType::DET:
    {
      return draw_results(orig_image, param, {}, {}, {},
                          cls_ids(), confs(), labels(), boxes(), track_ids(), track_points());
    }
    default:
      throw std::runtime_error("invalid task type in YoloResult!");
      break;
  }
}

std::string
YoloResult::save(const DrawParam& param)
{
  auto plot_img = plot(param);
  // save
}

int
YoloResult::show(bool block, float scale_f, const DrawParam& param, bool show_orig_img, bool show_input_img)
{
  auto plot_img = plot(param);
  scale_f = std::max(0.0f, std::min(1.0f, scale_f));
  if (std::abs(scale_f - 1.0f) > FLT_EPSILON)
    cv::resize(plot_img, plot_img, cv::Size(), scale_f, scale_f);

  cv::imshow("plot-img-" + std::to_string(id) + "(" + std::to_string(plot_img.cols) + "*" +
               std::to_string(plot_img.rows) + ")",
             plot_img);
  if (show_orig_img)
  {
    cv::imshow("orig-img-" + std::to_string(id) + "(" + std::to_string(orig_image.cols) + "*" +
                 std::to_string(orig_image.rows) + ")",
               orig_image);
  }
  if (show_input_img)
  {
    cv::imshow("input-img-" + std::to_string(id) + "(" + std::to_string(input_image.cols) + "*" +
                 std::to_string(input_image.rows) + ")",
               input_image);
  }

  auto delay = block ? 0 : 1;
  return cv::waitKey(delay);
}

std::string
YoloResult::to_csv(bool print)
{
  std::ostringstream oss;
  switch (task)
  {
    case YoloTaskType::CLS:
    {
      oss << "rank,cls_id,conf,label" << std::endl;
      for (size_t i = 0; i < classes.size(); i++)
      {
        auto& obj = classes[i];
        oss << (i + 1) << "," << obj.cls_id << "," << obj.conf << "," << obj.label;
        if (i + 1 != classes.size())
          oss << std::endl;
      }
      break;
    }
    case YoloTaskType::DET:
    {
      oss << "id,cls_id,conf,label,track_id" << std::endl;
      for (size_t i = 0; i < detections.size(); i++)
      {
        auto& obj = detections[i];
        oss << (i + 1) << "," << obj.cls_id << "," << obj.conf << "," << obj.label << "," << obj.track_id;
        if (i + 1 != detections.size())
          oss << std::endl;
      }
      break;
    }
    default:
      throw std::runtime_error("invalid task type in YoloResult!");
      break;
  }
  auto c_str = oss.str();
  if (print)
    std::cout << c_str << std::endl;

  return c_str;
}

std::string
YoloResult::to_json(bool print, bool indent)
{
  std::string j_str = "";
  auto indent_num = indent ? 4 : -1;
  switch (task)
  {
    case YoloTaskType::CLS:
    {
      json j = classes;
      j_str = j.dump(indent_num);
      break;
    }
    case YoloTaskType::DET:
    {
      json j = detections;
      j_str = j.dump(indent_num);
      break;
    }
    default:
      throw std::runtime_error("invalid task type in YoloResult!");
      break;
  }
  if (print)
    std::cout << j_str << std::endl;

  return j_str;
}

std::string
YoloResult::info(bool print)
{
  std::ostringstream oss;
  oss << "########### YoloResult ###########" << std::endl;
  oss << "id                : " << id << std::endl;
  oss << "task              : " << toString(task) << std::endl;
  oss << "yolo version      : " << toString(version) << std::endl;
  oss << "yolo runtime      : " << toString(target_rt) << std::endl;
  oss << "batch size        : " << batch_size << std::endl;
  oss << "input width       : " << input_w << std::endl;
  oss << "input height      : " << input_h << std::endl;
  oss << "original size     : " << orig_size.width << " * " << orig_size.height << std::endl;
  oss << "letterbox         : " << "scale: " << letterbox_info.scale << ", pad_w: " << letterbox_info.pad_w
      << ", pad_h: " << letterbox_info.pad_h << std::endl;
  oss << "speed             : " << "pre: " << speed[0] << "ms, infer: " << speed[1] << "ms, post: " << speed[2] << "ms"
      << std::endl;
  auto n = std::min(names.size(), static_cast<size_t>(5));
  auto top5_names = std::vector<std::string>(names.begin(), names.begin() + n);
  oss << "names(top5)       : " << top5_names << std::endl;
  if (task == YoloTaskType::CLS)
  {
    // labels&confs of top5: label0(conf0), label1(conf1), ...
    auto t5_labels = top5_labels();
    auto t5_confs = top5_confs();
    auto t5_out = t5_labels[0] + "(" + std::to_string(t5_confs[0]) + ")";
    for (size_t i = 1; i < t5_labels.size(); i++)
      t5_out += ", " + t5_labels[i] + "(" + std::to_string(t5_confs[i]) + ")";
    oss << "top5              : " << t5_out;
  }
  else
  {
    // number of boxes, which stand for the number of predicted objects.
    oss << "objects count     : " << boxes().size();
  }

  auto summary = oss.str();
  if (print)
    std::cout << summary << std::endl;
  return summary;
}

int
YoloResult::top1() const
{
  if (task != YoloTaskType::CLS || classes.empty())
  {
    throw std::runtime_error("could not get top1 from YoloResult, "
                             "it's not a classification task.");
  }
  return classes[0].cls_id;
}

float
YoloResult::top1_conf() const
{
  if (task != YoloTaskType::CLS || classes.empty())
  {
    throw std::runtime_error("could not get top1 conf from YoloResult, "
                             "it's not a classification task.");
  }
  return classes[0].conf;
}

std::string
YoloResult::top1_label() const
{
  if (task != YoloTaskType::CLS || classes.empty())
  {
    throw std::runtime_error("could not get top1 label from YoloResult, "
                             "it's not a classification task.");
  }
  return classes[0].label;
}

std::vector<int>
YoloResult::top5() const
{
  if (task != YoloTaskType::CLS || classes.empty())
  {
    throw std::runtime_error("could not get top5 from YoloResult, "
                             "it's not a classification task.");
  }
  // return the right number if size < 5
  auto n = std::min(classes.size(), static_cast<size_t>(5));
  std::vector<int> cls_ids;
  for (size_t i = 0; i < n; ++i)
    cls_ids.push_back(classes[i].cls_id);
  return cls_ids;
}

std::vector<float>
YoloResult::top5_confs() const
{
  if (task != YoloTaskType::CLS || classes.empty())
  {
    throw std::runtime_error("could not get top5 confs from YoloResult, "
                             "it's not a classification task.");
  }
  // return the right number if size < 5
  auto n = std::min(classes.size(), static_cast<size_t>(5));
  std::vector<float> confs;
  for (size_t i = 0; i < n; ++i)
    confs.push_back(classes[i].conf);
  return confs;
}

std::vector<std::string>
YoloResult::top5_labels() const
{
  if (task != YoloTaskType::CLS || classes.empty())
  {
    throw std::runtime_error("could not get top5 labels from YoloResult, "
                             "it's not a classification task.");
  }
  // return the right number if size < 5
  auto n = std::min(classes.size(), static_cast<size_t>(5));
  std::vector<std::string> labels;
  for (size_t i = 0; i < n; ++i)
    labels.push_back(classes[i].label);
  return labels;
}

std::vector<cv::Rect>
YoloResult::boxes() const
{
  if (task != YoloTaskType::DET)
  {
    throw std::runtime_error("could not get boxes from YoloResult, "
                             "it's not a detection|segmentation|pose task.");
  }
  std::vector<cv::Rect> boxes;

  // priority:
  // detection->segmentation->pose
  if (!detections.empty())
  {
    boxes.reserve(detections.size());
    for (size_t i = 0; i < detections.size(); ++i)
      boxes.emplace_back(detections[i].box);
  }
  return boxes;
}

std::vector<int>
YoloResult::cls_ids() const
{
  if (task != YoloTaskType::DET)
    throw std::runtime_error("could not get cls_ids from YoloResult: not a detection task.");
  std::vector<int> cls_ids;
  cls_ids.reserve(detections.size());
  for (const auto& d : detections)
    cls_ids.emplace_back(d.cls_id);
  return cls_ids;
}

std::vector<float>
YoloResult::confs() const
{
  if (task != YoloTaskType::DET)
    throw std::runtime_error("could not get confs from YoloResult: not a detection task.");
  std::vector<float> confs;
  confs.reserve(detections.size());
  for (const auto& d : detections)
    confs.emplace_back(d.conf);
  return confs;
}

std::vector<std::string>
YoloResult::labels() const
{
  if (task != YoloTaskType::DET)
    throw std::runtime_error("could not get labels from YoloResult: not a detection task.");
  std::vector<std::string> labels;
  labels.reserve(detections.size());
  for (const auto& d : detections)
    labels.emplace_back(d.label);
  return labels;
}

std::vector<int>
YoloResult::track_ids() const
{
  if (task != YoloTaskType::DET)
    throw std::runtime_error("could not get track ids from YoloResult: not a detection task.");
  std::vector<int> track_ids;
  track_ids.reserve(detections.size());
  for (const auto& d : detections)
    track_ids.emplace_back(d.track_id);
  return track_ids;
}

std::vector<std::vector<cv::Point>>
YoloResult::track_points() const
{
  if (task != YoloTaskType::DET)
    throw std::runtime_error("could not get track points from YoloResult: not a detection task.");
  std::vector<std::vector<cv::Point>> track_points;
  track_points.reserve(detections.size());
  for (const auto& d : detections)
    track_points.emplace_back(d.track_points);
  return track_points;
}

} // namespace yolo