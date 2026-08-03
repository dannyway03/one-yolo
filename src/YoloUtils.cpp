#include "YoloUtils.h"

namespace yolo {

auto
toString(const float f, const int precision) -> std::string
{
  std::ostringstream out;
  out.precision(precision);
  out << std::fixed << f;
  return out.str();
}

auto
drawResults(const cv::Mat& image, const DrawParam& param, const std::vector<int>& top5,
            const std::vector<float>& top5_confs, const std::vector<std::string>& top5_labels,
            const std::vector<int>& cls_ids, const std::vector<float>& confs, const std::vector<std::string>& labels,
            const std::vector<cv::Rect>& boxes, const std::vector<int>& track_ids,
            const std::vector<std::vector<cv::Point>>& tracks) -> cv::Mat
{
  auto canvas = image.clone();
  // resize or not

  auto colors = getColors48();
  auto colors_num = colors.size();
  auto font_face = param.font_face_;
  auto font_scale = param.font_scale_;
  auto font_color = param.font_color_;
  auto font_thickness = param.font_thickness_;
  auto font_offset_x = param.box_line_width_ / 2;
  auto font_padding = 8;

  // classification task
  if (!top5.empty())
  {
    assert(top5.size() == top5_confs.size());
    assert(top5.size() == top5_labels.size());

    auto num = param.top1_only_ ? 1 : top5.size();
    auto loc = cv::Point(font_padding, font_padding); // start point
    for (size_t i = 0; i < num; i++)
    {
      const auto& color = colors[top5[i] % colors_num];
      // [rank]: cls_id_, conf, label
      std::string txt = "[rank" + std::to_string(i + 1) +
                        "]: " + (param.cls_ids_ ? std::to_string(top5[i]) + ", " : "") + toString(top5_confs[i] * 100) +
                        "%, " + top5_labels[i];
      // calculate rectangle of txt
      auto baseline = 0;
      auto txt_size = cv::getTextSize(txt, font_face, font_scale, font_thickness, &baseline);
      cv::rectangle(
        canvas,
        cv::Rect(loc.x, loc.y, txt_size.width + font_padding * 2, txt_size.height + baseline + font_padding * 2), color,
        -1); // fill
      cv::putText(canvas, txt, cv::Point(loc.x + font_padding, loc.y + txt_size.height + font_padding), font_face,
                  font_scale, font_color, font_thickness);
      loc = cv::Point(loc.x,
                      loc.y + txt_size.height + baseline + font_padding * 2 +
                        font_padding); // add padding space in vertical
    }
  }

  // detection/segmentation/pose tasks
  if (!tracks.empty() && param.tracks_ && param.track_line_width_ > 0)
  {
    assert(boxes.size() == cls_ids.size());
    assert(boxes.size() == confs.size());
    assert(boxes.size() == labels.size());
    assert(boxes.size() == tracks.size());

    for (size_t i = 0; i < tracks.size(); i++)
    {
      auto& one_track = tracks[i];
      auto color = param.color_by_class_ ? colors[cls_ids[i] % colors_num] : colors[i % colors_num];

      for (size_t j = 0; j + 1 < one_track.size(); j++)
      {
        auto& p1 = one_track[j];
        auto& p2 = one_track[j + 1];
        cv::line(canvas, p1, p2, color, param.track_line_width_, cv::LINE_AA);

        // the last one is located point
        if (j + 2 == one_track.size() && param.loc_radius_ > 0)
          cv::circle(canvas, p2, param.loc_radius_, color, -1);
      }
    }
  }

  // detection task
  if (!boxes.empty() && param.boxes_ && param.box_line_width_ > 0)
  {
    assert(boxes.size() == cls_ids.size());
    assert(boxes.size() == confs.size());
    assert(boxes.size() == labels.size());
    assert(boxes.size() == track_ids.size());

    for (size_t i = 0; i < boxes.size(); i++)
    {
      auto color = param.color_by_class_ ? colors[cls_ids[i] % colors_num] : colors[i % colors_num];
      cv::rectangle(canvas, boxes[i], color, param.box_line_width_);

      std::string txt;
      if (param.track_ids_ && track_ids[i] >= 0)
      {
        // track_id == -1 means no tracked, ignored
        txt += "#" + std::to_string(track_ids[i]);
      }
      if (param.cls_ids_)
        txt += (!txt.empty() ? ", " : "") + std::to_string(cls_ids[i]);
      if (param.labels_)
        txt += (!txt.empty() ? ", " : "") + labels[i];
      if (param.confs_)
        txt += (!txt.empty() ? ", " : "") + toString(confs[i] * 100) + "%";

      // draw text
      if (!txt.empty())
      {
        // calculate rectangle of txt
        auto baseline = 0;
        auto txt_size = cv::getTextSize(txt, font_face, font_scale, font_thickness, &baseline);
        cv::rectangle(canvas,
                      cv::Rect(boxes[i].x - font_offset_x, boxes[i].y - txt_size.height - baseline - font_padding * 2,
                               txt_size.width + font_padding * 2, txt_size.height + baseline + font_padding * 2),
                      color, -1); // fill
        cv::putText(canvas, txt,
                    cv::Point(boxes[i].x - font_offset_x + font_padding, boxes[i].y - baseline - font_padding),
                    font_face, font_scale, font_color, font_thickness);
      }
    }
  }

  return canvas;
}

auto
getColors48() -> std::vector<cv::Scalar>
{
  return {
    cv::Scalar(0, 100, 0),    // DarkGreen
    cv::Scalar(0, 0, 139),    // DarkRed
    cv::Scalar(139, 0, 0),    // DarkBlue
    cv::Scalar(139, 0, 139),  // DarkMagenta
    cv::Scalar(0, 139, 139),  // Olive (dark)
    cv::Scalar(139, 139, 0),  // Teal (dark)
    cv::Scalar(130, 0, 75),   // Indigo
    cv::Scalar(128, 0, 128),  // Purple
    cv::Scalar(240, 32, 160), // DarkViolet
    cv::Scalar(211, 0, 148),  // DarkOrchid
    cv::Scalar(130, 0, 75),   // Indigo
    cv::Scalar(19, 69, 139),  // SaddleBrown
    cv::Scalar(45, 82, 160),  // Sienna
    cv::Scalar(33, 67, 101),  // DarkBrown
    cv::Scalar(47, 107, 85),  // DarkOliveGreen
    cv::Scalar(87, 139, 46),  // SeaGreen
    cv::Scalar(128, 128, 0),  // Teal
    cv::Scalar(150, 75, 0),   // Deep Blue
    cv::Scalar(75, 0, 150),   // Deep Magenta
    cv::Scalar(0, 150, 75),   // Deep Green
    cv::Scalar(0, 50, 100),   // Dark Orange-Brown
    cv::Scalar(100, 50, 0),   // Navy Green
    cv::Scalar(100, 0, 50),   // Deep Purple-Blue
    cv::Scalar(50, 0, 100),   // Maroon-ish
    cv::Scalar(0, 100, 50),   // Forest Green
    cv::Scalar(50, 100, 0),   // Green-Teal
    cv::Scalar(100, 50, 100), // Plum-like
    cv::Scalar(100, 100, 50), // Dark Cyan
    cv::Scalar(80, 40, 120),  // Wine
    cv::Scalar(120, 40, 80),  // Eggplant
    cv::Scalar(120, 80, 40),  // Steel Blue
    cv::Scalar(40, 80, 120),  // Mustard Brown
    cv::Scalar(60, 30, 110),  // Burgundy
    cv::Scalar(110, 30, 60),  // Royal Purple
    cv::Scalar(110, 60, 30),  // Ocean Blue
    cv::Scalar(30, 60, 110),  // Rust
    cv::Scalar(30, 110, 60),  // Lime Forest
    cv::Scalar(60, 110, 30),  // Jungle Green
    cv::Scalar(70, 20, 90),   // Deep Rose
    cv::Scalar(90, 20, 70),   // Velvet
    cv::Scalar(90, 70, 20),   // Deep Sky
    cv::Scalar(20, 70, 90),   // Bronze
    cv::Scalar(50, 25, 100),  // Crimson Dark
    cv::Scalar(100, 25, 50),  // Midnight Purple
    cv::Scalar(80, 120, 40),  // Moss Green
    cv::Scalar(40, 120, 80),  // Olive Drab
    cv::Scalar(20, 90, 70),   // Avocado
    cv::Scalar(70, 90, 20)    // Peacock
  };
}

YoloUtils::YoloUtils(/* args */) = default;

auto
YoloUtils::letterbox(const cv::Mat& img, int new_w, int new_h, LetterBoxInfo& info, const cv::Scalar& color) -> cv::Mat
{
  int w = img.cols;
  int h = img.rows;

  float r = std::min(static_cast<float>(new_w) / w, static_cast<float>(new_h) / h);

  int resized_w = std::round(w * r);
  int resized_h = std::round(h * r);

  cv::Mat resized;
  cv::resize(img, resized, cv::Size(resized_w, resized_h));

  int pad_w = new_w - resized_w;
  int pad_h = new_h - resized_h;

  int pad_left = pad_w / 2;
  int pad_right = pad_w - pad_left;
  int pad_top = pad_h / 2;
  int pad_bottom = pad_h - pad_top;

  cv::Mat padded;
  cv::copyMakeBorder(resized, padded, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, color);

  info.scale_ = r;
  info.pad_w_ = pad_left;
  info.pad_h_ = pad_top;

  return padded;
}

void
YoloUtils::classAwareNms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores,
                         const std::vector<int>& cls_ids, float conf_thresh, float nms_thresh,
                         std::vector<int>& keep_indices)
{
  keep_indices.clear();

  // class_id -> indices
  std::unordered_map<int, std::vector<int>> cls_map;
  for (int i = 0; i < static_cast<int>(cls_ids.size()); ++i)
    cls_map[cls_ids[i]].push_back(i);

  // per-class NMS
  for (const auto& kv : cls_map)
  {
    const auto& indices = kv.second;

    std::vector<cv::Rect> cls_boxes;
    std::vector<float> cls_scores;

    cls_boxes.reserve(indices.size());
    cls_scores.reserve(indices.size());

    for (int idx : indices)
    {
      cls_boxes.emplace_back(boxes[idx]);
      cls_scores.emplace_back(scores[idx]);
    }

    std::vector<int> cls_keep;
    cv::dnn::NMSBoxes(cls_boxes, cls_scores, conf_thresh, nms_thresh, cls_keep);

    // 映射回原索引
    for (int k : cls_keep)
      keep_indices.push_back(indices[k]);
  }
}

void
YoloUtils::classAwareNms(const std::vector<cv::RotatedRect>& rboxes, const std::vector<float>& scores,
                         const std::vector<int>& cls_ids, float conf_thresh, float nms_thresh,
                         std::vector<int>& keep_indices)
{
  keep_indices.clear();

  // class_id -> indices
  std::unordered_map<int, std::vector<int>> cls_map;
  for (int i = 0; i < static_cast<int>(cls_ids.size()); ++i)
    cls_map[cls_ids[i]].push_back(i);

  // per-class NMS
  for (const auto& kv : cls_map)
  {
    const auto& indices = kv.second;

    std::vector<cv::RotatedRect> cls_rboxes;
    std::vector<float> cls_scores;

    cls_rboxes.reserve(indices.size());
    cls_scores.reserve(indices.size());

    for (int idx : indices)
    {
      cls_rboxes.emplace_back(rboxes[idx]);
      cls_scores.emplace_back(scores[idx]);
    }

    std::vector<int> cls_keep;
    cv::dnn::NMSBoxes(cls_rboxes, cls_scores, conf_thresh, nms_thresh, cls_keep);

    // 映射回原索引
    for (int k : cls_keep)
      keep_indices.push_back(indices[k]);
  }
}

auto
YoloUtils::decodeBox(float cx, float cy, float w, float h, const LetterBoxInfo& lb, const cv::Size& orig_size)
  -> cv::Rect
{
  float x1 = cx - w * 0.5f;
  float y1 = cy - h * 0.5f;
  float x2 = cx + w * 0.5f;
  float y2 = cy + h * 0.5f;

  x1 = (x1 - lb.pad_w_) / lb.scale_;
  y1 = (y1 - lb.pad_h_) / lb.scale_;
  x2 = (x2 - lb.pad_w_) / lb.scale_;
  y2 = (y2 - lb.pad_h_) / lb.scale_;

  x1 = std::clamp(x1, 0.f, static_cast<float>(orig_size.width) - 1.f);
  y1 = std::clamp(y1, 0.f, static_cast<float>(orig_size.height) - 1.f);
  x2 = std::clamp(x2, 0.f, static_cast<float>(orig_size.width) - 1.f);
  y2 = std::clamp(y2, 0.f, static_cast<float>(orig_size.height) - 1.f);

  return {static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)};
}

} // namespace yolo