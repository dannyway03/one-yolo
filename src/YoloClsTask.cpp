#include "YoloClsTask.h"

#include <numeric>
#include <stdexcept>

namespace yolo {

YoloClsTask::YoloClsTask(const YoloConfig& cfg) : YoloTask(cfg) {}

YoloClsTask::~YoloClsTask() = default;

auto
YoloClsTask::softmax(const cv::Mat& logits) -> cv::Mat
{
  assert(logits.dims == 2);

  cv::Mat probs = cv::Mat::zeros(logits.size(), logits.type());

  for (int i = 0; i < logits.rows; ++i)
  {
    cv::Mat row = logits.row(i);

    double max_val;
    cv::minMaxLoc(row, nullptr, &max_val);

    cv::Mat exp_row;
    cv::exp(row - max_val, exp_row);

    double sum_exp = cv::sum(exp_row)[0];
    exp_row /= sum_exp;

    exp_row.copyTo(probs.row(i));
  }
  return probs;
}

auto
YoloClsTask::isProbDistribution(const cv::Mat& out, double eps) -> bool
{
  assert(out.dims == 2);

  for (int i = 0; i < out.rows; ++i)
  {
    cv::Mat row = out.row(i);

    double min_val, max_val;
    cv::minMaxLoc(row, &min_val, &max_val);
    double sum = cv::sum(row)[0];

    if (min_val < 0.0)
      return false;
    if (max_val > 1.0)
      return false;
    if (std::abs(sum - 1.0) > eps)
      return false;
  }
  return true;
}

void
YoloClsTask::postprocessOne(const std::vector<cv::Mat>& raw_outputs, int batch_id, cv::Size orig_size,
                            LetterBoxInfo lb_info, YoloResult& result)
{
  std::vector<float> scores;
  std::vector<int> cls_ids;
  if (raw_outputs.empty())
    return;

  /* just 1 output head for classification task in Yolo */
  auto& raw_output = raw_outputs[0];
  auto output = isProbDistribution(raw_output) ? raw_output : softmax(raw_output);

  int num_classes = 0;
  const float* scores_ptr = nullptr;

  if (output.dims == 2)
  {
    // [batch, num_classes]
    assert(batch_id < output.size[0]);
    num_classes = output.size[1];
    scores_ptr = output.ptr<float>(batch_id);
  }
  else if (output.dims == 1)
  {
    // [num_classes]
    num_classes = output.size[0];
    scores_ptr = output.ptr<float>();
  }
  else
  {
    throw std::runtime_error("unsupported output dims for classification task in Yolo!");
    return;
  }

  assert(num_classes == cfg_.num_classes_);
  std::vector<int> indices(num_classes);
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) -> bool
            {
              return scores_ptr[a] > scores_ptr[b];
            });

  scores.reserve(num_classes);
  cls_ids.reserve(num_classes);

  for (int idx : indices)
  {
    cls_ids.emplace_back(idx);
    scores.emplace_back(scores_ptr[idx]);
  }

  /* fill YoloResult with YoloClsObjs from high score to low */
  for (size_t i = 0; i < num_classes; i++)
  {
    YoloClsObj cls_obj{cls_ids[i], scores[i], cfg_.names_[cls_ids[i]]};
    result.classes_.emplace_back(cls_obj);
  }
}

} // namespace yolo