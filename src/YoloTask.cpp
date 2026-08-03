#include "YoloTask.h"

#include <chrono>

#include <stdexcept>
#include <utility>

#ifdef BUILD_WITH_ORT
  #include "ort/YoloONNXRT.h"
#endif
#ifdef BUILD_WITH_OVN
  #include "ovn/YoloOVNRT.h"
#endif
namespace yolo {

YoloTask::YoloTask(YoloConfig cfg) : cfg_(std::move(cfg))
{
  switch (cfg_.target_rt_)
  {
#ifdef BUILD_WITH_ORT
    case YoloTargetRT::ORT_CPU:
      rt_ = std::make_shared<yolo::YoloONNXRT>(cfg_.model_path_, false);
      break;
    case YoloTargetRT::ORT_CUDA:
      rt_ = std::make_shared<yolo::YoloONNXRT>(cfg_.model_path_, true);
      break;
#endif
#ifdef BUILD_WITH_OVN
    case YoloTargetRT::OVN_AUTO:
      rt_ = std::make_shared<yolo::YoloOVNRT>(cfg_.model_path_, "AUTO");
      break;
    case YoloTargetRT::OVN_CPU:
      rt_ = std::make_shared<yolo::YoloOVNRT>(cfg_.model_path_, "CPU");
      break;
    case YoloTargetRT::OVN_GPU:
      rt_ = std::make_shared<yolo::YoloOVNRT>(cfg_.model_path_, "GPU");
      break;
#endif
    default:
      throw std::invalid_argument("invalid(unsupported) YoloTargetRT parameter when initializing YoloTask!");
      ;
  }
}

YoloTask::~YoloTask() = default;

auto
YoloTask::preprocessOne(const cv::Mat& image) -> cv::Mat
{
  assert(!image.empty());

  if (cfg_.task_ == YoloTaskType::CLS)
  {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(cfg_.input_w_, cfg_.input_h_));
    orig_sizes_.push_back(image.size());
    letterbox_infos_.push_back(LetterBoxInfo{1.0f, 0, 0});
    input_images_.push_back(resized);
    return resized;
  }

  LetterBoxInfo info{};
  YoloUtils utils;
  cv::Mat lb = utils.letterbox(image, cfg_.input_w_, cfg_.input_h_, info);
  orig_sizes_.push_back(image.size());
  letterbox_infos_.push_back(info);
  input_images_.push_back(lb);
  return lb;
}

auto
YoloTask::preprocess(const std::vector<cv::Mat>& images) -> cv::Mat
{
  orig_sizes_.clear();
  letterbox_infos_.clear();
  input_images_.clear();

  std::vector<cv::Mat> letterboxes;
  letterboxes.reserve(images.size());
  for (const auto& image : images)
    letterboxes.push_back(preprocessOne(image));

  cv::Mat blob;
  cv::dnn::blobFromImages(letterboxes, blob, cfg_.scale_f_, cv::Size(), cv::Scalar(), cfg_.rgb_, false);

  if (cfg_.task_ == YoloTaskType::CLS && cfg_.mean_.size() == 3 && cfg_.std_.size() == 3)
  {
    int batch_size = blob.size[0];
    int ch_nr = blob.size[1];
    int height = blob.size[2];
    int width = blob.size[3];

    for (int n = 0; n < batch_size; ++n)
    {
      for (int c = 0; c < ch_nr; ++c)
      {
        auto* ptr = blob.ptr<float>(n, c);
        float m = cfg_.mean_[c];
        float s = cfg_.std_[c];

        int spatial = height * width;
        for (int i = 0; i < spatial; ++i)
          ptr[i] = (ptr[i] - m) / s;
      }
    }
  }

  if (!cfg_.nchw_)
  {
    cv::Mat blob_nhwc;
    std::vector<int> order = {0, 2, 3, 1};
    cv::transposeND(blob, order, blob_nhwc);
    return blob_nhwc;
  }

  return blob;
}

auto
YoloTask::inference(const cv::Mat& blob) -> std::vector<cv::Mat>
{
  return (*rt_).inference(blob);
}

auto
YoloTask::postprocess(const std::vector<cv::Mat>& raw_outputs, int batch_size) -> std::vector<yolo::YoloResult>
{
  std::vector<yolo::YoloResult> results(batch_size);
  for (int i = 0; i < batch_size; ++i)
  {
    /* MUST override in child class(extract structured data to fill YoloResult) */
    postprocessOne(raw_outputs, i, orig_sizes_[i], letterbox_infos_[i], results[i]);
  }
  return results;
}

auto
YoloTask::run(const std::vector<cv::Mat>& images) -> std::vector<yolo::YoloResult>
{
  /* step1. preprocess */
  auto t1 = std::chrono::system_clock::now();
  auto batch_blob = preprocess(images);
  /* step2. inference */
  auto t2 = std::chrono::system_clock::now();
  auto raw_outputs = inference(batch_blob);
  /* step3. postprocess */
  auto t3 = std::chrono::system_clock::now();
  auto results = postprocess(raw_outputs, images.size());
  auto t4 = std::chrono::system_clock::now();

  assert(images.size() == results.size());
  /* update properties for YoloResult */
  for (int i = 0; i < results.size(); ++i)
  {
    auto& r = results[i];

    /* note, it's batch cost time here since we do not know the time for single inference. */
    r.speed_.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
                       1000.0); // preprocess time(ms)
    r.speed_.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() /
                       1000.0); // inference time(ms)
    r.speed_.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() /
                       1000.0); // postprocess time(ms)

    r.id_ = i; // batch id
    r.names_ = cfg_.names_;
    r.batch_size_ = cfg_.batch_size_;
    r.task_ = cfg_.task_;
    r.version_ = cfg_.version_;
    r.target_rt_ = cfg_.target_rt_;
    r.input_w_ = cfg_.input_w_;
    r.input_h_ = cfg_.input_h_;
    r.letterbox_info_ = letterbox_infos_[i];
    r.input_image_ = input_images_[i];
    r.orig_size_ = orig_sizes_[i];
    r.orig_image_ = images[i];
  }

  /* return results */
  return results;
}

auto
YoloTask::operator()(const std::vector<cv::Mat>& images) -> std::vector<yolo::YoloResult>
{
  return run(images);
}

} // namespace yolo