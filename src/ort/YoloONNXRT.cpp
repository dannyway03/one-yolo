#include "ort/YoloONNXRT.h"

/**
 * tested for:
 * 1. ONNXRuntime==1.17.3
 */
namespace yolo {

YoloONNXRT::YoloONNXRT(const std::string& model_path, bool use_cuda) : YoloRuntime("ONNXRuntime")
{
  // options
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  if (use_cuda)
  {
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;

    session_options_.AppendExecutionProvider_CUDA(cuda_options);
  }

  session_ = Ort::Session(env_, model_path.c_str(), session_options_);

  memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  input_names_str_ = session_.GetInputNames();
  output_names_str_ = session_.GetOutputNames();
  input_names_.reserve(input_names_str_.size());
  for (const auto& s : input_names_str_)
    input_names_.push_back(s.c_str());
  output_names_.reserve(output_names_str_.size());
  for (const auto& s : output_names_str_)
    output_names_.push_back(s.c_str());
}

YoloONNXRT::~YoloONNXRT() = default;

auto
YoloONNXRT::inference(const cv::Mat& blob) -> std::vector<cv::Mat>
{
  std::vector<cv::Mat> outputs;

  // refer to cv::dnn::Net::forward
  ortForward(blob, outputs);
  return outputs;
}

void
YoloONNXRT::ortForward(const cv::Mat& input_4d, std::vector<cv::Mat>& outputs)
{
  // [batch, 3, input_h, input_w] or [batch, input_h, input_w, 3]
  assert(input_4d.isContinuous());
  assert(input_4d.type() == CV_32F);
  assert(input_4d.dims == 4);

  // get input shape
  std::vector<int64_t> input_shape(4);
  for (int i = 0; i < 4; ++i)
    input_shape[i] = input_4d.size[i];
  auto input_tensor_size = input_4d.total();

  // cv::Mat → ORT Tensor (zero copy)
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info_, (float*)input_4d.data, input_tensor_size,
                                                            input_shape.data(), input_shape.size());

  auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor, input_names_.size(),
                                     output_names_.data(), output_names_.size());

  // ORT Tensor → cv::Mat (should copy buffer data)
  outputs.clear();
  outputs.reserve(output_names_.size());
  for (size_t i = 0; i < output_names_.size(); ++i)
  {
    auto& out_tensor = output_tensors[i];

    auto shape_info = out_tensor.GetTensorTypeAndShapeInfo();
    auto shape = shape_info.GetShape();
    int dims = shape.size();

    std::vector<int> cv_sizes(dims);
    for (int d = 0; d < dims; ++d)
      cv_sizes[d] = static_cast<int>(shape[d]);

    auto* data_ptr = out_tensor.GetTensorMutableData<float>();
    cv::Mat out_mat(dims, cv_sizes.data(), CV_32F, data_ptr);

    // clone to own the buffer data!
    outputs.emplace_back(out_mat.clone());
  }
}

} // namespace yolo