#include "rkn/YoloRKNNRT.h"

#include <fstream>

/**
 * tested for:
 * 1. rknn_toolkit==2.3.2 on RK3588
 * 2. rknn_toolkit==2.3.2 on RV1126b
 */
namespace yolo {

YoloRKNNRT::YoloRKNNRT(const std::string& model_path) : YoloRuntime("RKNN")
{
  std::ifstream file(model_path, std::ios::binary);
  file.seekg(0, std::ios::end);
  size_t model_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> model_data(model_size);
  file.read(model_data.data(), model_size);
  file.close();

  // throw errors directly for C style API in rknn
  int ret = rknn_init(&ctx_, model_data.data(), model_size, 0, nullptr);
  if (ret < 0)
    throw std::runtime_error("rknn init failed!");

  if (!queryIO())
    throw std::runtime_error("query IO for rknn model failed!");
}

YoloRKNNRT::~YoloRKNNRT()
{
  if (ctx_)
  {
    rknn_destroy(ctx_);
    ctx_ = 0;
  }
}

auto
YoloRKNNRT::queryIO() -> bool
{
  int ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret < 0)
    return false;

  memset(&input_attr_, 0, sizeof(input_attr_));
  input_attr_.index = 0;

  ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));
  if (ret < 0)
    return false;

  output_attrs_.resize(io_num_.n_output);
  for (int i = 0; i < io_num_.n_output; ++i)
  {
    memset(&output_attrs_[i], 0, sizeof(rknn_tensor_attr));
    output_attrs_[i].index = i;

    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
    if (ret < 0)
      return false;
  }
  return true;
}

auto
YoloRKNNRT::inference(const cv::Mat& blob) -> std::vector<cv::Mat>
{
  // [batch, 3, input_h, input_w] or [batch, input_h, input_w, 3]
  assert(blob.isContinuous());
  assert(blob.type() == CV_32F);
  assert(blob.dims == 4);
  std::vector<cv::Mat> outputs;

  // cv::Mat -> rknn_input
  rknn_input input;
  memset(&input, 0, sizeof(input));

  input.index = 0;
  input.type = RKNN_TENSOR_FLOAT32;
  input.size = blob.total() * blob.elemSize();
  input.fmt = RKNN_TENSOR_NHWC;
  input.buf = (void*)blob.data;

  int ret = rknn_inputs_set(ctx_, 1, &input);
  if (ret < 0)
    return outputs;

  // run
  ret = rknn_run(ctx_, nullptr);
  if (ret < 0)
    return outputs;

  // get rknn_output
  std::vector<rknn_output> rknn_outputs(io_num_.n_output);
  for (int i = 0; i < io_num_.n_output; ++i)
  {
    memset(&rknn_outputs[i], 0, sizeof(rknn_output));
    rknn_outputs[i].index = i;
    rknn_outputs[i].want_float = 1;
  }

  ret = rknn_outputs_get(ctx_, io_num_.n_output, rknn_outputs.data(), nullptr);
  if (ret < 0)
    return outputs;

  // rknn_output -> cv::Mat
  for (int i = 0; i < io_num_.n_output; ++i)
  {
    const rknn_tensor_attr& attr = output_attrs_[i];

    std::vector<int> dims(attr.dims, attr.dims + attr.n_dims);
    auto* out_data = (float*)rknn_outputs[i].buf;

    // clone to own the buffer data
    cv::Mat out_mat = cv::Mat(dims.size(), dims.data(), CV_32F, out_data).clone();
    outputs.push_back(out_mat);
  }

  rknn_outputs_release(ctx_, io_num_.n_output, rknn_outputs.data());
  return outputs;
}

} // namespace yolo