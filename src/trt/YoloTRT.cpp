#include "trt/YoloTRT.h"

#include <cstddef>

#include <fstream>

/**
 * tested for:
 * 1. TensorRT==8.5 on RTX2060
 */
namespace yolo {

class TRTLogger : public nvinfer1::ILogger
{
  void
  log(Severity severity, const char* msg) noexcept override
  {
    if (severity <= Severity::kWARNING)
      std::cout << "[TensorRT] " << msg << std::endl;
  }
} g_logger;

YoloTRT::YoloTRT(const std::string& model_path) : YoloRuntime("TensorRT")
{
  std::ifstream file(model_path, std::ios::binary);

  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);
  file.close();

  // create runtime / engine / context
  runtime_ = nvinfer1::createInferRuntime(gLogger);
  engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
  context_ = engine_->createExecutionContext();

  allocate_buffers();
}

YoloTRT::~YoloTRT()
{
  for (void* buf : device_buffers_)
    cudaFree(buf);

  if (context_)
    context_->destroy();
  if (engine_)
    engine_->destroy();
  if (runtime_)
    runtime_->destroy();
}

void
YoloTRT::allocate_buffers()
{
  int nb_bindings = engine_->getNbBindings();
  device_buffers_.resize(nbBindings);

  for (int i = 0; i < nb_bindings; ++i)
  {
    auto dims = engine_->getBindingDimensions(i);
    size_t vol = 1;

    for (int d = 0; d < dims.nbDims; ++d)
      vol *= dims.d[d];

    size_t bytes = vol * sizeof(float);

    cudaMalloc(&device_buffers_[i], bytes);

    if (!engine_->bindingIsInput(i))
    {
      std::vector<int64_t> shape;
      for (int d = 0; d < dims.nbDims; ++d)
        shape.push_back(dims.d[d]);

      output_shapes_.push_back(shape);
    }
  }
}

auto
YoloTRT::inference(const cv::Mat& blob) -> std::vector<cv::Mat>
{
  // [batch, 3, input_h, input_w] or [batch, input_h, input_w, 3]
  assert(blob.isContinuous());
  assert(blob.type() == CV_32F);
  assert(blob.dims == 4);

  int d0 = blob.size[0];
  int d1 = blob.size[1];
  int d2 = blob.size[2];
  int d3 = blob.size[3];

  size_t input_bytes = static_cast<size_t>(d0) * d1 * d2 * d3 * sizeof(float);
  cudaMemcpy(device_buffers_[0], blob.data, input_bytes, cudaMemcpyHostToDevice);

  context_->enqueueV2(device_buffers_.data(),
                      0, // stream
                      nullptr);

  std::vector<cv::Mat> outputs;
  int output_index = 0;
  for (int i = 0; i < engine_->getNbBindings(); ++i)
  {
    if (engine_->bindingIsInput(i))
      continue;

    auto dims = engine_->getBindingDimensions(i);

    size_t vol = 1;
    for (int d = 0; d < dims.nbDims; ++d)
      vol *= dims.d[d];

    std::vector<float> host_buffer(vol);

    cudaMemcpy(host_buffer.data(), device_buffers_[i], vol * sizeof(float), cudaMemcpyDeviceToHost);

    // TensorRT dims → cv::Mat
    std::vector<int> mat_sizes;
    for (int d = 0; d < dims.nbDims; ++d)
      mat_sizes.push_back(dims.d[d]);

    cv::Mat out_mat(dims.nbDims, mat_sizes.data(), CV_32F, host_buffer.data());
    // clone to own the buffer data
    outputs.push_back(out_mat.clone());
    output_index++;
  }

  return outputs;
}

} // namespace yolo