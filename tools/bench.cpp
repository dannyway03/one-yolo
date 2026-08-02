/**
 * bench.cpp — end-to-end latency benchmark for one-yolo.
 *
 * Measures the full pipeline (preprocess + inference + postprocess) via the
 * YoloConfig / Yolo class, so numbers reflect real application cost.
 *
 * Usage:
 *   bench --model <path> --version <yolox|yolo26|yolo11|yolo8|yolo5|yolo5u>
 *         --backend <ort|ovn|dnn> [--device <cpu|gpu|auto|cuda>]
 *         [--task <det|cls|seg|pose|obb>]
 *         [--input-w 640] [--input-h 640] [--classes 80]
 *         [--warmup 10] [--iterations 100]
 *         [--image <path>]   # real image; random noise used when omitted
 *         [--save <path>]    # save annotated result (detection only)
 *         [--csv  <path>]    # append one summary row to CSV file
 */

#include <chrono>
#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Yolo.h"

using namespace yolo;

// ── helpers ──────────────────────────────────────────────────────────────────

struct Stats
{
  double avg_;
  double med_;
  double p95_;
  double p99_;
  double min_;
  double max_;
  double stddev_;
};

static auto
percentile(std::vector<double> v, double p) -> double
{
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  double idx = p / 100.0 * static_cast<double>(v.size() - 1);
  auto lo = static_cast<size_t>(idx);
  double frac = idx - static_cast<double>(lo);
  if (lo + 1 >= v.size())
    return v.back();
  return v[lo] * (1.0 - frac) + v[lo + 1] * frac;
}

static auto
computeStats(const std::vector<double>& v) -> Stats
{
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double avg = sum / static_cast<double>(v.size());
  double sq = 0.0;
  for (double x : v)
    sq += (x - avg) * (x - avg);
  return {avg,
          percentile(v, 50.0),
          percentile(v, 95.0),
          percentile(v, 99.0),
          *std::min_element(v.begin(), v.end()),
          *std::max_element(v.begin(), v.end()),
          std::sqrt(sq / static_cast<double>(v.size()))};
}

static auto
fmtMs(double ms) -> std::string
{
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2) << ms << " ms";
  return ss.str();
}

static void
printRow(const std::string& label, const Stats& s)
{
  std::cout << "  " << std::left << std::setw(14) << label << std::right << std::setw(10) << fmtMs(s.avg_)
            << std::setw(10) << fmtMs(s.med_) << std::setw(10) << fmtMs(s.p95_) << std::setw(10) << fmtMs(s.p99_)
            << std::setw(10) << fmtMs(s.min_) << std::setw(10) << fmtMs(s.max_) << "\n";
}

// ── argument parsing ──────────────────────────────────────────────────────────

struct Args
{
  std::string model_path_;
  std::string version_str_ = "yolox";
  std::string backend_str_ = "ort";
  std::string device_str_ = "cpu";
  std::string task_str_ = "det";
  int input_w_ = 640;
  int input_h_ = 640;
  int classes_ = 80;
  int warmup_ = 10;
  int iterations_ = 100;
  std::string image_path_;
  std::string save_path_;
  std::string csv_path_;
};

static void
usage(const char* prog)
{
  std::cerr << "usage: " << prog << "\n"
            << "  --model      <path>              model file (.onnx / .xml)\n"
            << "  --version    <yolox|yolo26|yolo11|yolo8|yolo5|yolo5u>  (default: yolox)\n"
            << "  --backend    <ort|ovn|dnn>       inference backend (default: ort)\n"
            << "  --device     <cpu|gpu|auto|cuda> device for backend (default: cpu)\n"
            << "  --task       <det|cls|seg|pose|obb> (default: det)\n"
            << "  --input-w    <int>               model input width  (default: 640)\n"
            << "  --input-h    <int>               model input height (default: 640)\n"
            << "  --classes    <int>               number of classes  (default: 80)\n"
            << "  --warmup     <int>               warmup runs        (default: 10)\n"
            << "  --iterations <int>               benchmark runs     (default: 100)\n"
            << "  --image      <path>              input image (random noise if omitted)\n"
            << "  --save       <path>              save annotated result image\n"
            << "  --csv        <path>              append summary row to CSV\n";
}

static auto
parse(int argc, char** argv) -> Args // NOLINT(modernize-avoid-c-arrays)
{
  Args a;
  for (int i = 1; i < argc; ++i)
  {
    std::string k = argv[i]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    auto need = [&]() -> std::string
    {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + k);
      return argv[++i]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    };
    if (k == "--model")
    {
      a.model_path_ = need();
    }
    else if (k == "--version")
    {
      a.version_str_ = need();
    }
    else if (k == "--backend")
    {
      a.backend_str_ = need();
    }
    else if (k == "--device")
    {
      a.device_str_ = need();
    }
    else if (k == "--task")
    {
      a.task_str_ = need();
    }
    else if (k == "--input-w")
    {
      a.input_w_ = std::stoi(need());
    }
    else if (k == "--input-h")
    {
      a.input_h_ = std::stoi(need());
    }
    else if (k == "--classes")
    {
      a.classes_ = std::stoi(need());
    }
    else if (k == "--warmup")
    {
      a.warmup_ = std::stoi(need());
    }
    else if (k == "--iterations")
    {
      a.iterations_ = std::stoi(need());
    }
    else if (k == "--image")
    {
      a.image_path_ = need();
    }
    else if (k == "--save")
    {
      a.save_path_ = need();
    }
    else if (k == "--csv")
    {
      a.csv_path_ = need();
    }
    else if (k == "--help" || k == "-h")
    {
      usage(argv[0]);
      std::exit(0);
    } // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    else
    {
      throw std::runtime_error("unknown argument: " + k);
    }
  }
  if (a.model_path_.empty())
    throw std::runtime_error("--model is required");
  return a;
}

// ── config builders ───────────────────────────────────────────────────────────

static auto
resolveVersion(const std::string& s) -> YoloVersion
{
  if (s == "yolo5")
    return YoloVersion::YOLO5;
  if (s == "yolo5u")
    return YoloVersion::YOLO5U;
  if (s == "yolo8")
    return YoloVersion::YOLO8;
  if (s == "yolo11")
    return YoloVersion::YOLO11;
  if (s == "yolo26")
    return YoloVersion::YOLO26;
  if (s == "yolox")
    return YoloVersion::YOLO5; // decoded YOLOX → YOLO5 decoder
  throw std::runtime_error("unknown --version: " + s);
}

static auto
resolveRuntime(const std::string& backend, const std::string& device) -> YoloTargetRT
{
  if (backend == "ort")
    return (device == "cuda") ? YoloTargetRT::ORT_CUDA : YoloTargetRT::ORT_CPU;
  if (backend == "ovn")
  {
    if (device == "gpu")
      return YoloTargetRT::OVN_GPU;
    if (device == "auto")
      return YoloTargetRT::OVN_AUTO;
    return YoloTargetRT::OVN_CPU;
  }
  if (backend == "dnn")
    return (device == "cuda") ? YoloTargetRT::OPENCV_CUDA : YoloTargetRT::OPENCV_CPU;
  throw std::runtime_error("unknown --backend: " + backend);
}

static auto
resolveTask(const std::string& s) -> YoloTaskType
{
  if (s == "det")
    return YoloTaskType::DET;
  if (s == "cls")
    return YoloTaskType::CLS;
  if (s == "seg")
    return YoloTaskType::SEG;
  if (s == "pose")
    return YoloTaskType::POSE;
  if (s == "obb")
    return YoloTaskType::OBB;
  throw std::runtime_error("unknown --task: " + s);
}

static auto
buildConfig(const Args& a) -> YoloConfig
{
  YoloConfig cfg;
  cfg.desc_ = a.version_str_ + " [" + a.backend_str_ + "/" + a.device_str_ + "] bench";
  cfg.model_path_ = a.model_path_;
  cfg.version_ = resolveVersion(a.version_str_);
  cfg.target_rt_ = resolveRuntime(a.backend_str_, a.device_str_);
  cfg.task_ = resolveTask(a.task_str_);
  cfg.input_w_ = a.input_w_;
  cfg.input_h_ = a.input_h_;
  cfg.batch_size_ = 1;
  cfg.num_classes_ = a.classes_;
  cfg.rgb_ = true;
  // YOLOX expects raw pixel values; all other versions expect normalised [0, 1]
  cfg.scale_f_ = (a.version_str_ == "yolox") ? 1.0f : 1.0f / 255.0f;
  cfg.names_ = (a.classes_ == 80) ? std::vector<std::string>{COCO_NAMES} :
                                   std::vector<std::string>(static_cast<size_t>(a.classes_), "obj");
  return cfg;
}

// ── input frame ──────────────────────────────────────────────────────────────

static auto
makeFrame(const Args& a) -> cv::Mat
{
  if (!a.image_path_.empty())
  {
    cv::Mat img = cv::imread(a.image_path_);
    if (img.empty())
      throw std::runtime_error("cannot read image: " + a.image_path_);
    return img;
  }
  cv::Mat noise(a.input_h_, a.input_w_, CV_8UC3);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  for (int r = 0; r < noise.rows; ++r)
  {
    for (int c = 0; c < noise.cols; ++c)
    {
      noise.at<cv::Vec3b>(r, c) = {static_cast<uint8_t>(dist(rng)), static_cast<uint8_t>(dist(rng)),
                                   static_cast<uint8_t>(dist(rng))};
    }
  }
  return noise;
}

// ── CSV ───────────────────────────────────────────────────────────────────────

static void
appendCsv(const std::string& path, const Args& a, const Stats& pre, const Stats& infer, const Stats& post,
          const Stats& total)
{
  bool write_header = false;
  {
    std::ifstream f(path);
    write_header = !f.good();
  }
  std::ofstream f(path, std::ios::app);
  if (write_header)
  {
    f << "version,backend,device,task,input_w,input_h,classes,iterations,"
      << "pre_avg,pre_p95,pre_p99,"
      << "infer_avg,infer_p95,infer_p99,"
      << "post_avg,post_p95,post_p99,"
      << "total_avg,total_p95,total_p99\n";
  }
  f << std::fixed << std::setprecision(3) << a.version_str_ << "," << a.backend_str_ << "," << a.device_str_ << ","
    << a.task_str_ << "," << a.input_w_ << "," << a.input_h_ << "," << a.classes_ << "," << a.iterations_ << ","
    << pre.avg_ << "," << pre.p95_ << "," << pre.p99_ << "," << infer.avg_ << "," << infer.p95_ << "," << infer.p99_
    << "," << post.avg_ << "," << post.p95_ << "," << post.p99_ << "," << total.avg_ << "," << total.p95_ << ","
    << total.p99_ << "\n";
}

// ── main ─────────────────────────────────────────────────────────────────────

auto
main(int argc, char** argv) -> int // NOLINT(modernize-avoid-c-arrays)
{
  try
  {
    Args a = parse(argc, argv);

    YoloConfig cfg = buildConfig(a);

    std::cout << "\n=== one-yolo bench ===\n";
    std::cout << "version    : " << a.version_str_ << "\n";
    std::cout << "backend    : " << a.backend_str_ << " / " << a.device_str_ << "\n";
    std::cout << "task       : " << a.task_str_ << "\n";
    std::cout << "model      : " << a.model_path_ << "\n";
    std::cout << "input      : " << a.input_w_ << "x" << a.input_h_ << "\n";
    std::cout << "classes    : " << a.classes_ << "\n";
    std::cout << "warmup     : " << a.warmup_ << "\n";
    std::cout << "iterations : " << a.iterations_ << "\n";
    std::cout << "image      : " << (a.image_path_.empty() ? "(random noise)" : a.image_path_) << "\n\n";

    auto model = Yolo(cfg);
    model.info();

    cv::Mat frame = makeFrame(a);
    std::vector<cv::Mat> batch{frame};

    // warm-up
    std::cout << "warming up (" << a.warmup_ << " runs)...\n";
    for (int i = 0; i < a.warmup_; ++i)
      model(batch);

    // benchmark
    std::cout << "benchmarking (" << a.iterations_ << " runs)...\n\n";

    std::vector<double> pre_ms, infer_ms, post_ms, total_ms;
    pre_ms.reserve(static_cast<size_t>(a.iterations_));
    infer_ms.reserve(static_cast<size_t>(a.iterations_));
    post_ms.reserve(static_cast<size_t>(a.iterations_));
    total_ms.reserve(static_cast<size_t>(a.iterations_));

    YoloResult last_result;
    for (int i = 0; i < a.iterations_; ++i)
    {
      auto results = model(batch);
      auto& r = results[0];
      if (r.speed.size() >= 3)
      {
        pre_ms.push_back(r.speed[0]);
        infer_ms.push_back(r.speed[1]);
        post_ms.push_back(r.speed[2]);
        total_ms.push_back(r.speed[0] + r.speed[1] + r.speed[2]);
      }
      if (i == a.iterations_ - 1)
        last_result = r;
    }

    auto pre = computeStats(pre_ms);
    auto infer = computeStats(infer_ms);
    auto post = computeStats(post_ms);
    auto total = computeStats(total_ms);

    const char* sep = "  ─────────────────────────────────────────────────────────────────────\n";
    std::cout << sep;
    std::cout << "  " << std::left << std::setw(14) << "phase" << std::right << std::setw(10) << "avg" << std::setw(10)
              << "p50" << std::setw(10) << "p95" << std::setw(10) << "p99" << std::setw(10) << "min" << std::setw(10)
              << "max"
              << "\n";
    std::cout << sep;
    printRow("preprocess", pre);
    printRow("inference", infer);
    printRow("postprocess", post);
    std::cout << sep;
    printRow("total", total);
    std::cout << sep;

    std::cout << "\nFPS (1000 / total_avg): " << std::fixed << std::setprecision(1)
              << (total.avg_ > 0.0 ? 1000.0 / total.avg_ : 0.0) << "\n";

    if (!last_result.detections.empty() || !last_result.classes.empty())
      std::cout << "detections (last frame): " << last_result.detections.size() << "\n";

    if (!a.save_path_.empty() && !a.image_path_.empty())
    {
      auto annotated = last_result.plot();
      cv::imwrite(a.save_path_, annotated);
      std::cout << "annotated image: " << a.save_path_ << "\n";
    }

    if (!a.csv_path_.empty())
    {
      appendCsv(a.csv_path_, a, pre, infer, post, total);
      std::cout << "csv row appended: " << a.csv_path_ << "\n";
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << "error: " << e.what() << "\n";
    usage(argv[0]); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return 1;
  }
  return 0;
}
