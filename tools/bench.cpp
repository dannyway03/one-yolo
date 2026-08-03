/**
 * bench.cpp — end-to-end latency benchmark for one-yolo.
 *
 * Measures the full pipeline (preprocess + inference + postprocess) via the
 * YoloConfig / Yolo class, so numbers reflect real application cost.
 *
 * Usage:
 *   bench <config.json> --img <path> [--backend ort|ovn] [--device cpu|gpu|auto|cuda]
 *         [--warmup 10] [--iterations 100] [--pipeline]
 */

#include <chrono>
#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "app_common.hpp"
#include "Yolo.h"
#include "YoloTask.h"
#ifdef BUILD_WITH_OVN
  #include "ovn/YoloOVNRT.h"
#endif

using namespace yolo;
using app::resolveRuntime;

// ── helpers ──────────────────────────────────────────────────────────────────

struct Stats
{
  double avg_;
  double med_;
  double p95_;
  double p99_;
  double min_;
  double max_;
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
  return {avg,
          percentile(v, 50.0),
          percentile(v, 95.0),
          percentile(v, 99.0),
          *std::min_element(v.begin(), v.end()),
          *std::max_element(v.begin(), v.end())};
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
  std::string config_;
  std::string backend_str_ = "ort";
  std::string device_str_ = "cpu";
  int warmup_ = 10;
  int iterations_ = 100;
  std::string image_path_;
  std::string save_path_;
  std::string csv_path_;
  bool pipeline_ = false;
};

static void
usage(const char* prog)
{
  std::cerr << "usage: " << prog << " <config.json> [options]\n"
            << "  --backend    <ort|ovn>             inference backend (default: ort)\n"
            << "  --device     <cpu|gpu|auto|cuda>   device for backend (default: cpu)\n"
            << "  --warmup     <int>                 warmup runs        (default: 10)\n"
            << "  --iterations <int>                 benchmark runs     (default: 100)\n"
            << "  --image      <path>                input image (random noise if omitted)\n"
            << "  --save       <path>                save annotated result image\n"
            << "  --csv        <path>                append summary row to CSV\n"
            << "  --pipeline                         double-buffer mode: overlap CPU pre with GPU infer (OVN only)\n";
}

static auto
parse(int argc, char** argv) -> Args // NOLINT(modernize-avoid-c-arrays)
{
  Args a;
  int positional = 0;
  for (int i = 1; i < argc; ++i)
  {
    std::string k = argv[i]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    auto need = [&]() -> std::string
    {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + k);
      return argv[++i]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    };
    if (k == "--backend")
    {
      a.backend_str_ = need();
    }
    else if (k == "--device")
    {
      a.device_str_ = need();
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
    else if (k == "--pipeline")
    {
      a.pipeline_ = true;
    }
    else if (k == "--help" || k == "-h")
    {
      usage(argv[0]);
      std::exit(0);
    } // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    else if (k.rfind("--", 0) == 0)
    {
      throw std::runtime_error("unknown argument: " + k);
    }
    else
    {
      if (positional == 0)
        a.config_ = k;
      else
        throw std::runtime_error("unexpected positional argument: " + k);
      ++positional;
    }
  }
  if (a.config_.empty())
    throw std::runtime_error("config.json is required");
  return a;
}

// ── config builder ────────────────────────────────────────────────────────────

static auto
buildConfig(const Args& a) -> YoloConfig
{
  auto cfg = YoloConfig::from_json(a.config_);
  cfg.target_rt_ = resolveRuntime(a.backend_str_, a.device_str_);
  return cfg;
}

// ── input frame ──────────────────────────────────────────────────────────────

static auto
makeFrame(const Args& a, int w, int h) -> cv::Mat
{
  if (!a.image_path_.empty())
  {
    cv::Mat img = cv::imread(a.image_path_);
    if (img.empty())
      throw std::runtime_error("cannot read image: " + a.image_path_);
    return img;
  }
  cv::Mat noise(h, w, CV_8UC3);
  std::mt19937 rng(42); // NOLINT
  std::uniform_int_distribution<int> dist(0, 255); // NOLINT
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
appendCsv(const std::string& path, const Args& a, const YoloConfig& cfg, const Stats& pre, const Stats& infer,
          const Stats& post, const Stats& total)
{
  bool write_header = false;
  {
    std::ifstream f(path);
    write_header = !f.good();
  }
  std::ofstream f(path, std::ios::app);
  if (write_header)
  {
    f << "version,backend,device,task,input_w,input_h,classes_,iterations,"
      << "pre_avg,pre_p95,pre_p99,"
      << "infer_avg,infer_p95,infer_p99,"
      << "post_avg,post_p95,post_p99,"
      << "total_avg,total_p95,total_p99\n";
  }
  f << std::fixed << std::setprecision(3) << toString(cfg.version_) << "," << a.backend_str_ << "," << a.device_str_
    << "," << toString(cfg.task_) << "," << cfg.input_w_ << "," << cfg.input_h_ << "," << cfg.num_classes_ << ","
    << a.iterations_ << "," << pre.avg_ << "," << pre.p95_ << "," << pre.p99_ << "," << infer.avg_ << "," << infer.p95_
    << "," << infer.p99_ << "," << post.avg_ << "," << post.p95_ << "," << post.p99_ << "," << total.avg_ << ","
    << total.p95_ << "," << total.p99_ << "\n";
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
    std::cout << "version    : " << toString(cfg.version_) << "\n";
    std::cout << "backend    : " << a.backend_str_ << " / " << a.device_str_ << "\n";
    std::cout << "task       : " << toString(cfg.task_) << "\n";
    std::cout << "model      : " << cfg.model_path_ << "\n";
    std::cout << "input      : " << cfg.input_w_ << "x" << cfg.input_h_ << "\n";
    std::cout << "classes_    : " << cfg.num_classes_ << "\n";
    std::cout << "warmup     : " << a.warmup_ << "\n";
    std::cout << "iterations : " << a.iterations_ << "\n";
    std::cout << "image      : " << (a.image_path_.empty() ? "(random noise)" : a.image_path_) << "\n\n";

    auto model = Yolo(cfg);
    model.info();

    cv::Mat frame = makeFrame(a, cfg.input_w_, cfg.input_h_);
    std::vector<cv::Mat> batch{frame};
    YoloResult last_result;

#ifdef BUILD_WITH_OVN
    if (a.pipeline_)
    {
      auto* ovn = dynamic_cast<yolo::YoloOVNRT*>(model.task()->runtime().get());
      if (ovn == nullptr)
        throw std::runtime_error("--pipeline requires --backend ovn");

      std::cout << "mode       : pipeline (submit/collect double-buffer)\n\n";

      // warm-up
      std::cout << "warming up (" << a.warmup_ << " runs)...\n";
      for (int i = 0; i < a.warmup_; ++i)
        model(batch);

      // Double-buffer blobs: GPU holds blobs[bidx^1] while CPU fills blobs[bidx].
      // Each blob must stay alive until its matching collect() returns (zero-copy).
      std::array<cv::Mat, 2> blobs;
      int bidx = 0;
      blobs[bidx] = model.task()->preprocess(batch);
      ovn->submit(blobs[bidx]);
      bidx ^= 1;

      std::cout << "benchmarking (" << a.iterations_ << " runs)...\n\n";

      using Clock = std::chrono::steady_clock;
      std::vector<double> pre_ms, wait_ms, post_ms, total_ms;
      pre_ms.reserve(static_cast<size_t>(a.iterations_));
      wait_ms.reserve(static_cast<size_t>(a.iterations_));
      post_ms.reserve(static_cast<size_t>(a.iterations_));
      total_ms.reserve(static_cast<size_t>(a.iterations_));

      for (int i = 0; i < a.iterations_; ++i)
      {
        auto t0 = Clock::now();
        blobs[bidx] = model.task()->preprocess(batch); // CPU, overlaps GPU on blobs[bidx^1]
        auto t1 = Clock::now();
        ovn->submit(blobs[bidx]);
        auto raw = ovn->collect(); // waits for blobs[bidx^1] — already safe to reuse next iter
        auto t2 = Clock::now();
        auto pipe_results = model.task()->postprocess(raw, 1);
        auto t3 = Clock::now();
        if (i == a.iterations_ - 1)
          last_result = pipe_results[0];
        bidx ^= 1;

        auto ms_of = [](auto aa, auto bb)
        -> auto {
          return std::chrono::duration<double, std::milli>(bb - aa).count();
        };
        pre_ms.push_back(ms_of(t0, t1));
        wait_ms.push_back(ms_of(t1, t2));
        post_ms.push_back(ms_of(t2, t3));
        total_ms.push_back(ms_of(t0, t3));
      }
      ovn->collect(); // drain final submit

      auto pre = computeStats(pre_ms);
      auto wait = computeStats(wait_ms);
      auto post = computeStats(post_ms);
      auto tot = computeStats(total_ms);

      const char* sep = "  ─────────────────────────────────────────────────────────────────────\n";
      std::cout << sep;
      std::cout << "  " << std::left << std::setw(14) << "phase" << std::right << std::setw(10) << "avg"
                << std::setw(10) << "p50" << std::setw(10) << "p95" << std::setw(10) << "p99" << std::setw(10) << "min"
                << std::setw(10) << "max"
                << "\n";
      std::cout << sep;
      printRow("preprocess", pre);
      printRow("gpu wait", wait);
      printRow("postprocess", post);
      std::cout << sep;
      printRow("total/iter", tot);
      std::cout << sep;
      std::cout << "\nFPS (1000 / total_avg): " << std::fixed << std::setprecision(1)
                << (tot.avg_ > 0.0 ? 1000.0 / tot.avg_ : 0.0) << "\n";
      std::cout << "note: 'gpu wait' ~ 0 means full overlap achieved; total ≈ max(pre+post, infer)\n";
      if (!last_result.detections_.empty() || !last_result.classes_.empty())
        std::cout << "detections (last frame): " << last_result.detections_.size() << "\n";
      return 0;
    }
#endif

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

    for (int i = 0; i < a.iterations_; ++i)
    {
      auto results = model(batch);
      auto& r = results[0];
      if (r.speed_.size() >= 3)
      {
        pre_ms.push_back(r.speed_[0]);
        infer_ms.push_back(r.speed_[1]);
        post_ms.push_back(r.speed_[2]);
        total_ms.push_back(r.speed_[0] + r.speed_[1] + r.speed_[2]);
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

    if (!last_result.detections_.empty() || !last_result.classes_.empty())
      std::cout << "detections (last frame): " << last_result.detections_.size() << "\n";

    if (!a.save_path_.empty() && !a.image_path_.empty())
    {
      auto annotated = last_result.plot();
      cv::imwrite(a.save_path_, annotated);
      std::cout << "annotated image: " << a.save_path_ << "\n";
    }

    if (!a.csv_path_.empty())
    {
      appendCsv(a.csv_path_, a, cfg, pre, infer, post, total);
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