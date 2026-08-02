#include "YoloRuntime.h"

#include <utility>

namespace yolo {

YoloRuntime::YoloRuntime(std::string rt_name) : rt_name_(std::move(rt_name)) {}

YoloRuntime::~YoloRuntime() = default;

auto
YoloRuntime::toString() -> std::string
{
  return rt_name_;
}

}