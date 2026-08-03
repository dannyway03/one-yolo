#       ____  _   _ ______  __     ______  _      ____  
#      / __ \| \ | |  ____| \ \   / / __ \| |    / __ \ 
#     | |  | |  \| | |__     \ \_/ / |  | | |   | |  | |
#     | |  | | . ` |  __|     \   /| |  | | |   | |  | |
#     | |__| | |\  | |____     | | | |__| | |___| |__| |
#      \____/|_| \_|______|    |_|  \____/|______\____/ 
#                                                       
#                                     @zhouzhi since 2025           
cmake_minimum_required(VERSION 3.10)
project(one-yolo VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
execute_process(COMMAND ln -sf "${CMAKE_BINARY_DIR}/compile_commands.json" "${CMAKE_SOURCE_DIR}/compile_commands.json")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -w -fdiagnostics-color=always -pthread")

option(BUILD_WITH_ORT "enable ONNXRuntime" ON)
option(BUILD_WITH_OVN "enable OpenVINO(Intel Platform)" ON)
#option(BUILD_WITH_CML "enable CoreML(Apple Platform)?" OFF)
#option(BUILD_WITH_PDL "enable PaddlePaddle?" OFF)
#option(BUILD_WITH_CAN "enable CANN(HuaWei Platform)?" OFF)
#option(BUILD_WITH_DEL "enable Denglin(Denglin Platform)?" OFF)
#option(BUILD_WITH_CAB "enable Cambricon(Cambricon Platform)?" OFF)

# OpenCV required
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("include")

##########################
# build one-yolo library #
##########################
file(GLOB BASE_SRCS "src/*.cpp")
file(GLOB_RECURSE ORT_SRCS "src/ort/*.cpp")
file(GLOB_RECURSE OVN_SRCS "src/ovn/*.cpp")
list(APPEND ONE_YOLO_DEPEND_SRCS ${BASE_SRCS})
list(APPEND ONE_YOLO_DEPEND_LIBS ${OpenCV_LIBS})

# enable onnxruntime, 
# make sure onnxruntime installed and include & libs path could be found by cmake
if(BUILD_WITH_ORT)
    #include_directories("include path to onnxruntime")
    #link_directories("libs path to onnxruntime")
    find_package(onnxruntime REQUIRED HINTS ${onnxruntime_DIR})
    add_definitions(-DBUILD_WITH_ORT)
    list(APPEND ONE_YOLO_DEPEND_SRCS ${ORT_SRCS})
    list(APPEND ONE_YOLO_DEPEND_LIBS onnxruntime::onnxruntime)
endif()
# enable openvino, 
# make sure openvino installed and include & libs path could be found by cmake
if(BUILD_WITH_OVN)
    #include_directories("include path to openvino")
    #link_directories("libs path to openvino") 
    find_package(OpenVINO REQUIRED HINTS ${OpenVINO_DIR})
    add_definitions(-DBUILD_WITH_OVN)
    list(APPEND ONE_YOLO_DEPEND_SRCS ${OVN_SRCS})
    list(APPEND ONE_YOLO_DEPEND_LIBS openvino::runtime)
endif()
find_package(nlohmann_json REQUIRED)
file(GLOB_RECURSE TRACK_SRCS "src/track/*.cpp")
list(APPEND ONE_YOLO_DEPEND_SRCS ${TRACK_SRCS})

add_library(${PROJECT_NAME} SHARED ${ONE_YOLO_DEPEND_SRCS})
target_link_libraries(${PROJECT_NAME} ${ONE_YOLO_DEPEND_LIBS} nlohmann_json::nlohmann_json)

##########################
# build one-yolo tools #
add_executable(test_yolox_img tools/test_yolox_img.cpp)
target_link_libraries(test_yolox_img ${PROJECT_NAME})

add_executable(bench tools/bench.cpp)
target_link_libraries(bench ${PROJECT_NAME})

##########################
# build one-yolo samples #
##########################
add_subdirectory(samples)
