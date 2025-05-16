cmake_minimum_required(VERSION 2.8)
project( digital )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# 添加 CUDA 支持
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 添加源文件
set(SOURCES
    src/main.cpp
    src/CuImage.cu
)

# 编译可执行文件
cuda_add_executable(main ${SOURCES})
target_link_libraries(main ${OpenCV_LIBS} ${CUDA_LIBRARIES})
