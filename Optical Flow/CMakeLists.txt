cmake_minimum_required(VERSION 2.8)
project(digital)

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 添加源文件（主可执行文件 + CUDA文件）
set(MAIN_SOURCES
    src/main.cpp
    src/CUfarneback.cu
)

# 添加新的显示程序源文件
set(DISPLAY_SOURCES
    src/FBdisplay.cpp
)

# 编译主可执行文件（带CUDA代码）
cuda_add_executable(main ${MAIN_SOURCES})
target_link_libraries(main ${OpenCV_LIBS} ${CUDA_LIBRARIES})

# 编译新的显示可执行文件（仅OpenCV）
add_executable(FBdisplay ${DISPLAY_SOURCES})
target_link_libraries(FBdisplay ${OpenCV_LIBS})