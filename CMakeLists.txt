project(Fast-Sample LANGUAGES CXX)

cmake_minimum_required(VERSION VERSION 3.15 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA QUIET)
IF(CUDA_FOUND)
    message(STATUS "CUDA found")
    set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
    enable_language(CUDA)
    set(CUDA_64_BIT_DEVICE_CODE ON FORCE)
    include_directories("${CUDA_INCLUDE_DIRS}")
ELSE()
    message("CUDA not found!")
ENDIF()

include_directories(include)
add_subdirectory(src)

add_subdirectory(test)
