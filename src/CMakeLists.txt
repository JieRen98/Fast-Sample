add_library(storage SHARED storage.cpp)
target_link_libraries(storage ${CUDA_LIBRARIES})
add_library(manager SHARED manager.cpp)
target_link_libraries(manager ${CUDA_LIBRARIES})

cuda_add_library(cu_manager SHARED manager.cu)
set_property(TARGET cu_manager PROPERTY CUDA_ARCHITECTURES "60" POSITION_INDEPENDENT_CODE ON)

cuda_add_library(cu_manager_func SHARED manager_func.cu)
set_property(TARGET cu_manager_func PROPERTY CUDA_ARCHITECTURES "60" POSITION_INDEPENDENT_CODE ON)