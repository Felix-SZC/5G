cmake_minimum_required(VERSION 3.10)

project(SmartCar_Racing)

# ============================================================
# 编译选项设置
# ============================================================
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# OpenMP 并行加速库
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found:")
    message(STATUS "  OpenMP version: ${OpenMP_CXX_VERSION}")
    message(STATUS "  OpenMP flags: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found, trying manual configuration...")
    # 手动添加 OpenMP 标志
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# 编译器优化标志
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -Wall")

# 设置输出目录为项目根目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# ============================================================
# 查找依赖库
# ============================================================

# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# 线程库
find_package(Threads REQUIRED)

# NCNN 库
find_package(ncnn QUIET)
if(NOT ncnn_FOUND)
    message(STATUS "ncnn not found by find_package, trying manual search...")
    # 手动查找 ncnn
    find_path(NCNN_INCLUDE_DIR
        NAMES ncnn/net.h
        PATHS 
            /usr/local/include
            /usr/include
            ${CMAKE_SOURCE_DIR}/3rdparty/ncnn/include
    )
    
    find_library(NCNN_LIBRARY
        NAMES ncnn
        PATHS
            /usr/local/lib
            /usr/lib
            ${CMAKE_SOURCE_DIR}/3rdparty/ncnn/lib
    )
    
    if(NCNN_INCLUDE_DIR AND NCNN_LIBRARY)
        message(STATUS "NCNN found manually:")
        message(STATUS "  Include: ${NCNN_INCLUDE_DIR}")
        message(STATUS "  Library: ${NCNN_LIBRARY}")
        
        # 创建导入目标
        add_library(ncnn SHARED IMPORTED)
        set_target_properties(ncnn PROPERTIES
            IMPORTED_LOCATION ${NCNN_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${NCNN_INCLUDE_DIR}
        )
    else()
        message(FATAL_ERROR "NCNN not found! Please install ncnn or specify NCNN_INCLUDE_DIR and NCNN_LIBRARY")
    endif()
else()
    message(STATUS "NCNN found by find_package")
endif()

# pigpio 库
set(PIGPIO_LIBRARY_PATHS
    /lib/libpigpio.so
    /usr/lib/libpigpio.so
    /usr/local/lib/libpigpio.so
)

foreach(lib_path ${PIGPIO_LIBRARY_PATHS})
    if(EXISTS ${lib_path})
        set(PIGPIO_LIBRARY ${lib_path})
        message(STATUS "pigpio found: ${PIGPIO_LIBRARY}")
        break()
    endif()
endforeach()

if(NOT PIGPIO_LIBRARY)
    message(FATAL_ERROR "pigpio library not found! Please install pigpio: sudo apt-get install pigpio")
endif()

# ============================================================
# 头文件目录
# ============================================================
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${NCNN_INCLUDE_DIR}
    /usr/include
    /usr/local/include
)

# ============================================================
# 源文件
# ============================================================
set(FASTESTDET_SOURCES
    ${CMAKE_SOURCE_DIR}/src/fastestdet.cpp
)

# ============================================================
# 主程序编译
# ============================================================
add_executable(SmartCar
    ${CMAKE_SOURCE_DIR}/main.cpp
    ${FASTESTDET_SOURCES}
)

add_executable(SmartCarFast
    ${CMAKE_SOURCE_DIR}/main_fast.cpp
    ${FASTESTDET_SOURCES}
)

foreach(target_name IN ITEMS SmartCar SmartCarFast)
    target_link_libraries(${target_name}
        ${PIGPIO_LIBRARY}
        ncnn
        ${OpenCV_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
        pthread
    )
endforeach()

# 链接 OpenMP 库
if(OpenMP_CXX_FOUND)
    target_link_libraries(SmartCar OpenMP::OpenMP_CXX)
    target_link_libraries(SmartCarFast OpenMP::OpenMP_CXX)
else()
    # 手动链接 OpenMP 库
    target_link_libraries(SmartCar gomp)
    target_link_libraries(SmartCarFast gomp)
endif()

# 设置输出位置为主目录
set_target_properties(SmartCar PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
set_target_properties(SmartCarFast PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

message(STATUS "==============================================")
message(STATUS "SmartCar will be built to: ${PROJECT_SOURCE_DIR}/SmartCar")
message(STATUS "==============================================")

# ============================================================
# 测试程序编译
# ============================================================
# test_sobel: ImageSobel函数测试程序
add_executable(test_sobel
    ${CMAKE_SOURCE_DIR}/test/test_sobel.cpp
)

target_link_libraries(test_sobel
    ${OpenCV_LIBS}
)

set_target_properties(test_sobel PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# test_banma: banma_get函数测试程序
add_executable(test_banma
    ${CMAKE_SOURCE_DIR}/test/test_banma.cpp
)

target_link_libraries(test_banma
    ${OpenCV_LIBS}
)

set_target_properties(test_banma PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "Test programs will be built to: ${CMAKE_BINARY_DIR}/")
message(STATUS "  - test_sobel: ImageSobel function demo")
message(STATUS "  - test_banma: banma_get function demo")

# ============================================================
# 安装规则（可选）
# ============================================================
install(TARGETS SmartCar
    RUNTIME DESTINATION bin
)

# ============================================================
# 编译信息输出
# ============================================================
message(STATUS "==============================================")
message(STATUS "Build Configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "  Output Directory: ${PROJECT_SOURCE_DIR}")
message(STATUS "==============================================")
message(STATUS "Dependencies:")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  NCNN: ${NCNN_LIBRARY}")
message(STATUS "  pigpio: ${PIGPIO_LIBRARY}")
message(STATUS "  Threads: ${CMAKE_THREAD_LIBS_INIT}")
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP: ${OpenMP_CXX_VERSION} (Enabled)")
else()
    message(STATUS "  OpenMP: Manual configuration (gomp)")
endif()
message(STATUS "==============================================")