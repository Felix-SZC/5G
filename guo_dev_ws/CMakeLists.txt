cmake_minimum_required(VERSION 3.10)

project(SmartCar_5G_fianl)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(PROTO_PB_DIR ${PROJECT_SOURCE_DIR}pio)

include_directories( 
    ${PROTO_PB_DIR} 
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include/MNN
    ${OpenCV_INCLUDE_DIRS}
    # 添加 pigpio 头文件可能的路径
    /usr/include
    /usr/local/include
)

# 添加库文件搜索路径
link_directories(
    /lib  # 添加 pigpio 所在的库目录
    /usr/lib
    /usr/local/lib
)

# 直接指定 pigpio 库的路径（根据你的实际输出）
set(PIGPIO_LIBRARY "/lib/libpigpio.so")

# 验证库是否存在
if(NOT EXISTS ${PIGPIO_LIBRARY})
    message(FATAL_ERROR "pigpio library not found at ${PIGPIO_LIBRARY}")
endif()

find_package(OpenCV REQUIRED)
find_package(Threads)

add_library(libmnn SHARED IMPORTED)
set_target_properties(libmnn PROPERTIES IMPORTED_LOCATION /usr/local/lib/libMNN.so)

aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC_LIST)


#add_executable(SmartCar
##    src/main.cpp
##    src/Yolo.cpp
#)
#
#target_link_libraries(SmartCar
#    ${PIGPIO_LIBRARY}  # 使用直接指定的 pigpio 库
#    libmnn
#    ${OpenCV_LIBS}
#    ${CMAKE_THREAD_LIBS_INIT}
#)

add_executable(motor_test src/motor_test.cpp)

target_link_libraries(motor_test 
    ${PIGPIO_LIBRARY}  # 使用直接指定的 pigpio 库
    ${CMAKE_THREAD_LIBS_INIT}
)

add_executable(track_line src/track_line.cpp)

target_link_libraries(track_line
    ${PIGPIO_LIBRARY}  # 使用直接指定的 pigpio 库
    ${OpenCV_LIBS}  
    ${CMAKE_THREAD_LIBS_INIT}
)


add_executable(test_sobel test/test_sobel.cpp)

target_link_libraries(test_sobel
    ${OpenCV_LIBS}
)

add_executable(test_cone_path test/test_cone_path.cpp)

target_link_libraries(test_cone_path
    ${OpenCV_LIBS}
)