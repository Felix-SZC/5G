cmake_minimum_required(VERSION 3.10)

project(SmartCar_5G_fianl)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/lib/cmake/pigpio/)
set(PROTO_PB_DIR ${PROJECT_SOURCE_DIR}pio)

include_directories( ${PROTO_PB_DIR} )
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(pigpio REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads)

add_library(libmnn SHARED IMPORTED)
set_target_properties(libmnn PROPERTIES IMPORTED_LOCATION /usr/local/lib/libMNN.so)

aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC_LIST)


add_executable(SmartCar
    src/main.cpp
    src/Yolo.cpp
)

target_link_libraries(SmartCar
    pigpio
    libmnn
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
)

add_executable(motor_test src/motor_test.cpp)

target_link_libraries(motor_test 
    pigpio
    ${CMAKE_THREAD_LIBS_INIT}
)

add_executable(track_line src/track_line.cpp)

target_link_libraries(track_line
    pigpio
    ${OpenCV_LIBS}  
    ${CMAKE_THREAD_LIBS_INIT}
)

add_executable(test_sobel test/test_sobel.cpp)

target_link_libraries(test_sobel
    ${OpenCV_LIBS}
)