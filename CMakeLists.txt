cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
    ${CMAKE_SOURCE_DIR}/libs/Eigen
    ${CMAKE_SOURCE_DIR}/libs/EigenRand
    ${CMAKE_SOURCE_DIR}/libs/googletest/googletest/include
    ${CMAKE_SOURCE_DIR}/libs/mnist/include
    ${CMAKE_SOURCE_DIR}/src
)

add_subdirectory(${CMAKE_SOURCE_DIR}/libs/googletest)

add_executable(test_net tests/test_net.cpp src/adam_optimizer.cpp src/net.cpp src/loss_function.cpp src/activation_function.cpp src/dataloader.cpp src/layer.cpp src/random_params.cpp tests/mnist_utils.cpp)
target_link_libraries(test_net PRIVATE gtest gtest_main)

enable_testing()
add_test(NAME test_mnist COMMAND test_mnist)



