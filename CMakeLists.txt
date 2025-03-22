cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


include_directories(
    ${CMAKE_SOURCE_DIR}/libs/eigen
    ${CMAKE_SOURCE_DIR}/libs/EigenRand
    ${CMAKE_SOURCE_DIR}/libs/googletest/googletest/include
    ${CMAKE_SOURCE_DIR}/libs/mnist/include
    ${CMAKE_SOURCE_DIR}/src/include
)


add_subdirectory(${CMAKE_SOURCE_DIR}/libs/googletest)
# add_subdirectory(libs/mnist)
# add_subdirectory(${CMAKE_SOURCE_DIR}/libs/EigenRand)
# add_subdirectory(${CMAKE_SOURCE_DIR}/libs/eigen)

add_executable(test_activation_function 
    tests/test_activation_function.cpp 
    src/activation_function.cpp  # Добавляем исходный файл с реализацией
)
target_link_libraries(test_activation_function PRIVATE gtest gtest_main)


add_executable(test_loss_function 
    tests/test_loss_function.cpp
    src/loss_function.cpp
)
target_link_libraries(test_loss_function PRIVATE gtest gtest_main)

add_executable(test_layer
    tests/test_layer.cpp
    src/layer.cpp
    src/activation_function.cpp)
target_link_libraries(test_layer PRIVATE gtest gtest_main)

add_executable(main src/main.cpp src/adam_optimizer.cpp src/net.cpp src/loss_function.cpp src/activation_function.cpp src/dataloader.cpp src/layer.cpp)




enable_testing()
add_test(NAME test_activation_function COMMAND test_activation_function)
add_test(NAME test_loss_function COMMAND test_loss_function)
add_test(NAME test_layer COMMAND test_layer)



