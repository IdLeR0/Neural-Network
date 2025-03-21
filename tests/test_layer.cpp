#include <gtest/gtest.h>
#include "activation_function.h"
#include "linalg.h"
#include "layer.h"

namespace network {
namespace tests_layer {

Matrix weights_matrix(3, 2);
Vector bias(3);
Matrix input(2, 3);
Matrix expected_linear_output(3, 3);
Matrix expected_forward_matrix(3, 3);
Matrix expected_backward_matrix(3, 2);
Vector expected_grad_b(3);
Matrix expected_grad_a(3, 2);
Matrix gradient(3, 3);
void SetTestParamers() {
    weights_matrix << 6, 1, 1, 1, 10, 7;
    bias << 1, 1, 1;
    input << 1, -2, 3, -4, 5, -5;
    expected_linear_output << 3, -6, 14, -2, 4, -1, -17, 16, -4;
    expected_forward_matrix << 3, 0, 14, 0, 4, 0, 0, 16, 0;
    gradient << 8, 10, 1, 7, 6, 6, -9, -5, 1;
    /////////////
    expected_backward_matrix << 48, 8, 66, 48, -54, -9;
    expected_grad_b << -1, 6, 6;
    expected_grad_a << -19, 13, -12, 30, -12, 30;
    expected_grad_b /= 3;
    expected_grad_a /= 3;
}

TEST(Correction, Forward) {
    Layer layer(weights_matrix, bias, ActivationFunc::Name::ReLU);
    Matrix output = layer.ApplyLinear(input);
    EXPECT_EQ(output, expected_linear_output);
    Matrix forward_matrix = layer.Forward(input);
    EXPECT_EQ(forward_matrix, expected_forward_matrix);
}
TEST(Correction, Backward) {
    Layer layer(weights_matrix, bias, ActivationFunc::Name::ReLU);
    Matrix bacward_matrix = layer.Backward(input, gradient);
    EXPECT_EQ(expected_backward_matrix, bacward_matrix);
}
TEST(Correction, Gradients) {
    Layer layer(weights_matrix, bias, ActivationFunc::Name::ReLU);
    WeightsBiasGradient grad_weights_bias = layer.GetWeightsBiasGradient(input, gradient);
    EXPECT_EQ(expected_grad_a, grad_weights_bias.weights);
    EXPECT_EQ(expected_grad_b, grad_weights_bias.bias);
}
// namespace network
}  // namespace tests_layer
}  // namespace network
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
