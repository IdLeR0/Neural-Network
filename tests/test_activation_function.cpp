#include <gtest/gtest.h>
#include "activation_function.h"






TEST(ActivationfuncTest, Relu){
    network::ActivationFunc func(network::NamesActivationFunc::ReLU);
    Eigen::VectorXd vector(2);
    vector[1] = 1;
    vector[0] = 1;
    Eigen::MatrixXd g = func.Activate(vector);
    EXPECT_EQ(vector, g);

}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}   