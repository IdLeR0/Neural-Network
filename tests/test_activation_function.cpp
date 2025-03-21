#include <gtest/gtest.h>
#include "linalg.h"
#include "activation_function.h"

namespace network {

TEST(ActivationfuncTest, Relu) {
    ActivationFunc act(ActivationFunc::Name::ReLU);
    Matrix a(2, 2);
    a << 1, -1, 2, -2;
    Matrix expected_apply(2, 2);
    expected_apply << 1, 0, 2, 0;
    EXPECT_EQ(expected_apply, act.Apply(a));
    Matrix expected_diff(2, 2);
    expected_diff << 1, 0, 0, 1;
    EXPECT_EQ(act.GetDifferential(a.col(0)), expected_diff);
}
}  // namespace network
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}