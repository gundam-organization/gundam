#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "JointProbability.h"

// Test that the joint probability constructor builds things.
TEST(jointProbability, Construction) {
    std::vector<std::string> names{
        "PoissonLLH",
        "LeastSquares",
        "BarlowLLH",
        "BarlowLLH_BANFF_OA2020",
        "BarlowLLH_BANFF_OA2021",
        "BarlowLLH_BANFF_OA2021_SFGD",
        "Chi2",
    };

    for (std::string prob : names) {
        std::unique_ptr<JointProbability::JointProbabilityBase>
            jointProb(JointProbability::makeJointProbability(prob));
        EXPECT_NE(jointProb,nullptr) << prob << " not constructed";
    }
}

TEST(jointProbability, LeastSquares_Continuity) {

    std::unique_ptr<JointProbability::JointProbabilityBase>
        jointProb(JointProbability::makeJointProbability(
                      "LeastSquares"));
    double data = 20.0;
    double origPred = std::max(1.0, data);
    double fracErr = 1.0/sqrt(origPred);
    double origErr = fracErr * origPred;;
    double predErr = fracErr * origPred;;

    // Check the continuity for values less than the observation
    double lastP = -1;
    double lastPred = 0.0;
    for (double pred = origPred; pred != lastPred; pred *= 0.9) {
        predErr = fracErr*pred;
        double newP = jointProb->eval(data, pred, predErr,0);
        ASSERT_FALSE(std::isnan(newP)) << "Joint probability is not a number";
        ASSERT_GE(newP,lastP) << "Joint probability should be monotonic toward zero";
        lastP = newP;
        lastPred = pred;
    }

    // Check the continuity for values greater than the observation
    lastP = -1;
    lastPred = 0.0;
    for (double pred = origPred; pred < 20*origPred; pred *= 1.01) {
        predErr = fracErr*pred;
        double newP = jointProb->eval(data, pred, predErr,0);
        ASSERT_FALSE(std::isnan(newP)) << "Joint probability is not a number";
        ASSERT_GE(newP,lastP) << "Joint probability should be monotonic toward infinity";
        lastP = newP;
        lastPred = pred;
    }

}
