#include "gtest/gtest.h"

#include <string>
#include <vector>
#include <limits>

#include "JointProbability.h"

// Test that the joint probability constructor builds things.
TEST(jointProbability, Construction) {
    std::vector<std::string> names{
        "LeastSquares",
        "Chi2",
        "PoissonLLH",
        "BarlowLLH",
        "BarlowLLH_BANFF_OA2020",
        "BarlowLLH_BANFF_OA2021",
        "BarlowLLH_BANFF_OA2021_SFGD",
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
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,lastP) << "Not monotonic for"
                              << " data: " << data
                              << " prediction: " << pred
                              << " error: " << predErr
                              << " prob: " << newP << " (" << lastP << ")";
        lastP = newP;
        lastPred = pred;
    }

    // Check the continuity for values greater than the observation
    lastP = -1;
    lastPred = 0.0;
    for (double pred = origPred; pred < 20*origPred; pred *= 1.01) {
        predErr = fracErr*pred;
        double newP = jointProb->eval(data, pred, predErr,0);
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,lastP) << "Not monotonic for"
                              << " data: " << data
                              << " prediction: " << pred
                              << " error: " << predErr
                              << " prob: " << newP << " (" << lastP << ")";
        lastP = newP;
        lastPred = pred;
    }

}

TEST(jointProbability, ChiSquared_Continuity) {

    std::unique_ptr<JointProbability::JointProbabilityBase>
        jointProb(JointProbability::makeJointProbability(
                      "Chi2"));
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
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,lastP) << "Not monotonic for"
                              << " data: " << data
                              << " prediction: " << pred
                              << " error: " << predErr
                              << " prob: " << newP << " (" << lastP << ")";
        lastP = newP;
        lastPred = pred;
    }

    // Check the continuity for values greater than the observation
    lastP = -1;
    lastPred = 0.0;
    for (double pred = origPred; pred < 20*origPred; pred *= 1.01) {
        predErr = fracErr*pred;
        double newP = jointProb->eval(data, pred, predErr,0);
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,lastP) << "Not monotonic for"
                              << " data: " << data
                              << " prediction: " << pred
                              << " error: " << predErr
                              << " prob: " << newP << " (" << lastP << ")";
        lastP = newP;
        lastPred = pred;
    }

}

TEST(jointProbability, PoissonLLH_Continuity) {

    std::unique_ptr<JointProbability::JointProbabilityBase>
        jointProb(JointProbability::makeJointProbability(
                      "PoissonLLH"));
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
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,lastP) << "Not monotonic for"
                              << " data: " << data
                              << " prediction: " << pred
                              << " error: " << predErr
                              << " prob: " << newP << " (" << lastP << ")";
        lastP = newP;
        lastPred = pred;
    }

    // Check the continuity for values greater than the observation
    lastP = -1;
    lastPred = 0.0;
    for (double pred = origPred; pred < 20*origPred; pred *= 1.01) {
        predErr = fracErr*pred;
        double newP = jointProb->eval(data, pred, predErr,0);
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,lastP) << "Not monotonic for"
                              << " data: " << data
                              << " prediction: " << pred
                              << " error: " << predErr
                              << " prob: " << newP << " (" << lastP << ")";
        lastP = newP;
        lastPred = pred;
    }

}

TEST(jointProbability, OA2021_NoUpdateErr_Continuity) {

    std::unique_ptr<JointProbability::JointProbabilityBase>
        jointProb(JointProbability::makeJointProbability(
                      "BarlowLLH_BANFF_OA2021"));
    double data = 20.0;
    double origPred = std::max(1.0, data);
    double overSample = 10.0;
    double fracErr = 1.0/sqrt(overSample*origPred);
    double origErr = fracErr * origPred;;
    double predErr = fracErr * origPred;;

    // Check the continuity for values less than the observation
    double lastP = -1;
    double lastPred = 0.0;
    for (double pred = origPred; pred != lastPred; pred *= 0.9) {
        predErr = origErr;
        double newP = jointProb->eval(data, pred, predErr,0);
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,(0.99999)*lastP)
            << "Not monotonic for"
            << " data: " << data
            << " prediction: " << pred
            << " error: " << predErr
            << " prob: " << newP << " (" << lastP << ")";
        lastP = newP;
        lastPred = pred;
    }

    // Check the continuity for values greater than the observation
    lastP = -1;
    lastPred = 0.0;
    for (double pred = origPred; pred < 20*origPred; pred *= 1.01) {
        predErr = origErr;
        double newP = jointProb->eval(data, pred, predErr,0);
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,lastP) << "Not monotonic for"
                              << " data: " << data
                              << " prediction: " << pred
                              << " error: " << predErr
                              << " prob: " << newP << " (" << lastP << ")";
        lastP = newP;
        lastPred = pred;
    }

    // Check the continuity for values greater than the observation with zero
    // data.
    lastP = -1;
    lastPred = 0.0;
    data = 0.0;
    for (double pred = 0.0; pred < 20.0;
         pred += std::numeric_limits<double>::min() + 1.01*pred) {
        predErr = origErr;
        double newP = jointProb->eval(data, pred, predErr,0);
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,lastP) << "Not monotonic for"
                              << " data: " << data
                              << " prediction: " << pred
                              << " error: " << predErr
                              << " prob: " << newP << " (" << lastP << ")";
        lastP = newP;
        lastPred = pred;
    }

}

TEST(jointProbability, OA2021_UpdateErr_Continuity) {

    std::unique_ptr<JointProbability::JointProbabilityBase>
        jointProb(JointProbability::makeJointProbability(
                      "BarlowLLH_BANFF_OA2021"));
    double data = 20.0;
    double origPred = std::max(1.0, data);
    double overSample = 10.0;
    double fracErr = 1.0/sqrt(overSample*origPred);
    double origErr = fracErr * origPred;;
    double predErr = fracErr * origPred;;

    // Check the continuity for values less than the observation
    double lastP = -1;
    double lastPred = 0.0;
    for (double pred = origPred; pred != lastPred; pred *= 0.9) {
        predErr = fracErr*pred;
        double newP = jointProb->eval(data, pred, predErr,0);
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,(0.99999)*lastP)
            << "Not monotonic for"
            << " data: " << data
            << " prediction: " << pred
            << " error: " << predErr
            << " prob: " << newP << " (" << lastP << ")";
        lastP = newP;
        lastPred = pred;
    }

    // Check the continuity for values greater than the observation
    lastP = -1;
    lastPred = 0.0;
    for (double pred = origPred; pred < 20*origPred; pred *= 1.01) {
        predErr = fracErr*pred;
        double newP = jointProb->eval(data, pred, predErr,0);
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,lastP) << "Not monotonic for"
                              << " data: " << data
                              << " prediction: " << pred
                              << " error: " << predErr
                              << " prob: " << newP << " (" << lastP << ")";
        lastP = newP;
        lastPred = pred;
    }

    // Check the continuity for values greater than the observation with zero
    // data.
    lastP = -1;
    lastPred = 0.0;
    data = 0.0;
    for (double pred = 0.0; pred < 20.0;
         pred += std::numeric_limits<double>::min() + 1.01*pred) {
        predErr = fracErr*pred;
        double newP = jointProb->eval(data, pred, predErr,0);
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,lastP) << "Not monotonic for"
                              << " data: " << data
                              << " prediction: " << pred
                              << " error: " << predErr
                              << " prob: " << newP << " (" << lastP << ")";
        lastP = newP;
        lastPred = pred;
    }

}

TEST(jointProbability, BarlowLLH_Continuity) {

    std::unique_ptr<JointProbability::JointProbabilityBase>
        jointProb(JointProbability::makeJointProbability(
                      "BarlowLLH"));
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
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,lastP) << "Not monotonic for"
                              << " data: " << data
                              << " prediction: " << pred
                              << " error: " << predErr
                              << " prob: " << newP << " (" << lastP << ")"
                              << " diff: " << newP-lastP;
        lastP = newP;
        lastPred = pred;
    }

    // Check the value for a prediction of zero.
    double zeroP = jointProb->eval(data, 0.0, 0.0, 0);
    ASSERT_FALSE(std::isnan(zeroP)) << "Not a number for"
                                    << " data: " << data
                                    << " prediction: " << 0.0
                                    << " error: " << 0.0;
    ASSERT_GE(zeroP,lastP) << "Not monotonic for"
                           << " data: " << data
                           << " prediction: " << 0.0
                           << " error: " << 0.0
                           << " prob: " << zeroP << " (" << lastP << ")"
                           << " diff: " << zeroP-lastP;


    // Check the continuity for values greater than the observation
    lastP = -1;
    lastPred = 0.0;
    for (double pred = origPred; pred < 20*origPred; pred *= 1.01) {
        predErr = fracErr*pred;
        double newP = jointProb->eval(data, pred, predErr,0);
        ASSERT_FALSE(std::isnan(newP)) << "Not a number for"
                                       << " data: " << data
                                       << " prediction: " << pred
                                       << " error: " << predErr;
        ASSERT_GE(newP,lastP) << "Not monotonic for"
                              << " data: " << data
                              << " prediction: " << pred
                              << " error: " << predErr
                              << " prob: " << newP << " (" << lastP << ")"
                              << " diff: " << newP-lastP;
        lastP = newP;
        lastPred = pred;
    }

}
