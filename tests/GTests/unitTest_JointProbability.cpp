#include "gtest/gtest.h"

#include <string>
#include <vector>
#include <limits>

#include "JointProbability.h"

namespace {
    void configurePredictionSample(Sample& sample_, const std::string& name_) {
        JsonType config{
            {"name", name_},
            {"binning", {{"binningDefinition", {{{"name", "X"}, {"edges", {0, 1}}}}}}}
        };
        sample_.configure(ConfigReader(config));
    }
}

TEST(jointProbability, PredictionValuesAreIndependentOfHostHistogram) {
    Sample model, data;
    configurePredictionSample(model, "model");
    configurePredictionSample(data, "data");
    SamplePair pair{&model, &data};
    model.getHistogram().getBinContentList()[0] = {20., 3.};
    data.getHistogram().getBinContentList()[0].sumWeights = 15.;
    std::unique_ptr<JointProbability::JointProbabilityBase> probability(
        JointProbability::makeJointProbability("PoissonLLH"));
    EXPECT_DOUBLE_EQ(probability->evalWithPrediction(pair, 17., 25., 11., 0),
                     probability->eval(17., 25., 11., 0));
    EXPECT_DOUBLE_EQ(model.getHistogram().getBinContentList()[0].sumWeights, 20.);
    EXPECT_DOUBLE_EQ(data.getHistogram().getBinContentList()[0].sumWeights, 15.);
}

TEST(jointProbability, PredictionPreservesBarlowBeestonUncertaintySettings) {
    for (bool fixedErrors : {false, true}) {
        SCOPED_TRACE(fixedErrors);
        Sample model, data;
        configurePredictionSample(model, "model");
        configurePredictionSample(data, "data");
        SamplePair pair{&model, &data};
        auto& content = model.getHistogram().getBinContentList()[0];
        content = {20., 3.};
        data.getHistogram().getBinContentList()[0].sumWeights = 15.;
        std::unique_ptr<JointProbability::JointProbabilityBase> probability(
            JointProbability::makeJointProbability("BarlowLLH_BANFF_OA2021"));
        probability->configure(ConfigReader(JsonType{{"BBNoUpdateWeights", fixedErrors}}));
        // The nominal MC uncertainty is captured before predictions vary.
        EXPECT_TRUE(std::isfinite(probability->eval(pair, 0)));
        content = {25., 11.};
        const double expected = probability->eval(pair, 0);
        // A detached prediction must use its supplied uncertainty, or the
        // cached nominal uncertainty when BBNoUpdateWeights requests it.
        content = {20., 3.};
        EXPECT_DOUBLE_EQ(probability->evalWithPrediction(pair, 15., 25., 11., 0), expected);
        EXPECT_DOUBLE_EQ(probability->evalWithPrediction(pair, 15., 25., 11., 0),
                         probability->eval(15., 25., fixedErrors ? 9. : 121., 0));
        EXPECT_DOUBLE_EQ(content.sumWeights, 20.);
        EXPECT_DOUBLE_EQ(content.sqrtSumSqWeights, 3.);
    }
}

TEST(jointProbability, PredictionPreservesDetectorUncertaintyFromSampleName) {
    for (const auto& name : {"SFGD FHC 0p", "SFGD RHC Nn", "WAGASCI #0pi PM-BM", "other"}) {
        SCOPED_TRACE(name);
        Sample model, data;
        configurePredictionSample(model, name);
        configurePredictionSample(data, "data");
        SamplePair pair{&model, &data};
        auto& content = model.getHistogram().getBinContentList()[0];
        content = {25., 11.};
        data.getHistogram().getBinContentList()[0].sumWeights = 15.;
        std::unique_ptr<JointProbability::JointProbabilityBase> probability(
            JointProbability::makeJointProbability("BarlowLLH_BANFF_OA2021_SFGD"));
        const double expected = probability->eval(pair, 0);
        content = {20., 3.};
        EXPECT_DOUBLE_EQ(probability->evalWithPrediction(pair, 15., 25., 11., 0), expected);
        EXPECT_DOUBLE_EQ(content.sumWeights, 20.);
        EXPECT_DOUBLE_EQ(content.sqrtSumSqWeights, 3.);
    }
}

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
