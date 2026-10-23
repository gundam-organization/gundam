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

TEST(jointProbability, BarlowBeestonResolvesPoissonDevianceNearTheMinimum) {
    std::unique_ptr<JointProbability::JointProbabilityBase> probability(
        JointProbability::makeJointProbability("BarlowLLH_BANFF_OA2021"));
    for (double data : {1.e-6, 1., 655.82092932695639, 1.e8, 1.e12}) {
        EXPECT_DOUBLE_EQ(probability->eval(data, data, 0., 0), 0.);
        for (double relativeStep : {-1.e-4, -1.e-7, -1.e-10, 1.e-10, 1.e-7, 1.e-4}) {
            const double prediction = data * (1. + relativeStep);
            const long double delta = (static_cast<long double>(prediction) - data) / data;
            // Independent expansion. The first omitted
            // term is O(delta^6), below the tolerance for these probes.
            const long double expected = 2.L * data * delta * delta
                * (.5L - delta / 3.L + delta * delta / 4.L - delta * delta * delta / 5.L);
            const double actual = probability->eval(data, prediction, 0., 0);
            SCOPED_TRACE(::testing::Message() << "data=" << data << ", step=" << relativeStep);
            EXPECT_GT(actual, 0.);
            EXPECT_NEAR(actual, double(expected), double(expected) * 5.e-13);
        }
    }
}

TEST(jointProbability, BarlowBeestonResolvesMcUncertaintyNearTheMinimum) {
    std::unique_ptr<JointProbability::JointProbabilityBase> probability(
        JointProbability::makeJointProbability("BarlowLLH_BANFF_OA2021"));
    for (bool poissonOnly : {false, true}) {
        probability->configure(ConfigReader(JsonType{{"usePoissonLikelihood", poissonOnly}}));
        for (double data : {1.e-6, 1., 3.6494374068937234, 655.82092932695639, 1.e12}) {
            for (double varianceRatio : {1.e-8, .1, 1., 10., 1.e8}) {
                const double variance = data * varianceRatio;
                // Exact equality is one point on the same stable curve.
                EXPECT_DOUBLE_EQ(probability->eval(data, data, variance, 0), 0.);
                for (double prediction : {std::nextafter(data, 0.), std::nextafter(data, 2.*data),
                                          data*(1.-1.e-10), data*(1.+1.e-10),
                                          data*(1.-.009), data*(1.+.009)}) {
                    SCOPED_TRACE(::testing::Message() << "data=" << data << ", prediction="
                        << prediction << ", variance ratio=" << varianceRatio
                        << ", poisson only=" << poissonOnly);
                    // Independent reference: Newton iteration on the beta shift
                    // in long double, followed by a Poisson deviance series.
                    const long double pred = prediction;
                    const long double fractionalSq = static_cast<long double>(variance) / (pred*pred);
                    const long double linear = 1.L + pred*fractionalSq;
                    const long double constant = (pred-data)*fractionalSq;
                    long double shift = 0.L;
                    for (int i = 0; i < 12; ++i) {
                        shift -= (shift*shift + linear*shift + constant) / (2.L*shift + linear);
                    }
                    const long double residual = -shift*(1.L+shift)/(fractionalSq*data);
                    long double series = 0.L;
                    long double power = residual*residual;
                    for (int order = 2; order <= 16; ++order) {
                        series += (order%2 == 0 ? power : -power)/order;
                        power *= residual;
                    }
                    const long double expected = 2.L*data*series
                        + (poissonOnly ? 0.L : shift*shift/fractionalSq);
                    const double actual = probability->eval(data, prediction, variance, 0);
                    EXPECT_GT(actual, 0.);
                    EXPECT_NEAR(actual, double(expected), double(expected)*2.e-12);
                }
            }
        }
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
