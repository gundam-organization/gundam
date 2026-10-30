#include "gtest/gtest.h"

#include <string>
#include <vector>
#include <limits>

#include "JointProbability.h"
#include "BarlowBeeston.h"
#include "BarlowBeestonLiteGaussian.h"
#include "BarlowBeestonLitePoisson.h"

namespace {
    void configurePredictionSample(Sample& sample_, const std::string& name_) {
        JsonType config{
            {"name", name_},
            {"binning", {{"binningDefinition", {{{"name", "X"}, {"edges", {0, 1}}}}}}}
        };
        sample_.configure(ConfigReader(config));
        sample_.setIndex(0);
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
        "BarlowBeeston",
        "BarlowBeestonLiteGaussian",
        "BarlowBeestonLitePoisson",
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


TEST(jointProbability, ExactBarlowBeestonSingleSource) {
    using BB = JointProbability::BarlowBeeston;
    for (double n : {0., 1., 20., 1000.}) {
        for (double a : {1., 20., 1000.}) {
            for (double p : {.001, .5, 2., 100.}) {
                const auto result = BB::evaluateSources(n, {{a, p}});
                const double fitted = (n+a)/(1+p);
                EXPECT_NEAR(result.fittedCounts[0], fitted, fitted*2.e-14);
                const double expected = 2*(p*fitted-n + (n == 0 ? 0 : n*std::log(n/(p*fitted)))
                    + fitted-a + a*std::log(a/fitted));
                EXPECT_NEAR(result.value, expected, 1.e-10*(1+expected));
            }
        }
    }
}

TEST(jointProbability, ExactBarlowBeestonEmptySourcesAndStationarity) {
    using BB = JointProbability::BarlowBeeston;
    // An empty template with the largest scale can acquire a positive mean.
    const auto boundary = BB::evaluateSources(10, {{0, 2}, {1, .1}});
    EXPECT_NEAR(boundary.prediction, 10./1.5, 1.e-13);
    EXPECT_GT(boundary.fittedCounts[0], 0);
    EXPECT_NEAR(boundary.fittedCounts[1], 1./.95, 1.e-13);
    EXPECT_DOUBLE_EQ(BB::evaluateSources(0, {{0, 2}}).value, 0);
    EXPECT_TRUE(std::isinf(BB::evaluateSources(10, {}).value));
    const std::vector<BB::Source> sources{{12, .2}, {3, 4}, {0, .8}};
    for (double n : {0., 1., 30., 100.}) {
        const auto result = BB::evaluateSources(n, sources, 2);
        for (size_t i = 0; i < sources.size(); ++i) {
            if (sources[i].count == 0) continue;
            const double a = result.fittedCounts[i];
            const double score = sources[i].scale*(1-n/result.prediction)+1-sources[i].count/a;
            EXPECT_NEAR(score, 0, 1.e-12);
            auto up = sources, down = sources;
            const double step = sources[i].scale*1.e-5;
            up[i].scale += step;
            down[i].scale -= step;
            EXPECT_NEAR(result.scaleGradients[i],
                (BB::evaluateSources(n, up, 2).value-BB::evaluateSources(n, down, 2).value)/(2*step), 1.e-6);
        }
    }
    BB probability;
    EXPECT_THROW((void)probability.eval(10, 12, 2, 0), std::logic_error);
    EXPECT_THROW((void)BB::evaluateSources(1, {{1.5, 1}}), std::invalid_argument);
}

TEST(jointProbability, LiteGaussianMatchesValidatedBanffAndHasAnalyticGradients) {
    using Gaussian = JointProbability::BarlowBeestonLiteGaussian;
    std::unique_ptr<JointProbability::JointProbabilityBase> legacy(
        JointProbability::makeJointProbability("BarlowLLH_BANFF_OA2021"));
    for (double n : {0., 1., 20., 1.e6}) {
        for (double mu : {.1, 2., 20., 1000.}) {
            for (double sigma : {0., .1, 2., 100.}) {
                const auto result = Gaussian::evaluate(n, mu, sigma);
                EXPECT_NEAR(result.value, legacy->eval(n, mu, sigma*sigma, 0), 2.e-9*(1+result.value));
                const double step = mu*1.e-5;
                EXPECT_NEAR(result.predictionGradient,
                    (Gaussian::evaluate(n, mu+step, sigma).value-Gaussian::evaluate(n, mu-step, sigma).value)/(2*step),
                    1.e-4*(1+std::abs(result.predictionGradient)));
                if (sigma == 0) continue;
                const double errorStep = sigma*1.e-5;
                EXPECT_NEAR(result.uncertaintyGradient,
                    (Gaussian::evaluate(n, mu, sigma+errorStep).value-Gaussian::evaluate(n, mu, sigma-errorStep).value)/(2*errorStep),
                    1.e-4*(1+std::abs(result.uncertaintyGradient)));
            }
        }
    }
}

TEST(jointProbability, LiteGaussianBoundariesAndSmallDisplacements) {
    using Gaussian = JointProbability::BarlowBeestonLiteGaussian;
    EXPECT_DOUBLE_EQ(Gaussian::evaluate(0, 0, 0).value, 0);
    EXPECT_TRUE(std::isinf(Gaussian::evaluate(1, 0, 0).value));
    EXPECT_TRUE(std::isinf(Gaussian::evaluate(-1, 2, 1).value));
    EXPECT_DOUBLE_EQ(Gaussian::evaluate(0, 1, 2).value, .25);
    for (double n : {1.e-100, 1., 1.e100}) {
        const double sigma = std::sqrt(n);
        EXPECT_DOUBLE_EQ(Gaussian::evaluate(n, n, sigma).value, 0);
        const double mu = n*(1+1.e-10);
        const double expected = (mu-n)*(mu-n)/(n+sigma*sigma);
        EXPECT_NEAR(Gaussian::evaluate(n, mu, sigma).value, expected, expected*1.e-8);
    }
    EXPECT_TRUE(std::isfinite(Gaussian::evaluate(10, 1.e-300, 1).value));
    EXPECT_TRUE(std::isfinite(Gaussian::evaluate(10, 0, 1.e-300).value));
    EXPECT_NEAR(Gaussian::evaluate(10, 2, 1.e-300).predictionGradient, -8., 1.e-13);
    EXPECT_TRUE(std::isfinite(Gaussian::evaluate(10, 1.e300, 1.e300).value));
}


TEST(jointProbability, LiteLikelihoodsUseCurrentErrorsAndAnalyticSampleGradients) {
    Sample model, data;
    configurePredictionSample(model, "model");
    configurePredictionSample(data, "data");
    SamplePair pair{&model, &data};
    auto& bin = model.getHistogram().getBinContentList()[0];
    data.getHistogram().getBinContentList()[0].sumWeights = 15;
    for (const auto& type : {"BarlowBeestonLiteGaussian", "BarlowBeestonLitePoisson"}) {
        SCOPED_TRACE(type);
        std::unique_ptr<JointProbability::JointProbabilityBase> probability(
            JointProbability::makeJointProbability(type));
        probability->configure(ConfigReader(JsonType::object()));
        bin = {25, 3};
        const double original = probability->eval(pair, 0);
        bin = {25, 11};
        EXPECT_NE(probability->eval(pair, 0), original);
        EXPECT_DOUBLE_EQ(probability->eval(pair, 0), probability->eval(15, 25, 11, 0));
        EXPECT_DOUBLE_EQ(probability->evalPredictionGradient(pair, 0), probability->evalPredictionGradient(15, 25, 11, 0));
        EXPECT_DOUBLE_EQ(probability->evalSqrtSumSqWeightsGradient(pair, 0), probability->evalSqrtSumSqWeightsGradient(15, 25, 11, 0));
        EXPECT_DOUBLE_EQ(probability->evalWithPrediction(pair, 15, 30, 7, 0), probability->eval(15, 30, 7, 0));
        EXPECT_DOUBLE_EQ(bin.sumWeights, 25);
        EXPECT_DOUBLE_EQ(bin.sqrtSumSqWeights, 11);
        probability->configure(ConfigReader(JsonType{{"ignoreBinsWithZeroPredictionAtPrior", true}}));
        bin = {0, 0};
        probability->updateZeroPredictionBinsAtPriorFlag({pair});
        bin = {25, 11};
        EXPECT_DOUBLE_EQ(probability->eval(pair, 0), 0);
        EXPECT_DOUBLE_EQ(probability->evalWithPrediction(pair, 15, 30, 7, 0), 0);
        EXPECT_DOUBLE_EQ(probability->evalPredictionGradient(pair, 0), 0);
        EXPECT_DOUBLE_EQ(probability->evalSqrtSumSqWeightsGradient(pair, 0), 0);
    }
}

TEST(jointProbability, LitePoissonMatchesExactSingleTemplate) {
    using Lite = JointProbability::BarlowBeestonLitePoisson;
    using Exact = JointProbability::BarlowBeeston;
    for (double count : {1., 5., 100., 1.e6}) {
        for (double scale : {.01, .2, 2., 100.}) {
            for (double n : {0., 1., 10., 1000.}) {
                SCOPED_TRACE(::testing::Message() << "n=" << n << ", a=" << count << ", scale=" << scale);
                const double mu = count*scale, sigma = std::sqrt(count)*scale;
                const auto actual = Lite::evaluate(n, mu, sigma);
                const auto expected = Exact::evaluateSources(n, {{count, scale}});
                EXPECT_NEAR(actual.value, expected.value, 2.e-12*(1+expected.value));
                // Along a source normalization variation, mu and sigma both vary.
                const double scaleDerivative = count*actual.predictionGradient + std::sqrt(count)*actual.uncertaintyGradient;
                EXPECT_NEAR(scaleDerivative, expected.scaleGradients[0], 2.e-10*(1+std::abs(expected.scaleGradients[0])));
            }
        }
    }
}

TEST(jointProbability, LitePoissonAnalyticGradients) {
    using Lite = JointProbability::BarlowBeestonLitePoisson;
    for (double n : {0., 1., 20., 1.e6}) {
        for (double mu : {.1, 2., 20., 1000.}) {
            for (double sigma : {.01, 2., 100.}) {
                SCOPED_TRACE(::testing::Message() << "n=" << n << ", mu=" << mu << ", sigma=" << sigma);
                const auto actual = Lite::evaluate(n, mu, sigma);
                const double h = mu*1.e-5, k = sigma*1.e-5;
                const double predDerivative = (Lite::evaluate(n, mu+h, sigma).value-Lite::evaluate(n, mu-h, sigma).value)/(2*h);
                const double errDerivative = (Lite::evaluate(n, mu, sigma+k).value-Lite::evaluate(n, mu, sigma-k).value)/(2*k);
                EXPECT_GE(actual.value, 0);
                EXPECT_NEAR(actual.predictionGradient, predDerivative, 2.e-5*(1+std::abs(predDerivative)));
                EXPECT_NEAR(actual.uncertaintyGradient, errDerivative, 2.e-5*(1+std::abs(errDerivative)));
            }
        }
    }
}

TEST(jointProbability, LitePoissonLimitsAndStability) {
    using Lite = JointProbability::BarlowBeestonLitePoisson;
    EXPECT_DOUBLE_EQ(Lite::evaluate(0, 0, 0).value, 0);
    EXPECT_TRUE(std::isinf(Lite::evaluate(1, 0, 0).value));
    EXPECT_TRUE(std::isinf(Lite::evaluate(0, 0, 1).value));
    EXPECT_TRUE(std::isinf(Lite::evaluate(-1, 2, 1).value));
    EXPECT_TRUE(std::isinf(Lite::evaluate(1, 2, -1).value));
    EXPECT_NEAR(Lite::evaluate(0, 5, 0).value, 10, 1.e-14);
    EXPECT_NEAR(Lite::evaluate(3, 5, 0).predictionGradient, .8, 1.e-14);
    for (double n : {1.e-100, 1., 1.e100}) {
        for (double sigma : {std::sqrt(n)*.1, std::sqrt(n), std::sqrt(n)*10}) {
            EXPECT_DOUBLE_EQ(Lite::evaluate(n, n, sigma).value, 0);
            for (double mu : {std::nextafter(n, 0.), std::nextafter(n, 2*n), n*(1+1.e-10)}) {
                const long double difference = static_cast<long double>(mu)-n;
                const double expected = double(difference*difference/(n+static_cast<long double>(sigma)*sigma));
                const auto actual = Lite::evaluate(n, mu, sigma);
                EXPECT_GT(actual.value, 0);
                EXPECT_NEAR(actual.value, expected, expected*1.e-9);
            }
        }
    }
    for (const auto& point : std::vector<std::vector<double>>{{10, 1.e-300, 1}, {10, 1.e300, 1.e300},
            {10, 2, 1.e-300}, {10, 2, 1.e300}, {0, 1.e300, 1.e300}}) {
        const auto actual = Lite::evaluate(point[0], point[1], point[2]);
        EXPECT_TRUE(std::isfinite(actual.value));
        EXPECT_TRUE(std::isfinite(actual.predictionGradient));
        EXPECT_TRUE(std::isfinite(actual.uncertaintyGradient));
        EXPECT_GE(actual.value, 0);
    }
    // Infinite MC-statistics limit.
    EXPECT_NEAR(Lite::evaluate(10, 2, 1.e-100).value, 2*(2-10+10*std::log(5)), 1.e-11);
}
