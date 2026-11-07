#include "FormulaEvaluator.h"
#include "FormulaUtils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <future>
#include <limits>
#include <stdexcept>
#include <vector>

TEST(FormulaEvaluator, ReorderedAndSparseInputs){
  FormulaEvaluator evaluator;
  evaluator.setFormulaStr("[delta]+[alpha]*(.056*[q3]-41.6)");
  evaluator.setInputNames({"q3", "unused", "delta", "alpha"});
  const std::vector<double> inputs{500., std::numeric_limits<double>::quiet_NaN(), 1., 2.};
  const double expected = 1. + 2. * (.056 * 500. - 41.6);
  EXPECT_NEAR(evaluator.eval(inputs), expected, 1E-12);
  EXPECT_NEAR(evaluator.evalWithInputs([&](int index_){
    EXPECT_NE(index_, 1); // The loader need not read unreferenced columns.
    return inputs.at(index_);
  }), expected, 1E-12);
}

TEST(FormulaEvaluator, LoaderExpressionConversion){
  FormulaEvaluator evaluator;
  evaluator.setFormulaStr(FormulaUtils::convertBareVariablesToFormulaParameters("offset+scale*TMath::Abs([q3])"));
  evaluator.setInputNames({"scale", "q3", "offset"});
  const std::vector<double> inputs{2., -3., 1.};
  EXPECT_DOUBLE_EQ(evaluator.eval(inputs), 7.);
  EXPECT_DOUBLE_EQ(evaluator.evalWithInputs([&](int index_){ return inputs.at(index_); }), 7.);
}

TEST(FormulaEvaluator, SharedConstantsAndConcurrentEvaluation){
  FormulaEvaluator evaluator;
  evaluator.setFormulaStr("[scale]*[X]");
  evaluator.setInputNames({"scale"}, {"X"});
  evaluator.getFormula().SetParameter("X", 2.5);
  const double storedScale = evaluator.getFormula().GetParameter("scale");
  EXPECT_DOUBLE_EQ(evaluator.eval({1.}), 2.5);
  std::vector<std::future<double>> responses;
  for( int i = 0 ; i < 16 ; ++i ){
    responses.emplace_back(std::async(std::launch::async, [&, i]{
      return evaluator.eval({double(i)}) + evaluator.evalWithInputs([&](int){ return double(i); });
    }));
  }
  for( int i = 0 ; i < int(responses.size()) ; ++i ){ EXPECT_DOUBLE_EQ(responses[i].get(), 5. * i); }
  EXPECT_DOUBLE_EQ(evaluator.getFormula().GetParameter("X"), 2.5);
  EXPECT_DOUBLE_EQ(evaluator.getFormula().GetParameter("scale"), storedScale);
}

TEST(FormulaEvaluator, ConstantsWithoutInputs){
  FormulaEvaluator evaluator;
  evaluator.setFormulaStr("2.");
  evaluator.setInputNames({});
  EXPECT_DOUBLE_EQ(evaluator.eval({}), 2.);
  EXPECT_DOUBLE_EQ(evaluator.evalWithInputs([](int){ ADD_FAILURE() << "Constant expression read an input"; return 0.; }), 2.);
  EXPECT_THROW(static_cast<void>(evaluator.eval({1.})), std::runtime_error);
}

TEST(FormulaEvaluator, InvalidBindingsAndReconfiguration){
  FormulaEvaluator evaluator;
  evaluator.setFormulaStr("[first]");
  EXPECT_THROW(static_cast<void>(evaluator.eval({1.})), std::runtime_error);
  EXPECT_THROW(evaluator.setInputNames({"other"}), std::runtime_error);
  EXPECT_THROW(static_cast<void>(evaluator.eval({1.})), std::runtime_error);
  EXPECT_THROW(evaluator.setInputNames({"first", "first"}), std::runtime_error);
  EXPECT_THROW(static_cast<void>(evaluator.eval({1., 2.})), std::runtime_error);
  evaluator.setInputNames({"first"});
  EXPECT_DOUBLE_EQ(evaluator.eval({2.}), 2.);
  evaluator.setFormulaStr("[next]");
  EXPECT_THROW(static_cast<void>(evaluator.eval({3.})), std::runtime_error);
  evaluator.setInputNames({"next"});
  EXPECT_DOUBLE_EQ(evaluator.eval({3.}), 3.);
  EXPECT_THROW(evaluator.setFormulaStr("x[0]"), std::runtime_error);
  EXPECT_THROW(static_cast<void>(evaluator.eval({3.})), std::runtime_error);
}


TEST(FormulaEvaluator, AnalyticGradientReorderedInputsAndConstants){
#ifndef R__HAS_CLAD
  GTEST_SKIP() << "ROOT was built without Clad";
#endif
  FormulaEvaluator evaluator;
  evaluator.setFormulaStr("[offset]+[scale]*sin([angle])+[scale]*[scale]*[C]");
  evaluator.setInputNames({"scale", "unused", "offset", "angle"}, {"C"});
  evaluator.getFormula().SetParameter("C", 2.5);
  evaluator.prepareGradient();
  const std::vector<double> inputs{1.3, std::numeric_limits<double>::quiet_NaN(), -.2, .4};
  for( int index : {0, 2, 3} ){
    auto up = inputs, down = inputs;
    up[index] += 1e-5;
    down[index] -= 1e-5;
    EXPECT_NEAR((evaluator.eval(up) - evaluator.eval(down)) / 2e-5,
                evaluator.evalGradient(inputs, index), 1e-8);
  }
  EXPECT_DOUBLE_EQ(0., evaluator.evalGradient(inputs, 1));
  EXPECT_DOUBLE_EQ(0., evaluator.evalGradient(inputs, -1));
  EXPECT_DOUBLE_EQ(0., evaluator.evalGradient(inputs, 4));
  EXPECT_DOUBLE_EQ(2.5, evaluator.getFormula().GetParameter("C"));
  // Constants remain read from the configured formula, even after preparation.
  evaluator.getFormula().SetParameter("C", 3.);
  EXPECT_NEAR(std::sin(.4) + 6. * 1.3, evaluator.evalGradient(inputs, 0), 1e-12);
}

TEST(FormulaEvaluator, AnalyticGradientConcurrentCopiesAndReconfiguration){
#ifndef R__HAS_CLAD
  GTEST_SKIP() << "ROOT was built without Clad";
#endif
  FormulaEvaluator evaluator;
  evaluator.setFormulaStr("[a]*[a]+[a]*[b]");
  evaluator.setInputNames({"b", "a"});
  evaluator.prepareGradient();
  FormulaEvaluator copy = evaluator;
  std::vector<std::future<void>> jobs;
  for( int i = 0; i < 8; ++i ){
    jobs.emplace_back(std::async(std::launch::async, [&, i]{
      for( int j = 0; j < 100; ++j ){
        const double a = .1 * i + .01 * j;
        const double b = -.3 * i + .02 * j;
        const auto& target = i % 2 ? evaluator : copy;
        EXPECT_NEAR(2. * a + b, target.evalGradient({b, a}, 1), 1e-12);
        EXPECT_NEAR(a, target.evalGradient({b, a}, 0), 1e-12);
        EXPECT_NEAR(a * a + a * b, target.eval({b, a}), 1e-12);
      }
    }));
  }
  for( auto& job : jobs ){ job.get(); }
  evaluator.setFormulaStr("[next]*[next]*[next]");
  evaluator.setInputNames({"next"});
  EXPECT_THROW(static_cast<void>(evaluator.evalGradient({2.}, 0)), std::runtime_error);
  evaluator.prepareGradient();
  EXPECT_DOUBLE_EQ(12., evaluator.evalGradient({2.}, 0));
  EXPECT_DOUBLE_EQ(7., copy.evalGradient({3., 2.}, 1));
  // Rebinding changes the returned input order without changing the formula.
  copy.setInputNames({"a", "b"});
  EXPECT_DOUBLE_EQ(7., copy.evalGradient({2., 3.}, 0));
}

TEST(FormulaEvaluator, AnalyticGradientConstantAndInvalidInputs){
#ifndef R__HAS_CLAD
  GTEST_SKIP() << "ROOT was built without Clad";
#endif
  FormulaEvaluator evaluator;
  evaluator.setFormulaStr("2.");
  evaluator.setInputNames({"unused"});
  evaluator.prepareGradient();
  EXPECT_DOUBLE_EQ(0., evaluator.evalGradient({1.}, 0));
  EXPECT_THROW(static_cast<void>(evaluator.evalGradient({}, 0)), std::runtime_error);
  evaluator.setFormulaStr("[C]");
  evaluator.setInputNames({}, {"C"});
  evaluator.getFormula().SetParameter("C", 3.);
  evaluator.prepareGradient();
  EXPECT_DOUBLE_EQ(0., evaluator.evalGradient({}, 0));
  EXPECT_DOUBLE_EQ(3., evaluator.eval({}));
}


TEST(FormulaEvaluator, AnalyticGradientRejectsNumericalFallback){
  FormulaEvaluator evaluator;
  evaluator.setFormulaStr("TMath::BesselJ0([a])");
  evaluator.setInputNames({"a"});
  // The external implementation has no body available to Clad. Its numerical
  // fallback must be rejected rather than advertised as an analytic gradient.
  EXPECT_THROW(evaluator.prepareGradient(), std::runtime_error);
  EXPECT_TRUE(std::isfinite(evaluator.eval({.3})));
  EXPECT_THROW(static_cast<void>(evaluator.evalGradient({.3}, 0)), std::runtime_error);
}
