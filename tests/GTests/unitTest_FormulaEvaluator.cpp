#include "FormulaEvaluator.h"
#include "FormulaUtils.h"

#include <gtest/gtest.h>

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
