#include "ExternalConstraints.h"
#include "GaussianConstraint.h"
#include "ParametersManager.h"

#include <TFormula.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace {
  JsonType constraintConfig(const std::string& formula_){
    return {{"name", "measurement"}, {"formula", formula_},
        {"parameters", {
            {{"name", "b"}, {"parameterSetName", "second"}, {"parameterIndex", 0}},
            {{"name", "aAgain"}, {"parameterSetName", "first"}, {"parameterName", "a"}},
            {{"name", "a"}, {"parameterSetName", "first"}, {"parameterIndex", 0}}}},
        {"type", {{"name", "gaussian"}, {"mean", .7}, {"stdDev", .4}}}};
  }

  void makeParameters(ParametersManager& manager_){
    const auto makeSet = [](const std::string& name_, const std::string& parameter_, double prior_){
      return JsonType{{"name", name_}, {"isEnabled", true}, {"nominalStepSize", .1},
          {"parameterDefinitions", {{{"name", parameter_}, {"priorValue", prior_}, {"priorType", "Flat"}}}}};
    };
    manager_.configure(ConfigReader(JsonType{{"parameterSetList", {
        makeSet("first", "a", .3), makeSet("second", "b", .8)}}}));
    manager_.initialize();
  }
}

TEST(ExternalConstraintGradient, NonlinearFormulaRepeatedAliasesSubsetsAndParameterFlags){
#ifndef GUNDAM_TEST_HAS_CLAD
  GTEST_SKIP() << "ROOT was built without Clad";
#endif
  ParametersManager manager;
  makeParameters(manager);
  auto& a = manager.getParameterSetsList()[0].getParameterList()[0];
  auto& b = manager.getParameterSetsList()[1].getParameterList()[0];
  GaussianConstraint constraint;
  constraint.configure(ConfigReader(constraintConfig("sin([a])+[aAgain]*[b]*[b]")));
  constraint.initialize();
  constraint.bindParameters(manager);
  constraint.validateGradient();
  a.setIsFixed(true);
  a.setIsPenaltyDisabled(true); // Neither flag removes an independent measurement.
  ParameterGradient gradient;
  gradient.resetValue(&a); gradient.resetValue(&b);
  constraint.evalGradient(gradient);
  const double residualScale = 2. * (std::sin(.3) + .3 * .8 * .8 - .7) / (.4 * .4);
  EXPECT_NEAR(residualScale * (std::cos(.3) + .8 * .8), gradient.getValue(&a), 1E-12);
  EXPECT_NEAR(residualScale * 2. * .3 * .8, gradient.getValue(&b), 1E-12);
  for( auto* parameter : {&a, &b} ){
    const double point = parameter->getParameterValue();
    parameter->setParameterValue(point + 1E-6);
    const double upper = constraint.eval();
    parameter->setParameterValue(point - 1E-6);
    const double lower = constraint.eval();
    parameter->setParameterValue(point);
    EXPECT_NEAR(gradient.getValue(parameter), (upper - lower) / 2E-6, 1E-8);
  }
  ParameterGradient subset;
  subset.resetValue(&a);
  subset.addValue(&a, 5.);
  constraint.evalGradient(subset);
  EXPECT_EQ(1U, subset.values.size());
  EXPECT_NEAR(5. + gradient.getValue(&a), subset.getValue(&a), 1E-12);
}

TEST(ExternalConstraintGradient, MultipleMeasurementsDisabledAndConstantConstraints){
#ifndef GUNDAM_TEST_HAS_CLAD
  GTEST_SKIP() << "ROOT was built without Clad";
#endif
  ParametersManager manager;
  makeParameters(manager);
  const auto& a = manager.getParameterSetsList()[0].getParameterList()[0];
  auto first = constraintConfig("[a]+[aAgain]+[b]");
  auto second = first;
  second["name"] = "otherMeasurement";
  second["type"]["mean"] = -.2;
  const JsonType constant{{"name", "constant"}, {"formula", "2."}, {"parameters", JsonType::array()},
      {"type", {{"name", "gaussian"}, {"mean", 1.}, {"stdDev", 2.}}}};
  ExternalConstraints constraints;
  constraints.configure(ConfigReader(JsonType{{"isEnabled", true}, {"constraintsList", {
      first, second, constant, {{"name", "disabled"}, {"isEnabled", false}}}}}));
  constraints.initialize();
  constraints.bindParameters(manager);
  constraints.validateGradient();
  ParameterGradient gradient;
  gradient.resetValue(&a);
  constraints.evalGradient(gradient);
  EXPECT_NEAR(2. * (1.4 - .7) / .16 * 2. + 2. * (1.4 + .2) / .16 * 2., gradient.getValue(&a), 1E-12);
  constraints.configure(ConfigReader(JsonType{{"isEnabled", false}}));
  constraints.initialize();
  constraints.validateGradient();
  const double before = gradient.getValue(&a);
  constraints.evalGradient(gradient);
  EXPECT_DOUBLE_EQ(before, gradient.getValue(&a));
}

TEST(ExternalConstraintGradient, RejectsUnsupportedGenerationAndNonFiniteDerivatives){
#ifndef GUNDAM_TEST_HAS_CLAD
  GTEST_SKIP() << "ROOT was built without Clad";
#endif
  ParametersManager manager;
  makeParameters(manager);
  auto& a = manager.getParameterSetsList()[0].getParameterList()[0];
  GaussianConstraint constraint;
  constraint.configure(ConfigReader(constraintConfig("TMath::BesselJ0([a])+[aAgain]+[b]")));
  constraint.initialize();
  constraint.bindParameters(manager);
  EXPECT_TRUE(std::isfinite(constraint.eval()));
  try{
    constraint.validateGradient();
    FAIL() << "Numerical fallback was accepted";
  }
  catch( const std::runtime_error& error ){
    EXPECT_NE(std::string::npos, std::string(error.what()).find("measurement"));
  }
  constraint.configure(ConfigReader(constraintConfig("sqrt([a])+[aAgain]+[b]")));
  constraint.initialize();
  constraint.bindParameters(manager);
  constraint.validateGradient();
  ParameterGradient gradient;
  gradient.resetValue(&a);
  a.setParameterValue(0.);
  EXPECT_TRUE(std::isfinite(constraint.eval()));
  EXPECT_THROW(constraint.evalGradient(gradient), std::runtime_error);
  a.setParameterValue(-1.);
  EXPECT_TRUE(std::isinf(constraint.eval()));
  EXPECT_THROW(constraint.evalGradient(gradient), std::runtime_error);
}
