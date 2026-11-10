#include "FitterEngine.h"
#include "RootMinimizer.h"
#include "GundamGlobals.h"

#include <Fit/ParameterSettings.h>
#include <TFile.h>
#include <TString.h>
#include <TSystem.h>
#include <TTree.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
  class RecordingMinimizer : public RootMinimizer {
  public:
    using RootMinimizer::RootMinimizer;
    struct Stage {
      std::string algorithm;
      double precision, tolerance;
      int strategy;
      ROOT::Math::Minimizer* minimizer;
      std::vector<ROOT::Fit::ParameterSettings> variables;
    };
    std::vector<Stage> stages;
    bool failSimplex{false};
    double simplexMinimum{std::numeric_limits<double>::infinity()};
    std::vector<double> simplexBestPoint;

    double evalFit(const double* values_) override {
      const auto& minimizer = getMinimizer();
      if( getMonitor().isEnabled and (stages.empty() or stages.back().minimizer != minimizer.get()) ){
        Stage stage{minimizer->Options().MinimizerAlgorithm(), minimizer->Precision(),
                    minimizer->Tolerance(), minimizer->Strategy(), minimizer.get(), {}};
        stage.variables.resize(minimizer->NDim());
        for( unsigned int i = 0; i < minimizer->NDim(); ++i ){
          minimizer->GetVariableSettings(i, stage.variables[i]);
        }
        stages.push_back(std::move(stage));
      }
      if( failSimplex and minimizer->Options().MinimizerAlgorithm() == "Simplex" ){
        throw std::runtime_error("Simplex objective failure");
      }
      const double value = RootMinimizer::evalFit(values_);
      if( minimizer->Options().MinimizerAlgorithm() == "Simplex" and value < simplexMinimum ){
        simplexMinimum = value;
        simplexBestPoint.assign(values_, values_ + minimizer->NDim());
      }
      return value;
    }
  };

  class RootMinimizerSimplex : public testing::Test {
  protected:
    void SetUp() override {
      TString name("gundam-simplex-test-");
      auto* handle = gSystem->TempFileName(name, nullptr, ".root");
      ASSERT_NE(handle, nullptr);
      std::fclose(handle);
      _path_ = name.Data();
      TFile file(_path_.c_str(), "RECREATE");
      double x = .5;
      TTree tree("events", "events");
      tree.Branch("X", &x, "X/D");
      for( int i = 0; i < 10; ++i ){ x = .05 + .09 * i; tree.Fill(); }
      tree.Write();
    }
    void TearDown() override { std::remove(_path_.c_str()); }

    void initializeEngine(FitterEngine& engine_, const std::string& likelihood_ = "LeastSquares",
                          const std::string& formula_ = "", bool eventInputs_ = false) {
      auto config = JsonType::parse(R"({
        "enablePca": false, "scaleParStepWithChi2Response": false,
        "generateSamplePlots": false, "generateOneSigmaPlots": false,
        "minimizerConfig": {"print_level": 0},
        "likelihoodInterfaceConfig": {
          "jointProbabilityConfig": {"type": "LeastSquares", "lsqPoissonianApproximation": true},
          "datasetList": [{"name": "input", "model": {"tree": "events"}}],
          "propagatorConfig": {
            "sampleSetConfig": {"sampleList": [{"name": "sample", "binning": {
              "binningDefinition": [{"name": "X", "edges": [0, 1]}]}}]},
            "parametersManagerConfig": {"parameterSetList": [{
              "name": "scales", "isEnabled": true, "nominalStepSize": 0.1,
              "parameterDefinitions": [
                {"name": "scale", "priorValue": 1.0, "stdDev": 0.2,
                 "parameterLimits": [0.2, 3.0],
                 "dialSetDefinitions": [{"dialType": "Normalization"}]},
                {"name": "nuisance", "priorValue": 1.0, "stdDev": 0.3},
                {"name": "fixed", "priorValue": 1.0, "stdDev": 0.4, "isFixed": true},
                {"name": "upper", "priorValue": 1.0, "stdDev": 0.5}
              ]}]} }
        }
      })");
      config["likelihoodInterfaceConfig"]["jointProbabilityConfig"]["type"] = likelihood_;
      auto& definitions = config["likelihoodInterfaceConfig"]["propagatorConfig"]
                               ["parametersManagerConfig"]["parameterSetList"][0]["parameterDefinitions"];
      for( auto& parameter : definitions ){
        parameter["dialSetDefinitions"] = JsonType::array({{{"dialType", "Normalization"}}});
        if( not formula_.empty() ){
          parameter["dialSetDefinitions"] = JsonType::array({{
              {"dialType", "Formula"}, {"options", {{"formulaStr", formula_}}},
              {"dialInputList", JsonType::array({{{"name", "value"},
                  {"parameterName", parameter.at("name")}}})}}});
          if( eventInputs_ ){
            auto& inputs = parameter["dialSetDefinitions"][0]["dialInputList"];
            inputs.insert(inputs.begin(), JsonType{{"name", "energy"}, {"eventVariable", "X"}});
            // Two named formula inputs can reference the same fit parameter.
            inputs.push_back({{"name", "alias"}, {"parameterName", parameter.at("name")}});
          }
        }
      }
      config["likelihoodInterfaceConfig"]["datasetList"][0]["model"]["filePathList"] = {_path_};
      engine_.configure(ConfigReader(config));
      engine_.getLikelihoodInterface().setDataType(LikelihoodInterface::DataType::Asimov);
      engine_.initialize();
    }
    std::string _path_;
  };
}

TEST_F(RootMinimizerSimplex, UsesSeparateAlgorithmsWithIdenticalPrecisionAndPreservedVariables) {
  FitterEngine engine;
  initializeEngine(engine);
  for( double precision : {1.1920928955078125e-7, 1e-6} ){
    RecordingMinimizer minimizer(&engine);
    minimizer.configure(ConfigReader(JsonType{
        {"enableSimplexBeforeMinimize", true},
        {"simplexStrategy", 2}, {"simplexToleranceLoose", 1000.},
        {"tolerance", .01}, {"strategy", 1}, {"print_level", 0}}));
    minimizer.initialize();
    auto* original = minimizer.getMinimizer().get();
    original->SetPrecision(precision);
    ASSERT_TRUE(original->SetVariableValue(0, -1.5));
    ASSERT_TRUE(original->FixVariable(2));
    ASSERT_TRUE(original->SetVariableLowerLimit(1, -2.));
    ASSERT_TRUE(original->SetVariableUpperLimit(3, 2.));
    std::vector<ROOT::Fit::ParameterSettings> variables(original->NDim());
    for( unsigned int i = 0; i < original->NDim(); ++i ){ original->GetVariableSettings(i, variables[i]); }
    minimizer.minimize();
    ASSERT_EQ(minimizer.stages.size(), 2);
    EXPECT_EQ(minimizer.stages[0].algorithm, "Simplex");
    EXPECT_EQ(minimizer.stages[1].algorithm, "Migrad");
    EXPECT_NE(minimizer.stages[0].minimizer, original);
    EXPECT_EQ(minimizer.stages[1].minimizer, original);
    EXPECT_EQ(minimizer.getMinimizer().get(), original);
    // ROOT may regularize the covariance while accepting the minimum (status 1).
    EXPECT_TRUE(original->Status() == 0 or original->Status() == 1);
    EXPECT_LT(original->Edm(), minimizer.getTargetEdm());
    EXPECT_LT(original->MinValue(), minimizer.getTargetEdm());
    for( const auto& stage : minimizer.stages ){
      EXPECT_EQ(stage.precision, precision);
      ASSERT_EQ(stage.variables.size(), variables.size());
      for( std::size_t i = 0; i < variables.size(); ++i ){
        EXPECT_EQ(stage.variables[i].Name(), variables[i].Name());
        EXPECT_EQ(stage.variables[i].IsFixed(), variables[i].IsFixed());
        EXPECT_EQ(stage.variables[i].HasLowerLimit(), variables[i].HasLowerLimit());
        EXPECT_EQ(stage.variables[i].HasUpperLimit(), variables[i].HasUpperLimit());
        if( variables[i].HasLowerLimit() ){ EXPECT_EQ(stage.variables[i].LowerLimit(), variables[i].LowerLimit()); }
        if( variables[i].HasUpperLimit() ){ EXPECT_EQ(stage.variables[i].UpperLimit(), variables[i].UpperLimit()); }
        if( not variables[i].IsFixed() ){ EXPECT_EQ(stage.variables[i].StepSize(), variables[i].StepSize()); }
      }
    }
    EXPECT_EQ(minimizer.stages[0].strategy, 2);
    EXPECT_EQ(minimizer.stages[0].tolerance, 10.);
    EXPECT_EQ(minimizer.stages[1].strategy, 1);
    EXPECT_EQ(minimizer.stages[1].tolerance, .01);
    ASSERT_EQ(minimizer.simplexBestPoint.size(), variables.size());
    EXPECT_NE(minimizer.stages[0].variables[0].Value(), minimizer.stages[1].variables[0].Value());
    for( std::size_t i = 0; i < variables.size(); ++i ){
      EXPECT_NEAR(minimizer.stages[1].variables[i].Value(), minimizer.simplexBestPoint[i], 1e-12);
    }
  }
}

TEST_F(RootMinimizerSimplex, RestoresMainMinimizerAfterAnObjectiveException) {
  FitterEngine engine;
  initializeEngine(engine);
  RecordingMinimizer minimizer(&engine);
  minimizer.configure(ConfigReader(JsonType{{"enableSimplexBeforeMinimize", true},
                                           {"print_level", 0}}));
  minimizer.initialize();
  auto* original = minimizer.getMinimizer().get();
  original->SetPrecision(1e-6);
  minimizer.failSimplex = true;
  EXPECT_THROW(minimizer.minimize(), std::runtime_error);
  EXPECT_EQ(minimizer.getMinimizer().get(), original);
  EXPECT_EQ(original->Precision(), 1e-6);
  EXPECT_EQ(original->Options().MinimizerAlgorithm(), "Migrad");
}


TEST_F(RootMinimizerSimplex, AnalyticGradientIsOptInAndValidatesConfiguration) {
  FitterEngine engine;
  initializeEngine(engine);
  auto* root = dynamic_cast<RootMinimizer*>(&engine.getMinimizer());
  ASSERT_NE(root, nullptr);
  EXPECT_FALSE(root->isUsingAnalyticGradient());
  EXPECT_EQ(root->getNbGradientCalls(), 0u);
  RecordingMinimizer invalid(&engine);
  EXPECT_THROW(invalid.configure(ConfigReader(JsonType{{"useAnalyticGradient", "true"}})), std::exception);
  invalid.configure(ConfigReader(JsonType{{"useAnalyticGradient", true}, {"minimizer", "Minuit"}}));
  EXPECT_THROW(invalid.initialize(), std::exception);
  auto& sample = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()[0];
  sample.setUseDynamicBinning(true);
  invalid.configure(ConfigReader(JsonType{{"useAnalyticGradient", true}, {"minimizer", "Minuit2"}}));
  EXPECT_THROW(invalid.initialize(), std::exception);
}


TEST_F(RootMinimizerSimplex, UnsupportedLikelihoodStopsAnalyticGradientInitialization) {
  FitterEngine engine;
  initializeEngine(engine, "BarlowLLH_BANFF_OA2021");
  RecordingMinimizer minimizer(&engine);
  minimizer.configure(ConfigReader(JsonType{{"useAnalyticGradient", true}}));
  try {
    minimizer.initialize();
    FAIL() << "Unsupported likelihood was accepted with an analytic gradient";
  }
  catch( const std::exception& error ){
    EXPECT_NE(std::string(error.what()).find("Analytic bin gradients are unsupported for likelihood"), std::string::npos);
  }
  EXPECT_EQ(minimizer.getNbGradientCalls(), 0u);
}


TEST_F(RootMinimizerSimplex, RootFormulaGradientMatchesFullLikelihoodFiniteDifferences) {
#ifndef GUNDAM_TEST_HAS_CLAD
  GTEST_SKIP() << "ROOT was built without Clad";
#endif
  FitterEngine engine;
  initializeEngine(engine, "BarlowBeestonLiteGaussian", "exp(.2*[value])+[value]*[value]");
  RecordingMinimizer minimizer(&engine);
  minimizer.configure(ConfigReader(JsonType{{"useAnalyticGradient", true}, {"useNormalizedFitSpace", false}}));
  minimizer.initialize();
  std::vector<double> point;
  for( auto* parameter : minimizer.getMinimizerFitParameterList() ){
    point.push_back(parameter->getParameterValue() + (parameter->isFixed() ? 0. : .15));
  }
  std::vector<double> gradient(point.size());
  minimizer.evalFitGradientAt(point.data(), gradient.data());
  for( size_t i = 0; i < point.size(); ++i ){
    if( minimizer.getMinimizerFitParameterList()[i]->isFixed() ){
      EXPECT_EQ(0., gradient[i]);
      continue;
    }
    auto up = point, down = point;
    up[i] += 1e-5;
    down[i] -= 1e-5;
    const double numeric = (minimizer.evalFit(up.data()) - minimizer.evalFit(down.data())) / 2e-5;
    EXPECT_NEAR(numeric, gradient[i], 1e-5);
  }
}

TEST_F(RootMinimizerSimplex, RootFormulaNumericalFallbackStopsAnalyticGradientInitialization) {
  FitterEngine engine;
  // Value-only initialization must remain usable for unsupported derivatives.
  initializeEngine(engine, "LeastSquares", "2+TMath::BesselJ0([value])");
  RecordingMinimizer minimizer(&engine);
  minimizer.configure(ConfigReader(JsonType{{"useAnalyticGradient", true}}));
  EXPECT_THROW(minimizer.initialize(), std::exception);
  EXPECT_EQ(0u, minimizer.getNbGradientCalls());
}


TEST_F(RootMinimizerSimplex, EventFormulaGradientMatchesFullLikelihoodFiniteDifferences) {
#ifndef GUNDAM_TEST_HAS_CLAD
  GTEST_SKIP() << "ROOT was built without Clad";
#endif
  const int previousThreads = GundamGlobals::getNbCpuThreads();
  for( int threads : {1, 4} ){
    GundamGlobals::setNumberOfThreads(threads);
    for( const std::string expression : {
        "exp(.2*[value])+[energy]*[alias]*[value]", "1+[energy]*[energy]"} ){
      FitterEngine engine;
      initializeEngine(engine, "BarlowBeestonLiteGaussian", expression, true);
      for( bool normalized : {false, true} ){
        RecordingMinimizer minimizer(&engine);
        minimizer.configure(ConfigReader(JsonType{{"useAnalyticGradient", true},
            {"useNormalizedFitSpace", normalized}}));
        minimizer.initialize();
        std::vector<double> point;
        for( auto* parameter : minimizer.getMinimizerFitParameterList() ){
          double value = parameter->getParameterValue() + (parameter->isFixed() ? 0. : .07);
          if( normalized ){ value = ParameterSet::toNormalizedParValue(value, *parameter); }
          point.push_back(value);
        }
        std::vector<double> gradient(point.size());
        minimizer.evalFitGradientAt(point.data(), gradient.data());
        for( size_t i = 0; i < point.size(); ++i ){
          if( minimizer.getMinimizerFitParameterList()[i]->isFixed() ){
            EXPECT_EQ(0., gradient[i]);
            continue;
          }
          auto up = point, down = point;
          up[i] += 1e-5;
          down[i] -= 1e-5;
          const double numeric = (minimizer.evalFit(up.data()) - minimizer.evalFit(down.data())) / 2e-5;
          EXPECT_NEAR(numeric, gradient[i], 1e-5)
              << expression << " threads=" << threads << " normalized=" << normalized << " input=" << i;
        }
      }
    }
  }
  GundamGlobals::setNumberOfThreads(previousThreads);
}

TEST_F(RootMinimizerSimplex, EventFormulaNumericalFallbackStopsAnalyticGradientInitialization) {
  FitterEngine engine;
  initializeEngine(engine, "LeastSquares", "2+TMath::BesselJ0([value])+[energy]", true);
  RecordingMinimizer minimizer(&engine);
  minimizer.configure(ConfigReader(JsonType{{"useAnalyticGradient", true}}));
  EXPECT_THROW(minimizer.initialize(), std::exception);
  EXPECT_EQ(0u, minimizer.getNbGradientCalls());
}
