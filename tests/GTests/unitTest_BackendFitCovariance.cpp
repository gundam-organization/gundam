#ifdef GUNDAM_USING_BACKENDS
#include "FitterEngine.h"
#include "RootMinimizer.h"
#include "BackendFactory.h"
#include "GundamGlobals.h"
#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif

#include <TClonesArray.h>
#include <TFile.h>
#include <TGraph.h>
#include <TMatrixDSymEigen.h>
#include <TString.h>
#include <TSystem.h>
#include <TTree.h>
#endif

#ifdef GUNDAM_TEST_HAS_CUDA
#include <cuda_runtime_api.h>
#endif
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  std::string cudaUnavailableReason() {
#ifndef GUNDAM_USING_BACKENDS
    return "GUNDAM was built without the backend library";
#elif !defined(GUNDAM_TEST_HAS_CUDA)
    return "GUNDAM was built without CUDA support";
#else
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if( status != cudaSuccess ){ return std::string("CUDA device enumeration unavailable: ") + cudaGetErrorString(status); }
    if( count == 0 ){ return "no CUDA device visible; request a GPU allocation"; }
    return {};
#endif
  }

#ifdef GUNDAM_USING_BACKENDS
  constexpr std::size_t parameterCount = 3;
  constexpr int binCount = 8;
  const std::array<double, parameterCount> priors{1., 1., 0.};
  const std::array<double, parameterCount> priorSigmas{.25, .3, .7};
  const std::array<double, parameterCount> startValues{1.18, .82, .35};

  struct FitEnvironment {
    int previousThreads{GundamGlobals::getNbCpuThreads()};
    bool previousLightOutput{GundamGlobals::isLightOutputMode()};
#ifdef GUNDAM_USING_CACHE_MANAGER
    bool previousCacheEnabled{Cache::Manager::IsCacheManagerEnabled()};
#endif
    FitEnvironment() {
      GundamGlobals::setNumberOfThreads(2);
      GundamGlobals::setLightOutputMode(true);
      FitterEngine::setRandomSeed(1729);
#ifdef GUNDAM_USING_CACHE_MANAGER
      Cache::Manager::SetIsEnabled(false);
#endif
    }
    ~FitEnvironment() {
      GundamGlobals::setNumberOfThreads(previousThreads);
      GundamGlobals::setLightOutputMode(previousLightOutput);
#ifdef GUNDAM_USING_CACHE_MANAGER
      Cache::Manager::SetIsEnabled(previousCacheEnabled);
#endif
    }
  };

  struct FitInput {
    std::string path;
    std::array<double, binCount> data{};
    std::array<std::array<double, parameterCount>, binCount> jacobian{};
    int eventCount{0};

    FitInput() {
      TString name("gundam-backend-fit-");
      auto* handle = gSystem->TempFileName(name, nullptr, ".root");
      if( not handle ){ throw std::runtime_error("Cannot create backend fit input."); }
      std::fclose(handle);
      path = name.Data();
      TFile file(path.c_str(), "RECREATE");
      if( file.IsZombie() ){ throw std::runtime_error("Cannot open backend fit input."); }
      double x = 0., weight = 1.;
      int category = 0;
      TClonesArray graphs("TGraph", 1);
      TTree tree("events", "Small deterministic backend fit");
      tree.Branch("X", &x, "X/D");
      tree.Branch("weight", &weight, "weight/D");
      tree.Branch("category", &category, "category/I");
      tree.Branch("shape", &graphs, 32000, 0);
      for( int bin = 0; bin < binCount; ++bin ){
        x = bin + .5;
        for( category = 0; category < 2; ++category ){
          const double slope = .025 * (bin - 3) * (bin - 3) - .1 + .02 * category;
          auto* graph = static_cast<TGraph*>(graphs.ConstructedAt(0));
          graph->Set(5);
          for( int knot = 0; knot < 5; ++knot ){ graph->SetPoint(knot, knot - 2., 1. + slope * (knot - 2.)); }
          const int repeats = category == 0 ? 18 + 4 * bin : 46 - 3 * bin;
          for( int i = 0; i < repeats; ++i ){
            weight = .8 + .05 * (i % 5);
            tree.Fill();
            data[bin] += weight;
            jacobian[bin][category] += weight;
            jacobian[bin][2] += weight * slope;
            ++eventCount;
          }
        }
      }
      tree.Write();
    }
    ~FitInput() { std::remove(path.c_str()); }
    FitInput(const FitInput&) = delete;
    FitInput& operator=(const FitInput&) = delete;

    TMatrixDSym expectedCovariance() const {
      // LeastSquares(lsqPoissonianApproximation) = 2*(prediction-data)^2/data.
      // At the Asimov minimum, Cov = inverse(Hessian/2). This independent
      // analytic anchor prevents a common CPU/CUDA error passing by agreement.
      TMatrixDSym information(parameterCount);
      information.Zero();
      for( std::size_t i = 0; i < parameterCount; ++i ){
        information(i, i) = 1. / (priorSigmas[i] * priorSigmas[i]);
        for( std::size_t j = 0; j < parameterCount; ++j ){
          for( int bin = 0; bin < binCount; ++bin ){
            information(i, j) += 2. * jacobian[bin][i] * jacobian[bin][j] / data[bin];
          }
        }
      }
      information.Invert();
      return information;
    }
  };

  JsonType makeConfig(const FitInput& input_, const std::string& backend_, const std::string& mode_,
                      const std::string& layout_ = "eventMajor") {
    auto config = JsonType::parse(R"({
      "enablePca": false, "scaleParStepWithChi2Response": false,
      "generateSamplePlots": false, "generateOneSigmaPlots": false,
      "minimizerConfig": {
        "minimizer": "Minuit2", "algorithm": "Migrad", "errors": "Hesse",
        "enablePostFitErrorFit": true, "useNormalizedFitSpace": false,
        "functionPrecision": 1e-12, "tolerance": 1e-7, "strategy": 1,
        "maxFcnCalls": 3000, "print_level": 0
      },
      "likelihoodInterfaceConfig": {
        "jointProbabilityConfig": {"type": "LeastSquares", "lsqPoissonianApproximation": true},
        "datasetList": [{"name": "input", "model": {"tree": "events", "nominalWeightFormula": "weight"}}],
        "propagatorConfig": {
          "sampleSetConfig": {"sampleList": [{"name": "spectrum", "binning": {
            "binningDefinition": [{"name": "X", "edges": [0,1,2,3,4,5,6,7,8]}]}}]},
          "parametersManagerConfig": {"parameterSetList": [{
            "name": "scales", "isEnabled": true, "enableEigenDecomp": false, "nominalStepSize": 0.03,
            "parameterDefinitions": [
              {"name": "normA", "priorValue": 1.0, "stdDev": 0.25,
               "dialSetDefinitions": [{"dialType": "Normalization", "applyCondition": "[category] == 0"}]},
              {"name": "normB", "priorValue": 1.0, "stdDev": 0.3,
               "dialSetDefinitions": [{"dialType": "Normalization", "applyCondition": "[category] == 1"}]},
              {"name": "shape", "priorValue": 0.0, "stdDev": 0.7,
               "dialSetDefinitions": [{"dialType": "Graph", "dialLeafName": "shape"}]}
            ]}]} }
      }
    })");
    auto& likelihood = config["likelihoodInterfaceConfig"];
    likelihood["datasetList"][0]["model"]["filePathList"] = {input_.path};
    if( backend_ != "Original" ){
      likelihood["backendManagerConfig"] = {{"isEnabled", true}, {"useBackend", backend_}};
      if( backend_ == "CUDA" ){
        likelihood["backendManagerConfig"]["options"] = JsonType::array({{{"name", backend_}, {"propagationMode", mode_}, {"weightLayout", layout_}}});
      }
    }
    return config;
  }

  struct FitResult {
    std::vector<std::string> names;
    std::vector<double> values, errors, sums, sumSquares;
    TMatrixDSym covariance{int(parameterCount)};
    int status{-1}, covarianceStatus{-1};
    double initialLikelihood{0.}, likelihood{0.}, statLikelihood{0.}, penaltyLikelihood{0.}, edm{0.}, targetEdm{0.};
    unsigned int calls{0};

    JsonType toJson() const {
      std::vector<std::vector<double>> matrix(parameterCount, std::vector<double>(parameterCount));
      auto correlation = matrix;
      for( std::size_t i = 0; i < parameterCount; ++i ){
        for( std::size_t j = 0; j < parameterCount; ++j ){
          matrix[i][j] = covariance(i, j);
          correlation[i][j] = covariance(i, j) / std::sqrt(covariance(i, i) * covariance(j, j));
        }
      }
      TMatrixDSymEigen decomposition(covariance);
      std::vector<double> eigenvalues(parameterCount);
      for( std::size_t i = 0; i < parameterCount; ++i ){ eigenvalues[i] = decomposition.GetEigenValues()[i]; }
      return {{"names", names}, {"values", values}, {"errors", errors}, {"covariance", matrix},
              {"correlation", correlation}, {"covarianceEigenvalues", eigenvalues},
              {"histogramSums", sums}, {"histogramSumSquares", sumSquares},
              {"status", status}, {"covarianceStatus", covarianceStatus},
              {"initialLikelihood", initialLikelihood}, {"likelihood", likelihood},
              {"statLikelihood", statLikelihood}, {"penaltyLikelihood", penaltyLikelihood},
              {"edm", edm}, {"targetEdm", targetEdm}, {"calls", calls}};
    }
  };

  FitResult fit(const FitInput& input_, const std::string& backend_, const std::string& mode_,
                const std::string& layout_ = "eventMajor", const std::string& probability_ = "") {
    FitterEngine engine;
    auto config = makeConfig(input_, backend_, mode_, layout_);
    if( !probability_.empty() ){
      config["likelihoodInterfaceConfig"]["jointProbabilityConfig"] = {{"type", probability_}};
    }
    if( backend_ == "MPS" ){
      config["likelihoodInterfaceConfig"]["backendManagerConfig"]["options"] = JsonType::array({{{"name", backend_}, {"propagationMode", mode_}}});
    }
    engine.configure(ConfigReader(config));
    engine.getLikelihoodInterface().setDataType(LikelihoodInterface::DataType::Asimov);
    engine.initialize();
    if( backend_ != "Original" and (not engine.getBackendsManager().hasBackend()
        or engine.getBackendsManager().getType() != backend_) ){
      throw std::runtime_error("Requested backend was not initialized: " + backend_);
    }
    auto* minimizer = dynamic_cast<RootMinimizer*>(&engine.getMinimizer());
    if( not minimizer ){ throw std::runtime_error("The fit test requires RootMinimizer."); }
    auto& parameters = engine.getLikelihoodInterface().getModelPropagator()
                             .getParametersManager().getParameterSetsList()[0].getParameterList();
    if( parameters.size() != parameterCount ){ throw std::runtime_error("Unexpected fit dimension."); }
    for( std::size_t i = 0; i < parameterCount; ++i ){ parameters[i].setParameterValue(startValues[i]); }
    minimizer->updateFitParameterSettings();
    engine.evaluateLikelihood();
    FitResult result;
    result.initialLikelihood = engine.getLikelihoodInterface().getLastLikelihood();
    engine.fit();
    engine.evaluateLikelihood();
    const auto& root = *minimizer->getMinimizer();
    if( root.NDim() != parameterCount ){ throw std::runtime_error("Unexpected minimizer dimension."); }
    result.status = root.Status();
    result.covarianceStatus = root.CovMatrixStatus();
    result.edm = root.Edm();
    result.targetEdm = minimizer->getTargetEdm();
    result.calls = root.NCalls();
    result.likelihood = engine.getLikelihoodInterface().getLastLikelihood();
    result.statLikelihood = engine.getLikelihoodInterface().getLastStatLikelihood();
    result.penaltyLikelihood = engine.getLikelihoodInterface().getLastPenaltyLikelihood();
    for( std::size_t i = 0; i < parameterCount; ++i ){
      result.names.push_back(root.VariableName(i));
      result.values.push_back(root.X()[i]);
      result.errors.push_back(root.Errors()[i]);
      for( std::size_t j = 0; j < parameterCount; ++j ){ result.covariance(i, j) = root.CovMatrix(i, j); }
    }
    const auto& histogram = engine.getLikelihoodInterface().getModelPropagator()
                                  .getSampleSet().getSampleList()[0].getHistogram();
    for( const auto& bin : histogram.getBinContentList() ){
      result.sums.push_back(bin.sumWeights);
      result.sumSquares.push_back(bin.sqrtSumSqWeights * bin.sqrtSumSqWeights);
    }
    return result;
  }

  void validateFit(const FitResult& fit_, const TMatrixDSym& expected_) {
    EXPECT_EQ(fit_.status, 0);
    EXPECT_EQ(fit_.covarianceStatus, 3) << "Hesse covariance must be accurate, not forced positive definite.";
    EXPECT_TRUE(std::isfinite(fit_.edm));
    EXPECT_LT(fit_.edm, fit_.targetEdm);
    EXPECT_GT(fit_.initialLikelihood, 1.);
    EXPECT_LT(fit_.likelihood, 1.e-6);
    EXPECT_GT(fit_.calls, 5u);
    TMatrixDSymEigen decomposition(fit_.covariance);
    for( std::size_t i = 0; i < parameterCount; ++i ){
      EXPECT_TRUE(std::isfinite(decomposition.GetEigenValues()[i]));
      EXPECT_GT(decomposition.GetEigenValues()[i], 0.);
      EXPECT_NEAR(fit_.values[i], priors[i], 2.e-4 * std::sqrt(expected_(i, i)));
      EXPECT_TRUE(std::isfinite(fit_.errors[i]));
      EXPECT_GT(fit_.errors[i], 0.);
      EXPECT_NEAR(fit_.errors[i] * fit_.errors[i], fit_.covariance(i, i), 1.e-8 * expected_(i, i));
      for( std::size_t j = 0; j < parameterCount; ++j ){
        const double scale = std::sqrt(expected_(i, i) * expected_(j, j));
        EXPECT_TRUE(std::isfinite(fit_.covariance(i, j)));
        EXPECT_NEAR(fit_.covariance(i, j), fit_.covariance(j, i), 1.e-12 * scale);
        EXPECT_NEAR(fit_.covariance(i, j), expected_(i, j), 1.e-3 * scale) << "Analytic covariance (" << i << ", " << j << ")";
      }
    }
  }

  void compareFits(const std::string& reference_, const std::string& candidate_, const std::string& mode_,
                    const std::string& layout_ = "eventMajor") {
    FitEnvironment environment;
    FitInput input;
    const auto reference = fit(input, reference_, mode_);
    const auto candidate = fit(input, candidate_, mode_, layout_);
    const auto expected = input.expectedCovariance();
    // Write the full matrices and all diagnostics even when comparisons fail.
    JsonType report{{"referenceBackend", reference_}, {"candidateBackend", candidate_}, {"mode", mode_},
                    {"weightLayout", layout_},
                    {"eventCount", input.eventCount}, {"binCount", binCount},
                    {"reference", reference.toJson()}, {"candidate", candidate.toJson()}};
    report["analyticCovariance"] = JsonType::array();
    for( std::size_t i = 0; i < parameterCount; ++i ){
      std::vector<double> row(parameterCount);
      for( std::size_t j = 0; j < parameterCount; ++j ){ row[j] = expected(i, j); }
      report["analyticCovariance"].push_back(row);
    }
    report["tolerances"] = {{"parameterDifferenceInPosteriorSigma", 2.e-4},
        {"relativeErrorDifference", 2.e-4}, {"covarianceDifferenceInSigmaProducts", 5.e-4},
        {"absoluteCorrelationDifference", 5.e-4}, {"relativeCovarianceEigenvalueDifference", 1.e-3},
        {"analyticCovarianceDifferenceInSigmaProducts", 1.e-3}};
    validateFit(reference, expected);
    validateFit(candidate, expected);
    EXPECT_EQ(candidate.names, reference.names);
    EXPECT_NEAR(candidate.likelihood, reference.likelihood, 1.e-7);
    EXPECT_NEAR(candidate.statLikelihood, reference.statLikelihood, 1.e-7);
    EXPECT_NEAR(candidate.penaltyLikelihood, reference.penaltyLikelihood, 1.e-7);
    double maxCovarianceDifference = 0., maxCorrelationDifference = 0.;
    TMatrixDSymEigen referenceEigen(reference.covariance), candidateEigen(candidate.covariance);
    for( std::size_t i = 0; i < parameterCount; ++i ){
      EXPECT_NEAR(candidate.values[i], reference.values[i], 2.e-4 * reference.errors[i]);
      EXPECT_NEAR(candidate.errors[i], reference.errors[i], 2.e-4 * reference.errors[i]);
      EXPECT_NEAR(candidateEigen.GetEigenValues()[i], referenceEigen.GetEigenValues()[i], 1.e-3 * referenceEigen.GetEigenValues()[i]);
      for( std::size_t j = 0; j < parameterCount; ++j ){
        const double scale = std::sqrt(reference.covariance(i, i) * reference.covariance(j, j));
        const double difference = std::abs(candidate.covariance(i, j) - reference.covariance(i, j)) / scale;
        const double correlationDifference = std::abs(candidate.covariance(i, j)
            / std::sqrt(candidate.covariance(i, i) * candidate.covariance(j, j)) - reference.covariance(i, j) / scale);
        maxCovarianceDifference = std::max(maxCovarianceDifference, difference);
        maxCorrelationDifference = std::max(maxCorrelationDifference, correlationDifference);
        EXPECT_LE(difference, 5.e-4) << "Covariance (" << i << ", " << j << ")";
        EXPECT_LE(correlationDifference, 5.e-4) << "Correlation (" << i << ", " << j << ")";
      }
    }
    EXPECT_EQ(candidate.sums.size(), reference.sums.size());
    for( std::size_t i = 0; i < std::min(candidate.sums.size(), reference.sums.size()); ++i ){
      EXPECT_NEAR(candidate.sums[i], reference.sums[i], 1.e-5 * std::max(1., std::abs(reference.sums[i])));
      EXPECT_NEAR(candidate.sumSquares[i], reference.sumSquares[i], 2.e-5 * std::max(1., reference.sumSquares[i]));
    }
    report["maxCovarianceDifferenceInSigmaProducts"] = maxCovarianceDifference;
    report["maxAbsoluteCorrelationDifference"] = maxCorrelationDifference;
    report["passed"] = not testing::Test::HasFailure();
    const auto path = "BackendFitCovariance_" + candidate_ + "_" + mode_
        + (layout_ == "eventMajor" ? "" : "_" + layout_) + ".json";
    std::ofstream output(path);
    ASSERT_TRUE(output.good()) << path;
    output << report.dump(2) << std::endl;
    std::cout << "Fit/covariance comparison: " << path << "; max normalized covariance difference = "
              << maxCovarianceDifference << "; max correlation difference = " << maxCorrelationDifference << std::endl;
  }
#endif

  class CudaFitCovariance : public testing::TestWithParam<const char*> {
  protected:
    void SetUp() override {
      const auto reason = cudaUnavailableReason();
      if( reason.empty() ){ return; }
      const auto* required = std::getenv("GUNDAM_REQUIRE_CUDA_TESTS");
      if( required != nullptr and std::string(required) == "1" ){ FAIL() << reason; }
      GTEST_SKIP() << reason;
    }
  };
}

TEST(BackendFitCovariance, CpuMatchesOriginalFitAndCovariance) {
#ifdef GUNDAM_USING_BACKENDS
  compareFits("Original", "CPU", "reference");
#else
  GTEST_SKIP() << "GUNDAM was built without the backend library";
#endif
}

TEST(BackendFitCovariance, ResidentReplacementPreservesLoadedModelAndRejectsStaleResults) {
#ifdef GUNDAM_USING_BACKENDS
  FitEnvironment environment;
  FitInput input;
  FitterEngine engine;
  engine.configure(ConfigReader(makeConfig(input, "CPU", "")));
  engine.getLikelihoodInterface().setDataType(LikelihoodInterface::DataType::Asimov);
  engine.initialize();
  auto& manager = engine.getBackendsManager();
  manager.setEnableAutoMaterialize(false);
  EXPECT_EQ(manager.getActiveBackendName(), "CPU");
  auto& parameters = engine.getLikelihoodInterface().getModelPropagator()
      .getParametersManager().getParameterSetsList()[0].getParameterList();
  const auto* parameter = &parameters[0];
  auto& histogram = engine.getLikelihoodInterface().getModelPropagator().getSampleSet()
      .getSampleList()[0].getHistogram();
  std::remove(input.path.c_str()); // Replacement must not reopen the input tree.
  for( std::size_t i = 0; i < parameterCount; ++i ){ parameters[i].setParameterValue(startValues[i]); }
  engine.evaluateLikelihood();
  const double reference = engine.getLikelihoodInterface().getLastLikelihood();
  manager.materialize(Backends::OutputRequest::Histograms);
  std::vector<double> sums;
  for( const auto& bin : histogram.getBinContentList() ){ sums.push_back(bin.sumWeights); }
  for( int iteration = 0; iteration < 3; ++iteration ){
    auto pending = manager.propagate();
    manager.activate("CPU", ConfigReader(JsonType::object()));
    EXPECT_THROW((void)pending.get(), std::exception);
    EXPECT_THROW(manager.materialize(Backends::OutputRequest::Histograms), std::exception);
    EXPECT_EQ(&parameters[0], parameter);
    EXPECT_FALSE(manager.isAutoMaterializeEnabled());
    engine.evaluateLikelihood();
    EXPECT_NEAR(engine.getLikelihoodInterface().getLastLikelihood(), reference, 1.e-12);
    manager.materialize(Backends::OutputRequest::Histograms);
    for( std::size_t i = 0; i < sums.size(); ++i ){
      EXPECT_NEAR(histogram.getBinContentList()[i].sumWeights, sums[i], 1.e-12);
    }
  }
  const auto* backend = manager.getBackend();
  EXPECT_THROW(manager.activate("invalid", ConfigReader(JsonType::object())), std::exception);
  EXPECT_EQ(manager.getBackend(), backend);
  EXPECT_THROW(manager.activate("CUDA", ConfigReader(JsonType{{"deviceIndex", -1}})), std::exception);
  EXPECT_EQ(manager.getBackend(), backend);
  engine.evaluateLikelihood();
  EXPECT_NEAR(engine.getLikelihoodInterface().getLastLikelihood(), reference, 1.e-12);
  const auto compiled = Backends::getCompiledBackendNames();
  if( std::find(compiled.begin(), compiled.end(), "CUDA") == compiled.end() ){
    EXPECT_THROW(manager.activate("CUDA", ConfigReader(JsonType::object())), std::exception);
    EXPECT_EQ(manager.getBackend(), backend);
    EXPECT_NO_THROW(engine.evaluateLikelihood());
    manager.activate("CPU", ConfigReader(JsonType::object()));
    engine.evaluateLikelihood();
    EXPECT_NEAR(engine.getLikelihoodInterface().getLastLikelihood(), reference, 1.e-12);
  }
#else
  GTEST_SKIP() << "GUNDAM was built without the backend library";
#endif
}

TEST_P(CudaFitCovariance, MatchesCpuFitAndCovariance) {
#ifdef GUNDAM_USING_BACKENDS
  compareFits("CPU", "CUDA", GetParam());
#endif
}

TEST_P(CudaFitCovariance, InterleavedMatchesCpuFitAndCovariance) {
#ifdef GUNDAM_USING_BACKENDS
  compareFits("CPU", "CUDA", GetParam(), "warpInterleaved");
#endif
}

TEST_P(CudaFitCovariance, ResidentSwitchMatchesCpu) {
#ifdef GUNDAM_USING_BACKENDS
  FitEnvironment environment;
  FitInput input;
  FitterEngine engine;
  engine.configure(ConfigReader(makeConfig(input, "CPU", "")));
  engine.getLikelihoodInterface().setDataType(LikelihoodInterface::DataType::Asimov);
  engine.initialize();
  auto& manager = engine.getBackendsManager();
  manager.setEnableAutoMaterialize(false);
  EXPECT_EQ(manager.getActiveBackendName(), "CPU");
  auto& parameters = engine.getLikelihoodInterface().getModelPropagator()
      .getParametersManager().getParameterSetsList()[0].getParameterList();
  for( std::size_t i = 0; i < parameterCount; ++i ){ parameters[i].setParameterValue(startValues[i]); }
  engine.evaluateLikelihood();
  const auto reference = engine.getLikelihoodInterface().getLastLikelihood();
  std::remove(input.path.c_str());
  for( const auto* type : {"CUDA", "CPU", "CUDA"} ){
    manager.activate(type, ConfigReader(std::string(type) == "CUDA"
        ? JsonType{{"propagationMode", GetParam()}} : JsonType::object()));
    engine.evaluateLikelihood();
    EXPECT_NEAR(engine.getLikelihoodInterface().getLastLikelihood(), reference, 1.e-10);
  }
#endif
}

INSTANTIATE_TEST_SUITE_P(Modes, CudaFitCovariance,
    testing::Values("cachedResponses", "recomputeResponses"),
    [](const testing::TestParamInfo<const char*>& info_){ return info_.param; });

#if defined(__APPLE__) && defined(GUNDAM_USING_BACKENDS)
TEST(BackendFitCovariance, DeviceStatLikelihoodMatchesOriginalFitAndHesse) {
  FitEnvironment environment;
  FitInput input;
  for( const auto* probability : {"PoissonLLH", "BarlowLLH_BANFF_OA2021"} ){
    const auto original = fit(input, "Original", "", "eventMajor", probability);
    ASSERT_EQ(original.status, 0);
    ASSERT_EQ(original.covarianceStatus, 3);
    for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
      SCOPED_TRACE(::testing::Message() << probability << ", " << mode);
      const auto device = fit(input, "MPS", mode, "eventMajor", probability);
      EXPECT_EQ(device.status, 0);
      EXPECT_EQ(device.covarianceStatus, 3);
      EXPECT_LT(device.edm, device.targetEdm);
      EXPECT_NEAR(device.likelihood, original.likelihood, 1e-7);
      for( std::size_t i = 0; i < parameterCount; ++i ){
        EXPECT_NEAR(device.values[i], original.values[i], 2e-4 * original.errors[i]);
        EXPECT_NEAR(device.errors[i], original.errors[i], 2e-4 * original.errors[i]);
        for( std::size_t j = 0; j < parameterCount; ++j ){
          EXPECT_NEAR(device.covariance(i, j), original.covariance(i, j),
                      5e-4 * original.errors[i] * original.errors[j]);
        }
      }
    }
  }
}
#endif
