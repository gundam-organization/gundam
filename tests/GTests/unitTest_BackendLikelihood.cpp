#include "CpuBackend.h"
#include "LikelihoodModel.h"
#include "Semantics/BackendLikelihood.h"
#include "JointProbability.h"
#include "JointProbabilitySnapshot.h"
#ifdef __APPLE__
#include "MpsBackend.h"
#endif
#ifdef GUNDAM_TEST_CUDA_DEVICE
#include "CudaBackend.h"
#endif

#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <limits>
#include <memory>

namespace {
  using namespace Backends;
  using Math = LikelihoodSemantics::DoubleMath;
  std::unique_ptr<JointProbability::JointProbabilityBase> reference(const char* type_, JsonType config_ = JsonType::object()) {
    std::unique_ptr<JointProbability::JointProbabilityBase> out(JointProbability::makeJointProbability(type_));
    out->configure(ConfigReader(config_));
    return out;
  }
  LikelihoodOptions<double> options(const JointProbability::JointProbabilityBase& reference_) {
    auto snapshot = JointProbability::snapshotJointProbability(reference_, {});
    LikelihoodOptions<double> out;
    out.kind = snapshot.kind == JointProbability::JointProbabilitySnapshot::Kind::Poisson
        ? LikelihoodKind::Poisson : LikelihoodKind::BarlowBeeston2022;
    out.usePoissonLikelihood = snapshot.usePoissonLikelihood;
    out.throwIfInfLlh = snapshot.throwIfInfLlh;
    out.fractionalErrorLimit = snapshot.fractionalErrorLimit;
    out.expectedValueMinimum = snapshot.expectedValueMinimum;
    return out;
  }
  void same(double actual_, double expected_) {
    if( std::isinf(expected_) ){ EXPECT_EQ(actual_, expected_); }
    else if( std::isnan(expected_) ){ EXPECT_TRUE(std::isnan(actual_)); }
    else{ EXPECT_NEAR(actual_, expected_, 3e-13 * std::max(1e-30, std::abs(expected_))); }
  }
}

TEST(BackendLikelihood, PoissonMatchesUntouchedReferenceIncludingEmptyAndNegativeBins) {
  auto ref = reference("PoissonLLH");
  const auto config = options(*ref);
  for( double data : {-1., 0., 1e-8, 1., 17., 1e8} ){
    for( double pred : {-1., 0., 1e-8, 1., 17., 1e8} ){
      same(LikelihoodSemantics::evaluate<Math>(data, pred, 99., config), ref->eval(data, pred, 0., 0));
    }
  }
}

TEST(BackendLikelihood, Barlow2022MatchesReferenceAcrossOptionsAndNearMinimum) {
  for( bool poissonOnly : {false, true} ){
    for( double limit : {.5, 1., 1e150} ){
      auto ref = reference("BarlowLLH_BANFF_OA2021", {{"usePoissonLikelihood", poissonOnly}, {"fractionalErrorLimit", limit}, {"verboseLevel", -1}});
      const auto config = options(*ref);
      for( double data : {0., 1e-8, 1., 17., 655.82092932695639, 1e8} ){
        for( double shift : {-.9, -.05, -.009, -1e-7, -1e-10, 0., 1e-10, 1e-7, .009, .05, 1.} ){
          const double pred = data > 0. ? data*(1.+shift) : 5.;
          for( double variance : {0., 1e-8, .1, 1., 100.} ){
            SCOPED_TRACE(::testing::Message() << data << ", " << pred << ", " << variance << ", " << limit);
            same(LikelihoodSemantics::evaluate<Math>(data, pred, variance, config), ref->eval(data, pred, variance, 0));
          }
        }
      }
    }
  }
}

TEST(BackendLikelihood, Barlow2022PreservesThresholdAndFailurePolicy) {
  for( bool throws : {false, true} ){
    auto ref = reference("BarlowLLH_BANFF_OA2021", {{"throwIfInfLlh", throws}, {"verboseLevel", -1}});
    auto config = options(*ref);
    same(LikelihoodSemantics::evaluate<Math>(1., 0., 0., config), ref->eval(1., 0., 0., 0));
    EXPECT_EQ(LikelihoodSemantics::evaluate<Math>(0., 0., 0., config), 0.);
    auto bad = LikelihoodSemantics::evaluate<Math>(1., 1., std::numeric_limits<double>::infinity(), config);
    EXPECT_TRUE(throws ? std::isnan(bad) : std::isinf(bad));
    config.expectedValueMinimum = .1;
    EXPECT_EQ(LikelihoodSemantics::evaluate<Math>(1., .05, 0., config), 0.);
    // The compatibility threshold returns +inf even when throwIfInfLlh is true.
    EXPECT_EQ(LikelihoodSemantics::evaluate<Math>(1., 0., 0., config), std::numeric_limits<double>::infinity());
  }
}

TEST(BackendLikelihood, SnapshotKeepsPreviouslyCapturedNominalErrors) {
  Sample model, data;
  for( auto* sample : {&model, &data} ){
    sample->configure(ConfigReader(JsonType{{"name", "sample"}, {"binning", {{"binningDefinition", {{{"name", "X"}, {"edges", {0, 1}}}}}}}}));
  }
  SamplePair pair{&model, &data};
  model.getHistogram().getBinContentList()[0] = {20., 3.};
  auto ref = reference("BarlowLLH_BANFF_OA2021", {{"BBNoUpdateWeights", true}});
  auto before = JointProbability::snapshotJointProbability(*ref, pair);
  EXPECT_EQ(before.nominalVariances, std::vector<double>{9.});
  EXPECT_TRUE(std::isfinite(ref->evalWithPrediction(pair, 17., 20., 3., 0))); // Capture nominal.
  model.getHistogram().getBinContentList()[0] = {40., 11.};
  auto after = JointProbability::snapshotJointProbability(*ref, pair);
  EXPECT_TRUE(after.useNominalVariance);
  EXPECT_EQ(after.nominalVariances, before.nominalVariances);
}

namespace {
  EngineView likelihoodView(LikelihoodKind kind_, bool nominal_ = false, bool poissonOnly_ = false) {
    EngineView view;
    auto& model = view.propagation;
    model.totalBins = 530;
    model.parameterCount = 1;
    model.dialInputs = {{0}};
    BackendDialDescriptor norm;
    norm.inputCount = 1;
    model.dials.push_back(norm);
    for( int sampleIndex = 0; sampleIndex < 2; ++sampleIndex ){
      LikelihoodSampleView sample;
      sample.binOffset = sampleIndex * 265;
      sample.portableOptions = {kind_, LikelihoodIndex(poissonOnly_), LikelihoodIndex(nominal_), 1, 1e150, -1.};
      // A broken callback proves supported likelihoods never call host code.
      sample.evalBin = [](double, double, double, int) -> double { throw std::runtime_error("Host likelihood called"); };
      for( int bin = 0; bin < 265; ++bin ){
        const double weight = 1. + .01*bin;
        sample.dataSums.push_back(2.*weight);
        sample.ignoredBins.push_back(bin % 17 == 0);
        sample.nominalVariances.push_back(2.*weight*weight);
        for( int i = 0; i < 2; ++i ){
          EventView event;
          event.globalBinIndex = sample.binOffset + bin;
          event.resultIndex = model.events.size();
          model.events.push_back(event);
          model.eventWeights.push_back({weight, model.eventDialIndices.size(), 1});
          model.eventDialIndices.push_back(0);
        }
      }
      view.likelihood.samples.push_back(sample);
    }
    return view;
  }

  void checkResidentLikelihood(Backend& gpu_, bool metal_) {
    for( auto kind : {LikelihoodKind::Poisson, LikelihoodKind::BarlowBeeston2022} ){
      for( int setting = 0; setting < 4; ++setting ){
        const bool nominal = setting & 1;
        const bool poissonOnly = setting & 2;
        auto view = likelihoodView(kind, nominal, poissonOnly);
        gpu_.build(view);
        const char* type = kind == LikelihoodKind::Poisson ? "PoissonLLH" : "BarlowLLH_BANFF_OA2021";
        auto ref = kind == LikelihoodKind::Poisson ? reference(type)
            : reference(type, {{"usePoissonLikelihood", poissonOnly}});
        auto stablePoisson = reference("BarlowLLH_BANFF_OA2021");
        PropagationToken previous;
        for( double parameter : {1.1, .8, 1., 1.+1e-6, 1.-1e-6} ){
          PropagationInputs inputs;
          inputs.parameters.values = {parameter};
          auto token = gpu_.requestPropagation(inputs);
          if( previous.isValid ){ EXPECT_FALSE(gpu_.isReady(previous)); }
          previous = token;
          double expected = 0.;
          for( const auto& sample : view.likelihood.samples ){
            for( int bin = 0; bin < int(sample.dataSums.size()); ++bin ){
              if( sample.ignoredBins[bin] ){ continue; }
              const double pred = sample.dataSums[bin] * parameter;
              const double variance = sample.nominalVariances[bin] * (nominal ? 1. : parameter*parameter);
              if( kind == LikelihoodKind::Poisson && std::abs(parameter - 1.) < .01 ){
                // The original Poisson expression cancels near the minimum;
                // the zero-variance 2022 reference provides a stable oracle.
                expected += stablePoisson->eval(sample.dataSums[bin], pred, 0., bin);
              }
              else{ expected += ref->eval(sample.dataSums[bin], pred, variance, bin); }
            }
          }
          const double actual = gpu_.getLikelihood(token);
          const double tolerance = metal_ ? (std::abs(parameter-1.) < 1e-5 ? 1e-15 : 1e-9) : 1e-11;
          EXPECT_NEAR(actual, expected, tolerance);
          auto timing = gpu_.getLastTimingSummary();
          EXPECT_EQ(timing.histogramReadbackBytes, 0);
          EXPECT_EQ(timing.eventWeightReadbackBytes, 0);
          EXPECT_EQ(timing.likelihoodReadbackBytes, 8);
          EXPECT_EQ(timing.likelihoodHostSeconds, 0.);
          EXPECT_EQ(gpu_.getStatus(token).histograms, OutputState::ReadyOnDevice);
          // Explicit diagnostics still work, and don't change the objective.
          gpu_.materialize(token, OutputRequest::Histograms);
          EXPECT_EQ(gpu_.getHistogramSumsHostView(token).size(), 530);
          EXPECT_GT(gpu_.getLastTimingSummary().histogramReadbackBytes, 0);
          EXPECT_EQ(gpu_.getLikelihood(token), actual);
        }
      }
    }
  }
}

TEST(BackendLikelihood, CpuUsesPortableEvaluatorWithoutCallback) {
  CpuBackend cpu;
  auto view = likelihoodView(LikelihoodKind::BarlowBeeston2022);
  cpu.build(view);
  PropagationInputs inputs;
  inputs.parameters.values = {1.};
  EXPECT_EQ(cpu.getLikelihood(cpu.requestPropagation(inputs)), 0.);
}

#ifdef __APPLE__
TEST(BackendLikelihood, MetalPreservesInfinityEmptyBinsAndHostFallback) {
  MpsBackend gpu;
  auto view = likelihoodView(LikelihoodKind::Poisson);
  PropagationInputs inputs;
  inputs.parameters.values = {0.};
  gpu.build(view);
  auto token = gpu.requestPropagation(inputs);
  EXPECT_EQ(gpu.getLikelihood(token), std::numeric_limits<double>::infinity());
  EXPECT_EQ(gpu.getLastTimingSummary().histogramReadbackBytes, 0);
  // The zero-prediction Barlow branch uses the binary64 logarithmic floor.
  double expected = 0.;
  auto barlow = reference("BarlowLLH_BANFF_OA2021", {{"verboseLevel", -1}});
  for( auto& sample : view.likelihood.samples ){
    sample.portableOptions.kind = LikelihoodKind::BarlowBeeston2022;
    for( std::size_t bin = 0; bin < sample.dataSums.size(); ++bin ){
      if( !sample.ignoredBins[bin] ){ expected += barlow->eval(sample.dataSums[bin], 0., 0., bin); }
    }
  }
  gpu.build(view);
  EXPECT_NEAR(gpu.getLikelihood(gpu.requestPropagation(inputs)), expected, expected * 1e-12);
  for( auto& sample : view.likelihood.samples ){
    sample.ignoredBins.assign(sample.dataSums.size(), true);
  }
  gpu.build(view);
  token = gpu.requestPropagation(inputs);
  EXPECT_EQ(gpu.getLikelihood(token), 0.);
  // Rebuilding with an unsupported function restores the host fallback.
  for( auto& sample : view.likelihood.samples ){
    sample.ignoredBins.assign(sample.dataSums.size(), false);
    sample.portableOptions.kind = LikelihoodKind::Host;
    sample.evalBin = [](double, double pred_, double, int){ return pred_; };
  }
  gpu.build(view);
  inputs.parameters.values = {1.};
  token = gpu.requestPropagation(inputs);
  EXPECT_GT(gpu.getLikelihood(token), 0.);
  EXPECT_GT(gpu.getLastTimingSummary().histogramReadbackBytes, 0);
  EXPECT_EQ(gpu.getLastTimingSummary().likelihoodReadbackBytes, 0);
  // A sample containing no bins still has a well-defined zero total.
  for( auto& sample : view.likelihood.samples ){
    sample.portableOptions.kind = LikelihoodKind::Poisson;
    sample.dataSums.clear();
  }
  gpu.build(view);
  EXPECT_EQ(gpu.getLikelihood(gpu.requestPropagation(inputs)), 0.);
}

TEST(BackendLikelihood, MetalReusesPredictionAndInvalidatesForExternalWeights) {
  using namespace Backends;
  auto view = likelihoodView(LikelihoodKind::BarlowBeeston2022);
  // The second parameter has no effect on the prediction (e.g. penalty only).
  view.propagation.parameterCount = 2;
  view.propagation.externalWeightBlocks = {{0, 1}};
  auto& dial = view.propagation.dials[0];
  dial.type = BackendDialType::ExternalWeight;
  dial.externalWeightIndex = 0;
  MpsBackend gpu;
  gpu.build(view);
  gpu.setDiagnosticStageProfiling(true);
  double weight = 1.;
  PropagationInputs inputs;
  inputs.parameters.values = {1., 0.};
  inputs.externalWeights = {{&weight, 1, 0, 1}};
  auto first = gpu.requestPropagation(inputs);
  const double original = gpu.getLikelihood(first);
  EXPECT_TRUE(JsonType::parse(gpu.getDiagnosticEventPropagationState()).at("updatedPrediction"));
  inputs.parameters.values[1] = .2;
  auto reused = gpu.requestPropagation(inputs);
  EXPECT_EQ(gpu.getLikelihood(reused), original);
  EXPECT_FALSE(JsonType::parse(gpu.getDiagnosticEventPropagationState()).at("updatedPrediction"));
  auto stages = JsonType::parse(gpu.getDiagnosticStageGpuSeconds());
  EXPECT_EQ(stages.count("reduce_histogram_chunks_parallel"), 0);
  EXPECT_EQ(stages.count("evaluate_likelihood_partials"), 0);
  EXPECT_EQ(gpu.getLastTimingSummary().likelihoodReadbackBytes, 8);
  EXPECT_EQ(gpu.getLastTimingSummary().histogramReadbackBytes, 0);
  gpu.materialize(reused, OutputRequest::Histograms);
  EXPECT_EQ(gpu.getHistogramSumsHostView(reused).size(), 530);
  weight = 1.1;
  inputs.externalWeights[0].generation++;
  const auto changed = gpu.requestPropagation(inputs);
  EXPECT_GT(gpu.getLikelihood(changed), original);
  EXPECT_TRUE(JsonType::parse(gpu.getDiagnosticEventPropagationState()).at("updatedPrediction"));
  // Explicit binning invalidation also prevents reuse, even at identical weights.
  inputs.forceBinningUpdate = true;
  gpu.requestPropagation(inputs);
  EXPECT_TRUE(JsonType::parse(gpu.getDiagnosticEventPropagationState()).at("updatedPrediction"));
}

TEST(BackendLikelihood, MetalBarlowResolvesLowRatesNearTheMinimum) {
  using namespace Backends;
  auto ref = reference("BarlowLLH_BANFF_OA2021");
  for( double data : {1e-12, 1e-8, 1e-4, 1., 1e8} ){
    for( double shift : {-1e-5, -1e-8, 0., 1e-8, 1e-5} ){
      for( double fractionalVariance : {1e-8, .1, 1.} ){
        SCOPED_TRACE(::testing::Message() << data << ", " << shift << ", " << fractionalVariance);
        const double prediction = data*(1.+shift);
        const double variance = prediction*prediction*fractionalVariance;
        EngineView view;
        view.propagation.totalBins = 1;
        EventView event;
        event.globalBinIndex = 0;
        view.propagation.events = {event};
        view.propagation.eventWeights = {{prediction, 0, 0}};
        LikelihoodSampleView sample;
        sample.dataSums = {data};
        sample.nominalVariances = {variance};
        sample.portableOptions = {LikelihoodKind::BarlowBeeston2022, 0, 1, 1, 1e150, -1.};
        view.likelihood.samples = {sample};
        MpsBackend gpu;
        gpu.build(view);
        const double expected = ref->eval(data, prediction, variance, 0);
        const double actual = gpu.getLikelihood(gpu.requestPropagation({}));
        EXPECT_GE(actual, 0.);
        // Float-pair inputs have ~48 significant bits. A relative 1e-8
        // displacement leaves fewer bits in the residual; test relative LLH
        // precision instead of a fixed absolute tolerance that masks tiny rates.
        EXPECT_NEAR(actual, expected, expected > 0. ? expected*2e-5 : data*1e-25);
      }
    }
  }
}

TEST(BackendLikelihood, MetalParallelReductionHandlesLargeAndUnevenBinCounts) {
  using namespace Backends;
  for( int count : {1, 127, 129, 4140, 8192, 8193, 65536} ){
    EngineView view;
    view.propagation.totalBins = count;
    LikelihoodSampleView sample;
    sample.portableOptions = {LikelihoodKind::BarlowBeeston2022, 0, 0, 1, 1e150, -1.};
    double expected = 0.;
    auto ref = reference("BarlowLLH_BANFF_OA2021");
    for( int i = 0; i < count; ++i ){
      double data = .1 + .001*i;
      double weight = data*(1. + .02*(i%7 - 3));
      sample.dataSums.push_back(data);
      sample.ignoredBins.push_back(i%19 == 0);
      EventView event;
      event.resultIndex = i;
      event.globalBinIndex = i;
      view.propagation.events.push_back(event);
      view.propagation.eventWeights.push_back({weight, 0, 0});
      if( !sample.ignoredBins.back() ){ expected += ref->eval(data, weight, weight*weight, i); }
    }
    view.likelihood.samples.push_back(sample);
    MpsBackend gpu;
    gpu.build(view);
    const auto token = gpu.requestPropagation({});
    EXPECT_NEAR(gpu.getLikelihood(token), expected, 1e-10 * std::max(1., expected));
    const auto diagnostic = JsonType::parse(gpu.getDiagnostics()).at("executor");
    EXPECT_EQ(diagnostic.at("likelihoodBinsPerThread"), count <= 8192 ? 1 : 8);
    EXPECT_EQ(gpu.getLastTimingSummary().histogramReadbackBytes, 0);
  }
}

TEST(BackendLikelihood, MetalKeepsHistogramsResidentInBothModes) {
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    MpsBackend gpu;
    ASSERT_NE(gpu.getCapabilities().deviceName, "Metal unavailable");
    gpu.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    checkResidentLikelihood(gpu, true);
  }
}
#endif
#ifdef GUNDAM_TEST_CUDA_DEVICE
TEST(BackendLikelihood, CudaKeepsHistogramsResidentInBothModes) {
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    CudaBackend gpu;
    gpu.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    checkResidentLikelihood(gpu, false);
  }
}
#endif
