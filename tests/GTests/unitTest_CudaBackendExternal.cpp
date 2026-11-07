#include "CudaBackend.h"
#include "CpuBackend.h"
#include "ExternalWeightDialFactory.h"
#include "ExternalWeightDispatcher.h"
#include "Kriged.h"
#include "Tabulated.h"
#include "Semantics/BackendDialSemantics.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  using namespace Backends;

  EngineView makeModel(bool externalOnly_) {
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = externalOnly_ ? 0 : 2;
    model.totalBins = 4; // Last bin is deliberately empty.
    model.externalWeightBlocks = {{7, 3}, {1, 2}, {0, 0}};
    BackendDialDescriptor external;
    external.type = BackendDialType::ExternalWeight;
    external.externalWeightIndex = 1;
    external.inputCount = 2; // Producer inputs are never read by the device.
    external.hasMinResponse = external.hasMaxResponse = true;
    external.minResponse = .1;
    external.maxResponse = 2.;
    auto tabulated = external;
    tabulated.type = BackendDialType::Tabulated;
    tabulated.tableTermCount = 2;
    auto kriged = external;
    kriged.type = BackendDialType::Kriged;
    kriged.externalBlockIndex = 1;
    kriged.firstTableTerm = 2;
    kriged.tableTermCount = 3;
    kriged.hasMinResponse = kriged.hasMaxResponse = false;
    auto empty = kriged;
    empty.externalBlockIndex = 2;
    empty.tableTermCount = 0;
    empty.hasMinResponse = true;
    empty.minResponse = .5;
    model.dials = {external, tabulated, kriged, empty};
    model.tableTerms = {{1, .25}, {2, .75}, {1, -.25}, {0, .5}, {1, .75}};
    BackendDialDescriptor shift;
    shift.type = BackendDialType::Shift;
    shift.payloadSize = 1;
    model.dialPayloads = {.9};
    model.dials.push_back(shift);
    // More than one CUDA response block per external dependency group.
    for( int i = 0; i < 600; ++i ){ model.dials.push_back(model.dials[i % 3]); }
    if( not externalOnly_ ){
      BackendDialDescriptor surface;
      surface.type = BackendDialType::Bilinear;
      surface.inputCount = 2;
      surface.payloadOffset = model.dialPayloads.size();
      surface.payloadSize = 10;
      model.dialPayloads.insert(model.dialPayloads.end(), {2., 2., -1., 1., -1., 1., .5, .75, 1.25, 1.5});
      model.dialInputs = {{0}, {1}};
      model.dials.push_back(surface);
      BackendDialDescriptor norm;
      norm.inputCount = 1;
      model.dials.push_back(norm);
    }
    model.events.resize(521);
    model.eventWeights.resize(521);
    for( std::size_t i = 0; i < model.events.size(); ++i ){
      const auto slot = (37 * i) % model.events.size();
      model.events[i].resultIndex = slot;
      model.events[i].globalBinIndex = i % 11 == 0 ? -1 : int(i % 3);
      auto& event = model.eventWeights[slot];
      event.baseWeight = i % 17 == 0 ? 0. : (i % 13 == 0 ? -.125 : .25);
      event.firstDial = model.eventDialIndices.size();
      for( auto dial : {0u, 1u, 2u, 0u, 3u, 4u, unsigned(5 + i % 600)} ){
        model.eventDialIndices.push_back(dial);
      }
      event.dialCount = 7;
      if( not externalOnly_ ){
        model.eventDialIndices.push_back(605);
        model.eventDialIndices.push_back(606);
        event.dialCount += 2;
      }
    }
    LikelihoodSampleView sample;
    sample.dataSums = {2., 1., 0., 3.};
    sample.evalBin = [](double data_, double prediction_, double error_, int){
      return (prediction_ - data_) * (prediction_ - data_) + error_ * error_;
    };
    view.likelihood.samples.push_back(sample);
    return view;
  }

  void expectClose(double actual_, double expected_) {
    EXPECT_NEAR(actual_, expected_, 3e-12 * std::max(1., std::abs(expected_)));
  }

  void compare(CudaBackend& cuda_, CpuBackend& cpu_, const PropagationInputs& inputs_,
               std::size_t blocks_, std::size_t values_, std::size_t candidates_) {
    const auto token = cuda_.requestPropagation(inputs_), reference = cpu_.requestPropagation(inputs_);
    ASSERT_TRUE(token.isValid);
    ASSERT_EQ(cuda_.getStatus(token).eventWeights, OutputState::ReadyOnDevice);
    const auto timing = cuda_.getLastTimingSummary();
    EXPECT_EQ(timing.externalWeightUploadBlocks, blocks_);
    EXPECT_EQ(timing.externalWeightUploadBytes, values_ * sizeof(double));
    EXPECT_EQ(timing.cacheCandidateCount, candidates_);
    const auto& actual = cuda_.getEventWeightsHostView(token);
    const auto& expected = cpu_.getEventWeightsHostView(reference);
    ASSERT_EQ(actual.size(), expected.size());
    for( std::size_t i = 0; i < actual.size(); ++i ){ expectClose(actual[i], expected[i]); }
    const auto& sums = cuda_.getHistogramSumsHostView(token);
    const auto& squares = cuda_.getHistogramSumSquaresHostView(token);
    for( std::size_t i = 0; i < sums.size(); ++i ){
      expectClose(sums[i], cpu_.getHistogramSumsHostView(reference)[i]);
      expectClose(squares[i], cpu_.getHistogramSumSquaresHostView(reference)[i]);
    }
    expectClose(cuda_.getLikelihood(token), cpu_.getLikelihood(reference));
  }

  class TestExternalWorker : public ExternalWeightWorker {
  public:
    double value{2.};
    bool fail{false};
  private:
    void evaluateImpl(const DialInputBuffer&) override {
      if( fail ){ throw std::runtime_error("Producer failed"); }
      std::fill_n(getWeightBuffer()->ptr, getWeightCount(), value);
    }
  };
}

TEST(CudaBackendExternal, MatchesCpuWithSelectiveGenerationUploadsAndMixedDials) {
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    for( const auto* layout : {"eventMajor", "warpInterleaved"} ){
      for( bool externalOnly : {false, true} ){
        SCOPED_TRACE(JsonType({mode, layout, externalOnly}).dump());
        const bool cached = std::string(mode) == "cachedResponses";
        auto model = makeModel(externalOnly);
        CudaBackend cuda;
        cuda.configure(ConfigReader(JsonType{{"propagationMode", mode}, {"weightLayout", layout}}));
        CpuBackend cpu;
        cuda.build(model);
        cpu.build(model);
        std::vector<double> first{10., 2., 4.}, second{3., 5.};
        PropagationInputs inputs;
        if( not externalOnly ){ inputs.parameters.values = {.4, -.2}; }
        inputs.externalWeights = {{first.data(), first.size(), 7, 1}, {second.data(), second.size(), 1, 1}, {nullptr, 0, 0, 1}};
        const auto all = model.propagation.dials.size();
        compare(cuda, cpu, inputs, 3, 5, all);
        compare(cuda, cpu, inputs, 0, 0, cached ? 0 : all);
        first[1] = -.2;
        first[2] = 1.5;
        ++inputs.externalWeights[0].generation;
        compare(cuda, cpu, inputs, 1, 3, cached ? 402 : all);
        second[0] = -2.;
        ++inputs.externalWeights[1].generation;
        compare(cuda, cpu, inputs, 1, 2, cached ? 201 : all);
        ++inputs.externalWeights[2].generation; // Empty table, zero terms, nonzero lower bound.
        compare(cuda, cpu, inputs, 1, 0, cached ? 1 : all);
        if( not externalOnly ){
          inputs.parameters.values[1] = .6; // Surface only.
          compare(cuda, cpu, inputs, 0, 0, cached ? 1 : all);
          inputs.parameters.values[0] = .7; // Surface + Norm, no external dependencies.
          compare(cuda, cpu, inputs, 0, 0, cached ? 3 : all); // Dense parameter traversal includes immutable Shift.
        }
        ++inputs.externalWeights[0].generation;
        ++inputs.externalWeights[1].generation;
        compare(cuda, cpu, inputs, 2, 5, cached ? 603 : all);
        const auto stale = cuda.requestPropagation(inputs);
        cuda.build(model);
        cpu.build(model);
        EXPECT_THROW((void)cuda.getEventWeightsHostView(stale), std::runtime_error);
        compare(cuda, cpu, inputs, 3, 5, all);
      }
    }
  }
}

TEST(CudaBackendExternal, TablesAgreeWithNativeDialsAndRecoverAfterInvalidInputs) {
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    auto view = makeModel(true);
    // Minimal table-only model with duplicate event references and an empty Kriged stencil.
    auto& model = view.propagation;
    model.dials = {model.dials[1], model.dials[2], model.dials[3]};
    model.events.resize(2);
    model.events[0].resultIndex = 0;
    model.events[0].globalBinIndex = 0;
    model.events[1].resultIndex = 1;
    model.events[1].globalBinIndex = 1;
    model.eventWeights = {{2., 0, 3}, {3., 3, 2}};
    model.eventDialIndices = {0, 1, 0, 0, 2};
    CudaBackend cuda;
    cuda.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    cuda.build(view);
    std::vector<double> first{10., 2., 4.}, second{3., 5.};
    Tabulated tabulated(&first, 1, .25);
    Kriged kriged(&second, 3, {1, 0, 1}, {-.25, .5, .75});
    DialInputBuffer buffer;
    PropagationInputs inputs;
    inputs.externalWeights = {{first.data(), 3, 7, 1}, {second.data(), 2, 1, 1}, {nullptr, 0, 0, 1}};
    auto check = [&]{
      const auto token = cuda.requestPropagation(inputs);
      const double a = Semantics::clampDialResponse(model.dials[0], tabulated.evalResponse(buffer));
      const double b = kriged.evalResponse(buffer);
      const auto& weights = cuda.getEventWeightsHostView(token);
      expectClose(weights[0], 2. * a * b * a);
      expectClose(weights[1], 3. * a * .5);
    };
    check();
    auto good = inputs;
    const auto stale = cuda.requestPropagation(inputs);
    for( int error = 0; error < 5; ++error ){
      inputs = good;
      if( error == 0 ){ inputs.externalWeights[0].generation = 0; }
      if( error == 1 ){ inputs.externalWeights[0].values = nullptr; }
      if( error == 2 ){ inputs.externalWeights[0].count = 1; }
      if( error == 3 ){ inputs.externalWeights[0].destinationOffset = 0; }
      if( error == 4 ){ inputs.externalWeights.pop_back(); }
      EXPECT_THROW(cuda.requestPropagation(inputs), std::runtime_error);
      EXPECT_FALSE(cuda.isReady(stale));
      inputs = good;
      check();
      EXPECT_EQ(cuda.getLastTimingSummary().externalWeightUploadBlocks, 3);
    }
    // A device failure must force response recomputation and upload on retry,
    // even when the caller restores values without publishing a new generation.
    second[0] = std::numeric_limits<double>::quiet_NaN();
    ++inputs.externalWeights[1].generation;
    EXPECT_THROW(cuda.requestPropagation(inputs), std::runtime_error);
    second[0] = -2.;
    check();
    EXPECT_EQ(cuda.getLastTimingSummary().externalWeightUploadBlocks, 3);
    EXPECT_EQ(cuda.getLastTimingSummary().cacheCandidateCount, 3);
    first[2] = .5;
    ++inputs.externalWeights[0].generation;
    check();
    cuda.build(view);
    check();
    EXPECT_EQ(cuda.getLastTimingSummary().externalWeightUploadBlocks, 3);
  }
}

TEST(CudaBackendExternal, DispatcherReadsPublishedWorkerStorage) {
  auto config = ConfigReader(JsonType::parse(R"({
    "useBinnedWeights": true,
    "inputEventVarList": ["x"],
    "binning": {"binningDefinition": [{"name": "x", "edges": [0, 1, 2]}]}
  })"));
  config.defineFields({{"useBinnedWeights"}, {"inputEventVarList"}, {"binning"}});
  auto worker = std::make_unique<TestExternalWorker>();
  worker->configure(config);
  worker->initialize();
  auto native = worker->makeBinnedDial(1);
  const auto source = worker->getWeightSource();
  EngineView view;
  auto& model = view.propagation;
  model.externalWeightBlocks = {{0, source->size()}};
  model.totalBins = 1;
  BackendDialDescriptor external;
  external.type = BackendDialType::ExternalWeight;
  external.externalWeightIndex = 1;
  model.dials = {external};
  model.events.resize(1);
  model.eventWeights = {{2., 0, 1}};
  model.eventDialIndices = {0};
  CudaBackend cuda;
  cuda.build(view);
  DialInputBuffer buffer;
  PropagationInputs inputs;
  auto check = [&](std::size_t uploads_){
    inputs.externalWeights = {{source->data(), source->size(), 0, source->getGeneration()}};
    const auto token = cuda.requestPropagation(inputs);
    expectClose(cuda.getEventWeightsHostView(token)[0], 2. * native->evalResponse(buffer));
    EXPECT_EQ(cuda.getLastTimingSummary().externalWeightUploadBlocks, uploads_);
  };
  inputs.externalWeights = {{source->data(), source->size(), 0, 0}};
  EXPECT_THROW(cuda.requestPropagation(inputs), std::runtime_error);
  worker->updateWeights(buffer);
  check(1);
  check(0);
  worker->value = .25;
  DialInputBuffer changed;
  worker->updateWeights(changed);
  check(1);
  worker->fail = true;
  EXPECT_THROW(worker->updateWeights(buffer), std::runtime_error);
  check(0); // Failed producer has not published a generation.
  worker->fail = false;
  worker->value = -.5;
  worker->updateWeights(buffer);
  check(1);
  worker.reset(); // Dispatcher/source keep the mapped storage alive.
  check(0);
}

TEST(CudaBackendExternal, EmptyKrigedStencilNeedsNoDeviceValues) {
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    EngineView view;
    auto& model = view.propagation;
    model.externalWeightBlocks = {{42, 0}};
    BackendDialDescriptor dial;
    dial.type = BackendDialType::Kriged;
    dial.hasMinResponse = true;
    dial.minResponse = .5;
    model.dials = {dial};
    model.events.resize(1);
    model.eventWeights = {{2., 0, 1}};
    model.eventDialIndices = {0};
    CudaBackend cuda;
    cuda.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    cuda.build(view);
    PropagationInputs inputs;
    inputs.externalWeights = {{nullptr, 0, 42, 1}};
    for( int i = 0; i < 3; ++i ){
      if( i == 2 ){ ++inputs.externalWeights[0].generation; }
      const auto token = cuda.requestPropagation(inputs);
      EXPECT_DOUBLE_EQ(cuda.getEventWeightsHostView(token)[0], 1.);
      EXPECT_EQ(cuda.getLastTimingSummary().externalWeightUploadBytes, 0);
      EXPECT_EQ(cuda.getLastTimingSummary().externalWeightUploadBlocks, i == 1 ? 0 : 1);
    }
  }
}
