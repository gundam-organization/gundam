#include "CpuBackend.h"
#include "DeviceBackendModel.h"
#ifdef __APPLE__
#include "MpsBackend.h"
#endif
#include "ExternalWeightDialFactory.h"
#include "ExternalWeightDispatcher.h"
#include "GundamGlobals.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
  Backends::EngineView makeExternalModel() {
    using namespace Backends;
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = 2;
    model.totalBins = 2;
    model.externalWeightBlocks = {{0, 2}, {2, 1}};
    model.dialInputs = {{0}, {1}};
    BackendDialDescriptor norm;
    norm.type = BackendDialType::Norm;
    norm.inputCount = 1;
    BackendDialDescriptor shared;
    shared.type = BackendDialType::ExternalWeight;
    shared.inputCount = 2; // Producer dependencies do not constrain device evaluation.
    shared.externalWeightIndex = 1;
    shared.hasMinResponse = true;
    shared.minResponse = 0.;
    shared.hasMaxResponse = true;
    shared.maxResponse = 2.;
    BackendDialDescriptor second = shared;
    second.externalBlockIndex = 1;
    second.externalWeightIndex = 0;
    second.hasMinResponse = false;
    second.hasMaxResponse = false;
    BackendDialDescriptor first = shared;
    first.externalWeightIndex = 0;
    model.dials = {norm, shared, second, first};
    model.eventDialIndices = {0, 1, 2, 1, 0, 3};
    for( std::size_t iEvent = 0; iEvent < 3; ++iEvent ){
      EventView event;
      event.resultIndex = iEvent;
      event.globalBinIndex = iEvent == 2 ? 1 : 0;
      event.weight.firstDial = iEvent == 0 ? 0 : (iEvent == 1 ? 3 : 4);
      event.weight.dialCount = iEvent == 0 ? 3 : (iEvent == 1 ? 1 : 2);
      model.events.emplace_back(event);
    }
    LikelihoodSampleView sample;
    sample.dataSums = {0., 0.};
    sample.evalBin = [](double data_, double pred_, double err_, int){
      return (pred_ - data_) * (pred_ - data_) + err_ * err_;
    };
    view.likelihood.samples.emplace_back(sample);
    return view;
  }

  void checkPropagation(Backends::Backend& backend_, const Backends::PropagationInputs& inputs_,
                        const std::vector<double>& expected_, bool device_, std::size_t blocks_, std::size_t values_) {
    using namespace Backends;
    auto token = backend_.requestPropagation(inputs_);
    ASSERT_TRUE(token.isValid);
    backend_.wait(token);
    if( device_ ){
      ASSERT_EQ(backend_.getStatus(token).eventWeights, OutputState::ReadyOnDevice);
      const auto timing = backend_.getLastTimingSummary();
      EXPECT_EQ(timing.externalWeightUploadBlocks, blocks_);
      EXPECT_EQ(timing.externalWeightUploadBytes, values_ * sizeof(float));
    }
    backend_.materialize(token, OutputRequest::EventWeights);
    const auto& weights = backend_.getEventWeightsHostView(token);
    ASSERT_EQ(weights.size(), expected_.size());
    for( std::size_t i = 0; i < weights.size(); ++i ){ EXPECT_NEAR(weights[i], expected_[i], 1e-6); }
    const auto& sums = backend_.getHistogramSumsHostView(token);
    const auto& squares = backend_.getHistogramSumSquaresHostView(token);
    ASSERT_EQ(sums.size(), 2);
    EXPECT_NEAR(sums[0], expected_[0] + expected_[1], 1e-6);
    EXPECT_NEAR(sums[1], expected_[2], 1e-6);
    EXPECT_NEAR(squares[0], expected_[0]*expected_[0] + expected_[1]*expected_[1], 1e-6);
    EXPECT_NEAR(squares[1], expected_[2]*expected_[2], 1e-6);
    EXPECT_NEAR(backend_.getLikelihood(token), sums[0]*sums[0] + sums[1]*sums[1] + squares[0] + squares[1], 1e-6);
  }

  void exerciseBackend(Backends::Backend& backend_, bool device_) {
    GundamGlobals::setNumberOfThreads(2);
    auto model = makeExternalModel();
    backend_.build(model);
    std::vector<double> first{0.25, 3.};
    std::vector<double> second{-0.5};
    Backends::PropagationInputs inputs;
    inputs.parameters.values = {2., 42.};
    inputs.externalWeights = {{first.data(), first.size(), 0, 1}, {second.data(), second.size(), 2, 1}};
    checkPropagation(backend_, inputs, {-2., 2., 0.5}, device_, 2, 3);
    checkPropagation(backend_, inputs, {-2., 2., 0.5}, device_, 0, 0);
    first[0] = 0.5;
    first[1] = 1.5;
    ++inputs.externalWeights[0].generation;
    checkPropagation(backend_, inputs, {-1.5, 1.5, 1.}, device_, 1, 2);
    inputs.parameters.values[0] = 4.;
    checkPropagation(backend_, inputs, {-3., 1.5, 2.}, device_, 0, 0);
    second[0] = 0.;
    ++inputs.externalWeights[1].generation;
    checkPropagation(backend_, inputs, {0., 1.5, 2.}, device_, 1, 1);
    // Rebuilding must invalidate every uploaded generation.
    backend_.build(model);
    checkPropagation(backend_, inputs, {0., 1.5, 2.}, device_, 2, 3);
    inputs.externalWeights[0].count = 1;
    EXPECT_THROW(backend_.requestPropagation(inputs), std::runtime_error);
    inputs.externalWeights[0].count = first.size();
    // External-only models need no device parameter buffer contents.
    model.propagation.parameterCount = 0;
    model.propagation.dialInputs.clear();
    model.propagation.eventDialIndices = {1, 2, 1, 3};
    model.propagation.events[0].weight.firstDial = 0;
    model.propagation.events[0].weight.dialCount = 2;
    model.propagation.events[1].weight.firstDial = 2;
    model.propagation.events[1].weight.dialCount = 1;
    model.propagation.events[2].weight.firstDial = 3;
    model.propagation.events[2].weight.dialCount = 1;
    inputs.parameters.values.clear();
    backend_.build(model);
    checkPropagation(backend_, inputs, {0., 1.5, 0.5}, device_, 2, 3);
  }

  class TestExternalWorker : public ExternalWeightWorker {
  public:
    double value{2.};
    bool fail{false};
    const double* getMappedValues() const { return getWeightBuffer()->ptr; }
  private:
    void evaluateImpl(const DialInputBuffer&) override {
      if( fail ){ throw std::runtime_error("Worker failure"); }
      std::fill_n(getWeightBuffer()->ptr, getWeightCount(), value);
    }
  };
}

TEST(ExternalWeightBackend, Cpu) {
  Backends::CpuBackend backend;
  exerciseBackend(backend, false);
}

#ifdef __APPLE__
TEST(ExternalWeightBackend, Mps) {
  Backends::MpsBackend backend;
  if( backend.getCapabilities().deviceName == "Metal unavailable" ){ GTEST_SKIP() << "Metal unavailable"; }
  exerciseBackend(backend, true);
}
#endif

TEST(ExternalWeightBackend, CommonPacking) {
  auto view = makeExternalModel();
  Backends::DevicePackedModel packed;
  Backends::BackendTimingSummary timing;
  std::string reason;
  ASSERT_TRUE(Backends::packDeviceBackendModel(view.propagation, true, packed, timing, reason, "test")) << reason;
  ASSERT_EQ(packed.externalDialOccurrences.size(), 4);
  EXPECT_EQ(packed.externalDialOccurrences[0].weightIndex, 1);
  EXPECT_EQ(packed.externalDialOccurrences[1].weightIndex, 2);
  EXPECT_EQ(packed.externalDialOccurrences[2].weightIndex, 1);
  EXPECT_EQ(packed.externalDialOccurrences[3].weightIndex, 0);
}

TEST(ExternalWeightBackend, SharedStorageAndGeneration) {
  auto config = ConfigReader(JsonType::parse(R"({
    "useBinnedWeights": true,
    "inputEventVarList": ["x"],
    "binning": {"binningDefinition": [{"name": "x", "edges": [0, 1, 2]}]}
  })"));
  config.defineFields({{"useBinnedWeights"}, {"inputEventVarList"}, {"binning"}});
  auto worker = std::make_unique<TestExternalWorker>();
  worker->configure(config);
  worker->initialize();
  TestExternalWorker other;
  other.configure(config);
  other.initialize();
  DialInputBuffer input;
  auto dial = worker->makeBinnedDial(1);
  auto source = worker->getWeightSource();
  EXPECT_EQ(source->getGeneration(), 0);
  worker->updateWeights(input);
  EXPECT_EQ(source->data(), worker->getMappedValues());
  EXPECT_EQ(source->getGeneration(), 1);
  EXPECT_DOUBLE_EQ(dial->evalResponse(input), 2.);
  // A second live worker must have independent shared-memory names and storage.
  other.value = 7.;
  other.updateWeights(input);
  EXPECT_NE(other.getWeightSource()->data(), source->data());
  EXPECT_DOUBLE_EQ(source->data()[1], 2.);
  input.update(); // No parameter changed.
  worker->value = 3.;
  worker->updateWeights(input);
  EXPECT_EQ(source->getGeneration(), 1);
  EXPECT_DOUBLE_EQ(dial->evalResponse(input), 2.);
  DialInputBuffer changed;
  worker->updateWeights(changed);
  EXPECT_EQ(source->getGeneration(), 2);
  EXPECT_DOUBLE_EQ(dial->evalResponse(input), 3.);
  worker->fail = true;
  EXPECT_THROW(worker->updateWeights(changed), std::runtime_error);
  EXPECT_EQ(source->getGeneration(), 2);
  worker->fail = false;
  worker->value = 4.;
  worker->updateWeights(input); // Retry even when parameter values did not change.
  EXPECT_EQ(source->getGeneration(), 3);
  worker.reset();
  EXPECT_DOUBLE_EQ(dial->evalResponse(input), 4.); // Dispatcher retains the mapping.
}
