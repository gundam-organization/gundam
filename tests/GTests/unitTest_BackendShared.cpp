#include "BinningGeometry.h"
#include "DialDependencies.h"
#include "DialDependencyGroups.h"
#include "HistogramEventIndex.h"
#ifdef __APPLE__
#include "MpsBackend.h"
#endif

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

TEST(BackendSharedBinning, StrictMembershipPreservesHalfOpenAndDegenerateIntervals) {
  // One continuous axis and one categorical axis, with a nonzero offset.
  const Backends::BinningInterval intervals[]{
      {0, -100., 100.}, {0, 0., 1.}, {1, 3., 3.}};
  const Backends::BinningBin bin{7, 1, 2};
  double values[]{0., 3.};
  EXPECT_TRUE(Backends::BinningSemantics::containsStrictBin(bin, intervals, values));
  values[0] = std::nextafter(1., 0.);
  EXPECT_TRUE(Backends::BinningSemantics::containsStrictBin(bin, intervals, values));
  values[0] = 1.;
  EXPECT_FALSE(Backends::BinningSemantics::containsStrictBin(bin, intervals, values));
  values[0] = 0.5;
  values[1] = std::nextafter(3., 4.);
  EXPECT_FALSE(Backends::BinningSemantics::containsStrictBin(bin, intervals, values));

  // The Metal descriptors use fixed-width indices and compensated scalars.
  const Backends::DeviceBinningInterval packedIntervals[]{
      {0, 0, 1, 0, 0}, {1, 3, 3, 0, 0}};
  const Backends::DeviceBinningBin packedBin{7, 0, 2, 0};
  const Backends::ObservableScalar packedValues[]{0.5, 3};
  EXPECT_TRUE(Backends::BinningSemantics::containsStrictBin(packedBin, packedIntervals, packedValues));
}

TEST(BackendSharedBinning, FractionPreservesProductsAndZeroShortCircuit) {
  const Backends::BinningInterval intervals[]{
      {0, 0., 1., 1., 1.}, {1, 0., 1., 1., 1.}};
  const Backends::BinningBin bin{0, 0, 2};
  const double atCorner[]{0., 0.};
  EXPECT_DOUBLE_EQ(Backends::BinningSemantics::binFraction(bin, intervals, atCorner), 0.25);
  const double outside[]{-2., std::numeric_limits<double>::quiet_NaN()};
  EXPECT_DOUBLE_EQ(Backends::BinningSemantics::binFraction(bin, intervals, outside), 0.);
}

TEST(BackendSharedBinning, TotalPreservesToleranceBoundaryAndCompensatedResidual) {
  using Backends::BinningSemantics::finalizeBinningTotal;
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double tolerance = 64 * epsilon;
  auto atLimit = finalizeBinningTotal(1. + tolerance, tolerance, true);
  EXPECT_TRUE(atLimit.isValid);
  EXPECT_DOUBLE_EQ(atLimit.total, 1.);
  EXPECT_DOUBLE_EQ(atLimit.excess, tolerance);
  EXPECT_FALSE(finalizeBinningTotal(std::nextafter(1. + tolerance, 2.), tolerance, true).isValid);
  EXPECT_FALSE(finalizeBinningTotal(std::numeric_limits<double>::infinity(), tolerance, false).isValid);
  EXPECT_FALSE(finalizeBinningTotal(std::numeric_limits<double>::quiet_NaN(), tolerance, false).isValid);
  const auto partial = finalizeBinningTotal(0.75, tolerance, true);
  EXPECT_DOUBLE_EQ(partial.total, 0.75);
  EXPECT_DOUBLE_EQ(partial.excess, 0.);

  // A residual smaller than float32 epsilon must survive normalization.
  using Scalar = Backends::ObservableScalar;
  const Scalar total(1.f, 0x1p-30f);
  const auto compensated = finalizeBinningTotal(total, 64 * std::numeric_limits<float>::epsilon(), true);
  EXPECT_TRUE(compensated.isValid);
  EXPECT_TRUE(compensated.total == Scalar(1));
  EXPECT_TRUE(compensated.excess == Scalar(0x1p-30f));
}

TEST(BackendSharedDependencies, PreserveInputOrderAndUsePublicationForProducers) {
  Backends::PropagationView model;
  model.dialInputs = {{4}, {1}, {4}};
  Backends::BackendDialDescriptor dial;
  dial.type = Backends::BackendDialType::Bicubic;
  dial.inputCount = 3;
  dial.externalBlockIndex = 2;
  std::vector<std::size_t> parameters, external;
  auto parameter = [&](const auto& input_){ parameters.push_back(input_.parameterIndex); };
  auto block = [&](std::size_t index_){ external.push_back(index_); };
  Backends::visitDialDependencies(model, dial, parameter, block);
  EXPECT_EQ(parameters, (std::vector<std::size_t>{4, 1, 4}));
  EXPECT_TRUE(external.empty());
  for( auto type : {Backends::BackendDialType::ExternalWeight, Backends::BackendDialType::Tabulated,
                    Backends::BackendDialType::Kriged} ){
    parameters.clear();
    external.clear();
    dial.type = type;
    Backends::visitDialDependencies(model, dial, parameter, block);
    EXPECT_TRUE(parameters.empty());
    EXPECT_EQ(external, (std::vector<std::size_t>{2}));
  }
}

template<typename Index>
void checkStableHistogramIndex() {
  std::vector<Backends::EventView> events(5);
  const int bins[]{2, -1, 0, 2, 0};
  const std::size_t slots[]{4, 2, 3, 0, 1};
  for( std::size_t i = 0; i < events.size(); ++i ){
    events[i].globalBinIndex = bins[i];
    events[i].resultIndex = slots[i];
  }
  std::vector<Index> offsets{0, 2, 0, 2}, indices;
  Backends::buildHistogramEventIndex(events, offsets, indices);
  EXPECT_EQ(offsets, (std::vector<Index>{0, 2, 2, 4}));
  EXPECT_EQ(indices, (std::vector<Index>{3, 1, 4, 0}));
  offsets = {0};
  Backends::buildHistogramEventIndex(std::vector<Backends::EventView>{}, offsets, indices);
  EXPECT_EQ(offsets, (std::vector<Index>{0}));
  EXPECT_TRUE(indices.empty());
}

TEST(BackendSharedHistogramIndex, PreservesTraversalOrderForBothIndexWidths) {
  checkStableHistogramIndex<std::uint32_t>();
  checkStableHistogramIndex<std::size_t>();
}

TEST(BackendSharedDependencies, GroupsPreserveSlotsAndSortedInputPairs) {
  using namespace Backends;
  PropagationView model;
  model.parameterCount = 5;
  model.dialInputs = {{3}, {1}, {1}, {3}, {2}, {2}, {0}, {2}};
  auto add = [&](BackendDialType type_, std::size_t first_, std::size_t count_, std::size_t block_ = 0){
    BackendDialDescriptor dial;
    dial.type = type_;
    dial.firstInput = first_;
    dial.inputCount = count_;
    dial.externalBlockIndex = block_;
    model.dials.push_back(dial);
  };
  add(BackendDialType::Bilinear, 0, 2);
  add(BackendDialType::Norm, 0, 1);
  add(BackendDialType::ExternalWeight, 0, 2, 2);
  add(BackendDialType::Bicubic, 2, 2); // Same dependency pair, reversed input order.
  add(BackendDialType::Shift, 99, 1); // Constant payloads have no input dependency.
  add(BackendDialType::Bilinear, 4, 2); // Repeated parameter remains a multi-input group.
  add(BackendDialType::Norm, 6, 1);
  add(BackendDialType::Tabulated, 4, 2, 0); // Producer inputs must not enter parameter groups.
  add(BackendDialType::Norm, 7, 1);
  add(BackendDialType::ExternalWeight, 0, 0, 2);
  add(BackendDialType::Bicubic, 4, 2);

  const auto groups = buildDialDependencyGroups(model, model.dials);
  EXPECT_EQ(groups.parameterOffsets, (std::vector<std::size_t>{0, 1, 1, 2, 3, 3}));
  EXPECT_EQ(groups.parameterDialIndices, (std::vector<std::uint32_t>{6, 8, 1}));
  ASSERT_EQ(groups.externalDials.size(), 2);
  EXPECT_EQ(groups.externalDials.at(0), (std::vector<std::uint32_t>{7}));
  EXPECT_EQ(groups.externalDials.at(2), (std::vector<std::uint32_t>{2, 9}));
  ASSERT_EQ(groups.multiParameterDials.size(), 2);
  EXPECT_EQ(groups.multiParameterDials.at({1, 3}), (std::vector<std::uint32_t>{0, 3}));
  EXPECT_EQ(groups.multiParameterDials.at({2, 2}), (std::vector<std::uint32_t>{5, 10}));
  EXPECT_EQ(model.dialInputs[0].parameterIndex, 3);
  EXPECT_EQ(model.dialInputs[1].parameterIndex, 1);

  // Packing may reorder descriptors before grouping; slots must refer to the
  // supplied descriptors, independently of their position in the source model.
  const std::vector<BackendDialDescriptor> reordered{
      model.dials[9], model.dials[6], model.dials[5], model.dials[0], model.dials[3]};
  const auto remapped = buildDialDependencyGroups<std::size_t>(model, reordered);
  EXPECT_EQ(remapped.parameterOffsets, (std::vector<std::size_t>{0, 1, 1, 1, 1, 1}));
  EXPECT_EQ(remapped.parameterDialIndices, (std::vector<std::size_t>{1}));
  EXPECT_EQ(remapped.externalDials.at(2), (std::vector<std::size_t>{0}));
  EXPECT_EQ(remapped.multiParameterDials.at({2, 2}), (std::vector<std::size_t>{2}));
  EXPECT_EQ(remapped.multiParameterDials.at({1, 3}), (std::vector<std::size_t>{3, 4}));
}

TEST(BackendSharedDependencies, EmptyAndConstantModelsHaveEmptyGroups) {
  using namespace Backends;
  PropagationView model;
  const auto empty = buildDialDependencyGroups(model, model.dials);
  EXPECT_EQ(empty.parameterOffsets, (std::vector<std::size_t>{0}));
  EXPECT_TRUE(empty.parameterDialIndices.empty());
  EXPECT_TRUE(empty.multiParameterDials.empty());
  EXPECT_TRUE(empty.externalDials.empty());
  model.parameterCount = 3;
  BackendDialDescriptor constant;
  constant.type = BackendDialType::Shift;
  model.dials.push_back(constant);
  const auto groups = buildDialDependencyGroups(model, model.dials);
  EXPECT_EQ(groups.parameterOffsets, (std::vector<std::size_t>{0, 0, 0, 0}));
  EXPECT_TRUE(groups.parameterDialIndices.empty());
  EXPECT_TRUE(groups.multiParameterDials.empty());
  EXPECT_TRUE(groups.externalDials.empty());
}

TEST(BackendMonitor, LabelsMemoryScopeAndGpuTimeShare) {
  using namespace Backends;
  BackendTimingSummary timing;
  EXPECT_TRUE(formatBackendMonitorLine("CPU", {}, timing).empty());
  BackendResourceUsage usage;
  usage.available = true;
  usage.deviceName = "Test device";
  usage.allocatedBytes = 1024 * 1024;
  usage.hasDeviceMemoryInfo = true;
  usage.freeDeviceBytes = 3 * 1024 * 1024;
  usage.totalDeviceBytes = 4 * 1024 * 1024;
  timing.devicePropagationSeconds = .002;
  timing.propagationWallSeconds = .008;
  const auto cuda = formatBackendMonitorLine("CUDA/cachedResponses", usage, timing);
  EXPECT_EQ(cuda, "Backend CUDA (Test device) / Memory: 1.05MB / On-device time: 25.0%");

  usage.processMemory = usage.unifiedMemory = true;
  usage.hasDeviceMemoryInfo = false;
  usage.recommendedDeviceBytes = 4 * 1024 * 1024;
  const auto mps = formatBackendMonitorLine("MPS/cachedResponses", usage, {});
  EXPECT_EQ(mps, "Backend MPS (Test device) / Memory: 1.05MB");

  timing.devicePropagationSeconds = .02;
  EXPECT_NE(formatBackendMonitorLine("MPS", usage, timing).find("On-device time: n/a"), std::string::npos);
  timing.devicePropagationSeconds = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(formatBackendMonitorLine("MPS", usage, timing).find("On-device time:"), std::string::npos);
}

#ifdef __APPLE__
TEST(BackendMonitor, MetalResourceQueriesPreservePropagationResultsInBothModes) {
  using namespace Backends;
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    MpsBackend backend;
    if( backend.getCapabilities().deviceName == "Metal unavailable" ){ GTEST_SKIP() << "Metal unavailable"; }
    EXPECT_TRUE(backend.getMonitorLine().empty());
    backend.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    EngineView view;
    view.propagation.totalBins = 1;
    EventView event;
    event.globalBinIndex = event.binIndex = 0;
    view.propagation.events = {event};
    view.propagation.eventWeights = {{2., 0, 0}};
    backend.build(view);
    auto usage = backend.getResourceUsage();
    ASSERT_TRUE(usage.available);
    EXPECT_TRUE(usage.processMemory);
    EXPECT_GT(usage.allocatedBytes, 0);
    EXPECT_GT(usage.recommendedDeviceBytes, 0);
    EXPECT_FALSE(usage.deviceName.empty());
    EXPECT_EQ(backend.getMonitorLine().find("On-device time:"), std::string::npos);
    const auto token = backend.requestPropagation({});
    EXPECT_TRUE(backend.isReady(token));
    EXPECT_GT(backend.getLastTimingSummary().propagationWallSeconds, 0.);
    EXPECT_NE(backend.getMonitorLine().find("On-device time:"), std::string::npos);
    EXPECT_TRUE(backend.isReady(token));
    backend.materialize(token, OutputRequest::EventWeights);
    EXPECT_EQ(backend.getEventWeightsHostView(token), (std::vector<double>{2.}));
  }
}
#endif
