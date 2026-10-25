#include "BinningGeometry.h"
#include "DialDependencies.h"
#include "HistogramEventIndex.h"

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
