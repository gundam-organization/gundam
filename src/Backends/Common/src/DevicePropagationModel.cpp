#include "DevicePropagationModel.h"
#include "ResponseSmoothingPacking.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace {
  Backends::DeviceIndex index(std::size_t value_) {
    if( value_ >= std::numeric_limits<Backends::DeviceIndex>::max() ){
      throw std::runtime_error("Device propagation exceeds the 32-bit indexing limit.");
    }
    return Backends::DeviceIndex(value_);
  }
  Backends::ObservableScalar scalar(double value_) {
    float result = float(value_);
    if( std::isfinite(value_) and not std::isfinite(result) ){
      throw std::runtime_error("Observable propagation value exceeds device float range.");
    }
    return Backends::ObservableScalar(value_);
  }
  std::size_t contributionBound(const Backends::BinningGeometry& layout_, const Backends::BinningIndex& index_) {
    if( not layout_.isSmoothed ){ return 1; }
    std::size_t largest{0};
    auto dimension = layout_.variableNames.size();
    auto bounds = index_.getBinBounds();
    for( auto& bound : bounds ){
      bound.lower = std::nextafter(float(bound.lower), -std::numeric_limits<float>::infinity());
      bound.upper = std::nextafter(float(bound.upper), std::numeric_limits<float>::infinity());
    }
    // Supports sharing a point necessarily intersect pairwise. Maximum degree
    // gives a conservative capacity without a dense event x bin allocation.
    for( std::size_t a = 0; a < layout_.bins.size(); ++a ){
      std::size_t count{0};
      for( std::size_t b = 0; b < layout_.bins.size(); ++b ){
        bool intersects{true};
        for( std::size_t axis = 0; axis < dimension; ++axis ){
          const auto& first = bounds[a * dimension + axis];
          const auto& second = bounds[b * dimension + axis];
          if( first.upper < second.lower or second.upper < first.lower ){ intersects = false; break; }
        }
        count += intersects;
      }
      largest = std::max(largest, count);
    }
    return largest + 1; // Includes the fraction outside the domain.
  }
}

void Backends::DevicePropagationModel::build(const PropagationView& model_) {
  *this = DevicePropagationModel();
  histogramBins = index(std::size_t(model_.totalBins) + model_.samples.size());
  for( double value : model_.variableValues ){ variables.push_back(scalar(value)); }
  for( double value : model_.dialPayloads ){ payloads.push_back(scalar(value)); }
  weightDialIndices = model_.eventDialIndices;
  variableDialIndices = model_.variableDialIndices;
  for( const auto& dial : model_.dials ){
    DevicePropagationDial packed{};
    packed.type = DeviceIndex(dial.type);
    packed.payloadOffset = index(dial.payloadOffset);
    packed.payloadSize = index(dial.payloadSize);
    packed.allowExtrapolation = dial.allowExtrapolation;
    packResponseLimits(dial, packed, scalar);
    if( dial.type == BackendDialType::ExternalWeight ){
      const auto& block = model_.externalWeightBlocks.at(dial.externalBlockIndex);
      if( dial.externalWeightIndex >= block.count ){ throw std::runtime_error("Invalid external observable dial index."); }
      packed.externalIndex = index(block.offset + dial.externalWeightIndex);
    }
    else if( dial.type != BackendDialType::Shift ){
      if( dial.inputCount != 1 ){ throw std::runtime_error("Device observable dials require exactly one parameter input."); }
      const auto& input = model_.dialInputs.at(dial.firstInput);
      packed.parameterIndex = index(input.parameterIndex);
      packed.useMirror = input.useMirror;
      packed.mirrorMin = scalar(input.mirrorMin);
      packed.mirrorRange = scalar(input.mirrorRange);
    }
    dials.push_back(packed);
  }
  for( const auto& response : model_.variableResponses ){
    variableResponses.push_back({index(response.variableIndex), scalar(response.referenceValue),
      response.overrideDial == std::size_t(-1) ? DeviceIndex(-1) : index(response.overrideDial),
      index(response.firstMultiplyDial), index(response.multiplyDialCount), index(response.firstAddDial), index(response.addDialCount)});
  }
  std::vector<std::size_t> capacities;
  for( const auto& sample : model_.samples ){
    const auto& layout = sample.binning;
    BinningIndex search;
    search.build(layout);
    capacities.push_back(contributionBound(layout, search));
    DeviceBinningSample packed{index(sample.binOffset), index(sample.binCount), index(nodes.size()),
      index(search.getNodes().size()), index(layout.variableNames.size()), DeviceIndex(layout.isSmoothed),
      DeviceIndex(sample.useDynamicBinning), DeviceIndex(sample.readHostVariables), index(model_.totalBins + samples.size())};
    std::size_t binOffset = bins.size();
    for( const auto& bin : layout.bins ){
      bins.push_back({bin.binIndex, index(intervals.size()), index(bin.intervalCount), index(bins.size() - binOffset)});
      for( std::size_t i = bin.firstInterval; i < bin.firstInterval + bin.intervalCount; ++i ){
        const auto& interval = layout.intervals.at(i);
        auto lower = scalar(interval.lowerBound), upper = scalar(interval.upperBound);
        auto lowerWidth = scalar(interval.lowerInverseWidth), upperWidth = scalar(interval.upperInverseWidth);
        if( (interval.lowerBound != interval.upperBound and lower == upper)
            or (interval.lowerBound == interval.upperBound and (double(lower.high) + lower.low) != interval.lowerBound)
            or (interval.lowerInverseWidth != 0 and (lowerWidth == ObservableScalar(0.f) or upperWidth == ObservableScalar(0.f))) ){
          throw std::runtime_error("Bin geometry cannot be represented with compensated observable precision.");
        }
        intervals.push_back({index(interval.variableIndex), lower, upper, lowerWidth, upperWidth});
      }
    }
    std::size_t nodeOffset = nodes.size(), boundOffset = bounds.size(), positionOffset = nodeBins.size();
    for( auto position : search.getBinPositions() ){ nodeBins.push_back(index(binOffset + position)); }
    for( const auto& bound : search.getNodeBounds() ){
      bounds.push_back({std::nextafter(float(scalar(bound.lower)), -std::numeric_limits<float>::infinity()),
                        std::nextafter(float(scalar(bound.upper)), std::numeric_limits<float>::infinity())});
    }
    for( const auto& node : search.getNodes() ){
      nodes.push_back({index(boundOffset + (nodes.size() - nodeOffset) * layout.variableNames.size()),
                       index(positionOffset + node.first), node.left < 0 ? index(node.count) : 0, 0});
    }
    std::function<void(int, std::size_t)> setSkip = [&](int node_, std::size_t skip_){
      nodes[nodeOffset + node_].skip = index(skip_);
      const auto& node = search.getNodes()[node_];
      if( node.left >= 0 ){
        setSkip(node.left, nodeOffset + node.right);
        setSkip(node.right, skip_);
      }
    };
    if( not search.getNodes().empty() ){ setSkip(0, nodes.size()); }
    samples.push_back(packed);
  }
  // Fixed-bin standalone views need not carry SampleView bindings.
  samples.push_back({0, index(model_.totalBins), 0, 0, 0, 0, 0, 0, 0});
  for( const auto& event : model_.events ){
    std::size_t sample = event.samplePosition < model_.samples.size() ? event.samplePosition : model_.samples.size();
    std::size_t capacity = sample < capacities.size() ? capacities[sample] : 1;
    DevicePropagationEvent packed{index(event.weight.firstDial), index(event.weight.dialCount), index(event.firstVariable),
      index(event.variableCount), index(event.firstVariableResponse), index(event.variableResponseCount), index(sample),
      contributionCapacity, index(capacity), sample == model_.samples.size() ? event.globalBinIndex : event.binIndex,
      float(scalar(event.weight.baseWeight))};
    contributionCapacity = index(std::size_t(contributionCapacity) + capacity);
    // Device outputs are indexed consistently with resultIndex on every backend.
    if( event.resultIndex != events.size() ){ throw std::runtime_error("Device propagation requires contiguous event result indices."); }
    events.push_back(packed);
  }
  histogramChunkCapacity = index(std::size_t(contributionCapacity) / 256 + histogramBins + 1);
}
