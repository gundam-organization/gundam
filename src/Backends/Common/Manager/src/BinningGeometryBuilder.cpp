#include "BinningGeometryBuilder.h"

#include "Histogram.h"

#include <algorithm>
#include <utility>

namespace Backends {
namespace {
BinningGeometry buildResponseGeometry(const BinningResponse& response_){
  BinningGeometry layout;
  layout.variableNames = response_.getVariableNames();
  layout.isSmoothed = true;
  auto appendBin = [&](const auto& responseBin_, const auto& widths_){
    BinningBin bin;
    bin.binIndex = responseBin_.binIndex;
    bin.firstInterval = layout.intervals.size();
    for( const auto& selection : responseBin_.selectionIntervals ){
      layout.intervals.push_back({selection.variableIndex, selection.lowerBound, selection.upperBound, 0, 0});
    }
    for( const auto& interval : responseBin_.intervals ){
      BinningInterval packed;
      packed.variableIndex = interval.variableIndex;
      packed.lowerBound = interval.lowerBound;
      packed.upperBound = interval.upperBound;
      auto widths = widths_(interval);
      packed.lowerInverseWidth = widths.first;
      packed.upperInverseWidth = widths.second;
      layout.intervals.emplace_back(packed);
    }
    bin.intervalCount = layout.intervals.size() - bin.firstInterval;
    layout.bins.emplace_back(bin);
  };
  if( response_.hasRelativeWidths() ){
    for( const auto& bin : response_.getRelativeResponseBins() ){
      appendBin(bin, [](const auto& interval_){ return std::make_pair(interval_.lowerInverseWidth, interval_.upperInverseWidth); });
    }
  }
  else{
    for( const auto& bin : response_.getResponseBins() ){
      appendBin(bin, [](const auto& interval_){ return std::make_pair(interval_.inverseWidth, interval_.inverseWidth); });
    }
  }
  return layout;
}

}

BinningGeometry buildBinningGeometry(const Histogram& histogram_){
  if( histogram_.isBinningResponseEnabled() ){ return buildResponseGeometry(histogram_.getBinningResponse()); }
  const auto& binContextList = histogram_.getBinContextList();
  BinningGeometry layout;
  if( binContextList.empty() ){ return layout; }
  layout.variableNames = binContextList.front().bin.buildVariableNameList();
  for( const auto& context : binContextList ){
    BinningBin bin;
    bin.binIndex = context.bin.getIndex();
    bin.firstInterval = layout.intervals.size();
    for( const auto& edge : context.bin.getEdgesList() ){
      auto position = std::find(layout.variableNames.begin(), layout.variableNames.end(), edge.varName);
      layout.intervals.push_back({std::size_t(position - layout.variableNames.begin()), edge.min, edge.max, 0, 0});
    }
    bin.intervalCount = layout.intervals.size() - bin.firstInterval;
    layout.bins.emplace_back(bin);
  }
  return layout;
}

}
