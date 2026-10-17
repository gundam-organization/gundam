#include "BinningResponse.h"

#include "DeviceMath/SharedBinningMath.h"

#include "Logger.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <utility>

void BinningResponse::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({{"isEnabled"}, {"variables"}});
  _config_.checkConfiguration();
  _config_.fillValue(_isEnabled_, "isEnabled");
  _axes_.clear();
  _hasRelativeWidths_ = false;
  for( auto axisConfig : _config_.loop("variables") ){
    axisConfig.clearFields();
    axisConfig.defineFields({{FieldFlag::MANDATORY, "name"}, {"width"}, {"widthFraction"}});
    axisConfig.checkConfiguration();
    Axis axis;
    axisConfig.fillValue(axis.name, "name");
    LogThrowIf(axis.name.empty(), "Binning response requires a variable name.");
    LogThrowIf(axisConfig.hasField("width") == axisConfig.hasField("widthFraction"),
               "Binning response requires exactly one of width and widthFraction for " << axis.name);
    if( axisConfig.hasField("widthFraction") ){
      axisConfig.fillValue(axis.widthFraction, "widthFraction");
      LogThrowIf(not std::isfinite(axis.widthFraction) or axis.widthFraction <= 0 or axis.widthFraction > 0.5,
                 "Binning response widthFraction must be in (0, 0.5] for " << axis.name);
      _hasRelativeWidths_ = true;
    }
    else{
      axisConfig.fillValue(axis.width, "width");
      LogThrowIf(not std::isfinite(axis.width) or axis.width <= 0 or not std::isfinite(1 / axis.width),
                 "Binning response requires a finite, positive width for " << axis.name);
    }
    LogThrowIf(findAxis(axis.name) != nullptr, "Duplicate binning response variable: " << axis.name);
    _axes_.emplace_back(std::move(axis));
  }
  LogThrowIf(_isEnabled_ and _axes_.empty(), "Enabled binning response requires at least one variable.");
}

const BinningResponse::Axis* BinningResponse::findAxis(const std::string& variableName_) const{
  for( const auto& axis : _axes_ ){
    if( axis.name == variableName_ ){ return &axis; }
  }
  return nullptr;
}

void BinningResponse::build(const std::vector<Bin>& binList_){
  _responseBins_.clear();
  _relativeResponseBins_.clear();
  _variableNames_.clear();
  if( not _isEnabled_ ){ return; }
  LogThrowIf(binList_.empty(), "Binning response requires a nonempty binning.");

  for( const auto& edge : binList_.front().getEdgesList() ){ _variableNames_.emplace_back(edge.varName); }

  for( const auto& bin : binList_ ){
    LogThrowIf(bin.getEdgesList().size() != binList_.front().getEdgesList().size(),
               "Binning response requires the same variables in every bin.");
    for( const auto& edge : binList_.front().getEdgesList() ){
      LogThrowIf(not bin.isVariableSet(edge.varName),
                 "Binning response requires the same variables in every bin: " << edge.varName);
    }
    for( const auto& axis : _axes_ ){
      const auto* edge = bin.getVarEdgesPtr(axis.name);
      LogThrowIf(edge == nullptr or edge->isConditionVar or not std::isfinite(edge->min)
                 or not std::isfinite(edge->max) or edge->min >= edge->max,
                 "Binning response requires finite, nonzero ranges for " << axis.name);
    }
  }

  for( size_t i = 0 ; i < binList_.size() ; ++i ){
    for( size_t j = 0 ; j < i ; ++j ){
      bool overlaps{true};
      for( const auto& edge : binList_[i].getEdgesList() ){
        const auto& other = binList_[j].getVarEdges(edge.varName);
        bool intersects;
        if( edge.min == edge.max ){ intersects = binList_[j].isBetweenEdges(other, edge.min); }
        else if( other.min == other.max ){ intersects = binList_[i].isBetweenEdges(edge, other.min); }
        else{ intersects = edge.min < other.max and other.min < edge.max; }
        if( not intersects ){ overlaps = false; break; }
      }
      LogThrowIf(overlaps, "Binning response does not support overlapping bins.");
    }
  }

  if( _hasRelativeWidths_ ){ buildRelativeResponse(binList_); }
  else{ buildAbsoluteResponse(binList_); }
}

std::vector<BinningResponse::SelectionInterval> BinningResponse::buildSelectionIntervals(const Bin& bin_) const{
  std::vector<SelectionInterval> intervals;
  for( size_t iVariable = 0 ; iVariable < _variableNames_.size() ; ++iVariable ){
    const auto& edge = bin_.getVarEdges(_variableNames_[iVariable]);
    if( findAxis(edge.varName) != nullptr ){ continue; }
    SelectionInterval interval;
    interval.variableIndex = iVariable;
    interval.lowerBound = edge.min;
    interval.upperBound = edge.max;
    intervals.emplace_back(interval);
  }
  return intervals;
}

bool BinningResponse::haveOverlappingSelections(const Bin& first_, const Bin& second_) const{
  for( const auto& edge : first_.getEdgesList() ){
    if( findAxis(edge.varName) != nullptr ){ continue; }
    const auto& other = second_.getVarEdges(edge.varName);
    bool intersects;
    if( edge.min == edge.max ){ intersects = second_.isBetweenEdges(other, edge.min); }
    else if( other.min == other.max ){ intersects = first_.isBetweenEdges(edge, other.min); }
    else{ intersects = edge.min < other.max and other.min < edge.max; }
    if( not intersects ){ return false; }
  }
  return true;
}

void BinningResponse::buildAbsoluteResponse(const std::vector<Bin>& binList_){
  _responseBins_.resize(binList_.size());
  for( size_t iBin = 0 ; iBin < binList_.size() ; ++iBin ){
    auto& responseBin = _responseBins_[iBin];
    responseBin.binIndex = binList_[iBin].getIndex();
    responseBin.selectionIntervals = buildSelectionIntervals(binList_[iBin]);
    for( const auto& axis : _axes_ ){
      const auto& edge = binList_[iBin].getVarEdges(axis.name);
      ResponseInterval interval;
      interval.variableIndex = std::distance(_variableNames_.begin(), std::find(_variableNames_.begin(), _variableNames_.end(), axis.name));
      interval.inverseWidth = 1 / axis.width;
      interval.lowerBound = edge.min;
      interval.upperBound = edge.max;
      responseBin.intervals.emplace_back(interval);
    }
  }
}

void BinningResponse::buildRelativeResponse(const std::vector<Bin>& binList_){
  struct AxisTransitions{
    std::vector<double> edges;
    std::vector<double> inverseWidths;
  };

  std::vector<size_t> selectionGroups(binList_.size());
  std::iota(selectionGroups.begin(), selectionGroups.end(), size_t(0));
  auto findGroup = [&](size_t index_){
    while( selectionGroups[index_] != index_ ){
      selectionGroups[index_] = selectionGroups[selectionGroups[index_]];
      index_ = selectionGroups[index_];
    }
    return index_;
  };
  for( size_t i = 0 ; i < binList_.size() ; ++i ){
    for( size_t j = 0 ; j < i ; ++j ){
      if( haveOverlappingSelections(binList_[i], binList_[j]) ){
        selectionGroups[findGroup(i)] = findGroup(j);
      }
    }
  }
  std::vector<std::vector<size_t>> groups(binList_.size());
  for( size_t iBin = 0 ; iBin < binList_.size() ; ++iBin ){ groups[findGroup(iBin)].emplace_back(iBin); }

  _relativeResponseBins_.resize(binList_.size());
  for( const auto& group : groups ){
    if( group.empty() ){ continue; }
    std::vector<AxisTransitions> axisTransitions(_axes_.size());
    for( size_t iAxis = 0 ; iAxis < _axes_.size() ; ++iAxis ){
      const auto& axis = _axes_[iAxis];
      if( axis.widthFraction == 0 ){ continue; }
      auto& transitions = axisTransitions[iAxis];
      for( size_t iBin : group ){
        const auto& edge = binList_[iBin].getVarEdges(axis.name);
        transitions.edges.emplace_back(edge.min);
        transitions.edges.emplace_back(edge.max);
      }
      std::sort(transitions.edges.begin(), transitions.edges.end());
      transitions.edges.erase(std::unique(transitions.edges.begin(), transitions.edges.end()), transitions.edges.end());
      for( size_t iEdge = 0 ; iEdge < transitions.edges.size() ; ++iEdge ){
        double spacing = std::numeric_limits<double>::infinity();
        if( iEdge > 0 ){ spacing = std::min(spacing, transitions.edges[iEdge] - transitions.edges[iEdge - 1]); }
        if( iEdge + 1 < transitions.edges.size() ){ spacing = std::min(spacing, transitions.edges[iEdge + 1] - transitions.edges[iEdge]); }
        double width = axis.widthFraction * spacing;
        LogThrowIf(not std::isfinite(width) or width <= 0 or not std::isfinite(1 / width),
                   "Invalid relative binning response width for " << axis.name << " at " << transitions.edges[iEdge]);
        transitions.inverseWidths.emplace_back(1 / width);
      }
    }

    for( size_t iBin : group ){
      auto& responseBin = _relativeResponseBins_[iBin];
      responseBin.binIndex = binList_[iBin].getIndex();
      responseBin.selectionIntervals = buildSelectionIntervals(binList_[iBin]);
      for( size_t iAxis = 0 ; iAxis < _axes_.size() ; ++iAxis ){
        const auto& axis = _axes_[iAxis];
        const auto& edge = binList_[iBin].getVarEdges(axis.name);
        RelativeResponseInterval interval;
        interval.variableIndex = std::distance(_variableNames_.begin(), std::find(_variableNames_.begin(), _variableNames_.end(), axis.name));
        interval.lowerBound = edge.min;
        interval.upperBound = edge.max;
        if( axis.widthFraction != 0 ){
          const auto& transitions = axisTransitions[iAxis];
          auto lowerIndex = std::distance(transitions.edges.begin(), std::lower_bound(transitions.edges.begin(), transitions.edges.end(), edge.min));
          auto upperIndex = std::distance(transitions.edges.begin(), std::lower_bound(transitions.edges.begin(), transitions.edges.end(), edge.max));
          interval.lowerInverseWidth = transitions.inverseWidths[lowerIndex];
          interval.upperInverseWidth = transitions.inverseWidths[upperIndex];
        }
        else{
          interval.lowerInverseWidth = 1 / axis.width;
          interval.upperInverseWidth = interval.lowerInverseWidth;
        }
        responseBin.intervals.emplace_back(interval);
      }
    }
  }
}

double BinningResponse::cumulativeKernel(double position_){
  return GundamBinningMath::cumulativeKernel(position_);
}

void BinningResponse::prepareEvaluation(const VariableCollection& variables_, EvaluationBuffer& buffer_) const{
  buffer_.variableIndices.clear();
  for( const auto& name : _variableNames_ ){ buffer_.variableIndices.emplace_back(variables_.findVarIndex(name)); }
  buffer_.values.resize(_variableNames_.size());
}

void BinningResponse::evaluate(const VariableCollection& variables_, EvaluationBuffer& buffer_) const{
  if( _hasRelativeWidths_ ){ evaluateRelative(variables_, buffer_); }
  else{ evaluateAbsolute(variables_, buffer_); }
}

void BinningResponse::evaluateAbsolute(const VariableCollection& variables_, EvaluationBuffer& buffer_) const{
  for( size_t iVariable = 0 ; iVariable < _variableNames_.size() ; ++iVariable ){
    double value = variables_.getVarList()[buffer_.variableIndices[iVariable]].getVarAsDouble();
    LogThrowIf(not std::isfinite(value), "Nonfinite binning variable: " << _variableNames_[iVariable]);
    buffer_.values[iVariable] = value;
  }
  auto& fractions = buffer_.response.binFractions;
  fractions.clear();
  double total{0};
  for( const auto& responseBin : _responseBins_ ){
    bool matches{true};
    for( const auto& interval : responseBin.selectionIntervals ){
      double value = buffer_.values[interval.variableIndex];
      bool contains = interval.lowerBound == interval.upperBound ? value == interval.lowerBound
          : value >= interval.lowerBound and value < interval.upperBound;
      if( not contains ){
        matches = false;
        break;
      }
    }
    if( not matches ){ continue; }
    double fraction{1};
    for( const auto& interval : responseBin.intervals ){
      double value = buffer_.values[interval.variableIndex];
      fraction *= std::max(0., cumulativeKernel((interval.upperBound - value) * interval.inverseWidth)
                            - cumulativeKernel((interval.lowerBound - value) * interval.inverseWidth));
      if( fraction == 0 ){ break; }
    }
    if( fraction == 0 ){ continue; }
    BinFraction binFraction;
    binFraction.binIndex = responseBin.binIndex;
    binFraction.fraction = fraction;
    fractions.emplace_back(binFraction);
    total += fraction;
  }
  double tolerance = 64 * std::numeric_limits<double>::epsilon() * std::max(size_t(1), fractions.size());
  LogThrowIf(not std::isfinite(total) or total > 1 + tolerance,
             "Total binning response exceeds one: " << total);
  if( total > 1 ){
    auto largest = std::max_element(fractions.begin(), fractions.end(), [](const BinFraction& first_, const BinFraction& second_){
      return first_.fraction < second_.fraction;
    });
    largest->fraction -= total - 1;
    total = 1;
  }
  buffer_.response.outsideFraction = 1 - total;
}

void BinningResponse::evaluateRelative(const VariableCollection& variables_, EvaluationBuffer& buffer_) const{
  for( size_t iVariable = 0 ; iVariable < _variableNames_.size() ; ++iVariable ){
    double value = variables_.getVarList()[buffer_.variableIndices[iVariable]].getVarAsDouble();
    LogThrowIf(not std::isfinite(value), "Nonfinite binning variable: " << _variableNames_[iVariable]);
    buffer_.values[iVariable] = value;
  }
  auto& fractions = buffer_.response.binFractions;
  fractions.clear();
  double total{0};
  total = evaluateRelativeResponse(buffer_);
  double tolerance = 64 * std::numeric_limits<double>::epsilon() * std::max(size_t(1), fractions.size());
  LogThrowIf(not std::isfinite(total) or total > 1 + tolerance,
             "Total binning response exceeds one: " << total);
  if( total > 1 ){
    auto largest = std::max_element(fractions.begin(), fractions.end(), [](const BinFraction& first_, const BinFraction& second_){
      return first_.fraction < second_.fraction;
    });
    largest->fraction -= total - 1;
    total = 1;
  }
  buffer_.response.outsideFraction = 1 - total;
}

double BinningResponse::evaluateRelativeResponse(EvaluationBuffer& buffer_) const{
  double total{0};
  for( const auto& responseBin : _relativeResponseBins_ ){
    bool matches{true};
    for( const auto& interval : responseBin.selectionIntervals ){
      double value = buffer_.values[interval.variableIndex];
      bool contains = interval.lowerBound == interval.upperBound ? value == interval.lowerBound
          : value >= interval.lowerBound and value < interval.upperBound;
      if( not contains ){
        matches = false;
        break;
      }
    }
    if( not matches ){ continue; }
    double fraction{1};
    for( const auto& interval : responseBin.intervals ){
      double value = buffer_.values[interval.variableIndex];
      fraction *= std::max(0., cumulativeKernel((interval.upperBound - value) * interval.upperInverseWidth)
                            - cumulativeKernel((interval.lowerBound - value) * interval.lowerInverseWidth));
      if( fraction == 0 ){ break; }
    }
    if( fraction == 0 ){ continue; }
    BinFraction binFraction;
    binFraction.binIndex = responseBin.binIndex;
    binFraction.fraction = fraction;
    buffer_.response.binFractions.emplace_back(binFraction);
    total += fraction;
  }
  return total;
}
