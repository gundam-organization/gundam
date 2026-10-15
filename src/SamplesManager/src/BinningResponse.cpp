#include "BinningResponse.h"

#include "Logger.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <utility>

void BinningResponse::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({{"isEnabled"}, {"variables"}});
  _config_.checkConfiguration();
  _config_.fillValue(_isEnabled_, "isEnabled");
  _axes_.clear();
  for( auto axisConfig : _config_.loop("variables") ){
    axisConfig.clearFields();
    axisConfig.defineFields({{FieldFlag::MANDATORY, "name"}, {FieldFlag::MANDATORY, "width"}});
    axisConfig.checkConfiguration();
    Axis axis;
    axisConfig.fillValue(axis.name, "name");
    axisConfig.fillValue(axis.width, "width");
    LogThrowIf(axis.name.empty() or not std::isfinite(axis.width) or axis.width <= 0 or not std::isfinite(1 / axis.width),
               "Binning response requires a variable name and a finite, positive width.");
    LogThrowIf(findWidth(axis.name) != 0, "Duplicate binning response variable: " << axis.name);
    _axes_.emplace_back(std::move(axis));
  }
  LogThrowIf(_isEnabled_ and _axes_.empty(), "Enabled binning response requires at least one variable.");
}

double BinningResponse::findWidth(const std::string& variableName_) const{
  for( const auto& axis : _axes_ ){
    if( axis.name == variableName_ ){ return axis.width; }
  }
  return 0;
}

void BinningResponse::build(const std::vector<Bin>& binList_){
  _responseBins_.clear();
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

  _responseBins_.resize(binList_.size());
  for( size_t iBin = 0 ; iBin < binList_.size() ; ++iBin ){
    auto& responseBin = _responseBins_[iBin];
    responseBin.binIndex = binList_[iBin].getIndex();
    for( size_t iVariable = 0 ; iVariable < _variableNames_.size() ; ++iVariable ){
      const auto& edge = binList_[iBin].getVarEdges(_variableNames_[iVariable]);
      if( findWidth(edge.varName) != 0 ){ continue; }
      SelectionInterval interval;
      interval.variableIndex = iVariable;
      interval.lowerBound = edge.min;
      interval.upperBound = edge.max;
      responseBin.selectionIntervals.emplace_back(interval);
    }
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

double BinningResponse::cumulativeKernel(double position_){
  if( position_ <= -1 ){ return 0; }
  if( position_ >= 1 ){ return 1; }
  double t = (position_ + 1) / 2;
  return t * t * t * (10 + t * (-15 + 6 * t));
}

void BinningResponse::prepareEvaluation(const VariableCollection& variables_, EvaluationBuffer& buffer_) const{
  buffer_.variableIndices.clear();
  for( const auto& name : _variableNames_ ){ buffer_.variableIndices.emplace_back(variables_.findVarIndex(name)); }
  buffer_.values.resize(_variableNames_.size());
}

void BinningResponse::evaluate(const VariableCollection& variables_, EvaluationBuffer& buffer_) const{
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
