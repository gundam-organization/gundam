#include "BinningResponse.h"

#include "Logger.h"

#include <algorithm>
#include <cmath>
#include <limits>

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
    LogThrowIf(axis.name.empty() or not std::isfinite(axis.width) or axis.width <= 0,
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

bool BinningResponse::haveSameUnsmoothedEdges(const Bin& first_, const Bin& second_) const{
  for( const auto& edge : first_.getEdgesList() ){
    if( findWidth(edge.varName) != 0 ){ continue; }
    const auto* other = second_.getVarEdgesPtr(edge.varName);
    if( other == nullptr or edge.min != other->min or edge.max != other->max ){ return false; }
  }
  return true;
}

void BinningResponse::build(const std::vector<Bin>& binList_){
  _responseBins_.clear();
  if( not _isEnabled_ ){ return; }
  LogThrowIf(binList_.empty(), "Binning response requires a nonempty binning.");

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

  std::vector<std::vector<size_t>> slices;
  for( size_t iBin = 0 ; iBin < binList_.size() ; ++iBin ){
    auto slice = std::find_if(slices.begin(), slices.end(), [&](const std::vector<size_t>& indices_){
      return haveSameUnsmoothedEdges(binList_[indices_.front()], binList_[iBin]);
    });
    if( slice == slices.end() ){ slices.push_back({iBin}); }
    else{ slice->push_back(iBin); }
  }

  for( size_t i = 0 ; i < slices.size() ; ++i ){
    for( size_t j = 0 ; j < i ; ++j ){
      const auto& first = binList_[slices[i].front()];
      const auto& second = binList_[slices[j].front()];
      bool overlaps{true};
      for( const auto& edge : first.getEdgesList() ){
        if( findWidth(edge.varName) != 0 ){ continue; }
        const auto& other = second.getVarEdges(edge.varName);
        bool intersects;
        if( edge.isConditionVar ){ intersects = second.isBetweenEdges(other, edge.min); }
        else if( other.isConditionVar ){ intersects = first.isBetweenEdges(edge, other.min); }
        else{ intersects = edge.min < other.max and other.min < edge.max; }
        if( not intersects ){ overlaps = false; break; }
      }
      LogThrowIf(overlaps, "Binning response does not support overlapping unsmoothed slices.");
    }
  }

  _responseBins_.resize(binList_.size());
  for( const auto& slice : slices ){
    std::vector<double> lowerBounds(_axes_.size(), std::numeric_limits<double>::infinity());
    std::vector<double> upperBounds(_axes_.size(), -std::numeric_limits<double>::infinity());
    double totalVolume{0};
    for( size_t iBin : slice ){
      double volume{1};
      for( size_t iAxis = 0 ; iAxis < _axes_.size() ; ++iAxis ){
        const auto& edge = binList_[iBin].getVarEdges(_axes_[iAxis].name);
        lowerBounds[iAxis] = std::min(lowerBounds[iAxis], edge.min);
        upperBounds[iAxis] = std::max(upperBounds[iAxis], edge.max);
        volume *= edge.max - edge.min;
      }
      totalVolume += volume;
    }
    double boundingVolume{1};
    for( size_t iAxis = 0 ; iAxis < _axes_.size() ; ++iAxis ){
      boundingVolume *= upperBounds[iAxis] - lowerBounds[iAxis];
    }
    LogThrowIf(not std::isfinite(boundingVolume) or not std::isfinite(totalVolume)
               or std::abs(totalVolume - boundingVolume) > 1e-12 * boundingVolume,
               "Binning response requires a rectangular domain without holes in each hard-variable slice.");

    for( size_t i = 0 ; i < slice.size() ; ++i ){
      for( size_t j = 0 ; j < i ; ++j ){
        bool overlaps{true};
        for( const auto& axis : _axes_ ){
          const auto& first = binList_[slice[i]].getVarEdges(axis.name);
          const auto& second = binList_[slice[j]].getVarEdges(axis.name);
          if( first.max <= second.min or second.max <= first.min ){ overlaps = false; break; }
        }
        LogThrowIf(overlaps, "Binning response does not support overlapping bins.");
      }
    }

    for( size_t iBin : slice ){
      auto& responseBin = _responseBins_[iBin];
      responseBin.bin = binList_[iBin];
      for( size_t iAxis = 0 ; iAxis < _axes_.size() ; ++iAxis ){
        const auto& axis = _axes_[iAxis];
        const auto& edge = binList_[iBin].getVarEdges(axis.name);
        ResponseInterval interval;
        interval.variableName = axis.name;
        interval.width = axis.width;
        interval.lowerBound = edge.min == lowerBounds[iAxis] ? -std::numeric_limits<double>::infinity() : edge.min;
        interval.upperBound = edge.max == upperBounds[iAxis] ? std::numeric_limits<double>::infinity() : edge.max;
        responseBin.intervals.emplace_back(std::move(interval));
      }
    }
  }
}

double BinningResponse::cumulativeKernel(double position_){
  if( position_ <= -1 ){ return 0; }
  if( position_ >= 1 ){ return 1; }
  double t = (position_ + 1) / 2;
  return t * t * t * (10 + t * (-15 + 6 * t));
}

void BinningResponse::evaluate(const VariableCollection& variables_, std::vector<BinFraction>& fractions_) const{
  fractions_.clear();
  double total{0};
  for( const auto& responseBin : _responseBins_ ){
    bool matches{true};
    for( const auto& edge : responseBin.bin.getEdgesList() ){
      if( findWidth(edge.varName) != 0 ){ continue; }
      double value = variables_.fetchVariable(edge.varName).getVarAsDouble();
      LogThrowIf(not std::isfinite(value), "Nonfinite binning variable: " << edge.varName);
      if( not responseBin.bin.isBetweenEdges(edge, value) ){ matches = false; break; }
    }
    if( not matches ){ continue; }
    double fraction{1};
    for( const auto& interval : responseBin.intervals ){
      double value = variables_.fetchVariable(interval.variableName).getVarAsDouble();
      LogThrowIf(not std::isfinite(value), "Nonfinite binning variable: " << interval.variableName);
      fraction *= std::max(0., cumulativeKernel((interval.upperBound - value) / interval.width)
                            - cumulativeKernel((interval.lowerBound - value) / interval.width));
    }
    if( fraction == 0 ){ continue; }
    BinFraction binFraction;
    binFraction.binIndex = responseBin.bin.getIndex();
    binFraction.fraction = fraction;
    fractions_.emplace_back(binFraction);
    total += fraction;
  }
  LogThrowIf(total <= 0 or not std::isfinite(total),
             "Event has no binning response. Check its unsmoothed variables against the sample binning.");
  for( auto& fraction : fractions_ ){ fraction.fraction /= total; }
  auto largest = std::max_element(fractions_.begin(), fractions_.end(), [](const BinFraction& first_, const BinFraction& second_){
    return first_.fraction < second_.fraction;
  });
  double others{0};
  for( const auto& fraction : fractions_ ){
    if( &fraction != &*largest ){ others += fraction.fraction; }
  }
  largest->fraction = 1 - others;
}
