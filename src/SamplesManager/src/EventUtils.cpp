//
// Created by Nadrino on 06/03/2024.
//

#include "EventUtils.h"

#include "GenericToolbox.Vector.h"
#include "GenericToolbox.String.h"
#include "GenericToolbox.Macro.h"
#include "Logger.h"

#include <sstream>

LoggerInit([]{
  Logger::getUserHeader() << "[EventUtils]";
});


/// Indices
namespace EventUtils{
  std::string Indices::getSummary() const{
    std::stringstream ss;
    ss << "dataset(" << dataset << ")";
    ss << ", " << "entry(" << entry << ")";
    ss << ", " << "sample(" << sample << ")";
    ss << ", " << "bin(" << bin << ")";
    return ss.str();
  }
}


/// Weights
namespace EventUtils{
  std::string Weights::getSummary() const{
    std::stringstream ss;
    ss << "base(" << base << ")";
    ss << ", " << "current(" << current << ")";
    return ss.str();
  }
}


/// Variables
namespace EventUtils{

  void Variables::Variable::set(const GenericToolbox::LeafForm& leafForm_){
    if( leafForm_.getTreeFormulaPtr() != nullptr ){ leafForm_.fillLocalBuffer(); }
    memcpy(
        var.getPlaceHolderPtr()->getVariableAddress(),
        leafForm_.getDataAddress(), leafForm_.getDataSize()
    );
    updateCache();
  }

  void Variables::setVarNameList( const std::shared_ptr<std::vector<std::string>> &nameListPtr_ ){
    LogThrowIf(nameListPtr_ == nullptr, "Invalid commonNameListPtr_ provided.");
    _nameListPtr_ = nameListPtr_;
    _varList_.clear();
    _varList_.resize(_nameListPtr_->size());
  }

  // memory
  void Variables::allocateMemory( const std::vector<const GenericToolbox::LeafForm*>& leafFormList_){
    LogThrowIf( _nameListPtr_ == nullptr, "var name list not set." );
    LogThrowIf( _nameListPtr_->size() != leafFormList_.size(), "size mismatch." );

    auto nLeaf{_nameListPtr_->size()};
    for(size_t iVar = 0 ; iVar < nLeaf ; iVar++ ){
      _varList_[iVar].set(GenericToolbox::leafToAnyType( leafFormList_[iVar]->getLeafTypeName() ));
    }
  }
  void Variables::copyData( const std::vector<const GenericToolbox::LeafForm*>& leafFormList_){
    size_t nLeaf{leafFormList_.size()};
    for( size_t iLeaf = 0 ; iLeaf < nLeaf ; iLeaf++ ){
      _varList_[iLeaf].set( *leafFormList_[iLeaf] );
    }
  }

  int Variables::findVarIndex( const std::string& leafName_, bool throwIfNotFound_) const{
    LogThrowIf(_nameListPtr_ == nullptr, "Can't " << __METHOD_NAME__ << " while _commonLeafNameListPtr_ is empty.");
    int out{GenericToolbox::findElementIndex(leafName_, *_nameListPtr_)};
    LogThrowIf(throwIfNotFound_ and out == -1, leafName_ << " not found in: " << GenericToolbox::toString(*_nameListPtr_));
    return out;
  }
  const Variables::Variable& Variables::fetchVariable(const std::string& name_) const{
    int index = this->findVarIndex(name_, true);
    return _varList_[index];
  }
  Variables::Variable& Variables::fetchVariable(const std::string& name_){
    int index = this->findVarIndex(name_, true);
    return _varList_[index];
  }

  // bin tools
  bool Variables::isInBin( const DataBin& bin_) const{
    return std::all_of(
        bin_.getEdgesList().begin(), bin_.getEdgesList().end(),
        [&](const DataBin::Edges& edges_){
          return bin_.isBetweenEdges(
              edges_,
              ( edges_.varIndexCache != -1 ?
                _varList_[edges_.varIndexCache].getVarAsDouble():    // use directly the index if available
                this->fetchVariable(edges_.varName).getVarAsDouble() // look for the name otherwise
              )
          );
        }
    );
  }
  int Variables::findBinIndex(const std::vector<DataBin>& binList_) const{
    if( binList_.empty() ){ return -1; }

    auto dialItr = std::find_if(
        binList_.begin(), binList_.end(),
        [&](const DataBin& bin_){ return this->isInBin(bin_); }
    );

    if ( dialItr == binList_.end() ){ return -1; }
    return int( std::distance( binList_.begin(), dialItr ) );
  }
  int Variables::findBinIndex( const DataBinSet& binSet_) const{ return this->findBinIndex( binSet_.getBinList() ); }

  // formula
  double Variables::evalFormula( const TFormula* formulaPtr_, std::vector<int>* indexDict_) const{
    LogThrowIf(formulaPtr_ == nullptr, GET_VAR_NAME_VALUE(formulaPtr_));

    std::vector<double> parArray(formulaPtr_->GetNpar());
    for( int iPar = 0 ; iPar < formulaPtr_->GetNpar() ; iPar++ ){
      if(indexDict_ != nullptr){ parArray[iPar] = _varList_[(*indexDict_)[iPar]].getVarAsDouble(); }
      else                     { parArray[iPar] = this->fetchVariable(formulaPtr_->GetParName(iPar)).getVarAsDouble(); }
    }

    return formulaPtr_->EvalPar(nullptr, &parArray[0]);
  }

  // printout
  std::string Variables::getSummary() const{
    std::stringstream ss;
    for( int iVar = 0 ; iVar < int(_varList_.size()) ; iVar++ ){
      if( not ss.str().empty() ){ ss << std::endl; }
      ss << "  { name: " << _nameListPtr_->at(iVar);
      ss << ", var: " << _varList_.at(iVar).get();
      ss << " }";
    }
    return ss.str();
  }

}


#ifdef GUNDAM_USING_CACHE_MANAGER
/// Cache
namespace EventUtils{
  double Cache::getWeight() const{
    if( isValidPtr != nullptr and not(*isValidPtr)){
      // This is slowish, but will make sure that the cached result is
      // updated when the cache has changed.  The values pointed to by
      // _CacheManagerValue_ and _CacheManagerValid_ are inside
      // of the weights cache (a bit of evil coding here), and are
      // updated by the cache.  The update is triggered by
      // (*_CacheManagerUpdate_)().
      if( updateCallbackPtr != nullptr ){ (*updateCallbackPtr)(); }
    }
#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning CACHE_MANAGER_SLOW_VALIDATION used in PhysicsEvent::getEventWeight
      do {
          static double maxDelta = 1.0E-20;
          static double sumDelta = 0.0;
          static double sum2Delta = 0.0;
          static long long int numDelta = 0;
          double res = *_CacheManagerValue_;
          double avg = 0.5*(std::abs(res) + std::abs(_eventWeight_));
          if (avg < getTreeWeight()) avg = getTreeWeight();
          double delta = std::abs(res - _eventWeight_);
          delta /= avg;
          maxDelta = std::max(maxDelta,delta);
          if (delta < 1e-4) {
              sumDelta += delta;
              sum2Delta += delta*delta;
              ++numDelta;
              if (numDelta < 0) throw std::runtime_error("validation wrap");
              if ((numDelta % 1000000) == 0) {
                  LogInfo << "VALIDATION: Average event weight delta: "
                          << sumDelta/numDelta
                          << " +/- " << std::sqrt(
                              sum2Delta/numDelta
                              - sumDelta*sumDelta/numDelta/numDelta)
                          << " Maximum: " << maxDelta
                          << " " << numDelta
                          << std::endl;
              }
          }
          if (maxDelta < 1E-5) break;
          if (delta > 100.0*sumDelta/numDelta) break;
          LogWarning << "WARNING: Event weight difference: " << delta
                     << " Cache: " << res
                     << " Dial: " << _eventWeight_
                     << " Tree: " << getTreeWeight()
                     << " Delta: " << delta
                     << " Max: " << maxDelta
                     << std::endl;
      } while(false);
#endif
#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning CACHE_MANAGER_SLOW_VALIDATION force CPU _eventWeight_
      // When the slow validation is running, the "CPU" event weight is
      // calculated after Cache::Manager::Fill
      return _eventWeight_;
#endif
    LogThrowIf(not std::isfinite(*valuePtr), "NaN weight");
    return *valuePtr;
  }
}
#endif

