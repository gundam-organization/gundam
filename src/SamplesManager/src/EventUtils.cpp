//
// Created by Nadrino on 06/03/2024.
//

#include "EventUtils.h"

#include "GenericToolbox.Vector.h"
#include "GenericToolbox.String.h"
#include "GenericToolbox.Macro.h"
#include "Logger.h"

#include <sstream>


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
      ss << ", value: " << _varList_.at(iVar).get();
      ss << " }";
    }
    return ss.str();
  }
}


#ifdef GUNDAM_USING_CACHE_MANAGER
/// Cache
namespace EventUtils {
  void Cache::update() const {
    if( valuePtr and isValidPtr and not (*isValidPtr)) {
      // This is slowish, but will make sure that the cached result is updated
      // when the cache has changed.  The value pointed to by isValidPtr is
      // inside of the weights cache (a bit of evil coding here), and are
      // updated by the cache.  The update is triggered by
      // (*updateCallbackPtr)().
      if(updateCallbackPtr) { (*updateCallbackPtr)(); }
    }
  }

  bool Cache::valid() const {
    // Check that the valuePtr points to a value that exists, and is valid.
    return valuePtr and isValidPtr and *isValidPtr;
  }

  double Cache::getWeight() const {
    // The value pointed to by valuePtr limes inside of the weights cache (a
    // bit of evil coding here).
    return *valuePtr;
  }
}
#endif
