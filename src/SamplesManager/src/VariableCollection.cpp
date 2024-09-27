//
// Created by Nadrino on 27/09/2024.
//

#include "VariableCollection.h"

#include "Logger.h"



void VariableCollection::setVarNameList( const std::shared_ptr<std::vector<std::string>> &nameListPtr_ ){
  LogThrowIf(nameListPtr_ == nullptr, "Invalid commonNameListPtr_ provided.");
  _nameListPtr_ = nameListPtr_;
  _varList_.clear();
  _varList_.resize(_nameListPtr_->size());
}

// memory
void VariableCollection::allocateMemory( const std::vector<const GenericToolbox::LeafForm*>& leafFormList_){
  LogThrowIf( _nameListPtr_ == nullptr, "var name list not set." );
  LogThrowIf( _nameListPtr_->size() != leafFormList_.size(), "size mismatch." );

  auto nLeaf{_nameListPtr_->size()};
  for(size_t iVar = 0 ; iVar < nLeaf ; iVar++ ){
    _varList_[iVar].set(GenericToolbox::leafToAnyType( leafFormList_[iVar]->getLeafTypeName() ));
  }
}
void VariableCollection::copyData( const std::vector<const GenericToolbox::LeafForm*>& leafFormList_){
  size_t nLeaf{leafFormList_.size()};
  for( size_t iLeaf = 0 ; iLeaf < nLeaf ; iLeaf++ ){
    _varList_[iLeaf].set( *leafFormList_[iLeaf] );
  }
}

int VariableCollection::findVarIndex( const std::string& leafName_, bool throwIfNotFound_) const{
  LogThrowIf(_nameListPtr_ == nullptr, "Can't " << __METHOD_NAME__ << " while _commonLeafNameListPtr_ is empty.");
  int out{GenericToolbox::findElementIndex(leafName_, *_nameListPtr_)};
  LogThrowIf(throwIfNotFound_ and out == -1, leafName_ << " not found in: " << GenericToolbox::toString(*_nameListPtr_));
  return out;
}
const VariableHolder& VariableCollection::fetchVariable( const std::string& name_) const{
  int index = this->findVarIndex(name_, true);
  return _varList_[index];
}
VariableHolder& VariableCollection::fetchVariable( const std::string& name_){
  int index = this->findVarIndex(name_, true);
  return _varList_[index];
}

// bin tools
bool VariableCollection::isInBin( const Bin& bin_) const{
  return std::all_of(
      bin_.getEdgesList().begin(), bin_.getEdgesList().end(),
      [&](const Bin::Edges& edges_){
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
int VariableCollection::findBinIndex( const std::vector<Bin>& binList_) const{
  if( binList_.empty() ){ return -1; }

  auto dialItr = std::find_if(
      binList_.begin(), binList_.end(),
      [&](const Bin& bin_){ return this->isInBin(bin_); }
  );

  if ( dialItr == binList_.end() ){ return -1; }
  return int( std::distance( binList_.begin(), dialItr ) );
}
int VariableCollection::findBinIndex( const BinSet& binSet_) const{ return this->findBinIndex(binSet_.getBinList() ); }

// formula
double VariableCollection::evalFormula( const TFormula* formulaPtr_, std::vector<int>* indexDict_) const{
  LogThrowIf(formulaPtr_ == nullptr, GET_VAR_NAME_VALUE(formulaPtr_));

  std::vector<double> parArray(formulaPtr_->GetNpar());
  for( int iPar = 0 ; iPar < formulaPtr_->GetNpar() ; iPar++ ){
    if(indexDict_ != nullptr){ parArray[iPar] = _varList_[(*indexDict_)[iPar]].getVarAsDouble(); }
    else                     { parArray[iPar] = this->fetchVariable(formulaPtr_->GetParName(iPar)).getVarAsDouble(); }
  }

  return formulaPtr_->EvalPar(nullptr, &parArray[0]);
}

// printout
std::string VariableCollection::getSummary() const{
  std::stringstream ss;
  for( int iVar = 0 ; iVar < int(_varList_.size()) ; iVar++ ){
    if( not ss.str().empty() ){ ss << std::endl; }
    ss << "  { name: " << _nameListPtr_->at(iVar);
    ss << ", value: " << _varList_.at(iVar).get();
    ss << " }";
  }
  return ss.str();
}
