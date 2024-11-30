//
// Created by Nadrino on 28/09/2024.
//

#include "LoaderUtils.h"

#include "GundamUtils.h"

#include "Logger.h"


namespace LoaderUtils{

  void allocateMemory(Event& event_, const std::vector<const GenericToolbox::LeafForm*>& leafFormList_){
    LogThrowIf( event_.getVariables().getNameListPtr() == nullptr, "var name list not set." );
    LogThrowIf( event_.getVariables().getNameListPtr()->size() != leafFormList_.size(), "size mismatch." );

    auto nLeaf{event_.getVariables().getNameListPtr()->size()};
    for(size_t iVar = 0 ; iVar < nLeaf ; iVar++ ){
      event_.getVariables().getVarList()[iVar].set(GenericToolbox::leafToAnyType( leafFormList_[iVar]->getLeafTypeName() ));
    }
  }
  void copyData(Event& event_, const std::vector<const GenericToolbox::LeafForm*>& leafFormList_){
    size_t nLeaf{leafFormList_.size()};
    for( size_t iLeaf = 0 ; iLeaf < nLeaf ; iLeaf++ ){
      if( leafFormList_[iLeaf]->getTreeFormulaPtr() != nullptr ){ leafFormList_[iLeaf]->fillLocalBuffer(); }
      event_.getVariables().getVarList()[iLeaf].set( leafFormList_[iLeaf]->getDataAddress(), leafFormList_[iLeaf]->getDataSize() );
    }
  }
  void fillBinIndex(Event& event_, const std::vector<Histogram::BinContext>& binList_){
    for( auto& binContext : binList_ ){
      if( event_.getVariables().isInBin(binContext.bin) ){
        event_.getIndices().bin = binContext.bin.getIndex();
        return;
      }
    }
    event_.getIndices().bin = -1;
  }
  double evalFormula(const Event& event_, const TFormula* formulaPtr_, std::vector<int>* indexDict_){
    LogThrowIf(formulaPtr_ == nullptr, GET_VAR_NAME_VALUE(formulaPtr_));

    std::vector<double> parArray(formulaPtr_->GetNpar());
    for( int iPar = 0 ; iPar < formulaPtr_->GetNpar() ; iPar++ ){
      if(indexDict_ != nullptr){ parArray[iPar] = event_.getVariables().getVarList()[(*indexDict_)[iPar]].getVarAsDouble(); }
      else                     { parArray[iPar] = event_.getVariables().fetchVariable(formulaPtr_->GetParName(iPar)).getVarAsDouble(); }
    }

    return formulaPtr_->EvalPar(nullptr, &parArray[0]);
  }

}


