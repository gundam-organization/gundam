//
// Created by Nadrino on 28/09/2024.
//

#include "LoaderUtils.h"

#include "GundamUtils.h"

#include "Logger.h"


namespace LoaderUtils{

  void copyData(const Event& src_, Event& dst_){
    dst_.getIndices() = src_.getIndices();
    dst_.getWeights() = src_.getWeights();

    // variables
    LogThrowIf( dst_.getVariables().getNameListPtr() == nullptr, "var name list not set." );
    size_t nVars{dst_.getVariables().getNameListPtr()->size()};
    for( size_t iVar = 0 ; iVar < nVars ; iVar++ ){
      auto& var{src_.getVariables().fetchVariable( dst_.getVariables().getNameListPtr()->at(iVar) )};
      dst_.getVariables().getVarList()[iVar].set( var.get() );
    }
  }
  void copyData(Event& event_, const std::vector<const GenericToolbox::TreeBuffer::ExpressionBuffer*>& expList_){
    size_t nLeaf{expList_.size()};
    for( size_t iLeaf = 0 ; iLeaf < nLeaf ; iLeaf++ ){
      event_.getVariables().getVarList()[iLeaf].set( expList_[iLeaf]->getBuffer() );
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
  void applyVarTransforms(Event& event_, const std::vector<EventVarTransformLib*>& transformList_){
    for( auto* varTransformPtr : transformList_ ){
      varTransformPtr->evalAndStore(event_);
    }
  }

}


