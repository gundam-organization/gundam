//
// Created by Adrien BLANCHET on 18/05/2022.
//

#include "TreeEventBuffer.h"


TreeEventBuffer::TreeEventBuffer() = default;
TreeEventBuffer::~TreeEventBuffer() = default;

void TreeEventBuffer::hook(TTree* tree_){
  _leafContentList_.clear();
  _leafContentList_.resize(_leafNameList_.size());
  int iLeaf{0};
  for( auto& leafName : _leafNameList_ ){
    _leafContentList_[iLeaf++].hook(tree_, leafName);
  }
}

void TreeEventBuffer::setLeafNameList(const std::vector<std::string> &leafNameList) {
  _leafNameList_ = leafNameList;
}

const std::vector<GenericToolbox::LeafHolder> &TreeEventBuffer::getLeafContentList() const {
  return _leafContentList_;
}

int TreeEventBuffer::fetchLeafIndex(const std::string& leafName_) const{
  return GenericToolbox::findElementIndex(leafName_, _leafNameList_);
}
const GenericToolbox::LeafHolder& TreeEventBuffer::getLeafContent(const std::string& leafName_) const{
  int i = this->fetchLeafIndex(leafName_);
  if(i==-1){ throw std::runtime_error(leafName_ + ": not found."); }
  return _leafContentList_[i];
}
std::string TreeEventBuffer::getSummary(){
  std::stringstream ss;
  ss << "TreeEventBuffer:" << std::endl << "_leafContentList_ = {";
  if( not _leafNameList_.empty() ){
    for( size_t iLeaf = 0 ; iLeaf < _leafNameList_.size() ; iLeaf++ ){
      ss<< std::endl << "  " << _leafContentList_[iLeaf];
    }
    ss << std::endl << "}";
  }
  else{
    ss << "}";
  }
  return ss.str();
}
