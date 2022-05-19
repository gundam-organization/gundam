//
// Created by Adrien BLANCHET on 18/05/2022.
//

#include "TreeEventBuffer.h"


TreeEventBuffer::TreeEventBuffer() {}
TreeEventBuffer::~TreeEventBuffer() {}

void TreeEventBuffer::hookToTree(TTree* tree_){
  _leafContentList_.clear();
  _leafContentList_.resize(_leafNameList_.size());
  int iLeaf{0};
  for( auto& leafName : _leafNameList_ ){
    TLeaf* l = tree_->GetLeaf(leafName.c_str());
    if(l==nullptr) throw std::logic_error("Could not find leaf: " + leafName);
    std::cout << "HOOK: " << leafName << std::endl;
    l->GetBranch()->SetStatus(true);
//    tree_->SetBranchStatus(tree_->GetLeaf(leafName.c_str())->GetBranch()->GetName(), true);
    _leafContentList_[iLeaf++].hookToTree(tree_, leafName);
  }
}

void TreeEventBuffer::setLeafNameList(const std::vector<std::string> &leafNameList) {
  _leafNameList_ = leafNameList;
}

const std::vector<GenericToolbox::LeafHolder> &TreeEventBuffer::getLeafContentList() const {
  return _leafContentList_;
}
const GenericToolbox::LeafHolder& TreeEventBuffer::getLeafContent(const std::string& leafName_) const{
  int i = GenericToolbox::findElementIndex(leafName_, _leafNameList_);
  if(i==-1){ throw std::runtime_error(leafName_ + ": not found."); }
  return _leafContentList_[i];
}
