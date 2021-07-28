//
// Created by Nadrino on 22/07/2021.
//

#include "Logger.h"

#include "PhysicsEvent.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[PhysicsEvent]");
})

PhysicsEvent::PhysicsEvent() { this->reset(); }
PhysicsEvent::~PhysicsEvent() { this->reset(); }

void PhysicsEvent::reset() {
  _commonLeafNameListPtr_ = nullptr;
  _leafContentList_.clear();
}

void PhysicsEvent::setLeafNameListPtr(const std::vector<std::string> *leafNameListPtr) {
  _commonLeafNameListPtr_ = leafNameListPtr;
}

void PhysicsEvent::hookToTree(TTree* tree_){
  LogThrowIf(not _leafContentList_.empty(), "Can't " << __METHOD_NAME__ << " while leafContentList is not empty." );
  LogThrowIf(_commonLeafNameListPtr_ == nullptr, "_commonLeafNameListPtr_ is not set.");

  _leafContentList_.resize(_commonLeafNameListPtr_->size());
  for( size_t iLeaf = 0 ; iLeaf < _commonLeafNameListPtr_->size() ; iLeaf++ ){
    _leafContentList_.at(iLeaf).hookToTree(tree_, _commonLeafNameListPtr_->at(iLeaf));
  }
}

Int_t PhysicsEvent::fetchIntValue(const std::string &leafName_) {
  return 0;
}

std::string PhysicsEvent::getSummary(){
  std::stringstream ss;
  ss << __CLASS_NAME__ << " :";
  if( _leafContentList_.empty() ){
    ss << "empty";
  }
  else{
    for( size_t iLeaf = 0 ; iLeaf < _leafContentList_.size() ; iLeaf++ ){
      ss << std::endl;
      if(_commonLeafNameListPtr_ != nullptr and _commonLeafNameListPtr_->size() == _leafContentList_.size()) {
        ss << _commonLeafNameListPtr_->at(iLeaf) << " -> ";
      }
      ss << _leafContentList_.at(iLeaf);
    }
  }
  return ss.str();
}
void PhysicsEvent::print() {
  LogInfo << this->getSummary() << std::endl;
}
