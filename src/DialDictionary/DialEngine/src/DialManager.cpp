//
// Created by Nadrino on 07/03/2025.
//

#include "DialManager.h"


void DialManager::configureImpl(){
  LogThrowIf(_parametersManagerPtr_==nullptr, "Parameters manager not set.");

  _dialCollectionList_.clear();
  for(size_t iParSet = 0 ; iParSet < _parametersManagerPtr_->getParameterSetsList().size() ; iParSet++ ){
    if( not _parametersManagerPtr_->getParameterSetsList()[iParSet].isEnabled() ){ continue; }
    // DEV / DialCollections
    if( not _parametersManagerPtr_->getParameterSetsList()[iParSet].getDialSetDefinitions().empty() ){
      for( auto& dialSetDef : _parametersManagerPtr_->getParameterSetsList()[iParSet].getDialSetDefinitions().loop() ){
        _dialCollectionList_.emplace_back(&_parametersManagerPtr_->getParameterSetsList());
        _dialCollectionList_.back().setIndex(int(_dialCollectionList_.size()) - 1);
        _dialCollectionList_.back().setSupervisedParameterSetIndex(int(iParSet) );
        _dialCollectionList_.back().configure( dialSetDef );
      }
    }
    else{

      for( auto& par : _parametersManagerPtr_->getParameterSetsList()[iParSet].getParameterList() ){
        if( not par.isEnabled() ){ continue; }

        // Check if no definition is present -> disable the parameter in that case
        if( par.getDialDefinitionsList().empty() ) {
          LogAlert << "Disabling \"" << par.getFullTitle() << "\": no dial definition." << std::endl;
          par.setIsEnabled(false);
          continue;
        }

        for( const auto& dialDefinitionConfig : par.getDialDefinitionsList() ){
          _dialCollectionList_.emplace_back(&_parametersManagerPtr_->getParameterSetsList());
          _dialCollectionList_.back().setIndex(int(_dialCollectionList_.size()) - 1);
          _dialCollectionList_.back().setSupervisedParameterSetIndex(int(iParSet) );
          _dialCollectionList_.back().setSupervisedParameterIndex(par.getParameterIndex() );
          _dialCollectionList_.back().configure( dialDefinitionConfig );
        }
      }
    }
  }
}
void DialManager::initializeImpl(){
  for( auto& dialCollection : _dialCollectionList_ ){
    dialCollection.initialize();
  }
}

void DialManager::invalidateInputBuffers(){
  // be extra sure the dial input will request an update
  for( auto& dialCollection : _dialCollectionList_ ){
    dialCollection.invalidateCachedInputBuffers();
  }
}
void DialManager::shrinkDialContainers(){
  LogInfo << "Resizing dial containers..." << std::endl;
  for( auto& dialCollection : _dialCollectionList_ ) {
    if( dialCollection.isEventByEvent() ){ dialCollection.resizeContainers(); }
  }
}
void DialManager::clearEventByEventDials(){
  for( auto& dialCollection: _dialCollectionList_ ) {
    if( not dialCollection.getDialLeafName().empty() ) { dialCollection.clear(); }
  }
  invalidateInputBuffers();
}
void DialManager::updateDialState(){
  for( auto& dialCollection : _dialCollectionList_ ) {
    dialCollection.updateInputBuffers();
    dialCollection.update();
  }
}
void DialManager::printSummaryTable() const{
  GenericToolbox::TablePrinter t;
  t << "Dial Collection" << GenericToolbox::TablePrinter::NextColumn;
  t << "Type" << GenericToolbox::TablePrinter::NextColumn;
  t << "Options" << GenericToolbox::TablePrinter::NextLine;
  int parSetIdx{-1};
  for( auto& dialCollection : _dialCollectionList_ ) {
    if( dialCollection.isEventByEvent() ){ t.setColorBuffer(LOGGER_STR_COLOR_BLUE_BG); }
    else{ t.setColorBuffer(""); }

    if( parSetIdx != dialCollection.getSupervisedParameterSetIndex() ) {
      t.addSeparatorLine();
      parSetIdx = dialCollection.getSupervisedParameterSetIndex();
    }
    t << dialCollection.getTitle() << GenericToolbox::TablePrinter::NextColumn;
    t << dialCollection.getDialType() << GenericToolbox::TablePrinter::NextColumn;
    t << dialCollection.getDialOptions() << GenericToolbox::TablePrinter::NextColumn;
  }

  LogInfo << "Dial manager has " << _dialCollectionList_.size() << " dial collections:" << std::endl;
  t.printTable();
  LogInfo << LOGGER_STR_COLOR_BLUE_BG << "      " << LOGGER_STR_COLOR_RESET << ": Event-by-event dials." << std::endl;
}
