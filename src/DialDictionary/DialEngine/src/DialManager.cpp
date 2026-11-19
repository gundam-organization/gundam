//
// Created by Nadrino on 07/03/2025.
//

#include "DialManager.h"
#include "CompositeDial.h"
#include "GundamUtils.h"

#include <set>


void DialManager::configureImpl(){
  LogThrowIf(_parametersManagerPtr_==nullptr, "Parameters manager not set.");

  _dialCollectionList_.clear();
  auto& parameterSets = _parametersManagerPtr_->getParameterSetsList();
  const auto addCollection = [&](const ConfigReader& config_, int setIndex_, int parameterIndex_){
    auto& collection = _dialCollectionList_.emplace_back(&parameterSets);
    collection.setIndex(int(_dialCollectionList_.size()) - 1);
    collection.setSupervisedParameterSetIndex(setIndex_);
    collection.setSupervisedParameterIndex(parameterIndex_);
    collection.configure(config_);
  };
  for( size_t iSet = 0; iSet < parameterSets.size(); ++iSet ){
    auto& set = parameterSets[iSet];
    if( not set.isEnabled() ){ continue; }
    if( not set.getDialSetDefinitions().empty() ){
      for( const auto& config : set.getDialSetDefinitions().loop() ){
        addCollection(config, int(iSet), -1);
      }
      continue;
    }
    for( const auto& parameter : set.getParameterList() ){
      if( not parameter.isEnabled() ){ continue; }
      for( const auto& config : parameter.getDialDefinitionsList() ){
        addCollection(config, int(iSet), parameter.getParameterIndex());
      }
    }
  }
  const auto& definitions = _parametersManagerPtr_->getDialSetDefinitions();
  LogThrowIf(not definitions.empty() and not definitions.getConfig().is_array(),
             "parametersManagerConfig/dialSetDefinitions must be a list");
  for( const auto& config : definitions.loop() ){
    addCollection(config, -1, -1);
  }

  // Parameters with no local dial may still be explicit inputs of another collection.
  std::set<std::pair<int, int>> referencedParameters;
  for( auto& collection : _dialCollectionList_ ){
    if( not collection.isEnabled() ){ continue; }
    for( const auto& buffer : collection.getDialInputBufferList() ){
      for( const auto& input : buffer.getInputParameterIndicesList() ){
        referencedParameters.emplace(input.parSetIndex, input.parIndex);
      }
    }
  }
  for( size_t iSet = 0; iSet < parameterSets.size(); ++iSet ){
    auto& set = parameterSets[iSet];
    if( not set.isEnabled() or not set.getDialSetDefinitions().empty() ){ continue; }
    for( auto& parameter : set.getParameterList() ){
      if( not parameter.isEnabled() or not parameter.getDialDefinitionsList().empty()
          or referencedParameters.count({int(iSet), parameter.getParameterIndex()}) ){ continue; }
      LogAlert << "Disabling \"" << parameter.getFullTitle() << "\": no dial definition." << std::endl;
      parameter.setIsEnabled(false);
    }
  }

  // Keep node collections in the normal loading/update pipeline. They are
  // consumed only when EventDialCache binds their parent composite.
  std::vector<bool> inheritedMirroring(_dialCollectionList_.size(), false);
  for( size_t iCollection = 0 ; iCollection < _dialCollectionList_.size() ; ++iCollection ){
    auto& collection = _dialCollectionList_[iCollection];
    if( not collection.isEnabled() or collection.getDialType() != DialCollection::DialType::Composite ){ continue; }
    auto* composite = dynamic_cast<CompositeDial*>(collection.getDialInterfaceList().at(0).getDialBaseRef());
    const int parSetIndex = collection.getSupervisedParameterSetIndex();
    const bool mirroredInputs = inheritedMirroring[iCollection]
                                or collection.getConfig().fetchValue("useMirrorDial", false);
    const size_t nodeCount = composite->getNodes().size();
    for( size_t iNode = 0 ; iNode < nodeCount ; ++iNode ){
      // Growing the collection vector can deep-copy DialBaseObject. Resolve
      // the graph again after every insertion; never retain a node reference.
      composite = dynamic_cast<CompositeDial*>(_dialCollectionList_[iCollection]
          .getDialInterfaceList().at(0).getDialBaseRef());
      auto& node = composite->getNodes()[iNode];
      const auto nodeName = node.name;
      auto config = *node.config;
      DialCollection::prepareConfig(config);
      LogThrowIf(not config.fetchValue("isEnabled", true), "Composite node " << node.name << " must be enabled");
      LogThrowIf(config.hasField("applyToVariable") or config.hasField("variableResponseMode"),
                 "Composite node " << node.name << ": put variable response options on the composite");
      LogThrowIf(config.hasField("dialsDefinitions") or config.hasField("parametersBinningPath"),
                 "Composite node " << node.name << ": use a direct dial definition with explicit inputs");
      const auto type = config.fetchValue<DialCollection::DialType>("dialType");
      LogThrowIf((type == DialCollection::DialType::Norm or type == DialCollection::DialType::Graph
                  or type == DialCollection::DialType::Spline) and node.inputs.size() != 1,
                 "Composite node " << node.name << ": " << type << " requires exactly one input");
      LogThrowIf(type == DialCollection::DialType::Surface and node.inputs.size() != 2,
                 "Composite node " << node.name << ": Surface requires exactly two inputs");
      const bool prepared = type == DialCollection::DialType::Tabulated
                            or type == DialCollection::DialType::Kriged
                            or type == DialCollection::DialType::ExternalWeight;
      for( const auto& input : node.inputs ){
        const bool dynamic = not input.eventVariable.empty() or input.nodeIndex >= 0 or mirroredInputs;
        LogThrowIf(prepared and dynamic,
                   "Composite node " << node.name << ": " << type
                   << " requires parameter inputs; response/event inputs and composite mirroring need per-event factory updates");
        LogThrowIf(not input.eventVariable.empty() and type != DialCollection::DialType::Formula,
                   "Composite node " << node.name << ": eventVariable inputs are only supported by Formula dials");
      }
      node.collectionIndex = int(_dialCollectionList_.size());
      const int childIndex = node.collectionIndex;
      _dialCollectionList_.emplace_back(&_parametersManagerPtr_->getParameterSetsList());
      auto& child = _dialCollectionList_.back();
      child.setIndex(childIndex);
      child.setSupervisedParameterSetIndex(parSetIndex);
      child.setCompositeParentIndex(int(iCollection));
      child.setIsCompositeNode(true);
      child.configure(config);
      LogThrowIf(not child.isEnabled(), "Composite: could not configure node " << nodeName);
      auto* parent = dynamic_cast<CompositeDial*>(_dialCollectionList_[iCollection]
          .getDialInterfaceList().at(0).getDialBaseRef());
      parent->getNodes()[iNode].mirrorEdges = child.getEventInputMirrorEdges();
      inheritedMirroring.emplace_back(mirroredInputs);
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
  LogInfo << "RAM before resizing dial containers: " << GundamUtils::getProcessMemoryUsageSummary() << std::endl;
  LogInfo << "Resizing dial containers..." << std::endl;
  for( auto& dialCollection : _dialCollectionList_ ) {
    if( dialCollection.isEventByEvent() ){ dialCollection.resizeContainers(); }
  }
  LogInfo << "RAM after resizing dial containers: " << GundamUtils::getProcessMemoryUsageSummary() << std::endl;
}
void DialManager::clearEventByEventDials(){
  for( auto& dialCollection: _dialCollectionList_ ) {
    dialCollection.resetForLoad();
  }
  invalidateInputBuffers();
}
void DialManager::updateDialState(){
  auto s{_updateTimer_.scopeTime()};
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
