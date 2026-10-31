//
// Created by Nadrino on 07/03/2025.
//

#include "DialManager.h"
#include "CompositeDial.h"


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
      JsonType inputs = JsonType::array();
      for( const auto& input : node.inputs ){
        const bool dynamic = not input.eventVariable.empty() or input.nodeIndex >= 0
                             or mirroredInputs;
        LogThrowIf(prepared and dynamic,
                   "Composite node " << node.name << ": " << type
                   << " requires parameter inputs; response/event inputs and composite mirroring need per-event factory updates");
        if( not input.eventVariable.empty() ){
          LogThrowIf(type != DialCollection::DialType::Formula,
                     "Composite node " << node.name << ": eventVariable inputs are only supported by Formula dials");
          inputs.emplace_back(JsonType{{"name", input.name}, {"eventVariable", input.eventVariable}});
          continue;
        }
        // Intermediate inputs are evaluated by CompositeDial, not by this
        // buffer. A real parameter placeholder keeps normal loaders usable.
        JsonType parameter = input.parameterName.empty()
            ? JsonType{{"parameterIndex", 0}} : JsonType{{"parameterName", input.parameterName}};
        if( type == DialCollection::DialType::Formula ){ parameter["name"] = input.name; }
        inputs.emplace_back(std::move(parameter));
      }
      // Nested graphs already carry their inferred parameter inputs.
      if( type != DialCollection::DialType::Composite ){ config.getConfig()["dialInputList"] = inputs; }
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
  LogInfo << "Resizing dial containers..." << std::endl;
  for( auto& dialCollection : _dialCollectionList_ ) {
    if( dialCollection.isEventByEvent() ){ dialCollection.resizeContainers(); }
  }
}
void DialManager::clearEventByEventDials(){
  for( auto& dialCollection: _dialCollectionList_ ) {
    if( dialCollection.hasDialBranchData() ) { dialCollection.clear(); }
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
