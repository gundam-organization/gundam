//
// Created by Nadrino on 07/03/2025.
//

/// @file DialManager.cpp
/// @brief Implements DialManager: builds the DialCollection objects from the
///        parameter definitions and keeps their dials up to date.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in DialManager.h. The
/// comments here describe how each method does its job.

#include "DialManager.h"
#include "CompositeDial.h"
#include "GundamUtils.h"


/// @par Implementation
/// Loops over the parameter sets in configuration order and appends one
/// DialCollection per dial-set definition. Every new collection gets a
/// pointer to the full list of parameter sets, its own position in
/// `_dialCollectionList_` as index, the index of the set it supervises and,
/// for per-parameter definitions, the index of that parameter. It is then
/// configured with its definition block, which creates the binned dials.
/// Event-by-event dials are only created later, while the data is loaded.
///
/// Composite nodes are added by a second loop over the list, which also
/// visits the nodes it appends, so nested composites get their children
/// too. The parent's CompositeDial is looked up again after every insertion,
/// because growing `_dialCollectionList_` can copy the dials. Each node's
/// configuration is checked, its `dialInputList` is rebuilt from the node
/// inputs (an intermediate result gets a placeholder parameter, since
/// CompositeDial supplies its value), and `useMirrorDial` is passed on to
/// nested nodes.
void DialManager::configureImpl(){
  LogThrowIf(_parametersManagerPtr_==nullptr, "Parameters manager not set.");

  // Rebuild from scratch, in case configure() runs more than once.
  _dialCollectionList_.clear();
  for(size_t iParSet = 0 ; iParSet < _parametersManagerPtr_->getParameterSetsList().size() ; iParSet++ ){
    if( not _parametersManagerPtr_->getParameterSetsList()[iParSet].isEnabled() ){ continue; }
    // DEV / DialCollections
    // Set-wide form: the ParameterSet itself lists dial-set definitions.
    // Each one becomes a collection with one dial per parameter of the set,
    // so no supervised parameter index is given.
    if( not _parametersManagerPtr_->getParameterSetsList()[iParSet].getDialSetDefinitions().empty() ){
      for( auto& dialSetDef : _parametersManagerPtr_->getParameterSetsList()[iParSet].getDialSetDefinitions().loop() ){
        _dialCollectionList_.emplace_back(&_parametersManagerPtr_->getParameterSetsList());
        _dialCollectionList_.back().setIndex(int(_dialCollectionList_.size()) - 1);
        _dialCollectionList_.back().setSupervisedParameterSetIndex(int(iParSet) );
        _dialCollectionList_.back().configure( dialSetDef );
      }
    }
    else{
      // Per-parameter form: each enabled parameter carries its own
      // definitions, and each definition becomes a collection whose dials all
      // read that one parameter.

      for( auto& par : _parametersManagerPtr_->getParameterSetsList()[iParSet].getParameterList() ){ // mutable, since it may be disabled below
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
/// @par Implementation
/// Delegates to DialCollection::initialize() for each collection, in order.
void DialManager::initializeImpl(){
  for( auto& dialCollection : _dialCollectionList_ ){
    dialCollection.initialize();
  }
}

/// @par Implementation
/// Delegates to DialCollection::invalidateCachedInputBuffers(), which sets
/// every cached input value of the collection to NaN. NaN never compares
/// equal, so the next DialInputBuffer::update() always flags a change.
void DialManager::invalidateInputBuffers(){
  // be extra sure the dial input will request an update
  for( auto& dialCollection : _dialCollectionList_ ){
    dialCollection.invalidateCachedInputBuffers();
  }
}
/// @par Implementation
/// Only event-by-event collections are resized: binned collections have a
/// fixed number of dials set at configuration, so there is nothing to trim.
/// The process memory use is logged before and after.
void DialManager::shrinkDialContainers(){
  LogInfo << "RAM before resizing dial containers: " << GundamUtils::getProcessMemoryUsageSummary() << std::endl;
  LogInfo << "Resizing dial containers..." << std::endl;
  for( auto& dialCollection : _dialCollectionList_ ) {
    if( dialCollection.isEventByEvent() ){ dialCollection.resizeContainers(); }
  }
  LogInfo << "RAM after resizing dial containers: " << GundamUtils::getProcessMemoryUsageSummary() << std::endl;
}
/// @par Implementation
/// Every collection is reset; DialCollection::resetForLoad() decides what
/// each one keeps. The buffers are invalidated afterwards so the next update
/// re-evaluates every dial that remains.
void DialManager::clearEventByEventDials(){
  for( auto& dialCollection: _dialCollectionList_ ) {
    dialCollection.resetForLoad();
  }
  invalidateInputBuffers();
}
/// @par Implementation
/// The scoped timer starts here and stops when the function returns, so the
/// whole loop is timed. Input buffers are refreshed before the callbacks run,
/// because the tabulated and kriged callbacks read the refreshed buffers.
void DialManager::updateDialState(){
  auto s{_updateTimer_.scopeTime()}; // scope guard; stops _updateTimer_ when it goes out of scope
  for( auto& dialCollection : _dialCollectionList_ ) {
    dialCollection.updateInputBuffers();
    dialCollection.update();
  }
}
/// @par Implementation
/// Builds a three-column GenericToolbox::TablePrinter. configureImpl()
/// creates the collections set by set, with composite nodes appended at the
/// end, so a separator is added whenever the supervised set index changes.
void DialManager::printSummaryTable() const{
  GenericToolbox::TablePrinter t; // the table being built, printed at the end
  t << "Dial Collection" << GenericToolbox::TablePrinter::NextColumn;
  t << "Type" << GenericToolbox::TablePrinter::NextColumn;
  t << "Options" << GenericToolbox::TablePrinter::NextLine;
  int parSetIdx{-1}; // parameter set of the previous row; -1 so the first row gets a separator
  for( auto& dialCollection : _dialCollectionList_ ) {
    // Blue background marks event-by-event collections (see the legend below the table).
    if( dialCollection.isEventByEvent() ){ t.setColorBuffer(LOGGER_STR_COLOR_BLUE_BG); }
    else{ t.setColorBuffer(""); }

    // New parameter set: draw a separator line before its first collection.
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
