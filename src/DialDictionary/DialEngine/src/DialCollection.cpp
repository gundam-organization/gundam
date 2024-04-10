//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "GundamGlobals.h"
#include "DialCollection.h"
#include "DialBaseFactory.h"

#include "GenericToolbox.Json.h"
#include "Logger.h"

#include <sstream>


LoggerInit([]{
  Logger::setUserHeaderStr("[DialCollection]");
});


void DialCollection::readConfigImpl() {

  _dataSetNameList_ = GenericToolbox::Json::fetchValue<std::vector<std::string>>(
    _config_, "applyOnDataSets", std::vector<std::string>());
  if( _dataSetNameList_.empty() ){ _dataSetNameList_.emplace_back("*"); }

  _isEnabled_ = GenericToolbox::Json::fetchValue(_config_, "isEnabled", _isEnabled_);
  if( not _isEnabled_ ){ return; }

  if( GundamGlobals::isDisableDialCache() ){ _disableDialCache_ = true; }

  // Dials are directly defined with a binning file?
  if     ( initializeNormDialsWithParBinning() ){}
  else if( initializeDialsWithDefinition() ){}
  else {
    LogAlert << std::endl << "No valid definition for DialCollection. Disabling." << std::endl;
    _isEnabled_ = false;
  }

  // if "dialInputList" is not present, it will be setup during the initialize sequence.
  if( GenericToolbox::Json::doKeyExist(_config_, "dialInputList") ){
    auto dialInputList = GenericToolbox::Json::fetchValue<JsonType>(_config_, "dialInputList");

    LogThrowIf(_supervisedParameterSetIndex_ == -1, "Can't initialize dialInputList with out setting _supervisedParameterSetIndex_");

    _dialInputBufferList_.emplace_back();
    _dialInputBufferList_.back().setParSetRef( _parameterSetListPtr_ );

    // add the ref of each selected parameter
    for( auto& dialInput : dialInputList ){
      DialInputBuffer::ParameterReference p;
      p.parSetIndex = _supervisedParameterSetIndex_;

      if( GenericToolbox::Json::doKeyExist(dialInput, "name") ){
        auto parName{GenericToolbox::Json::fetchValue<std::string>(dialInput, "name")};
        auto* parPtr{_parameterSetListPtr_->at( _supervisedParameterSetIndex_ ).getParameterPtr( parName )};
        LogThrowIf(parPtr == nullptr, "Could not find parameter: " << parName);
        p.parIndex = parPtr->getParameterIndex();
      }

      _dialInputBufferList_.back().addParameterReference( p );
    }

  }

}
void DialCollection::initializeImpl() {
  LogInfo << "Initialising dial collection \"" << this->getTitle() << "\"" << std::endl;

  LogThrowIf(_index_==-1, "Index not set.");
  this->setupDialInterfaceReferences();
}

// non-trivial getters
bool DialCollection::isDatasetValid(const std::string& datasetName_) const{
  if( GenericToolbox::doesElementIsInVector(datasetName_, _dataSetNameList_) ){ return true; }

  // If not found, find general dialSet
  if( _dataSetNameList_.size() == 1 ){
    if(    GenericToolbox::doesElementIsInVector("", _dataSetNameList_)
           or GenericToolbox::doesElementIsInVector("*", _dataSetNameList_)
        ){
      return true;
    }
  }

  return false;
}
std::string DialCollection::getTitle() const {

  auto* parPtr{this->getSupervisedParameter()};
  if( parPtr != nullptr ){ return parPtr->getFullTitle(); }

  auto* parSetPtr{this->getSupervisedParameterSet()};
  if( parSetPtr != nullptr ){ return parSetPtr->getName(); }

  return {"UnsetParameterSuperVision"};
}
std::string DialCollection::getSummary(bool shallow_){
  std::stringstream ss;
  ss << "DialCollection: ";
  ss << this->getTitle();
  ss << " / nDials=" << _dialInterfaceList_.size();

  if( not shallow_ ){
    // print parameters
    for( auto& dialInterface : _dialInterfaceList_ ){
      if( _isBinned_ ){
        ss << std::endl << "  " << dialInterface.getSummary();
      }
    }
  }

  return ss.str();
}
Parameter* DialCollection::getSupervisedParameter() const {
  auto* parSetPtr = this->getSupervisedParameterSet();
  if( parSetPtr == nullptr ) return nullptr;
  if( _supervisedParameterIndex_ < 0 ) return nullptr;
  if( _supervisedParameterIndex_ > parSetPtr->getParameterList().size() ) return nullptr;
  return &parSetPtr->getParameterList().at(_supervisedParameterIndex_);
}
ParameterSet* DialCollection::getSupervisedParameterSet() const{
  if( _supervisedParameterSetIndex_ < 0 ) return nullptr;
  if( _supervisedParameterSetIndex_ > _parameterSetListPtr_->size() ) return nullptr;
  return &_parameterSetListPtr_->at(_supervisedParameterSetIndex_);
}

// core
void DialCollection::clear(){
  _dialInterfaceList_.clear();
  _dialBaseList_.clear();
  _dialInterfaceList_.shrink_to_fit();
  _dialBaseList_.shrink_to_fit();
  _dialFreeSlot_.setValue(0);
}
void DialCollection::resizeContainers(){
  LogInfo << "Resizing containers of the dial collection \"" << this->getTitle() << "\" from "
          << _dialInterfaceList_.size() << " to " << _dialFreeSlot_.getValue() << std::endl;
  _dialInterfaceList_.resize(_dialFreeSlot_.getValue());
  _dialBaseList_.resize(_dialFreeSlot_.getValue());
  _dialInterfaceList_.shrink_to_fit();
  _dialBaseList_.shrink_to_fit();
  this->setupDialInterfaceReferences();
}
void DialCollection::updateInputBuffers(){
  std::for_each(_dialInputBufferList_.begin(), _dialInputBufferList_.end(), [](DialInputBuffer& i_){
    i_.update();
  });
}
void DialCollection::setupDialInterfaceReferences(){
  LogThrowIf(_supervisedParameterSetIndex_==-1, "par set index not set.");
  LogThrowIf(_supervisedParameterSetIndex_>_parameterSetListPtr_->size(), "invalid selected parset index: " << _supervisedParameterSetIndex_);

  // set it up is not already done
  if( _dialInputBufferList_.empty() ){
    if( _supervisedParameterIndex_ == -1 ){
      // one dial interface per parameter
      LogThrowIf(_dialBaseList_.size() != _parameterSetListPtr_->at(_supervisedParameterSetIndex_).getParameterList().size(),
                 "Nb of dial base don't match the number of parameters of the selected set: nDials="
                 << _dialBaseList_.size() << " != " << "nPars="
                 << _parameterSetListPtr_->at(_supervisedParameterSetIndex_).getParameterList().size()
                 << std::endl << "is the defined dial binning matching the number of parameters?"
      );
      _dialInputBufferList_.resize(_dialBaseList_.size());
      for( int iDial = 0 ; iDial < int(_dialBaseList_.size()) ; iDial++ ){
        DialInputBuffer::ParameterReference p{};
        p.parSetIndex = _supervisedParameterSetIndex_;
        p.parIndex = iDial;
        _dialInputBufferList_[iDial].addParameterReference(p);
      }
    }
    else{
      // one parameter for every dial of the collection
      _dialInputBufferList_.resize(1);
      DialInputBuffer::ParameterReference p{};
      p.parSetIndex = _supervisedParameterSetIndex_;
      p.parIndex = _supervisedParameterIndex_;
      _dialInputBufferList_.back().addParameterReference(p);
    }
  }

  for( auto& inputBuffer : _dialInputBufferList_ ){ inputBuffer.setParSetRef( _parameterSetListPtr_ ); }

  if( _useMirrorDial_ ){
    for( auto& inputBuffer : _dialInputBufferList_ ){
      for( auto & inputParRef : inputBuffer.getInputParameterIndicesList() ){
        inputParRef.mirrorEdges.minValue = _mirrorLowEdge_;
        inputParRef.mirrorEdges.range = _mirrorRange_;
      }
    }
  }

  // initialize the input buffers
  for( auto& inputBuffer : _dialInputBufferList_ ){ inputBuffer.initialise(); }

  // Initializing dial interfaces:
  if( _dialInterfaceList_.size() != _dialBaseList_.size() ){
    _dialInterfaceList_.clear();
    _dialInterfaceList_.resize( _dialBaseList_.size() );
  }
  for( size_t iDial = 0 ; iDial < _dialBaseList_.size() ; iDial++ ){
    // Dial base reference
    _dialInterfaceList_[iDial].setDialBaseRef( _dialBaseList_[iDial].get() );

    // Input buffers
    if( _dialInputBufferList_.size() == 1 ){
      _dialInterfaceList_[iDial].setInputBufferRef( &_dialInputBufferList_[0] );
    }
    else if( _dialInputBufferList_.size() == _dialInterfaceList_.size() ){
      _dialInterfaceList_[iDial].setInputBufferRef( &_dialInputBufferList_[iDial] );
    }
    else{
      LogThrow("DEV: size mismatch between input buffers and dial interfaces."
                   << std::endl << "interface = " << _dialInterfaceList_.size()
                   << std::endl << "input = " << _dialInputBufferList_.size()
      );
    }

    // Input buffers
    if( not _dialBinSet_.getBinList().empty() ){
      if( _dialBinSet_.getBinList().size() == 1 ){
        _dialInterfaceList_[iDial].setDialBinRef( &_dialBinSet_.getBinList()[0] );
      }
      else if( _dialBinSet_.getBinList().size() == _dialInterfaceList_.size() ){
        _dialInterfaceList_[iDial].setDialBinRef( &_dialBinSet_.getBinList()[iDial] );
      }
      else{
        LogThrow("DEV: size mismatch between bins and dial interfaces."
                     << std::endl << "interface = " << _dialInterfaceList_.size()
                     << std::endl << "bins = " << _dialBinSet_.getBinList().size()
        );
      }
    }


    // Supervisor reference
    if( _dialResponseSupervisorList_.size() == 1 ){
      _dialInterfaceList_[iDial].setResponseSupervisorRef( &_dialResponseSupervisorList_[0] );
    }
    else if( _dialResponseSupervisorList_.size() == _dialInterfaceList_.size() ){
      _dialInterfaceList_[iDial].setResponseSupervisorRef( &_dialResponseSupervisorList_[iDial] );
    }
    else{
      LogThrow("DEV: size mismatch between response supervisors and dial interfaces."
                   << std::endl << "interface = " << _dialInterfaceList_.size()
                   << std::endl << "supervisor = " << _dialResponseSupervisorList_.size()
      );
    }
  }


}
size_t DialCollection::getNextDialFreeSlot(){
  return _dialFreeSlot_++;
}


// init protected
void DialCollection::readGlobals(const JsonType &config_) {
  // globals for the dialSet
  _enableDialsSummary_ = GenericToolbox::Json::fetchValue<bool>(_config_, "printDialsSummary", _enableDialsSummary_);

  _globalDialType_ = GenericToolbox::Json::fetchValue(config_, {{"dialsType"}, {"dialType"}}, _globalDialType_);

  _globalDialSubType_ =  GenericToolbox::Json::fetchValue<std::string>(config_, "dialSubType", _globalDialSubType_);

  _globalDialLeafName_ = GenericToolbox::Json::fetchValue<std::string>(config_, "dialLeafName", _globalDialLeafName_);

  if (GenericToolbox::Json::doKeyExist(config_, "applyCondition")) {
    _applyConditionStr_ = GenericToolbox::Json::fetchValue<std::string>(config_, "applyCondition");
  }
  else if (GenericToolbox::Json::doKeyExist(config_, "applyConditions")) {
    std::vector<std::string> conditionsList;

    for (auto &condEntry: GenericToolbox::Json::fetchValue<std::vector<JsonType>>(config_, "applyConditions")) {
      if (condEntry.is_string()) {
        conditionsList.emplace_back(condEntry.get<std::string>());
      } else if (condEntry.is_structured()) {
        auto expression = GenericToolbox::Json::fetchValue<std::string>(condEntry, {{"exp"},
                                                                                    {"expression"},
                                                                                    {"var"},
                                                                                    {"variable"}});
        std::stringstream ssCondEntry;

        // allowedRanges
        {
          auto allowedRanges = GenericToolbox::Json::fetchValue(condEntry, "allowedRanges",
                                                                std::vector<std::pair<double, double>>());
          if (not allowedRanges.empty()) {
            std::vector<std::string> allowedRangesCond;
            for (auto &allowedRange: allowedRanges) {
              LogThrowIf(allowedRange.first >= allowedRange.second,
                         "Invalid range bounds: min(" << allowedRange.first << ") max(" << allowedRange.second << ")")
              std::stringstream condSs;
              condSs << "(" << expression << " >= " << allowedRange.first;
              condSs << " && " << expression << " < " << allowedRange.second << ")";
              allowedRangesCond.emplace_back(condSs.str());
            }
            ssCondEntry << GenericToolbox::joinVectorString(allowedRangesCond, " || ");
          }
        }

        // allowedValues
        {
          auto allowedValues = GenericToolbox::Json::fetchValue(condEntry, "allowedValues", std::vector<double>());
          if (not allowedValues.empty()) {
            std::vector<std::string> allowedValuesCond;
            for (auto &allowedValue: allowedValues) {
              std::stringstream condSs;
              condSs << expression << " == " << allowedValue;
              allowedValuesCond.emplace_back(condSs.str());
            }
            if (not ssCondEntry.str().empty()) ssCondEntry << " || "; // allowed regions are linked with "OR"
            ssCondEntry << GenericToolbox::joinVectorString(allowedValuesCond, " || ");
          }
        }

        auto excludedRanges = GenericToolbox::Json::fetchValue(condEntry, "excludedRanges",
                                                               std::vector<std::pair<double, double>>());
        auto excludedValues = GenericToolbox::Json::fetchValue(condEntry, "excludedValues", std::vector<double>());
        if (not excludedRanges.empty() or not excludedValues.empty()) {
          if (not ssCondEntry.str().empty()) {
            // exclusion ranges are linked with &&: they are supposed to prevail
            ssCondEntry.str("(" + ssCondEntry.str() + ")");
            // after that no parenthesis needed since only && will be used
          }

          {
            if (not excludedRanges.empty()) {
              std::vector<std::string> excludedRangesCond;
              for (auto &excludedRange: excludedRanges) {
                LogThrowIf(excludedRange.first >= excludedRange.second,
                           "Invalid range bounds: min(" << excludedRange.first << ") max(" << excludedRange.second
                                                        << ")")
                std::stringstream condSs;
                condSs << expression << " < " << excludedRange.first << " && ";
                condSs << expression << " >= " << excludedRange.second;
                excludedRangesCond.emplace_back(condSs.str());
              }
              if (not ssCondEntry.str().empty()) ssCondEntry << " && "; // allowed regions are linked with "OR"
              ssCondEntry << GenericToolbox::joinVectorString(excludedRangesCond, " && ");
            }
          }

          {
            if (not excludedValues.empty()) {
              std::vector<std::string> excludedValuesCond;
              for (auto &excludedValue: excludedValues) {
                std::stringstream condSs;
                condSs << expression << " == " << excludedValue;
                excludedValuesCond.emplace_back(condSs.str());
              }
              if (not ssCondEntry.str().empty()) ssCondEntry << " && "; // allowed regions are linked with "OR"
              ssCondEntry << GenericToolbox::joinVectorString(excludedValuesCond, " && ");
            }
          }
        }

        LogThrowIf(ssCondEntry.str().empty(), "Could not parse condition entry: " << condEntry)
        conditionsList.emplace_back(ssCondEntry.str());
      } else {
        LogThrow("Could not recognise condition entry: " << condEntry);
      }
    }

    LogThrowIf(conditionsList.empty(), "No apply condition was recognised.")
    _applyConditionStr_ = "( ";
    _applyConditionStr_ += GenericToolbox::joinVectorString(conditionsList, " ) && ( ");
    _applyConditionStr_ += " )";
  }

  if (not _applyConditionStr_.empty()) {
    _applyConditionFormula_ = std::make_shared<TFormula>("_applyConditionFormula_", _applyConditionStr_.c_str());
    LogThrowIf(not _applyConditionFormula_->IsValid(),
               "\"" << _applyConditionStr_ << "\": could not be parsed as formula expression.")
  }

  _minDialResponse_ = GenericToolbox::Json::fetchValue(config_, {{"minDialResponse"}, {"minimumSplineResponse"}}, _minDialResponse_);
  _maxDialResponse_ = GenericToolbox::Json::fetchValue(config_, "maxDialResponse", _maxDialResponse_);

  _useMirrorDial_ = GenericToolbox::Json::fetchValue(config_, "useMirrorDial", _useMirrorDial_);
  if (_useMirrorDial_) {
    _mirrorLowEdge_ = GenericToolbox::Json::fetchValue(config_, "mirrorLowEdge", _mirrorLowEdge_);
    _mirrorHighEdge_ = GenericToolbox::Json::fetchValue(config_, "mirrorHighEdge", _mirrorHighEdge_);
    _mirrorRange_ = _mirrorHighEdge_ - _mirrorLowEdge_;
    LogThrowIf(_mirrorRange_ < 0, GET_VAR_NAME_VALUE(_mirrorHighEdge_) << " < " << GET_VAR_NAME_VALUE(_mirrorLowEdge_))
  }

  _allowDialExtrapolation_ = GenericToolbox::Json::fetchValue(config_, "allowDialExtrapolation", _allowDialExtrapolation_);
}
bool DialCollection::initializeNormDialsWithParBinning() {
  auto binning = GenericToolbox::Json::fetchValue(_config_, "parametersBinningPath", JsonType());
  if( binning.empty() ){ return false; } // not defined

  // Get global parameters from the main config
  this->readGlobals(_config_);

  // Read the binning
  _dialBinSet_ = DataBinSet();
  _dialBinSet_.setName("parameterBinning");
  _dialBinSet_.readBinningDefinition( binning );

  // By default use min dial response for norm dials
  _dialResponseSupervisorList_.resize( 1 );
  _dialResponseSupervisorList_[0].setMinResponse( GenericToolbox::Json::fetchValue(_config_, {{"minDialResponse"}, {"minimumSplineResponse"}}, double(0.)) );
  _dialResponseSupervisorList_[0].setMaxResponse( GenericToolbox::Json::fetchValue(_config_, "maxDialResponse", _maxDialResponse_) );

  _dialBaseList_.reserve( _dialBinSet_.getBinList().size() );
  DialBaseFactory factory;
  for(const auto & bin : _dialBinSet_.getBinList()) {
    _dialBaseList_.emplace_back(DialBaseObject(factory.makeDial(getTitle(), "Normalization","",nullptr,false)));
  }

  return true;
}
bool DialCollection::initializeDialsWithDefinition() {

  JsonType dialsDefinition = _config_;
  if( GenericToolbox::Json::doKeyExist(dialsDefinition, "dialsDefinitions") ) {
    // Fetch the dialSet corresponding to the selected parameter
    dialsDefinition = this->fetchDialsDefinition(GenericToolbox::Json::fetchValue<JsonType>(_config_, "dialsDefinitions"));
  }

  if( dialsDefinition.empty() ) {return false;}

  if( not GenericToolbox::Json::fetchValue<bool>(dialsDefinition, "isEnabled", true) ){
    LogDebug << "DialSet is disabled." << std::endl;
    return true;
  }

  this->readGlobals( dialsDefinition );
  DialBaseFactory dialBaseFactory;

  if( _globalDialType_ == "Norm" or _globalDialType_ == "Normalization" ) {
    // This dial collection is a normalization, so there is a single dial.
    // Create it here.
    _dialBaseList_.emplace_back(
      DialBaseObject(dialBaseFactory.makeDial(getTitle(),"Normalization","",nullptr,false)));
  }
  else if( _globalDialType_ == "Formula" or _globalDialType_ == "RootFormula" ){
    DialBaseFactory f;
    _dialBaseList_.emplace_back( DialBaseObject( f.makeDial( dialsDefinition ) ) );
  }
  else if( _globalDialType_ == "CompiledLibDial" ){
    DialBaseFactory f;
    _dialBaseList_.emplace_back( DialBaseObject( f.makeDial( dialsDefinition ) ) );
  }
  else {
    if     (not _globalDialLeafName_.empty()) {
      // The dialLeafName field has been provided, so this is an event by
      // event dial.  The generation of the dials will be handled in
      // DataDispenser.
      _isBinned_ = false;
    }
    else if( GenericToolbox::Json::doKeyExist(dialsDefinition, "binningFilePath") ) {
      // A binning file has been provided, so this is a binned dial.  Create
      // the dials for each bin here.  The dials will be assigned to the
      // events in DataDispenser.
      auto binningFilePath = GenericToolbox::Json::fetchValue<std::string>(dialsDefinition, "binningFilePath");

      _dialBinSet_ = DataBinSet();
      _dialBinSet_.setName(binningFilePath);
      _dialBinSet_.readBinningDefinition(binningFilePath);

      // Get the filename for a file with the object array of dials (graphs)
      // that will be applied based on the binning.
      auto filePath = GenericToolbox::Json::fetchValue<std::string>(dialsDefinition, "dialsFilePath");
      LogThrowIf(not GenericToolbox::doesTFileIsValid(filePath), "Could not open: " << filePath);
      TFile* dialsTFile = TFile::Open(filePath.c_str());
      LogThrowIf(dialsTFile==nullptr, "Could not open: " << filePath);

      if      ( GenericToolbox::Json::doKeyExist(dialsDefinition, "dialsList") ) {
        auto* dialsList = dialsTFile->Get<TObjArray>(GenericToolbox::Json::fetchValue<std::string>(dialsDefinition, "dialsList").c_str());

        LogThrowIf(
          dialsList==nullptr,
          "Could not find dialsList: " << GenericToolbox::Json::fetchValue<std::string>(dialsDefinition, "dialsList")
          );
        LogThrowIf(
          dialsList->GetEntries() != _dialBinSet_.getBinList().size(),
          this->getTitle() << ": Number of dials (" << dialsList->GetEntries()
          << ") don't match the number of bins " << _dialBinSet_.getBinList().size()
          );

        std::vector<int> excludedBins{};
        for( int iBin = 0 ;
             iBin < int(_dialBinSet_.getBinList().size()) ; ++iBin ) {
          TObject* binnedInitializer = dialsList->At(iBin);

          DialBase *dialBase = dialBaseFactory.makeDial(
              getTitle(),
              getGlobalDialType(),
              getGlobalDialSubType(),
              binnedInitializer,
              false);
          if (dialBase == nullptr) {
              LogAlert << "Invalid dial for " << getTitle() << " -> "
                       << _dialBinSet_.getBinList()[iBin].getSummary()
                       << std::endl;
              excludedBins.emplace_back(iBin);
              continue;
          }
          dialBase->setAllowExtrapolation(_allowDialExtrapolation_);
          _dialBaseList_.emplace_back(DialBaseObject(dialBase));
        }

        if( not excludedBins.empty() ){
          LogInfo << "Removing invalid bin dials..." << std::endl;
          for( int iBin = int(_dialBinSet_.getBinList().size()) ; iBin >= 0 ; iBin-- ){
            if( GenericToolbox::doesElementIsInVector(iBin, excludedBins) ){
              _dialBinSet_.getBinList().erase(_dialBinSet_.getBinList().begin() + iBin);
            }
          }
        }

        dialsTFile->Close();

      }

      ///////////////////////////////////////////////////////////////////////
      else if ( GenericToolbox::Json::doKeyExist(dialsDefinition, "dialsTreePath") ) {
        // Deprecated: A tree with event binning has beenprovided, so this is
        // a binned dial.  Create the dials for each bin here.  The dials will
        // be assigned to the events in DataDispenser.
        auto objPath = GenericToolbox::Json::fetchValue<std::string>(dialsDefinition, "dialsTreePath");
        auto* dialsTTree = (TTree*) dialsTFile->Get(objPath.c_str());
        LogThrowIf(dialsTTree== nullptr, objPath << " within " << filePath << " could not be opened.")

        Int_t kinematicBin;
        TSpline3* splinePtr = nullptr;
        TGraph* graphPtr = nullptr;

        // searching for additional split var
        std::vector<std::string> splitVarNameList;
        for( int iKey = 0 ; iKey < dialsTTree->GetListOfLeaves()->GetEntries() ; iKey++ ){
          std::string leafName = dialsTTree->GetListOfLeaves()->At(iKey)->GetName();
          if(leafName != "kinematicBin" and leafName != "spline" and leafName != "graph"){
            splitVarNameList.emplace_back(leafName);
          }
        }

        // Hooking to the tree
        std::vector<Int_t> splitVarValueList(splitVarNameList.size(), 0);
        std::vector<std::pair<int, int>> splitVarBoundariesList(splitVarNameList.size(), std::pair<int, int>());
        std::vector<std::vector<int>> splitVarValuesList(splitVarNameList.size(), std::vector<int>());
        dialsTTree->SetBranchAddress("kinematicBin", &kinematicBin);
        if( _globalDialType_ == "Spline" ) dialsTTree->SetBranchAddress("spline", &splinePtr);
        if( _globalDialType_ == "Graph" ) dialsTTree->SetBranchAddress("graph", &graphPtr);
        for( size_t iSplitVar = 0 ; iSplitVar < splitVarNameList.size() ; iSplitVar++ ){
          dialsTTree->SetBranchAddress(splitVarNameList[iSplitVar].c_str(), &splitVarValueList[iSplitVar]);
        }

        Long64_t nSplines = dialsTTree->GetEntries();
        LogWarning << "Reading dials in \"" << dialsTFile->GetName() << "\"" << std::endl;
        for( Long64_t iSpline = 0 ; iSpline < nSplines ; iSpline++ ){
          dialsTTree->GetEntry(iSpline);
          auto* dialBin = &_dialBinSet_.getBinList()[kinematicBin];
          dialBin->setIsZeroWideRangesTolerated(true);
          for( size_t iSplitVar = 0 ; iSplitVar < splitVarNameList.size() ; iSplitVar++ ){
            if( splitVarBoundariesList.at(iSplitVar).second < splitVarValueList.at(iSplitVar) or iSpline == 0 ){
              splitVarBoundariesList.at(iSplitVar).second = splitVarValueList.at(iSplitVar);
            }
            if( splitVarBoundariesList.at(iSplitVar).first > splitVarValueList.at(iSplitVar) or iSpline == 0 ){
              splitVarBoundariesList.at(iSplitVar).first = splitVarValueList.at(iSplitVar);
            }
            if( not GenericToolbox::doesElementIsInVector(splitVarValueList.at(iSplitVar), splitVarValuesList.at(iSplitVar)) ){
              splitVarValuesList.at(iSplitVar).emplace_back(splitVarValueList.at(iSplitVar));
            }
            dialBin->addBinEdge(splitVarNameList.at(iSplitVar), splitVarValueList.at(iSplitVar), splitVarValueList.at(iSplitVar));
          }

          TObject* dialInitializer{nullptr};
          if (getGlobalDialType() == "Spline") dialInitializer = splinePtr;
          if (getGlobalDialType() == "Graph") dialInitializer = graphPtr;
          DialBaseFactory factory;
          DialBase *dialBase = factory.makeDial(
              getTitle(),
              getGlobalDialType(),
              getGlobalDialSubType(),
              dialInitializer,
              false);
          if (dialBase) _dialBaseList_.emplace_back(DialBaseObject(dialBase));
        } // iSpline (in TTree)
        dialsTFile->Close();
      } // Splines in TTree
      else{
        LogError << "Neither dialsTreePath nor dialsList are provided..." << std::endl;
      }
    }
    else{
      LogError << "The dial is neither event-by-event nor binned..." << std::endl;
    }
  }

  _dialResponseSupervisorList_.emplace_back();
  _dialResponseSupervisorList_.back().setMinResponse(
      GenericToolbox::Json::fetchValue(_config_, {{"minDialResponse"}, {"minimumSplineResponse"}}, double(0.))
  );
  _dialResponseSupervisorList_.back().setMaxResponse(
      GenericToolbox::Json::fetchValue(_config_, "maxDialResponse", _maxDialResponse_)
  );

  return true;
}
JsonType DialCollection::fetchDialsDefinition(const JsonType &definitionsList_) {
  auto* parSetPtr = this->getSupervisedParameterSet();
  LogThrowIf(parSetPtr == nullptr, "Can't fetch dial definition of parameter: par ref not set.");
  auto* par = &parSetPtr->getParameterList()[_supervisedParameterIndex_];
  for(size_t iDial = 0 ; iDial < definitionsList_.size() ; iDial++ ){
    if( par->getName().empty() ){
      if( par->getParameterIndex() == iDial ){
        return definitionsList_.at(iDial);
      }
    }
    else if( par->getName() == GenericToolbox::Json::fetchValue<std::string>(definitionsList_.at(iDial), {{"name"}, {"parameterName"}}, "") ){
      return definitionsList_.at(iDial);
    }
  }
  return {};
}


//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
