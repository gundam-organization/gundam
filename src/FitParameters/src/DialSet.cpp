//
// Created by Nadrino on 21/05/2021.
//

#include "DialSet.h"
#include "JsonUtils.h"
#include "DataBinSet.h"
#include "NormDial.h"
#include "SplineDial.h"
#include "GraphDial.h"
#include "FitParameter.h"

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

#include "TFile.h"
#include "TTree.h"


bool DialSet::verboseMode{false};

LoggerInit([]{
  Logger::setUserHeaderStr("[DialSet]");
});


DialSet::DialSet(const FitParameter* owner_): _owner_(owner_){}

void DialSet::readConfigImpl(){
  LogThrowIf(_config_.empty(), "Config not set for dial set.");

  _dataSetNameList_ = JsonUtils::fetchValue<std::vector<std::string>>(
      _config_, "applyOnDataSets", std::vector<std::string>()
  );
  if( _dataSetNameList_.empty() ){
    _dataSetNameList_.emplace_back("*");
  }

  // Dials are directly defined with a binning file?
  if     ( initializeNormDialsWithParBinning() ){ /* LogInfo << "DialSet initialised with parameter binning definition." << std::endl; */  }
    // Dials are individually defined?
  else if( initializeDialsWithDefinition() )   { /* LogInfo << "DialSet initialised with config definition." << std::endl; */ }
    // Dials definition not found?
  else{
    LogWarning << "Could not fetch dials definition for parameter: #" << _owner_->getParameterIndex();
    if( not _owner_->getName().empty() ) LogWarning << " (" << _owner_->getName() << ")";
    LogWarning << std::endl << "Disabling dialSet." << std::endl;
    _isEnabled_ = false;
  }
}
void DialSet::initializeImpl() {
  LogThrowIf(_owner_==nullptr, "Owner address not set.");
}

void DialSet::setOwner(const FitParameter* owner_){
  _owner_ = owner_;
}

bool DialSet::isEnabled() const {
  return _isEnabled_;
}
const std::vector<std::string> &DialSet::getDataSetNameList() const {
  return _dataSetNameList_;
}
std::vector<DialWrapper> &DialSet::getDialList() {
  return _dialList_;
}
TFormula *DialSet::getApplyConditionFormula() const {
  return _applyConditionFormula_.get();
}
const std::string &DialSet::getDialSubType() const {
  return _globalDialSubType_;
}
const std::string &DialSet::getDialLeafName() const {
  return _globalDialLeafName_;
}
DialType::DialType DialSet::getGlobalDialType() const {
  return _globalDialType_;
}
double DialSet::getMinDialResponse() const {
  return _minDialResponse_;
}
double DialSet::getMaxDialResponse() const {
  return _maxDialResponse_;
}
bool DialSet::useMirrorDial() const {
  return _useMirrorDial_;
}
double DialSet::getMirrorLowEdge() const {
  return _mirrorLowEdge_;
}
double DialSet::getMirrorHighEdge() const {
  return _mirrorHighEdge_;
}
double DialSet::getMirrorRange() const {
  return _mirrorRange_;
}

std::string DialSet::getSummary() const {
  std::stringstream ss;
  ss << "DialSet: Datasets: " << GenericToolbox::parseVectorAsString(_dataSetNameList_);

  if( not _applyConditionStr_.empty() ) {
    ss << " / applyCondition:\"" << _applyConditionStr_ << "\"";
  }

  if( _enableDialsSummary_ ){
    for( const auto& dialPtr: _dialList_ ){
      ss << std::endl << GenericToolbox::indentString(dialPtr->getSummary(), 2);
    }
  }

  return ss.str();
}

// Protected
void DialSet::readGlobals(const nlohmann::json &config_){
  // globals for the dialSet
  _enableDialsSummary_ = JsonUtils::fetchValue<bool>(_config_, "printDialsSummary", _enableDialsSummary_);

  std::string dialTypeStr = JsonUtils::fetchValue(config_, "dialsType", "");
  if( not dialTypeStr.empty() ){
    if( dialTypeStr == "Normalization" ) dialTypeStr = "Norm"; // old name
    _globalDialType_ = DialType::DialTypeEnumNamespace::toEnum(dialTypeStr);
    LogThrowIf(_globalDialType_==DialType::DialType_OVERFLOW, "Invalid dial type provided: " << dialTypeStr)
  }

  if     ( JsonUtils::doKeyExist(config_, "applyCondition") ){
    _applyConditionStr_ = JsonUtils::fetchValue<std::string>(config_, "applyCondition");
  }
  else if( JsonUtils::doKeyExist(config_, "applyConditions") ){
    std::vector<std::string> conditionsList;

    for( auto& condEntry : JsonUtils::fetchValue<std::vector<nlohmann::json>>(config_, "applyConditions") ){
      if( condEntry.is_string() ){
        conditionsList.emplace_back(condEntry.get<std::string>());
      }
      else if( condEntry.is_structured() ){
        auto expression = JsonUtils::fetchValue<std::string>(condEntry, {{"exp"}, {"expression"}, {"var"}, {"variable"}});
        std::stringstream ssCondEntry;

        // allowedRanges
        {
          auto allowedRanges = JsonUtils::fetchValue(condEntry, "allowedRanges", std::vector<std::pair<double,double>>());
          if( not allowedRanges.empty() ){
            std::vector<std::string> allowedRangesCond;
            for( auto& allowedRange : allowedRanges ){
              LogThrowIf(allowedRange.first >= allowedRange.second, "Invalid range bounds: min(" << allowedRange.first << ") max(" << allowedRange.second << ")" )
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
          auto allowedValues = JsonUtils::fetchValue(condEntry, "allowedValues", std::vector<double>());
          if( not allowedValues.empty() ){
            std::vector<std::string> allowedValuesCond;
            for( auto& allowedValue : allowedValues ){
              std::stringstream condSs;
              condSs << expression << " == " << allowedValue;
              allowedValuesCond.emplace_back(condSs.str());
            }
            if( not ssCondEntry.str().empty() ) ssCondEntry << " || "; // allowed regions are linked with "OR"
            ssCondEntry << GenericToolbox::joinVectorString(allowedValuesCond, " || ");
          }
        }

        auto excludedRanges = JsonUtils::fetchValue(condEntry, "excludedRanges", std::vector<std::pair<double,double>>());
        auto excludedValues = JsonUtils::fetchValue(condEntry, "excludedValues", std::vector<double>());
        if( not excludedRanges.empty() or not excludedValues.empty() ){
          if( not ssCondEntry.str().empty() ){
            // exclusion ranges are linked with &&: they are supposed to prevail
            ssCondEntry.str("(" + ssCondEntry.str() + ")");
            // after that no parenthesis needed since only && will be used
          }

          {
            if( not excludedRanges.empty() ){
              std::vector<std::string> excludedRangesCond;
              for( auto& excludedRange : excludedRanges ){
                LogThrowIf(excludedRange.first >= excludedRange.second, "Invalid range bounds: min(" << excludedRange.first << ") max(" << excludedRange.second << ")" )
                std::stringstream condSs;
                condSs << expression << " < " << excludedRange.first << " && ";
                condSs << expression << " >= " << excludedRange.second;
                excludedRangesCond.emplace_back(condSs.str());
              }
              if( not ssCondEntry.str().empty() ) ssCondEntry << " && "; // allowed regions are linked with "OR"
              ssCondEntry << GenericToolbox::joinVectorString(excludedRangesCond, " && ");
            }
          }

          {
            if( not excludedValues.empty() ){
              std::vector<std::string> excludedValuesCond;
              for( auto& excludedValue : excludedValues ){
                std::stringstream condSs;
                condSs << expression << " == " << excludedValue;
                excludedValuesCond.emplace_back(condSs.str());
              }
              if( not ssCondEntry.str().empty() ) ssCondEntry << " && "; // allowed regions are linked with "OR"
              ssCondEntry << GenericToolbox::joinVectorString(excludedValuesCond, " && ");
            }
          }
        }

        LogThrowIf(ssCondEntry.str().empty(), "Could not parse condition entry: " << condEntry)
        conditionsList.emplace_back(ssCondEntry.str());
      }
      else{
        LogThrow("Could not recognise condition entry: " << condEntry);
      }
    }

    LogThrowIf(conditionsList.empty(), "No apply condition was recognised.")
    _applyConditionStr_ = "( ";
    _applyConditionStr_ += GenericToolbox::joinVectorString(conditionsList, " ) && ( ");
    _applyConditionStr_ += " )";
  }

  if( not _applyConditionStr_.empty() ){
//    LogWarning << "Apply condition: " << _applyConditionStr_ << std::endl;
    _applyConditionFormula_ = std::make_shared<TFormula>("_applyConditionFormula_", _applyConditionStr_.c_str());
    LogThrowIf(not _applyConditionFormula_->IsValid(),
               "\"" << _applyConditionStr_ << "\": could not be parsed as formula expression.")
  }

  _minDialResponse_ = JsonUtils::fetchValue(config_, {{"minDialResponse"}, {"minimumSplineResponse"}}, _minDialResponse_);
  _maxDialResponse_ = JsonUtils::fetchValue(config_, "maxDialResponse", _maxDialResponse_);

  _useMirrorDial_   = JsonUtils::fetchValue(config_, "useMirrorDial", _useMirrorDial_);
  if( _useMirrorDial_ ){
    _mirrorLowEdge_ = JsonUtils::fetchValue(config_, "mirrorLowEdge", _mirrorLowEdge_);
    _mirrorHighEdge_ = JsonUtils::fetchValue(config_, "mirrorHighEdge", _mirrorHighEdge_);
    _mirrorRange_ = _mirrorHighEdge_ - _mirrorLowEdge_;
    LogThrowIf(_mirrorRange_ < 0, GET_VAR_NAME_VALUE(_mirrorHighEdge_) << " < " << GET_VAR_NAME_VALUE(_mirrorLowEdge_))
  }

  _allowDialExtrapolation_ = JsonUtils::fetchValue(config_, "allowDialExtrapolation", _allowDialExtrapolation_);
}
bool DialSet::initializeNormDialsWithParBinning() {

  auto parameterBinningPath = JsonUtils::fetchValue<std::string>(_config_, "parametersBinningPath", "");
  if( parameterBinningPath.empty() ){ return false; }

  this->readGlobals(_config_);

  DataBinSet binning;
  binning.setName("parameterBinning");
  DataBinSet::setVerbosity(static_cast<int>(Logger::LogLevel::ERROR)); // only print errors if any
  binning.readBinningDefinition(parameterBinningPath);
  DataBinSet::setVerbosity(static_cast<int>(Logger::getMaxLogLevel())); // take back the log level with this instance

  LogThrowIf(
      _owner_->getParameterIndex() >= binning.getBinsList().size(),
      "Can't fetch parameter index #" << _owner_->getParameterIndex() << " while binning size is: " << binning.getBinsList().size()
      )

  _binningCacheList_.emplace_back();
  _binningCacheList_.back().addBin(binning.getBinsList().at(_owner_->getParameterIndex()));

  NormDial dial(this);
  dial.setApplyConditionBin( &_binningCacheList_.back().getBinsList()[0] );
  dial.initialize();
  _dialList_.emplace_back( std::make_unique<NormDial>(dial) );

  // By default use min dial response for norm dials
  _minDialResponse_ = JsonUtils::fetchValue(_config_, {{"minDialResponse"}, {"minimumSplineResponse"}}, 0);
  _maxDialResponse_ = JsonUtils::fetchValue(_config_, "maxDialResponse", _maxDialResponse_);

  return true;
}
bool DialSet::initializeDialsWithDefinition() {

  nlohmann::json dialsDefinition = _config_;
  if( JsonUtils::doKeyExist(dialsDefinition, "dialsDefinitions") ){
    // Fetch the dialSet corresponding to the selected parameter
    dialsDefinition = this->fetchDialsDefinition(JsonUtils::fetchValue<nlohmann::json>(_config_, "dialsDefinitions"));
  }
  if( dialsDefinition.empty() ){ return false; }
  if( not JsonUtils::fetchValue<bool>(dialsDefinition, "isEnabled", true) ){
    LogDebug << "DialSet is disabled." << std::endl;
    return true;
  }

  this->readGlobals(dialsDefinition);

  if( _globalDialType_ == DialType::Norm ){
    NormDial dial(this);
    dial.initialize();
    _dialList_.emplace_back( std::make_unique<NormDial>(dial) );

    // By default use min dial response for norm dials
    _minDialResponse_ = JsonUtils::fetchValue(_config_, {{"minDialResponse"}, {"minimumSplineResponse"}}, 0);
    _maxDialResponse_ = JsonUtils::fetchValue(_config_, "maxDialResponse", _maxDialResponse_);
  }
  else if( _globalDialType_ == DialType::Spline or _globalDialType_ == DialType::Graph ){

    if ( JsonUtils::doKeyExist(dialsDefinition, "dialSubType") ) {
      _globalDialSubType_ =  JsonUtils::fetchValue<std::string>(dialsDefinition, "dialSubType");
    }

    if     ( JsonUtils::doKeyExist(dialsDefinition, "dialLeafName") ){
      _globalDialLeafName_ = JsonUtils::fetchValue<std::string>(dialsDefinition, "dialLeafName");
      // nothing to do here, the dials list will be filled while reading the datasets
    }
    else if( JsonUtils::doKeyExist(dialsDefinition, "binningFilePath") ){

      auto binningFilePath = JsonUtils::fetchValue<std::string>(dialsDefinition, "binningFilePath");

      _binningCacheList_.emplace_back();
      _binningCacheList_.back().setName(binningFilePath);
      _binningCacheList_.back().readBinningDefinition(binningFilePath);

      auto filePath = JsonUtils::fetchValue<std::string>(dialsDefinition, "dialsFilePath");
      LogThrowIf(not GenericToolbox::doesTFileIsValid(filePath), "Could not open: " << filePath)
      TFile* dialsTFile = TFile::Open(filePath.c_str());
      LogThrowIf(dialsTFile==nullptr, "Could not open: " << filePath)

      if      ( JsonUtils::doKeyExist(dialsDefinition, "dialsList") ) {
        auto* dialsList = dialsTFile->Get<TObjArray>(JsonUtils::fetchValue<std::string>(dialsDefinition, "dialsList").c_str());
        LogThrowIf(dialsList==nullptr, "Could not find dialsList: " << JsonUtils::fetchValue<std::string>(dialsDefinition, "dialsList"));

        LogThrowIf(dialsList->GetSize() != _binningCacheList_.back().getBinsList().size(), "Number of dials (" << dialsList->GetSize() << ") don't match the number of bins "
                   << _binningCacheList_.back().getBinsList().size() << "");

        for( int iBin = 0 ; iBin < _binningCacheList_.back().getBinsList().size() ; iBin++ ){
          if     ( _globalDialType_ == DialType::Spline ){
            SplineDial s(this);
            s.setApplyConditionBin(&_binningCacheList_.back().getBinsList()[iBin]);
            s.copySpline((TSpline3*) dialsList->At(iBin));
            s.initialize();
            _dialList_.emplace_back( std::make_unique<SplineDial>(s) );
          }
          else if( _globalDialType_ == DialType::Graph ){
            GraphDial g(this);
            g.setApplyConditionBin(&_binningCacheList_.back().getBinsList()[iBin]);
            g.setGraph(*(TGraph*) dialsList->At(iBin));
            g.initialize();
            _dialList_.emplace_back( std::make_unique<GraphDial>(g) );
          }
          else{
            LogThrow("Should not be here???")
          }
        }

        dialsTFile->Close();

      }
      else if ( JsonUtils::doKeyExist(dialsDefinition, "dialsTreePath") ) {
        // OLD
        auto objPath = JsonUtils::fetchValue<std::string>(dialsDefinition, "dialsTreePath");
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
        if( _globalDialType_ == DialType::Spline ) dialsTTree->SetBranchAddress("spline", &splinePtr);
        if( _globalDialType_ == DialType::Graph ) dialsTTree->SetBranchAddress("graph", &graphPtr);
        for( size_t iSplitVar = 0 ; iSplitVar < splitVarNameList.size() ; iSplitVar++ ){
          dialsTTree->SetBranchAddress(splitVarNameList[iSplitVar].c_str(), &splitVarValueList[iSplitVar]);
        }

        Long64_t nSplines = dialsTTree->GetEntries();
        LogWarning << "Reading dials in \"" << dialsTFile->GetName() << "\"" << std::endl;
        for( Long64_t iSpline = 0 ; iSpline < nSplines ; iSpline++ ){
          dialsTTree->GetEntry(iSpline);
          auto* dialBin = &_binningCacheList_.back().getBinsList()[kinematicBin];
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
          if      ( _globalDialType_ == DialType::Spline ){
            SplineDial s(this);
            s.setApplyConditionBin(dialBin);
            s.copySpline(splinePtr);
            s.initialize();
            _dialList_.emplace_back( std::make_unique<SplineDial>(s) );
          }
          else if( _globalDialType_ == DialType::Graph ){
            LogThrow("TTree loading of DialType::Graph not implemented.")
            // TODO
          }
        } // iSpline (in TTree)
        dialsTFile->Close();
      } // Splines in TTree
      else{
        LogError << "Neither dialsTreePath nor dialsList are provided..." << std::endl;
      }
    }
    else {
      LogError << "The dial is neither even-by-event nor binned..." << std::endl;
    }
  } // Spline ? Graph ?
  else {
    LogError << "dialsType is not supported yet: " << DialType::DialTypeEnumNamespace::toString(_globalDialType_)
             << "(" << _globalDialType_ << ")" << std::endl;
    throw std::logic_error("dialsType is not supported");
  }

  return true;
}
nlohmann::json DialSet::fetchDialsDefinition(const nlohmann::json &definitionsList_) {
  for(size_t iDial = 0 ; iDial < definitionsList_.size() ; iDial++ ){
    if( _owner_->getName().empty() ){
      if( _owner_->getParameterIndex() == iDial ){
        return definitionsList_.at(iDial);
      }
    }
    else if( _owner_->getName() == JsonUtils::fetchValue<std::string>(definitionsList_.at(iDial), "parameterName", "") ){
      return definitionsList_.at(iDial);
    }
  }
  return {};
}

bool DialSet::isAllowDialExtrapolation() const {
  return _allowDialExtrapolation_;
}
