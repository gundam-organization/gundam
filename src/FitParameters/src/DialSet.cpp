//
// Created by Nadrino on 21/05/2021.
//

#include "TFile.h"
#include "TTree.h"

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

#include "DialSet.h"
#include "JsonUtils.h"
#include "DataBinSet.h"

#include "NormalizationDial.h"
#include "SplineDial.h"

bool DialSet::_verboseMode_{false};

LoggerInit([](){
  Logger::setUserHeaderStr("[DialSet]");
  if( not DialSet::_verboseMode_ ) Logger::setMaxLogLevel(Logger::LogLevel::INFO);
})


DialSet::DialSet() {
  this->reset();
}
DialSet::~DialSet() {
  this->reset();
}

void DialSet::reset() {
  _dataSetNameList_.clear();
  _dialList_.clear();
  _dialSetConfig_ = nlohmann::json();
  _parameterIndex_ = -1;
  _parameterName_ = "";
  _enableDialsSummary_ = false;
  _isEnabled_ = true;
  _workingDirectory_ = ".";
}

void DialSet::setDialSetConfig(const nlohmann::json &dialSetConfig) {
  _dialSetConfig_ = dialSetConfig;
  JsonUtils::forwardConfig(_dialSetConfig_);
}
void DialSet::setParameterIndex(int parameterIndex) {
  _parameterIndex_ = parameterIndex;
}
void DialSet::setWorkingDirectory(const std::string &workingDirectory) {
  _workingDirectory_ = workingDirectory;
}
void DialSet::setParameterName(const std::string &parameterName) {
  _parameterName_ = parameterName;
}
void DialSet::setAssociatedParameterReference(void *associatedParameterReference) {
  _associatedParameterReference_ = associatedParameterReference;
}

void DialSet::initialize() {
  LogTrace << __METHOD_NAME__ << std::endl;

  // Sanity checks
  if( _parameterName_.empty() and _parameterIndex_ == -1 ){
    LogError << "_parameterIndex_ is not set." << std::endl;
    throw std::logic_error("_parameterIndex_ is not set.");
  }
  else if( _dialSetConfig_.empty() ){
    LogError << "_dialSetConfig_ is not set." << std::endl;
    throw std::logic_error("_dialSetConfig_ is not set.");
  }

  _dataSetNameList_ = JsonUtils::fetchValue<std::vector<std::string>>(
    _dialSetConfig_, "applyOnDataSets", std::vector<std::string>()
  );
  if( _dataSetNameList_.empty() ){
    _dataSetNameList_.emplace_back("*");
  }
  else { }

  _enableDialsSummary_ = JsonUtils::fetchValue<bool>(_dialSetConfig_, "printDialsSummary", false);

  std::string dialTypeStr = JsonUtils::fetchValue<std::string>(_dialSetConfig_, "dialType", "");
  if( not dialTypeStr.empty() ){
    _globalDialType_ = DialType::toDialType(dialTypeStr);
  }

  // Dials are directly defined with a binning file?
  if( initializeNormDialsWithBinning() ){ LogDebug << "DialSet initialised with binning definition." << std::endl;  }
  // Dials are individually defined?
  else if( initializeDialsWithDefinition() ) { LogDebug << "DialSet initialised with config definition." << std::endl; }
  // Dials definition not found?
  else{
    LogWarning << "Could not fetch dials definition for parameter: #" << _parameterIndex_;
    if( not _parameterName_.empty() ) LogWarning << " (" << _parameterName_ << ")";
    LogWarning << std::endl << "Disabling dialSet." << std::endl;
    _isEnabled_ = false;
  }

}

bool DialSet::isEnabled() const {
  return _isEnabled_;
}
const nlohmann::json &DialSet::getDialSetConfig() const {
  return _dialSetConfig_;
}
const std::vector<std::string> &DialSet::getDataSetNameList() const {
  return _dataSetNameList_;
}
std::vector<std::shared_ptr<Dial>> &DialSet::getDialList() {
  return _dialList_;
}
TFormula *DialSet::getApplyConditionFormula() const {
  return _applyConditionFormula_;
}
void *DialSet::getAssociatedParameterReference() {
  return _associatedParameterReference_;
}
const std::string &DialSet::getDialLeafName() const {
  return _dialLeafName_;
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

nlohmann::json DialSet::fetchDialsDefinition(const nlohmann::json &definitionsList_) {

  for(size_t iDial = 0 ; iDial < definitionsList_.size() ; iDial++ ){
    if( _parameterName_.empty() ){
      if( _parameterIndex_ == iDial ){
        return definitionsList_.at(iDial);
      }
    }
    else if( _parameterName_ == JsonUtils::fetchValue<std::string>(definitionsList_.at(iDial), "parameterName", "") ){
      return definitionsList_.at(iDial);
    }
  }

  return nlohmann::json();
}

bool DialSet::initializeNormDialsWithBinning() {

  std::string parameterBinningPath = JsonUtils::fetchValue<std::string>(_dialSetConfig_, "parametersBinningPath", "");
  if( parameterBinningPath.empty() ){ return false; }

  LogTrace << "Initializing dials with binning file..." << std::endl;

  DataBinSet binning;
  binning.setName("parameterBinning");
  DataBinSet::setVerbosity(static_cast<int>(Logger::LogLevel::ERROR)); // only print errors if any
  binning.readBinningDefinition(_workingDirectory_ + "/" + parameterBinningPath);
  DataBinSet::setVerbosity(static_cast<int>(Logger::getMaxLogLevel())); // take back the log level with this instance
  if( _parameterIndex_ >= binning.getBinsList().size() ){
    LogError << "Can't fetch parameter index #" << _parameterIndex_ << " while binning size is: " << binning.getBinsList().size() << std::endl;
    throw std::runtime_error("Can't fetch parameter index.");
  }

  auto* dialPtr = new NormalizationDial();
  dialPtr->setApplyConditionBin( binning.getBinsList().at( _parameterIndex_ ) );
  dialPtr->setAssociatedParameterReference(_associatedParameterReference_);
  dialPtr->initialize();
  _dialList_.emplace_back( std::shared_ptr<NormalizationDial>(dialPtr) );

  return true;
}
bool DialSet::initializeDialsWithDefinition() {

  nlohmann::json dialsDefinition = _dialSetConfig_;
  if( JsonUtils::doKeyExist(dialsDefinition, "dialsDefinitions") ){
    // Fetch the dialSet corresponding to the selected parameter
    dialsDefinition = this->fetchDialsDefinition(JsonUtils::fetchValue<nlohmann::json>(_dialSetConfig_, "dialsDefinitions"));
  }
  if( dialsDefinition.empty() ){ return false; }
  if( not JsonUtils::fetchValue<bool>(dialsDefinition, "isEnabled", true) ){
    LogDebug << "DialSet is disabled." << std::endl;
    return true;
  }

  // Fetch dial type
  DialType::DialType dialsType = _globalDialType_;
  std::string dialTypeStr = JsonUtils::fetchValue<std::string>(dialsDefinition, "dialsType");
  if( not dialTypeStr.empty() ){
    dialsType = DialType::toDialType(dialTypeStr);
  }

  _applyConditionStr_ = JsonUtils::fetchValue(dialsDefinition, "applyCondition", std::string(""));
  if( not _applyConditionStr_.empty() ){
    LogWarning << "Found apply condition: " << _applyConditionStr_ << std::endl;
    _applyConditionFormula_ = new TFormula("_applyConditionFormula_", _applyConditionStr_.c_str());
    if( not _applyConditionFormula_->IsValid() ){
      LogError << _applyConditionStr_ << ": could not be parsed as formula expression" << std::endl;
      throw std::runtime_error("invalid formula expression");
    }
  }

  if( dialsType == DialType::Normalization ){
    auto* dialPtr = new NormalizationDial();
    dialPtr->setAssociatedParameterReference(_associatedParameterReference_);
    dialPtr->initialize();
    _dialList_.emplace_back(std::make_shared<NormalizationDial>(*dialPtr) );
  }
  else if( dialsType == DialType::Spline or dialsType == DialType::Graph ){

    if     ( JsonUtils::doKeyExist(dialsDefinition, "dialLeafName") ){
      _dialLeafName_ = JsonUtils::fetchValue<std::string>(dialsDefinition, "dialLeafName");
    }
    else if( JsonUtils::doKeyExist(dialsDefinition, "binningFilePath") ){

      auto binningFilePath = JsonUtils::fetchValue<std::string>(dialsDefinition, "binningFilePath");
      DataBinSet binning;
      binning.setName(binningFilePath);
      binning.readBinningDefinition(_workingDirectory_ + "/" + binningFilePath);

      auto binList = binning.getBinsList();

      std::string filePath = _workingDirectory_ + "/" + JsonUtils::fetchValue<std::string>(dialsDefinition, "dialsFilePath");
      LogThrowIf(not GenericToolbox::doesTFileIsValid(filePath), "Could not open: " << filePath)
      TFile* dialsTFile = TFile::Open(filePath.c_str());
      LogThrowIf(dialsTFile==nullptr, "Could not open: " << filePath)

      if ( JsonUtils::doKeyExist(dialsDefinition, "dialsList") ) {
        auto* dialsList = dialsTFile->Get<TObjArray>(JsonUtils::fetchValue<std::string>(dialsDefinition, "dialsList").c_str());
        LogThrowIf(dialsList==nullptr, "Could not find dialsList: " << JsonUtils::fetchValue<std::string>(dialsDefinition, "dialsList"))

        LogThrowIf(dialsList->GetSize() != binList.size(), "Number of dials (" << dialsList->GetSize() << ") don't match the number of bins " << binList.size() << "")

      for( int iBin = 0 ; iBin < binList.size() ; iBin++ ){
        if      ( dialsType == DialType::Spline ){
          auto* dialPtr = new SplineDial();
          dialPtr->setApplyConditionBin(binList.at(iBin));
          dialPtr->copySpline((TSpline3*) dialsList->At(iBin));
          if( JsonUtils::doKeyExist(dialsDefinition, "minimunSplineResponse") ){
            dialPtr->setMinimumSplineResponse(
                JsonUtils::fetchValue<double>(dialsDefinition, "minimunSplineResponse")
              );
            }
            dialPtr->setAssociatedParameterReference(_associatedParameterReference_);
            dialPtr->initialize();
            _dialList_.emplace_back(std::make_shared<SplineDial>(*dialPtr) );
          }
          else if( dialsType == DialType::Graph ){
            // TODO
          }
        }

        dialsTFile->Close();

      }
      else if ( JsonUtils::doKeyExist(dialsDefinition, "dialsTreePath") ) {
        std::string objPath = JsonUtils::fetchValue<std::string>(dialsDefinition, "dialsTreePath");
        auto* dialsTTree = (TTree*) dialsTFile->Get(objPath.c_str());
        if( dialsTTree == nullptr ){
          LogError << objPath << " within " << filePath << " could not be opened." << std::endl;
          throw std::runtime_error("dialsTTree could not be opened.");
        }

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
        if( dialsType == DialType::Spline ) dialsTTree->SetBranchAddress("spline", &splinePtr);
        if( dialsType == DialType::Graph ) dialsTTree->SetBranchAddress("graph", &graphPtr);
        for( size_t iSplitVar = 0 ; iSplitVar < splitVarNameList.size() ; iSplitVar++ ){
          dialsTTree->SetBranchAddress(splitVarNameList[iSplitVar].c_str(), &splitVarValueList[iSplitVar]);
        }

        Long64_t nSplines = dialsTTree->GetEntries();
        LogDebug << "Reading dials in \"" << dialsTFile->GetName() << "\"" << std::endl;
        for( Long64_t iSpline = 0 ; iSpline < nSplines ; iSpline++ ){
          dialsTTree->GetEntry(iSpline);
          auto dialBin = binList.at(kinematicBin); // copy
          dialBin.setIsZeroWideRangesTolerated(true);
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
            dialBin.addBinEdge(splitVarNameList.at(iSplitVar), splitVarValueList.at(iSplitVar), splitVarValueList.at(iSplitVar));
          }
          if      ( dialsType == DialType::Spline ){
            auto* dialPtr = new SplineDial();
            dialPtr->setApplyConditionBin(dialBin);
            dialPtr->setSplinePtr((TSpline3*) splinePtr->Clone());
            dialPtr->setAssociatedParameterReference(_associatedParameterReference_);
            dialPtr->initialize();
            _dialList_.emplace_back(std::make_shared<SplineDial>(*dialPtr) );
          }
          else if( dialsType == DialType::Graph ){
            // TODO
          }
          dialBin.addBinEdge(splitVarNameList.at(iSplitVar), splitVarValueList.at(iSplitVar), splitVarValueList.at(iSplitVar));
        }
        if      ( dialsType == DialType::Spline ){
          _dialList_.emplace_back( std::make_shared<SplineDial>() );
          _dialList_.back()->setApplyConditionBin(dialBin);
          _dialList_.back()->setAssociatedParameterReference(_associatedParameterReference_);
          dynamic_cast<SplineDial*>(_dialList_.back().get())->copySpline(splinePtr);
          dynamic_cast<SplineDial*>(_dialList_.back().get())->initialize();
        }
        else if( dialsType == DialType::Graph ){
          // TODO
        }
      } // iSpline

        dialsTFile->Close();

      }


      else{
        LogError << "Neither dialsTreePath nor dialsList are provided..." << std::endl;
      }
    }
    else {
      LogError << "The dial is neither even-by-event nor binned..." << std::endl;
    }
  } // Spline ? Graph ?
  else {
    LogError << "dialsType is not supported yet: " << dialTypeStr << "(" << dialsType << ")" << std::endl;
    throw std::logic_error("dialsType is not supported");
  }

  return true;
}
