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

LoggerInit([](){
  Logger::setUserHeaderStr("[DialSet]");
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
  while( _dialSetConfig_.is_string() ){
    // forward
    _dialSetConfig_ = JsonUtils::readConfigFile(_dialSetConfig_.get<std::string>());
  }
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
const std::vector<std::string> &DialSet::getDataSetNameList() const {
  return _dataSetNameList_;
}
std::vector<std::shared_ptr<Dial>> &DialSet::getDialList() {
  return _dialList_;
}
TFormula *DialSet::getApplyConditionFormula() const {
  return _applyConditionFormula_;
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

  nlohmann::json dialsDefinition = fetchDialsDefinition(JsonUtils::fetchValue<nlohmann::json>(_dialSetConfig_, "dialsDefinitions"));
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
    std::string binningFilePath = JsonUtils::fetchValue<std::string>(dialsDefinition, "binningFilePath");

    DataBinSet binning;
    binning.setName("dialsBinning");
    binning.readBinningDefinition(_workingDirectory_ + "/" + binningFilePath);
    auto binList = binning.getBinsList();

    std::string filePath = _workingDirectory_ + "/" + JsonUtils::fetchValue<std::string>(dialsDefinition, "dialsFilePath");
    TFile* dialsTFile = TFile::Open(filePath.c_str());
    if( dialsTFile == nullptr ){
      LogError << filePath << " could not be opened." << std::endl;
      throw std::runtime_error("dialsTFile could not be opened.");
    }
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
      if(    leafName != "kinematicBin"
             and leafName != "spline"
             and leafName != "graph"
        ){
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
    } // iSpline

    dialsTFile->Close();

    if( _applyConditionFormula_ == nullptr ) {
      // Create additional selection for helping the dial indexing
      std::vector<std::string> splitVarFormulaConditionList;
      for (size_t iSplitVar = 0; iSplitVar < splitVarNameList.size(); iSplitVar++) {

        // searching for gaps in the int values
        std::vector<std::pair<int, int>> excludedBands;
        for (int intVal = splitVarBoundariesList.at(iSplitVar).first + 1;
             intVal < splitVarBoundariesList.at(iSplitVar).second; intVal++) {
          if (not GenericToolbox::doesElementIsInVector(intVal, splitVarValuesList.at(iSplitVar))) {
            if (excludedBands.empty() or intVal - 1 != excludedBands.back().second) {
              excludedBands.emplace_back(std::pair<int, int>(intVal, intVal));
            } else {
              excludedBands.back().second = intVal;
            }
          }
        }

        std::stringstream ss;
        ss << "[" << splitVarNameList.at(iSplitVar) << "] >= " << splitVarBoundariesList.at(iSplitVar).first;
        ss << " && [" << splitVarNameList.at(iSplitVar) << "] <= " << splitVarBoundariesList.at(iSplitVar).second;
        if (not excludedBands.empty()) {
          for (auto &exclBand : excludedBands) {
            if (exclBand.first == exclBand.second) {
              // NOT THE VALUE
              ss << " && [" << splitVarNameList.at(iSplitVar) << "] != " << exclBand.first;
            } else {
              // NOT IN THE BAND
              ss << " && [" << splitVarNameList.at(iSplitVar) << "] <= " << exclBand.first;
              ss << " && [" << splitVarNameList.at(iSplitVar) << "] >= " << exclBand.second;
            }
          }
        }

        splitVarFormulaConditionList.emplace_back(ss.str());
      } // SplitVar

      std::string addFormulaCondition = GenericToolbox::joinVectorString(splitVarFormulaConditionList, " && ");
      _applyConditionFormula_ = new TFormula("_applyConditionFormula_", addFormulaCondition.c_str());
      if( not _applyConditionFormula_->IsValid() ){
        LogError << "\"" << addFormulaCondition << "\"" << ": could not be parsed as formula expression" << std::endl;
        _applyConditionFormula_->Print("ALL");
        throw std::runtime_error("invalid formula expression");
      }

    } // Formula ?
  } // Spline ? Graph ?
  else {
    LogError << "dialsType is not supported yet: " << dialTypeStr << "(" << dialsType << ")" << std::endl;
    throw std::logic_error("dialsType is not supported");
  }

  return true;
}



