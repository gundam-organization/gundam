//
// Created by Adrien BLANCHET on 21/05/2021.
//

#include "TFile.h"
#include "TTree.h"

#include "Logger.h"


#include "DialSet.h"
#include "JsonUtils.h"
#include "DataBinSet.h"

#include "NormalizationDial.h"
#include "SplineDial.h"

DialSet::DialSet() {
  Logger::setUserHeaderStr("[DialSet]");
  this->reset();
}
DialSet::~DialSet() {
  this->reset();
}

void DialSet::reset() {
  _name_ = "";
  for( auto& dial : _dialList_ ) delete dial;
  _dialList_.clear();
  _dialSetConfig_ = nlohmann::json();
  _parameterIndex_ = -1;
  _parameterName_ = "";
}

void DialSet::setDialSetConfig(const nlohmann::json &dialSetConfig) {
  _dialSetConfig_ = dialSetConfig;
}
void DialSet::setParameterIndex(int parameterIndex) {
  _parameterIndex_ = parameterIndex;
}

void DialSet::initialize() {

  if( _parameterName_.empty() and _parameterIndex_ == -1 ){
    LogError << "_parameterIndex_ is not set." << std::endl;
    throw std::logic_error("_parameterIndex_ is not set.");
  }
  if( _dialSetConfig_.empty() ){
    LogError << "_dialSetConfig_ is not set." << std::endl;
    throw std::logic_error("_dialSetConfig_ is not set.");
  }

  _name_ = JsonUtils::fetchValue<std::string>(_dialSetConfig_, "applyOnDataSets");

  std::string enumStr;
  int enumIndex;

  enumStr = JsonUtils::fetchValue<std::string>(_dialSetConfig_, "dialType", "");
  if( not enumStr.empty() ){
    enumIndex = DialType::DialTypeEnumNamespace::toEnumInt(enumStr);
    if( enumIndex == DialType::DialTypeEnumNamespace::enumOffSet - 1 ){
      LogError << "\"" << enumStr << "\" unrecognized  dial type." << std::endl;
      LogError << "Expecting: { " << DialType::DialTypeEnumNamespace::enumNamesAgregate << " }" << std::endl;
      throw std::runtime_error("Unrecognized  dial type.");
    }
    _globalDialType_ = static_cast<DialType::DialType>(enumIndex);
  }

  std::string parameterBinningPath = JsonUtils::fetchValue<std::string>(_dialSetConfig_, "parametersBinningPath", "");
  if( not parameterBinningPath.empty() ){

    // Sanity checks
    if( _parameterIndex_ < 0 ){
      LogError << "_parameterIndex_ needs to be set while parameters are defined with a binning file." << std::endl;
      throw std::logic_error("_parameterIndex_ needs to be set while parameters are defined with a binning file.");
    }

    DataBinSet binning;
    binning.setName("parameterBinning");
    binning.readBinningDefinition(parameterBinningPath);
    if( _parameterIndex_ >= binning.getBinsList().size() ){
      LogError << "Can't fetch parameter index #" << _parameterIndex_ << " while binning size is: " << binning.getBinsList().size() << std::endl;
      throw std::runtime_error("Can't fetch parameter index.");
    }
    _dialList_.emplace_back( new NormalizationDial() );
    _dialList_.back()->setApplyConditionBin(binning.getBinsList().at(_parameterIndex_));
    _dialList_.back()->initialize();
  }
  else{

    auto dialsDefinitions = JsonUtils::fetchValue<nlohmann::json>(_dialSetConfig_, "dialsDefinitions");
    nlohmann::json dialsDefinition;
    for(size_t iDial = 0 ; iDial < dialsDefinitions.size() ; iDial++ ){
      if( _parameterName_.empty() ){
        if( _parameterIndex_ == iDial ){
          dialsDefinition = dialsDefinitions.at(iDial);
          break;
        }
      }
      else if( _parameterName_ == JsonUtils::fetchValue<std::string>(_dialSetConfig_, "parameterName", "") ){
        dialsDefinition = dialsDefinitions.at(iDial);
        break;
      }
    }

    if( dialsDefinition.empty() ){
      LogError << "Could not fetch dialsDefinition associated with parameter." << std::endl;
      throw std::runtime_error("Could not fetch dialsDefinition associated with parameter.");
    }

    if( JsonUtils::fetchValue<bool>(dialsDefinition, "isEnabled", true) ){
      DialType::DialType dialsType = _globalDialType_;
      std::string dialTypeStr = JsonUtils::fetchValue<std::string>(dialsDefinition, "dialsType");
      if( not dialTypeStr.empty() ){
        int enumIndex = DialType::DialTypeEnumNamespace::toEnumInt(dialTypeStr);
        if( enumIndex == DialType::DialTypeEnumNamespace::enumOffSet - 1 ){
          LogError << "\"" << dialTypeStr << "\" unrecognized  dial type." << std::endl;
          LogError << "Expecting: { " << DialType::DialTypeEnumNamespace::enumNamesAgregate << " }" << std::endl;
          throw std::runtime_error("Unrecognized dial type.");
        }
        dialsType = static_cast<DialType::DialType>(enumIndex);
      }
      if( dialsType == DialType::Normalization ){
        _dialList_.emplace_back( new NormalizationDial() );
      }
      else if( dialsType == DialType::Spline or dialsType == DialType::Graph ){
        std::string binningFilePath = JsonUtils::fetchValue<std::string>(dialsDefinition, "binningFilePath");

        DataBinSet binning;
        binning.setName("dialsBinning");
        binning.readBinningDefinition(binningFilePath);
        auto binList = binning.getBinsList();

        std::string filePath = JsonUtils::fetchValue<std::string>(dialsDefinition, "dialsFilePath");
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

        Int_t kinematicBin;
        TSpline3* splinePtr;
        TGraph* graphPtr;
        std::vector<Int_t> splitVarValueList;

        // Hooking to the tree
        dialsTTree->SetBranchAddress("kinematicBin", &kinematicBin);
        if( dialsType == DialType::Spline ) dialsTTree->SetBranchAddress("spline", &splinePtr);
        if( dialsType == DialType::Graph ) dialsTTree->SetBranchAddress("graph", &graphPtr);
        for( size_t iSplitVar = 0 ; iSplitVar < splitVarNameList.size() ; iSplitVar++ ){
          splitVarValueList.emplace_back(0);
          dialsTTree->SetBranchAddress(splitVarNameList[iSplitVar].c_str(), &splitVarValueList[iSplitVar]);
        }

        Long64_t nSplines = dialsTTree->GetEntries();
        LogInfo << "Reading dials in " << dialsTFile->GetName() << std::endl;
        for( Long64_t iSpline = 0 ; iSpline < nSplines ; iSpline++ ){
          dialsTTree->GetEntry(iSpline);
          auto dialBin = binList.at(kinematicBin);
          dialBin.setIsZeroWideRangesTolerated(true);
          for( size_t iSplitVar = 0 ; iSplitVar < splitVarNameList.size() ; iSplitVar++ ){
            dialBin.addBinEdge(splitVarNameList.at(iSplitVar), splitVarValueList.at(iSplitVar), splitVarValueList.at(iSplitVar));
          }
          if      ( dialsType == DialType::Spline ){
            auto* splineDialPtr = new SplineDial();
            splineDialPtr->setSplinePtr(splinePtr);
            splineDialPtr->setApplyConditionBin(dialBin);
            _dialList_.emplace_back( splineDialPtr );
          }
          else if( dialsType == DialType::Graph ){
            // TODO
          }
        }

      }
      else {
        LogError << "dialsType is invalid" << std::endl;
        throw std::logic_error("dialsType is invalid");
      }

    }
  }



}

const std::string &DialSet::getName() const {
  return _name_;
}

void DialSet::reweightEvent(AnaEvent *eventPtr_) {

}
