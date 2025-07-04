//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialCollection.h"

#include "TabulatedDialFactory.h"
#include "KrigedDialFactory.h"

#include "RootFormula.h"
#include "Shift.h"
#include "RootGraph.h"
#include "Graph.h"
#include "Norm.h"
#include "MonotonicSpline.h"
#include "CompactSpline.h"
#include "UniformSpline.h"
#include "GeneralSpline.h"
#include "RootSpline.h"
#include "Bilinear.h"
#include "Bicubic.h"
#include "CompiledLibDial.h"

#include "DialUtils.h"
#include "GundamGlobals.h"

#include "Logger.h"

#include <sstream>

void DialCollection::prepareConfig(ConfigReader &config_){
  config_.clearFields();
  config_.defineFields({
      {"name"},
      {"isEnabled"},
      {"applyOnDataSets"},
      {"dialInputList"},
      {"printDialSummary", {"printDialsSummary"}},
      {"dialType", {"type", "dialsType"}},
      {"options", {"dialSubType"}},
      {"treeExpression", {"dialLeafName"}},
      {"minDialResponse", {"minimumSplineResponse"}},
      {"maxDialResponse"},
      {"useMirrorDial"},
      {"mirrorLowEdge"},
      {"mirrorHighEdge"},
      {"allowDialExtrapolation"},
      {"applyCondition"},
      {"definitionRange"},
      {"mirrorDefinitionRange"},
      {"applyConditions"},
      {"parametersBinningPath"},
      {"binning", {"binningFilePath"}},
      {"dialsFilePath"},
      {"dialsList"},
      {"dialsTreePath"},
      {"dialsDefinitions"},
      {"tableConfig"},
      {FieldFlag::DEPRECATED, "parameterLimits", "\"parameterLimits\" should be set in the parameter definition section. Not the dial definition. Support has been removed."},
    });
  config_.checkConfiguration();
}
void DialCollection::configureImpl() {
  prepareConfig(_config_);

  _config_.fillValue(_dataSetNameList_, "applyOnDataSets");
  _config_.fillValue(_isEnabled_, "isEnabled");
  if( not _isEnabled_ ){ return; }

  // Dials are directly defined with a binning file?
  if     ( initializeNormDialsWithParBinning() ){}
  else if( initializeDialsWithDefinition() ){}
  else {
    LogAlert << std::endl << "No valid definition for DialCollection. Disabling." << std::endl;
    _isEnabled_ = false;
  }

  // if "dialInputList" is not present, it will be setup during the initialize sequence.
  if( _config_.hasField("dialInputList") ){
    auto dialInputList = _config_.fetchValue<ConfigReader>("dialInputList");

    LogThrowIf(_supervisedParameterSetIndex_ == -1, "Can't initialize dialInputList without setting _supervisedParameterSetIndex_");

    _dialInputBufferList_.emplace_back();
    _dialInputBufferList_.back().setParSetRef( _parameterSetListPtr_ );

    // add the ref of each selected parameter
    for( auto& dialInput : dialInputList.getConfig() ){
      DialInputBuffer::ParameterReference p;
      p.parSetIndex = _supervisedParameterSetIndex_;

      if( GenericToolbox::Json::doKeyExist(dialInput, "name") ){
        auto parName = GenericToolbox::Json::fetchValue<std::string>(dialInput, "name");
        auto* parPtr{_parameterSetListPtr_->at( _supervisedParameterSetIndex_ ).getParameterPtr( parName )};
        LogThrowIf(parPtr == nullptr, "Could not find parameter: " << parName);
        p.parIndex = parPtr->getParameterIndex();
      }

      _dialInputBufferList_.back().addParameterReference( p );
    }

  }

}
void DialCollection::initializeImpl() {
  LogThrowIf(_index_==-1, "Index not set.");
  this->setupDialInterfaceReferences();
}

// non-trivial getters
bool DialCollection::isDatasetValid(const std::string& datasetName_) const{
  if( _dataSetNameList_.empty() ){ return true; }

  auto datasetNameLowerCase = GenericToolbox::toLowerCase(datasetName_);
  bool matching = std::any_of( _dataSetNameList_.begin(), _dataSetNameList_.end(), [&](const std::string& datasetPattern){
    return GenericToolbox::isMatching(datasetNameLowerCase, GenericToolbox::toLowerCase(datasetPattern));
  });
  return matching;
}

std::string DialCollection::getTitle() const {

  auto* parPtr{this->getSupervisedParameter()};
  if( parPtr != nullptr ){ return parPtr->getFullTitle(); }

  auto* parSetPtr{this->getSupervisedParameterSet()};
  if( parSetPtr != nullptr ){ return parSetPtr->getName(); }

  return {"UnsetParameterSuperVision"};
}
std::string DialCollection::getSummary(bool shallow_) const{
  std::stringstream ss;
  ss << "DialCollection: ";
  ss << this->getTitle();
  if( _dialType_ != DialType::Unset ){ ss << " / " << _dialType_; }
  if( not _dialOptions_.empty() ){ ss << ":\"" << _dialOptions_ << "\""; }
  if( not _dialLeafName_.empty() ){ ss << " / dialLeafName:" << _dialLeafName_; }
  if( _definitionRange_.hasBound() ){ ss << " / definitionRange:" << _definitionRange_; }
  if( _mirrorDefinitionRange_.hasBound() ){ ss << " / mirrorDefinitionRange:" << _mirrorDefinitionRange_; }

  if( not shallow_ ){
    // print parameters
    for( auto& dialInterface : _dialInterfaceList_ ){
      if( not isEventByEvent() ){
        ss << std::endl << "  " << dialInterface.getSummary(shallow_);
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
  _dialInterfaceList_.shrink_to_fit();
  _dialFreeSlot_.setValue(0);
}

void DialCollection::resizeContainers(){
  LogDebugIf(GundamGlobals::isDebug()) << "Resizing containers of the dial collection \"" << this->getTitle() << "\" from "
                                       << _dialInterfaceList_.size() << " to " << _dialFreeSlot_.getValue() << std::endl;

  _dialInterfaceList_.resize(_dialFreeSlot_.getValue());
  _dialInterfaceList_.shrink_to_fit();
  this->setupDialInterfaceReferences();
}

void DialCollection::updateInputBuffers(){
  for( auto& inputBuffer : _dialInputBufferList_ ){ inputBuffer.update(); }
}

void DialCollection::setupDialInterfaceReferences(){
  LogThrowIf(_supervisedParameterSetIndex_==-1, "par set index not set.");
  LogThrowIf(_supervisedParameterSetIndex_>_parameterSetListPtr_->size(), "invalid selected parset index: " << _supervisedParameterSetIndex_);

  // set it up is not already done
  if( _dialInputBufferList_.empty() ){
    if( _supervisedParameterIndex_ == -1 ){
      // one dial interface per parameter
      if (_dialInterfaceList_.size() != _parameterSetListPtr_->at(_supervisedParameterSetIndex_).getParameterList().size()) {
        LogError << "Nb of dial base don't match the number of parameters of the selected set: nDials="
                 << _dialInterfaceList_.size() << " != " << "nPars="
                 << _parameterSetListPtr_->at(_supervisedParameterSetIndex_).getParameterList().size()
                 << std::endl << "is the defined dial binning matching the number of parameters?" << std::endl;
        LogExit("Bad dial definition");
      }

      _dialInputBufferList_.resize(_dialInterfaceList_.size());
      for( int iDial = 0 ; iDial < int(_dialInterfaceList_.size()) ; iDial++ ){
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
  for( size_t iDial = 0 ; iDial < _dialInterfaceList_.size() ; iDial++ ){
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

// init protected
void DialCollection::readParametersFromConfig(const ConfigReader &config_) {
  // globals for the dialSet
  config_.fillValue(_enableDialsSummary_, "printDialSummary");
  config_.fillValue(_dialType_, "dialType");
  config_.fillValue(_dialOptions_, "options");
  config_.fillValue(_dialLeafName_, "treeExpression");
  config_.fillValue(_minDialResponse_, "minDialResponse");
  config_.fillValue(_maxDialResponse_, "maxDialResponse");
  config_.fillValue(_useMirrorDial_, "useMirrorDial");
  config_.fillValue(_mirrorLowEdge_, "mirrorLowEdge");
  config_.fillValue(_mirrorHighEdge_, "mirrorHighEdge");
  config_.fillValue(_allowDialExtrapolation_, "allowDialExtrapolation");
  config_.fillValue(_applyConditionStr_, "applyCondition");
  config_.fillValue(_definitionRange_, "definitionRange");
  config_.fillValue(_mirrorDefinitionRange_, "mirrorDefinitionRange");

  if( config_.hasField("applyConditions") ){
    std::vector<std::string> conditionsList;

    for (auto &condEntry: config_.loop("applyConditions")) {
      condEntry.defineFields({
        {"exp", {"expression", "var", "variable"}},
        {"allowedRanges"},
        {"allowedValues"},
        {"excludedRanges"},
        {"excludedValues"},
      });

      if (condEntry.getConfig().is_string()) {
        conditionsList.emplace_back(condEntry.getConfig().get<std::string>());
      }
      else if (condEntry.getConfig().is_structured()) {
        auto expression = condEntry.fetchValue<std::string>("exp");
        std::stringstream ssCondEntry;

        // allowedRanges
        {
          std::vector<GenericToolbox::Range> allowedRanges;
          condEntry.fillValue(allowedRanges, "allowedRanges");
          if (not allowedRanges.empty()) {
            std::vector<std::string> allowedRangesCond;
            for (auto &allowedRange: allowedRanges) {
              LogThrowIf(allowedRange.min >= allowedRange.max,
                         "Invalid range bounds: min(" << allowedRange.min << ") max(" << allowedRange.max << ")")
              std::stringstream condSs;
              condSs << "(" << expression << " >= " << allowedRange.min;
              condSs << " && " << expression << " < " << allowedRange.max << ")";
              allowedRangesCond.emplace_back(condSs.str());
            }
            ssCondEntry << GenericToolbox::joinVectorString(allowedRangesCond, " || ");
          }
        }

        // allowedValues
        {
          std::vector<double> allowedValues{};
          condEntry.fillValue(allowedValues, "allowedValues");
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

        std::vector<GenericToolbox::Range> excludedRanges;
        std::vector<double> excludedValues;
        condEntry.fillValue(excludedRanges, "excludedRanges");
        condEntry.fillValue(excludedValues, "excludedValues");
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
                LogThrowIf(excludedRange.min >= excludedRange.max,
                           "Invalid range bounds: min(" << excludedRange.min << ") max(" << excludedRange.max
                                                        << ")")
                std::stringstream condSs;
                condSs << expression << " < " << excludedRange.min << " && ";
                condSs << expression << " >= " << excludedRange.max;
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

  if (_useMirrorDial_) {
    _mirrorRange_ = _mirrorHighEdge_ - _mirrorLowEdge_;
    LogThrowIf(_mirrorRange_ < 0, GET_VAR_NAME_VALUE(_mirrorHighEdge_) << " < " << GET_VAR_NAME_VALUE(_mirrorLowEdge_))
  }

  // safety: many users ask why do my parameter limits aren't taken into account? -> because the parameters limits should be
  // handled in the parameter definition section
  if( config_.hasField("parameterLimits") ){
    LogError << "\"parameterLimits\" should be set in the parameter definition section. Not the dial definition." << std::endl;
    LogError << "Please move it to the appropriate section in: ";
    if( _parameterSetListPtr_ != nullptr and _supervisedParameterSetIndex_ != -1 ){

      if( _supervisedParameterIndex_ != -1 ){
        LogError << (*_parameterSetListPtr_)[_supervisedParameterSetIndex_].getParameterList()[_supervisedParameterIndex_].getFullTitle();
      }
      else{
        LogError << (*_parameterSetListPtr_)[_supervisedParameterSetIndex_].getName();
      }

    }
    else{
      LogError << "[handled parameter unspecified]";
    }

    LogError << std::endl;
    LogThrow("parameterLimits should be specified in the parameter definition.");
  }
}
bool DialCollection::initializeNormDialsWithParBinning() {

  if( not _config_.hasField("parametersBinningPath") ){
    return false; // not defined
  }

  // Get global parameters from the main config
  this->readParametersFromConfig(_config_);

  // Read the binning
  LogDebugIf(GundamGlobals::isDebug()) << "Defining binned dials for " << getTitle() << std::endl;
  _dialBinSet_ = BinSet();
  _dialBinSet_.setName("parameterBinning");
  _dialBinSet_.configure( _config_.fetchValue<ConfigReader>("parametersBinningPath") );

  // By default use min dial response for norm dials
  _dialResponseSupervisorList_.resize( 1 );
  _dialResponseSupervisorList_[0].setMinResponse( _config_.fetchValue("minDialResponse", double(0.)) );
  _dialResponseSupervisorList_[0].setMaxResponse( _config_.fetchValue("maxDialResponse", _maxDialResponse_) );

  _dialInterfaceList_.reserve( _dialBinSet_.getBinList().size() );
  for(const auto & bin : _dialBinSet_.getBinList()) {
    _dialInterfaceList_.emplace_back();
    _dialInterfaceList_.back().setDial( DialBaseObject(this->makeDial()) );
  }

  return true;
}

bool DialCollection::initializeDialsWithTabulation(const ConfigReader& dialsDefinition_) {
  // Initialize the Tabulated type.  That yaml for this is
  // tableConfig:
  //    - name: <table-name>                A table name passed to the library.
  //      libraryPath: <path-to-library>    Location of the library
  //      initFunction: <init-func-name>    Function called for initialization
  //      updateFunction: <update-func-name>    Function called to update table
  //      binningFunction: <bin-func-name>  Function to find bin index
  //      initArguments: [<arg1>, ...]      List of argument strings (e.g.
  //                                           input file names)
  //      bins:  <number>                   Number of bins in the table
  //      variables: [<var1>, <var2>, ... ] Variables used for binning the
  //                                           table "X" coordinate by the
  //                                           binning function.
  //
  //
  // If initFunction is provided it will be called with the signature:
  //
  //    extern "C"
  //    int initFunc(char* name, int argc, char* argv[], int bins)
  //
  //        name -- The name of the table
  //        argc -- number of arguments
  //        argv -- argument strings (0 is table name)
  //                arguments 1+ are defined by the library
  //        bins -- The size of the table.
  //
  // The function should return 0 for success, and any other value for failure
  //
  // The updateFunction signature is:
  //
  //    extern "C"
  //    int updateFunc(char* name,
  //                 double table[], int bins,
  //                 double par[], int npar)
  //
  //        name  -- table name
  //        table -- address of the table to update
  //        bins  -- The size of the table
  //        par   -- The parameters.  Must match parameters
  //                   define in the dial definition.
  //        npar  -- number of parameters
  //
  // The function should return 0 for success, and any other value for failure
  //
  // The table will be filled with "bins" values calculated with uniform
  // spacing between "low" and "high".  If bins is one, there must be one
  // value calculated for "low", if bins is two or more, then the first point
  // is located at "low", and the last point is located at "high".  The step
  // between the bins is (high-low)/(bins-1).  Examples:
  //
  //    bins = 2, low = 1.0, high = 6.0
  //       values calculated at 1.0 and 6.0
  //
  //    bins = 3, low = 1.0, high = 6.0
  //       values calculated at 1.0, 3.5, and 6.0
  //
  //    extern "C"
  //    double binFunc(char* name, int nvar, double varv[], int bins);
  //        name -- table name
  //        nvar -- number of (truth) variables used to find bin
  //        varv -- array of (truth) variables used to find bin
  //        bins -- The number of bins in the table.
  //
  // The function should return a double giving the fractional bin number
  // greater or equal to zero and LESS THAN the maximum number of bins.  The
  // integer part determines the index of the value below the value to be
  // interpolated, and the fractional part determines the interpolation
  // between the indexed bin, and the next.
  //
  // The code should be compiled with
  // gcc -fPIC -rdynamic --shared -o libLibraryName.so

  // Create a unique copy of this dial data so that it gets deleted if
  // there is a problem during initialization.
  std::unique_ptr<TabulatedDialFactory> tabulated
      = std::make_unique<TabulatedDialFactory>(dialsDefinition_);

  // Save the new object (the move releases the pointer).
  _dialCollectionData_.emplace_back(std::move(tabulated));

  // Get the index of the new dial collection data entry.  This is "back()",
  // but the index will be needed for the update closure, so use that instead.
  int index = _dialCollectionData_.size()-1;

  for (const std::string& var :
       getCollectionData<TabulatedDialFactory>(index)->getBinningVariables()) {
    addExtraLeafName(var);
  }

  addUpdate(
      [index](DialCollection* dc){
        dc->getCollectionData<TabulatedDialFactory>(index)
        ->updateTable(dc->getDialInputBufferList().front());
      });

  return true;
}
bool DialCollection::initializeDialsWithKriging(const ConfigReader& dialsDefinition_){
  // Initialize the Tabulated type.  That yaml for this is
  // tableConfig:
  //    - name: <table-name>                A table name passed to the library.
  //      libraryPath: <path-to-library>    Location of the library
  //      initFunction: <init-func-name>    Function called for initialization
  //      updateFunction: <update-func-name>    Function called to update table
  //      binningFunction: <bin-func-name>  Function to find bin index
  //      initArguments: [<arg1>, ...]      List of argument strings (e.g.
  //                                           input file names)
  //      bins:  <number>                   Number of bins in the table
  //      variables: [<var1>, <var2>, ... ] Variables used for binning the
  //                                           table "X" coordinate by the
  //                                           binning function.
  //
  //
  // If initFunction is provided it will be called with the signature:
  //
  //    extern "C"
  //    int initFunc(char* name, int argc, char* argv[], int bins)
  //
  //        name -- The name of the table
  //        argc -- number of arguments
  //        argv -- argument strings (0 is table name)
  //                arguments 1+ are defined by the library
  //        bins -- The size of the table.
  //
  // The function should return 0 for success, and any other value for failure
  //
  // The updateFunction signature is:
  //
  //    extern "C"
  //    int updateFunc(char* name,
  //                 double table[], int bins,
  //                 double par[], int npar)
  //
  //        name  -- table name
  //        table -- address of the table to update
  //        bins  -- The size of the table
  //        par   -- The parameters.  Must match parameters
  //                   define in the dial definition.
  //        npar  -- number of parameters
  //
  // The function should return 0 for success, and any other value for failure
  //
  // The table will be filled with "bins" values calculated with uniform
  // spacing between "low" and "high".  If bins is one, there must be one
  // value calculated for "low", if bins is two or more, then the first point
  // is located at "low", and the last point is located at "high".  The step
  // between the bins is (high-low)/(bins-1).  Examples:
  //
  //    bins = 2, low = 1.0, high = 6.0
  //       values calculated at 1.0 and 6.0
  //
  //    bins = 3, low = 1.0, high = 6.0
  //       values calculated at 1.0, 3.5, and 6.0
  //
  //    extern "C"
  //    double weightFunc(char* name, int bins,
  //                      int nvar, double varv[],
  //                      int maxEntries,
  //                      int indices[], double weights[]);
  //        name -- table name
  //        bins -- The number of bins in the table.
  //        nvar -- number of (truth) variables used to find bin
  //        varv -- array of (truth) variables used to find bin
  //        maxEntries -- size of the indices and weights arrays
  //        indices -- array of returned indices
  //        weights -- array of returned weights.
  //
  // The function should fill the indices and weights arrays with the
  // kriging weights.  The indices should always be inside the table since
  // the dial won't do bounds checking (for efficiency).
  //
  // The code should be compiled with
  // gcc -fPIC -rdynamic --shared -o libLibraryName.so

  // Create a unique copy of this dial data so that it gets deleted if
  // there is a problem during initialization.
  std::unique_ptr<KrigedDialFactory> kriged
      = std::make_unique<KrigedDialFactory>(dialsDefinition_);

  // Save the new object (the move releases the pointer).
  _dialCollectionData_.emplace_back(std::move(kriged));

  // Get the index of the new dial collection data entry.  This is "back()",
  // but the index will be needed for the update closure, so use that instead.
  int index = _dialCollectionData_.size()-1;

  for (const std::string& var :
       getCollectionData<KrigedDialFactory>(index)->getWeightVariables()) {
    addExtraLeafName(var);
  }

  addUpdate(
      [index](DialCollection* dc){
        dc->getCollectionData<KrigedDialFactory>(index)
        ->updateTable(dc->getDialInputBufferList().front());
      });

  LogDebug << "Initialize dial with kriging" << std::endl;

  return true;
}

bool DialCollection::initializeDialsWithBinningFile(const ConfigReader& dialsDefinition) {
  if( not dialsDefinition.hasField("binning") ){ return false; }

  // A binning file has been provided, so this is a binned dial.  Create
  // the dials for each bin here.  The dials will be assigned to the
  // events in DataDispenser.
  auto binningFilePath = dialsDefinition.fetchValue("binning", ConfigReader());

  LogInfo << "Defining binned dials for " << getTitle() << std::endl;
  _dialBinSet_ = BinSet();
  _dialBinSet_.setName("dial binning");
  _dialBinSet_.configure(binningFilePath);
  // NOTE: DON'T SORT THE DIALS AS THE ORDERING IS MATCHING THE SPLINE FILE!

  // Get the filename for a file with the object array of dials (graphs)
  // that will be applied based on the binning.
  auto filePath = dialsDefinition.fetchValue<std::string>("dialsFilePath");
  filePath = GenericToolbox::expandEnvironmentVariables(filePath);

  LogThrowIf(not GenericToolbox::doesTFileIsValid(filePath), "Could not open: " << filePath);
  std::unique_ptr<TFile> dialsTFile{TFile::Open(filePath.c_str())};
  LogThrowIf(dialsTFile==nullptr, "Could not open: " << filePath);

  if      ( dialsDefinition.hasField("dialsList") ) {
    auto* dialsList = dialsTFile->Get<TObjArray>(dialsDefinition.fetchValue<std::string>("dialsList").c_str());

    LogThrowIf(
        dialsList==nullptr,
        "Could not find dialsList: " << dialsDefinition.fetchValue<std::string>("dialsList")
    );
    LogThrowIf(
        dialsList->GetEntries() != _dialBinSet_.getBinList().size(),
        this->getTitle() << ": Number of dials (" << dialsList->GetEntries()
                         << ") don't match the number of bins " << _dialBinSet_.getBinList().size()
    );

    std::vector<int> excludedBins{};
    int nBins(static_cast<int>(_dialBinSet_.getBinList().size()));
    _dialInterfaceList_.reserve( nBins ); // at most
    for( int iBin = 0 ; iBin < nBins ; iBin++ ){
      TObject* binnedInitializer = dialsList->At(iBin);

      _verboseShortCircuit_ = true;
      auto dial = DialBaseObject( this->makeDial( dialsList->At(iBin) ) );
      _verboseShortCircuit_ = false;

      if( dial.get() == nullptr ) {
        LogDebugIf(GundamGlobals::isDebug()) << getTitle() << " -> "
                 << _dialBinSet_.getBinList()[iBin].getSummary()
                 << " -> " << _verboseShortCircuitStr_ << std::endl;
        excludedBins.emplace_back(iBin);
        continue;
      }

      _dialInterfaceList_.emplace_back();
      _dialInterfaceList_.back().setDial( dial );
    }

      if( not excludedBins.empty() ){
      LogWarning << "Removing " << excludedBins.size() << " null dials out of " << nBins << " / " << 100*double(excludedBins.size())/double(nBins) << "%... (--debug for more info)" << std::endl;
      for( int iBin = nBins ; iBin >= 0 ; iBin-- ){
        if( GenericToolbox::doesElementIsInVector(iBin, excludedBins) ){
          _dialBinSet_.getBinList().erase(_dialBinSet_.getBinList().begin() + iBin);
        }
      }
    }

  }

    ///////////////////////////////////////////////////////////////////////
  else if ( dialsDefinition.hasField("dialsTreePath") ) {
    // Deprecated: A tree with event binning has been provided, so this is
    // a binned dial.  Create the dials for each bin here.  The dials will
    // be assigned to the events in DataDispenser.
    auto objPath = dialsDefinition.fetchValue<std::string>("dialsTreePath");
    auto* dialsTTree = (TTree*) dialsTFile->Get(objPath.c_str());
    LogThrowIf(dialsTTree== nullptr, objPath << " within " << filePath << " could not be opened.")

    Int_t kinematicBin;
    TSpline3* splinePtr = nullptr;
    TGraph* graphPtr = nullptr;

    // searching for additional split var
    std::vector<std::string> splitVarNameList;
    for( int iKey = 0 ; iKey < dialsTTree->GetListOfLeaves()->GetEntries() ; iKey++ ){
      std::string leafName = dialsTTree->GetListOfLeaves()->At(iKey)->GetName();
      if(leafName != "kinematicBin" and leafName != "Spline" and leafName != "Graph"){
        splitVarNameList.emplace_back(leafName);
      }
    }

    // Hooking to the tree
    std::vector<Int_t> splitVarValueList(splitVarNameList.size(), 0);
    std::vector<std::pair<int, int>> splitVarBoundariesList(splitVarNameList.size(), std::pair<int, int>());
    std::vector<std::vector<int>> splitVarValuesList(splitVarNameList.size(), std::vector<int>());
    dialsTTree->SetBranchAddress("kinematicBin", &kinematicBin);
    if( _dialType_ == DialType::Spline ) dialsTTree->SetBranchAddress("Spline", &splinePtr);
    if( _dialType_ == DialType::Graph ) dialsTTree->SetBranchAddress("Graph", &graphPtr);
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
      if( _dialType_ == DialType::Spline ){ dialInitializer = splinePtr;}
      if( _dialType_ == DialType::Graph ){ dialInitializer = graphPtr;}

      auto dial = makeDial(dialInitializer);
      if( dial != nullptr ){
        _dialInterfaceList_.emplace_back();
        DialBaseObject obj;
        obj.dialPtr = std::move(dial);
        _dialInterfaceList_.back().setDial( obj );
      }
    } // iSpline (in TTree)
  } // Splines in TTree
  else{
    LogError << "Neither dialsTreePath nor dialsList are provided..." << std::endl;
    return false;
  }
  return true;
}

bool DialCollection::initializeDialsWithDefinition() {

  auto dialsDefinition = _config_;
  if( _config_.hasField("dialsDefinitions") ) {
    // Fetch the dialSet corresponding to the selected parameter
    dialsDefinition = this->fetchDialsDefinition(_config_.fetchValue<ConfigReader>("dialsDefinitions"));
    prepareConfig(dialsDefinition);
  }

  if( dialsDefinition.getConfig().empty() ) {return false;}

  if( not dialsDefinition.fetchValue("isEnabled", true) ){
    LogDebug << "DialSet is disabled." << std::endl;
    return true;
  }

  this->readParametersFromConfig( ConfigReader(dialsDefinition) );

  if     ( _dialType_ == DialType::Norm ) {
    // This dial collection is a normalization, so there is a single dial.
    // Create it here.
    _isEventByEvent_ = false;
    _dialInterfaceList_.emplace_back();
    _dialInterfaceList_.back().setDial( DialBaseObject(makeDial()) );
  }
  else if( _dialType_ == DialType::Formula ){
    // This dial collection calculates a function of the parameter values, so it
    // is a single dial for all events.  Create the dial here.
    _isEventByEvent_ = false;

    if( dialsDefinition.hasField("binning") ){
      auto binning = dialsDefinition.fetchValue("binning", ConfigReader());

      LogInfo << "Defining binned dials for " << getTitle() << std::endl;
      _dialBinSet_ = BinSet();
      _dialBinSet_.setName( "formula binning" );
      _dialBinSet_.configure( ConfigReader(binning) );

      _dialInterfaceList_.reserve( _dialBinSet_.getBinList().size() );
      for( auto& bin : _dialBinSet_.getBinList() ){

        _dialInterfaceList_.emplace_back();
        _dialInterfaceList_.back().setDial( DialBaseObject(makeDial( dialsDefinition )) );

        for( auto& var : bin.getEdgesList() ){
          ((RootFormula*) _dialInterfaceList_.back().getDialBaseRef())->getFormula().SetParameter(
              var.varName.c_str(), var.getCenterValue()
          );
        }
      }

    }
    else{
      _dialInterfaceList_.emplace_back();
      _dialInterfaceList_.back().setDial( DialBaseObject(makeDial( dialsDefinition )) );
    }

  }
  else if( _dialType_ == DialType::CompiledLibDial ){
    // This dial collection calculates a function of the parameter values so it
    // is a single dial for all events.  Create the dial here.
    _isEventByEvent_ = false;

    _dialInterfaceList_.emplace_back();
    _dialInterfaceList_.back().setDial( DialBaseObject(makeDial( dialsDefinition )) );
  }
  else if( _dialType_ == DialType::Tabulated ) {
    // This dial uses a precalculated table to apply weight to each event
    // (e.g. it might be used to implement neutrino osillations).  It has a
    // different weight for each event.
    _isEventByEvent_ = true;
    LogThrowIf(not initializeDialsWithTabulation(dialsDefinition),
               "Error initializing dials with tabulation");
  }
  else if( _dialType_ == DialType::Kriged ) {
    // This dial uses a precalculated table to apply weight to each event
    // (e.g. it might be used to implement neutrino osillations).  It has a
    // different weight for each event.
    _isEventByEvent_ = true;
    LogThrowIf(not initializeDialsWithKriging(dialsDefinition),
               "Error initializing dials with kriging");
  }
  else if( dialsDefinition.hasField("binning") ) {
    // This dial collection is binned with different weights for each bin.
    // Create the dials here.
    _isEventByEvent_ = false;

      LogThrowIf(not initializeDialsWithBinningFile(dialsDefinition),
               "Error initializing dials with binning file");
    }
  else if (not _dialLeafName_.empty()) {
    // None of the other dial types are matched, and a dialLeafName field has
    // been provided, so this is an event by event dial with one TGraph (or
    // TSpline3) per event.  The generation of the dials will be handled in
    // DataDispenser.
    _isEventByEvent_ = true;
  }
  else{
    LogError << "The dial does not match a known dial type." << std::endl;
    LogError << "  dialType:     " << _dialType_ << std::endl;
    LogError << "  dialOptions:  " << _dialOptions_ << std::endl;
    LogError << "  dialLeafName: " << _dialLeafName_ << std::endl;
    LogThrow("Invalid dial type");
  }

  _dialResponseSupervisorList_.emplace_back();
  _dialResponseSupervisorList_.back().setMinResponse(
      _config_.fetchValue("minDialResponse", double(0.))
  );
  _dialResponseSupervisorList_.back().setMaxResponse(
      _config_.fetchValue("maxDialResponse", _maxDialResponse_)
  );

  return true;
}

ConfigReader DialCollection::fetchDialsDefinition(const ConfigReader& definitionsList_) const {
  auto* parSetPtr = this->getSupervisedParameterSet();
  LogThrowIf(parSetPtr == nullptr, "Can't fetch dial definition of parameter: par ref not set.");
  auto* par = &parSetPtr->getParameterList()[_supervisedParameterIndex_];
  for(size_t iDial = 0 ; iDial < definitionsList_.getConfig().size() ; iDial++ ){
    if( par->getName().empty() ){
      if( par->getParameterIndex() == iDial ){
        return ConfigReader(definitionsList_.getConfig().at(iDial));
      }
    }
    else if( par->getName() == GenericToolbox::Json::fetchValue<std::string>(definitionsList_.getConfig().at(iDial), {{"name"}, {"parameterName"}}, "") ){
      return ConfigReader(definitionsList_.getConfig().at(iDial));
    }
  }
  return {};
}

void DialCollection::update() {
  for( auto& func : _dialCollectionCallbacks_ ){
    func(this);
  }
}

void DialCollection::addUpdate(const std::function<void(DialCollection* dc)>& callback) {
  _dialCollectionCallbacks_.emplace_back(callback);
}

void DialCollection::printConfiguration() const {

  LogInfo << "DialCollection: " << this->getTitle() << std::endl;

}



std::unique_ptr<DialBase> DialCollection::makeDial() const{ return std::make_unique<Norm>(); }
std::unique_ptr<DialBase> DialCollection::makeDial(const TObject* src_) const {
  std::unique_ptr<DialBase> out{nullptr};

  // always returns an invalid ptr if
  if( src_ == nullptr ){ return out; }

  if     ( _dialType_ == DialType::Graph   ){ out = makeGraphDial(src_); }
  else if( _dialType_ == DialType::Spline  ){ out = makeSplineDial(src_); }
  else if( _dialType_ == DialType::Surface ){ out = makeSurfaceDial(src_); }
  else{
    LogError << this->getSummary() << std::endl;
    LogThrow(this->getTitle() << ": invalid dial type to init with TObject: " << _dialType_);
  }

  if( out != nullptr ){
    out->setAllowExtrapolation( _allowDialExtrapolation_ );
  }

  return out;
}
std::unique_ptr<DialBase> DialCollection::makeDial(const ConfigReader& config_) const{
  std::unique_ptr<DialBase> dialBase{nullptr};
  std::string dialType{};

  config_.fillValue(dialType, "dialType");

  if( dialType == "Formula" or dialType == "RootFormula" ){
    dialBase = std::make_unique<RootFormula>();
    auto* rootFormulaPtr{(RootFormula*) dialBase.get()};
    auto formulaConfig = config_.fetchValue("dialConfig", JsonType());
    rootFormulaPtr->setFormulaStr( GenericToolbox::Json::fetchValue<std::string>(formulaConfig, "formulaStr") );
  }
  else if( dialType == "CompiledLibDial" ){
    dialBase = std::make_unique<CompiledLibDial>();
    auto* compiledLibDialPtr{(CompiledLibDial*) dialBase.get()};

    auto formulaConfig(config_.fetchValue<JsonType>("dialConfig"));

    bool success = compiledLibDialPtr->loadLibrary( GenericToolbox::Json::fetchValue<std::string>(formulaConfig, "libraryFile") );
    if( not success ){
      LogThrow("Could not load CompiledLibDial. " << GenericToolbox::Json::fetchValue(formulaConfig, "messageOnError", std::string("")) );
    }
  }
  else{ LogThrow("Unknown dial type: " << dialType); }

  return dialBase;
}

std::unique_ptr<DialBase> DialCollection::makeGraphDial(const TObject* src_) const {
  // always returns an invalid ptr if
  if( src_ == nullptr ){ return nullptr; }

  std::vector<DialUtils::DialPoint> splinePointList{};
  if( src_->InheritsFrom(TGraph::Class()) ){
    splinePointList = DialUtils::getPointListNoSlope((TGraph *)src_);
  }
  else{
    // try something else (from TSpline3 for instance):
    splinePointList = DialUtils::getPointList(src_);
  }

  checkDialPointList(splinePointList);

  std::unique_ptr<DialBase> out{nullptr};
  if( makeDialShortCircuit(splinePointList, out) ){ return out; }

  return makeGraphDial(splinePointList);
}
std::unique_ptr<DialBase> DialCollection::makeSplineDial(const TObject* src_) const{
  // always returns an invalid ptr if
  if( src_ == nullptr ){ return nullptr; }

  // The types of cubic splines are "not-a-knot", "natural", "catmull-rom",
  // and "ROOT".  The "not-a-knot" spline will give the same curve as ROOT
  // (and might be implemented with at TSpline3).  The "ROOT" spline will use
  // an actual TSpline3 (and is slow).  The "natural" and "catmull-rom"
  // splines are just as expected (you can see the underlying math on
  // Wikipedia or another source).  Be careful about the order since later
  // conditionals can override earlier ones.
  std::string splType = "not-a-knot";  // The default.
  if( _dialOptions_.find("akima") != std::string::npos ) splType = "akima";
  if( _dialOptions_.find("catmull") != std::string::npos) splType = "catmull-rom";
  if( _dialOptions_.find("natural") != std::string::npos) splType = "natural";
  if( _dialOptions_.find("not-a-knot") != std::string::npos) splType = "not-a-knot";
  if( _dialOptions_.find("pixar") != std::string::npos) {
    splType = "catmull-rom";
    // sneaky output... logger would tattle on me.
    static bool woody=true;
    if (woody) std::cout << std::endl << std::endl << "You got a friend in me!" << std::endl;
    woody=false;
  }
  if( _dialOptions_.find("ROOT") != std::string::npos) splType = "ROOT";

  bool isMonotonic = ( _dialOptions_.find("monotonic") != std::string::npos );

  // Get the numeric tolerance for when a uniform spline can be used.  We
  // should be able to set this in the DialSubType.
  const double defUniformityTolerance{16*std::numeric_limits<float>::epsilon()};
  double uniformityTolerance{defUniformityTolerance};
  if (_dialOptions_.find("uniformity(") != std::string::npos) {
    std::size_t bg = _dialOptions_.find("uniformity(");
    bg = _dialOptions_.find('(',bg);
    std::size_t en = _dialOptions_.find(')',bg);
    LogThrowIf(en == std::string::npos,
               "Invalid spline uniformity with dialSubType: " << _dialOptions_
               << " dial: " << getTitle());
    en = en - bg;
    std::string uniformityString = _dialOptions_.substr(bg+1,en-1);
    std::istringstream unif(uniformityString);
    unif >> uniformityTolerance;
  }

  auto splinePointList = DialUtils::getPointList(src_);
  checkDialPointList(splinePointList);

  std::unique_ptr<DialBase> out;

  if( makeDialShortCircuit(splinePointList, out) ) {
    // dial short circuit has been processed
    return out;
  }

  // If there are only two points, then force a catmull-rom.  This could be
  // handled using a graph, but Catmull-Rom is fast, and works better with the
  // GPU.  The isMonotonic is forced to false so that this uses CompactSpline
  // instead of MonotonicSpline.
  if( splinePointList.size() <= 2 ){ splType = "catmull-rom"; isMonotonic = false; }

  ////////////////////////////////////////////////////////////////
  // Check if the spline slope calculation should be updated.  The slopes for
  // not-a-knot and natural splines are calculated by FillFromGraph and
  // FillFromSpoline using ROOT code.  That means we need to fill in the
  // slopes for the other types ("catmull-rom", "akima")
  if     ( splType == "catmull-rom" ){ DialUtils::fillCatmullRomSlopes(splinePointList); }
  else if( splType == "akima" ){ DialUtils::fillAkimaSlopes(splinePointList); }

  ////////////////////////////////////////////////////////////////
  // Check if the spline is supposed to be monotonic and condition the slopes
  // if necessary.  This is ignored by "ROOT" splines.  The Catmull-Rom
  // splines have a special implementation for monotonic splines, so save a
  // flag that can be checked later.
  ////////////////////////////////////////////////////////////////
  if( isMonotonic ){ DialUtils::applyMonotonicCondition(splinePointList); }

  ///////////////////////////////////////////////////////////
  // Create the right kind low level spline class base on all of the previous
  // queries.  This is pushing the if-elseif-elseif limit so watch for when it
  // should change to the do-while-false idiom.  Make sure the individual
  // conditionals are less than 10 lines.
  ///////////////////////////////////////////////////////////
  if      ( splType == "ROOT" ){
    // The ROOT implementation of the spline has been explicitly requested, so
    // use it.
    auto rootSpline = std::make_unique<RootSpline>();
    rootSpline->buildDial(splinePointList);
    out = std::move(rootSpline);
  }
  else if( splType == "catmull-rom" ){
    // Catmull-Rom is handled as a special case because it ignores the slopes,
    // and has an explicit monotonic implementatino.  It also must have
    // uniformly spaced knots.
    if( not DialUtils::isUniform(splinePointList, uniformityTolerance) ){
      LogError << "Catmull-rom splines need a uniformly spaced points"
               << " Dial: " << getTitle()
               << std::endl;
      double step = (splinePointList.back().x-splinePointList.front().x)/(static_cast<double>(splinePointList.size())-1.);
      for (int i = 0; i<splinePointList.size()-1; ++i) {
        LogError << i << " --  X: " << splinePointList[i].x
                 << " X+1: " << splinePointList[i+1].x
                 << " step: " << step
                 << " error: " << splinePointList[i+1].x - splinePointList[i].x - step
                 << std::endl;
      }
      // If the user specified a tolerance then crash, otherwise trust the
      // user knows that it's not uniform and continue.
      LogThrowIf(uniformityTolerance != defUniformityTolerance, "Invalid catmull-rom inputs -- Nonuniform spacing");
    }

    if( isMonotonic ) {
      auto monotonicSpline = std::make_unique<MonotonicSpline>();
      monotonicSpline->buildDial(splinePointList);
      out = std::move(monotonicSpline);
    }
    else {
      auto compactSpline = std::make_unique<CompactSpline>();
      compactSpline->buildDial(splinePointList);
      out = std::move(compactSpline);
    }

  }
  else if( DialUtils::isUniform(splinePointList, uniformityTolerance) ){
    // Haven't matched a specific special case, but we have uniformly spaced
    // knots so we can use the faster UniformSpline implementation.
    auto uniformSpline = std::make_unique<UniformSpline>();
    uniformSpline->buildDial(splinePointList);
    out = std::move(uniformSpline);
  }
  else {
    // Haven't matched a specific special case, and the knots are not
    // uniformly spaced, so we have to use the GeneralSpline implemenatation
    // which can handle any kind of cubic spline.
    auto generalSpline = std::make_unique<GeneralSpline>();
    generalSpline->buildDial(splinePointList);
    out = std::move(generalSpline);
  }

  // Pass the ownership without any constraints!
  return out;

}
std::unique_ptr<DialBase> DialCollection::makeSurfaceDial(const TObject* src_) const{
  // always returns an invalid ptr if
  if( src_ == nullptr ){ return nullptr; }

  auto* srcObject = dynamic_cast<const TH2*>(src_);

  LogThrowIf(srcObject == nullptr, "Surface dial initializers must be a TH2");

  // Stuff the created dial into a unique_ptr, so it will be properly deleted
  // in the event of an exception.
  std::unique_ptr<DialBase> dialBase;

  if (_dialOptions_ == "Bilinear") {
    // Basic coding: Give a hint to the reader and put likely branch "first".
    // Do we really expect the cached version more than the uncached?
    auto bilinear = std::make_unique<Bilinear>();
    bilinear->buildDial(*srcObject);
    dialBase = std::move(bilinear);
  }
  else if (_dialOptions_ == "Bicubic") {
    auto bicubic = std::make_unique<Bicubic>();
    bicubic->buildDial(*srcObject);
    dialBase = std::move(bicubic);
  }

  if (not dialBase) {
    LogError << "Invalid dialSubType value: " << _dialOptions_ << std::endl;
    LogError << "Valid dialSubType values are: Bilinear, Bicubic" << std::endl;
    LogThrow("Invalid Surface dialSubType");
  }

  // Pass the ownership without any constraints!
  return dialBase;
}

void DialCollection::checkDialPointList(std::vector<DialUtils::DialPoint>& pointList_) const{

  if( _definitionRange_.hasBound() and not std::all_of(pointList_.begin(), pointList_.end(), [&](DialUtils::DialPoint dialPoint){ return _definitionRange_.isInBounds(dialPoint.x); }) ){
    auto temp{pointList_}; temp.clear();
    for( auto& point : pointList_ ) {
      if( not _definitionRange_.isInBounds(point.x) ){ continue; }
      temp.emplace_back(point);
    }
    pointList_ = std::move(temp);
  }

  std::sort(pointList_.begin(), pointList_.end(),
    [](const DialUtils::DialPoint& a_, const DialUtils::DialPoint& b_){
      return a_.x < b_.x; // a goes first?
    });

  if( _mirrorDefinitionRange_.hasBound() ){
    // TODO: process mirroring options
    LogExit("_mirrorDefinitionRange_ not implemented yet.");
  }

}
bool DialCollection::makeDialShortCircuit(const std::vector<DialUtils::DialPoint>& pointList_, std::unique_ptr<DialBase>& dial_) const{
  dial_ = nullptr;

  if( pointList_.empty() ){ return true; }

  if( pointList_.size() == 1 or DialUtils::isFlat(pointList_) ){
    // it's flat, let's shortcut

    if( std::abs( pointList_[0].y - 1.0 ) < 2 * std::numeric_limits<float>::epsilon() ){
      // one! no need for a dial
      if(_verboseShortCircuit_){ _verboseShortCircuitStr_ = "No dial: response is always 1."; }
      return true;
    }

    // Do the unique_ptr dance in case there are exceptions.
    auto dialBase = std::make_unique<Shift>();
    dialBase->setShiftValue(pointList_[0].y);
    dial_ = std::move(dialBase);
    if(_verboseShortCircuit_){ _verboseShortCircuitStr_ = "Shift dial: response is flat."; }
    return true;
  }

#define SHORT_CIRCUIT_SMALL_SPLINES
#ifdef  SHORT_CIRCUIT_SMALL_SPLINES
  if( pointList_.size() <= 2 ) {
    if(_verboseShortCircuit_){ _verboseShortCircuitStr_ = "Graph dial: 2 points dial."; }
    dial_ = makeGraphDial(pointList_);
    return true;
  }
#endif

  return false;
}
std::unique_ptr<DialBase> DialCollection::makeGraphDial(const std::vector<DialUtils::DialPoint>& pointList_) const{

  if( pointList_.empty() or pointList_.size() == 1 ){
    // for sure, it's a short circuit
    std::unique_ptr<DialBase> out{nullptr};
    makeDialShortCircuit(pointList_, out);
    return out;
  }

  auto dial = std::make_unique<Graph>();
  dial->buildDial(pointList_);
  return dial;
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
// End:
