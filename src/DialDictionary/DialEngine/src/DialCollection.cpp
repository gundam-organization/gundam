//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialCollection.h"

#include <SplineDialBaseFactory.h>

#include "TabulatedDialFactory.h"
#include "RootFormula.h"

#include "Shift.h"
#include "RootGraph.h"
#include "Graph.h"
#include "Norm.h"
#include "Spline.h"
#include "SimpleSpline.h"
#include "CompactSpline.h"
#include "UniformSpline.h"
#include "GeneralSpline.h"
#include "MonotonicSpline.h"
#include "MakeMonotonicSpline.h"
#include "Bilinear.h"
#include "Bicubic.h"
#include "CompiledLibDial.h"

#include "GundamGlobals.h"

#include "Logger.h"

#include <sstream>


void DialCollection::configureImpl() {

  _dataSetNameList_ = GenericToolbox::Json::fetchValue<std::vector<std::string>>(
      _config_, "applyOnDataSets", std::vector<std::string>());
  if( _dataSetNameList_.empty() ){ _dataSetNameList_.emplace_back("*"); }

  _isEnabled_ = GenericToolbox::Json::fetchValue(_config_, "isEnabled", _isEnabled_);
  if( not _isEnabled_ ){ return; }

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

    LogThrowIf(_supervisedParameterSetIndex_ == -1, "Can't initialize dialInputList without setting _supervisedParameterSetIndex_");

    _dialInputBufferList_.emplace_back();
    _dialInputBufferList_.back().setParSetRef( _parameterSetListPtr_ );

    // add the ref of each selected parameter
    for( auto& dialInput : dialInputList ){
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
  ss << " / nDialInterfaces=" << _dialInterfaceList_.size();
  ss << " / lastDialFreeSlot=" << getDialFreeSlotIndex();

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
      LogThrowIf(_dialInterfaceList_.size() != _parameterSetListPtr_->at(_supervisedParameterSetIndex_).getParameterList().size(),
                 "Nb of dial base don't match the number of parameters of the selected set: nDials="
                     << _dialInterfaceList_.size() << " != " << "nPars="
                     << _parameterSetListPtr_->at(_supervisedParameterSetIndex_).getParameterList().size()
                     << std::endl << "is the defined dial binning matching the number of parameters?"
      );
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
void DialCollection::readGlobals(const JsonType &config_) {
  // globals for the dialSet
  GenericToolbox::Json::fillValue(config_, _enableDialsSummary_, "printDialsSummary");
  GenericToolbox::Json::fillValue(config_, _globalDialType_, {{"dialsType"}, {"dialType"}});
  GenericToolbox::Json::fillValue(config_, _globalDialSubType_, "dialSubType");
  GenericToolbox::Json::fillValue(config_, _globalDialLeafName_, "dialLeafName");
  GenericToolbox::Json::fillValue(config_, _minDialResponse_, {{"minDialResponse"}, {"minimumSplineResponse"}});
  GenericToolbox::Json::fillValue(config_, _maxDialResponse_, "maxDialResponse");
  GenericToolbox::Json::fillValue(config_, _useMirrorDial_, "useMirrorDial");
  GenericToolbox::Json::fillValue(config_, _mirrorLowEdge_, "mirrorLowEdge");
  GenericToolbox::Json::fillValue(config_, _mirrorHighEdge_, "mirrorHighEdge");
  GenericToolbox::Json::fillValue(config_, _allowDialExtrapolation_, "allowDialExtrapolation");
  GenericToolbox::Json::fillValue(config_, _applyConditionStr_, "applyCondition");

  if( GenericToolbox::Json::doKeyExist(config_, "applyConditions") ){
    std::vector<std::string> conditionsList;

    for (auto &condEntry: GenericToolbox::Json::fetchValue<std::vector<JsonType>>(config_, "applyConditions")) {
      if (condEntry.is_string()) {
        conditionsList.emplace_back(condEntry.get<std::string>());
      }
      else if (condEntry.is_structured()) {
        auto expression = GenericToolbox::Json::fetchValue<std::string>(condEntry, {{"exp"},{"expression"},{"var"},{"variable"}});
        std::stringstream ssCondEntry;

        // allowedRanges
        {
          std::vector<GenericToolbox::Range> allowedRanges;
          GenericToolbox::Json::fillValue(condEntry, allowedRanges, "allowedRanges");
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

        std::vector<GenericToolbox::Range> excludedRanges;
        GenericToolbox::Json::fillValue(condEntry, excludedRanges, "excludedRanges");
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
  if( GenericToolbox::Json::doKeyExist(config_, "parameterLimits") ){
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
  auto binning = GenericToolbox::Json::fetchValue(_config_, "parametersBinningPath", JsonType());
  if( binning.empty() ){ return false; } // not defined

  // Get global parameters from the main config
  this->readGlobals(_config_);

  // Read the binning
  LogInfo << "Defining binned dials for " << getTitle() << std::endl;
  _dialBinSet_ = BinSet();
  _dialBinSet_.setName("parameterBinning");
  _dialBinSet_.configure( binning );

  // By default use min dial response for norm dials
  _dialResponseSupervisorList_.resize( 1 );
  _dialResponseSupervisorList_[0].setMinResponse( GenericToolbox::Json::fetchValue(_config_, {{"minDialResponse"}, {"minimumSplineResponse"}}, double(0.)) );
  _dialResponseSupervisorList_[0].setMaxResponse( GenericToolbox::Json::fetchValue(_config_, "maxDialResponse", _maxDialResponse_) );

  _dialInterfaceList_.reserve( _dialBinSet_.getBinList().size() );
  for(const auto & bin : _dialBinSet_.getBinList()) {
    _dialInterfaceList_.emplace_back();
    _dialInterfaceList_.back().setDial( DialBaseObj(this->makeDial()) );
  }

  return true;
}

bool DialCollection::initializeDialsWithTabulation(const JsonType& dialsDefinition_) {
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
  //    double binFunc(char* name, int nvar, double varv[]);
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

bool DialCollection::initializeDialsWithBinningFile(const JsonType& dialsDefinition) {
  if(not GenericToolbox::Json::doKeyExist(dialsDefinition, "binningFilePath") ) return false;
  
  // A binning file has been provided, so this is a binned dial.  Create
  // the dials for each bin here.  The dials will be assigned to the
  // events in DataDispenser.
  auto binningFilePath = GenericToolbox::Json::fetchValue(dialsDefinition, "binningFilePath", JsonType());

  LogInfo << "Defining binned dials for " << getTitle() << std::endl;
  _dialBinSet_ = BinSet();
  _dialBinSet_.setName(binningFilePath);
  _dialBinSet_.configure(binningFilePath);
  // NOTE: DON'T SORT THE DIALS AS THE ORDERING IS MATCHING THE SPLINE FILE!

  // Get the filename for a file with the object array of dials (graphs)
  // that will be applied based on the binning.
  auto filePath = GenericToolbox::Json::fetchValue<std::string>(dialsDefinition, "dialsFilePath");
  filePath = GenericToolbox::expandEnvironmentVariables(filePath);

  LogThrowIf(not GenericToolbox::doesTFileIsValid(filePath), "Could not open: " << filePath);
  std::unique_ptr<TFile> dialsTFile{TFile::Open(filePath.c_str())};
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
    int nBins(static_cast<int>(_dialBinSet_.getBinList().size()));
    _dialInterfaceList_.reserve( nBins ); // at most
    for( int iBin = 0 ; iBin < nBins ; iBin++ ){
      TObject* binnedInitializer = dialsList->At(iBin);

      auto dial = DialBaseObj( this->makeDial( dialsList->At(iBin) ) );
      if( dial.get() == nullptr ) {
        LogAlert << "Invalid dial for " << getTitle() << " -> "
                 << _dialBinSet_.getBinList()[iBin].getSummary()
                 << std::endl;
        excludedBins.emplace_back(iBin);
        continue;
      }

      _dialInterfaceList_.emplace_back();
      _dialInterfaceList_.back().setDial( dial );
    }

      if( not excludedBins.empty() ){
      LogInfo << "Removing invalid bin dials..." << std::endl;
      for( int iBin = nBins ; iBin >= 0 ; iBin-- ){
        if( GenericToolbox::doesElementIsInVector(iBin, excludedBins) ){
          _dialBinSet_.getBinList().erase(_dialBinSet_.getBinList().begin() + iBin);
        }
      }
    }

  }

    ///////////////////////////////////////////////////////////////////////
  else if ( GenericToolbox::Json::doKeyExist(dialsDefinition, "dialsTreePath") ) {
    // Deprecated: A tree with event binning has been provided, so this is
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
      if(leafName != "kinematicBin" and leafName != "Spline" and leafName != "Graph"){
        splitVarNameList.emplace_back(leafName);
      }
    }

    // Hooking to the tree
    std::vector<Int_t> splitVarValueList(splitVarNameList.size(), 0);
    std::vector<std::pair<int, int>> splitVarBoundariesList(splitVarNameList.size(), std::pair<int, int>());
    std::vector<std::vector<int>> splitVarValuesList(splitVarNameList.size(), std::vector<int>());
    dialsTTree->SetBranchAddress("kinematicBin", &kinematicBin);
    if( _globalDialType_ == "Spline" ) dialsTTree->SetBranchAddress("Spline", &splinePtr);
    if( _globalDialType_ == "Graph" ) dialsTTree->SetBranchAddress("Graph", &graphPtr);
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

      auto dial = makeDial(dialInitializer);
      if( dial != nullptr ){
        _dialInterfaceList_.emplace_back();
        DialBaseObj obj;
        obj.dialPtr = std::unique_ptr<DialBase>(dial.release());
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

  if     ( _globalDialType_ == "Norm" or _globalDialType_ == "Normalization" ) {
    // This dial collection is a normalization, so there is a single dial.
    // Create it here.
    _isEventByEvent_ = false;
    _dialInterfaceList_.emplace_back();
    _dialInterfaceList_.back().setDial( DialBaseObj(makeDial()) );
  }
  else if( _globalDialType_ == "Formula" or _globalDialType_ == "RootFormula" ){
    // This dial collection calculates a function of the parameter values, so it
    // is a single dial for all events.  Create the dial here.
    _isEventByEvent_ = false;

    if( GenericToolbox::Json::doKeyExist(dialsDefinition, "binning") ){
      auto binning = GenericToolbox::Json::fetchValue(dialsDefinition, "binning", JsonType());

      LogInfo << "Defining binned dials for " << getTitle() << std::endl;
      _dialBinSet_ = BinSet();
      _dialBinSet_.setName( "formula binning" );
      _dialBinSet_.configure(binning);

      _dialInterfaceList_.reserve( _dialBinSet_.getBinList().size() );
      for( auto& bin : _dialBinSet_.getBinList() ){

        _dialInterfaceList_.emplace_back();
        _dialInterfaceList_.back().setDial( DialBaseObj(makeDial( dialsDefinition )) );

        for( auto& var : bin.getEdgesList() ){
          ((RootFormula*) _dialInterfaceList_.back().getDialBaseRef())->getFormula().SetParameter(
              var.varName.c_str(), var.getCenterValue()
          );
        }
      }

    }
    else{
      _dialInterfaceList_.emplace_back();
      _dialInterfaceList_.back().setDial( DialBaseObj(makeDial( dialsDefinition )) );
    }

  }
  else if( _globalDialType_ == "CompiledLibDial" ){
    // This dial collection calculates a function of the parameter values so it
    // is a single dial for all events.  Create the dial here.
    _isEventByEvent_ = false;

    _dialInterfaceList_.emplace_back();
    _dialInterfaceList_.back().setDial( DialBaseObj(makeDial( dialsDefinition )) );
  }
  else if( _globalDialType_ == "Tabulated" ) {
    // This dial uses a precalculated table to apply weight to each event (e.g. it
    // might be used to implement neutrino osillations).  It has a different
    // weight for each event.
    _isEventByEvent_ = true;
    LogThrowIf(not initializeDialsWithTabulation(dialsDefinition),
               "Error initializing dials with tabulation");
  }
  else if( GenericToolbox::Json::doKeyExist(dialsDefinition, "binningFilePath") ) {
    // This dial collection is binned with different weights for each bin.
    // Create the dials here.
    _isEventByEvent_ = false;

      LogThrowIf(not initializeDialsWithBinningFile(dialsDefinition),
               "Error initializing dials with binning file");
    }
  else if (not _globalDialLeafName_.empty()) {
    // None of the other dial types are matched, and a dialLeafName field has
    // been provided, so this is an event by event dial with one TGraph (or
    // TSpline3) per event.  The generation of the dials will be handled in
    // DataDispenser.
    _isEventByEvent_ = true;
  }
  else{
    LogError << "The dial does not match a known dial type." << std::endl;
    LogError << "  dialType:     " << getGlobalDialType() << std::endl;
    LogError << "  dialSubType:  " << getGlobalDialSubType() << std::endl;
    LogError << "  dialLeafName: " << getGlobalDialLeafName() << std::endl;
    LogThrow("Invalid dial type");
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

void DialCollection::update() {
  for (std::function<void(DialCollection*)>& func : _dialCollectionCallbacks_) {
    func(this);
  }
}

void DialCollection::addUpdate(std::function<void(DialCollection*)> callback) {
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

  if     ( _globalDialType_ == "Graph"   ){ out = makeGraphDial(src_); }
  else if( _globalDialType_ == "Spline"  ){ out = makeSplineDial(src_); }
  else if( _globalDialType_ == "surface" ){ out = makeSurfaceDial(src_); }
  else{ LogThrow("Invalid dial type to init with TObject: " << _globalDialType_); }

  if( out != nullptr ){
    out->setAllowExtrapolation( _allowDialExtrapolation_ );
  }

  return out;
}
std::unique_ptr<DialBase> DialCollection::makeDial(const JsonType& config_) const{
  std::unique_ptr<DialBase> dialBase{nullptr};
  std::string dialType{};

  dialType = GenericToolbox::Json::fetchValue(config_, {{"dialType"}, {"dialsType"}}, dialType);

  if( dialType == "Formula" or dialType == "RootFormula" ){
    dialBase = std::make_unique<RootFormula>();
    auto* rootFormulaPtr{(RootFormula*) dialBase.get()};

    auto formulaConfig(GenericToolbox::Json::fetchValue<JsonType>(config_, "dialConfig"));

    rootFormulaPtr->setFormulaStr( GenericToolbox::Json::fetchValue<std::string>(formulaConfig, "formulaStr") );
  }
  else if( dialType == "CompiledLibDial" ){
    dialBase = std::make_unique<CompiledLibDial>();
    auto* compiledLibDialPtr{(CompiledLibDial*) dialBase.get()};

    auto formulaConfig(GenericToolbox::Json::fetchValue<JsonType>(config_, "dialConfig"));

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

  auto* srcGraph = dynamic_cast<const TGraph*>(src_);

  if( srcGraph->GetN() == 0 ) { return {}; }

  if( _globalDialSubType_ == "root" ) {
    auto dial = std::make_unique<RootGraph>();
    dial->buildDial(*srcGraph);
    return dial;
  }

  if( srcGraph->GetN() == 1 ){
    // For one point graph, just use a scale. Do the unique_ptr dance in case
    // there are exceptions.
    double value = srcGraph->GetY()[0];

    // is flat?
    if (std::abs(value-1.0) < 2*std::numeric_limits<float>::epsilon()) { return {}; }

    auto dial = std::make_unique<Shift>();
    dial->buildDial(value);
    return dial;
  }

  // srcGraph->GetN() >= 2, default
  auto dial = std::make_unique<Graph>();
  dial->buildDial(*srcGraph);
  return dial;
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
  if( _globalDialSubType_.find("akima") != std::string::npos ) splType = "akima";
  if( _globalDialSubType_.find("catmull") != std::string::npos) splType = "catmull-rom";
  if( _globalDialSubType_.find("natural") != std::string::npos) splType = "natural";
  if( _globalDialSubType_.find("not-a-knot") != std::string::npos) splType = "not-a-knot";
  if( _globalDialSubType_.find("pixar") != std::string::npos) {
    splType = "catmull-rom";
    // sneaky output... logger would tattle on me.
    static bool woody=true;
    if (woody) std::cout << std::endl << std::endl << "You got a friend in me!" << std::endl;
    woody=false;
  }
  if( _globalDialSubType_.find("ROOT") != std::string::npos) splType = "ROOT";

  // Get the numeric tolerance for when a uniform spline can be used.  We
  // should be able to set this in the DialSubType.
  const double defUniformityTolerance{16*std::numeric_limits<float>::epsilon()};
  double uniformityTolerance{defUniformityTolerance};
  if (_globalDialSubType_.find("uniformity(") != std::string::npos) {
    std::size_t bg = _globalDialSubType_.find("uniformity(");
    bg = _globalDialSubType_.find('(',bg);
    std::size_t en = _globalDialSubType_.find(')',bg);
    LogThrowIf(en == std::string::npos,
               "Invalid spline uniformity with dialSubType: " << _globalDialSubType_
               << " dial: " << getTitle());
    en = en - bg;
    std::string uniformityString = _globalDialSubType_.substr(bg+1,en-1);
    std::istringstream unif(uniformityString);
    unif >> uniformityTolerance;
  }

  std::vector<double> _xPointListBuffer_;
  std::vector<double> _yPointListBuffer_;
  std::vector<double> _slopeListBuffer_;

  _xPointListBuffer_.clear();
  _yPointListBuffer_.clear();
  _slopeListBuffer_.clear();

  SplineDialBaseFactory factory{};

  ///////////////////////////////////////////////////////////////////////
  // Side-effect programming alert.  The conditionals are doing the actual
  // work and setting xPoint, yPoint and slopePoint.  We only have a couple
  // of types of input for now, but watch out for the evil
  // if-then-elseif-elseif-elseif-elseif idiom.  Change this to do-while-false
  // if the number of kinds of initializers is more than a few.
  ///////////////////////////////////////////////////////////////////////
  if( factory.FillFromGraph(_xPointListBuffer_, _yPointListBuffer_, _slopeListBuffer_, (TObject*) src_, splType) ) {
    // The points were from a ROOT graph like object and the points were
    // filled, don't check any further.
  }
  else if ( factory.FillFromSpline(_xPointListBuffer_, _yPointListBuffer_, _slopeListBuffer_,
                          (TObject*) src_, splType) ) {
    // The points were from a ROOT TSpline3 like object and the points were
    // filled, don't check any further.
  }
  else {
    // If we get to this point, we don't know how to grab the spline points
    // out of the initializer, so flag this as an invalid dial.
    return nullptr;
  }

  // Check that we got at least some points!  A single point will be treated
  // as a constant value, but it's not an error.
  if (_xPointListBuffer_.empty()) {
    LogAlertOnce << "Splines must have at least one point." << std::endl;
    return nullptr;
  }

  // Check that there are equal numbers of X and Y.  There have to be an equal
  // number of points or something is very wrong.
  LogThrowIf( _xPointListBuffer_.size() != _yPointListBuffer_.size(),
              "INVALID Spline Input: "
              << "must have the same number of X and Y points "
              << "for dial " << getTitle() );

  // Check that the X points are in strictly increasing order (sorted order is
  // not sufficient) with explicit comparisons in case we want to add more
  // detailed logic.  This shouldn't happen, and indicates there is a problem
  // with the inputs.  Don't try to continue!
  for (int i = 1; i<_xPointListBuffer_.size(); ++i) {
    LogThrowIf(_xPointListBuffer_[i] <= _xPointListBuffer_[i-1],
               "INVALID Spline Input: points are not in increasing order "
               << " for dial " << getTitle());
  }

  // Check if the spline is flat.  Flat functions won't be handled with
  // splines.
  bool isFlat{true};
  for (double y : _yPointListBuffer_) {
    // Use a tolerance based on float in case the data when through a float.
    // Keep this inside the loop so it has the right scope, and depend on the
    // compiler to do the right thing.
    const double toler{2*std::numeric_limits<float>::epsilon()};
    const double delta{std::abs(y-_yPointListBuffer_[0])};
    if (delta > toler) isFlat = false;
  }

  // If the function is flat AND equal to one, the drop it.  Compare against
  // float accuracy in case the value was actually calculated against with a
  // float.
  if (std::abs(_yPointListBuffer_[0]-1.0)
      < 2*std::numeric_limits<float>::epsilon()
      and isFlat) {
    return nullptr;
  }

  // If the function is equal to a constant (but not to one) then we don't
  // need a Spline so use the faster "Shift" dial (which applies a "scale"
  // factor, and not an additive shift).
  if (isFlat) {
    // Do the unique_ptr dance in case there are exceptions.
    std::unique_ptr<DialBase> dialBase = std::make_unique<Shift>();
    dialBase->buildDial(_yPointListBuffer_[0]);
    return dialBase;
  }

#define SHORT_CIRCUIT_SMALL_SPLINES
#ifdef  SHORT_CIRCUIT_SMALL_SPLINES
  if (_xPointListBuffer_.size() <= 2) { return this->makeGraphDial(src_); }
#endif

  // Sanity check.  By the time we get here, there can't be fewer than two
  // points, and it should have been trapped above by other conditionals
  // (e.g. a single point "Spline" should have been flagged as flat).
  LogThrowIf((_xPointListBuffer_.size() < 2),
             "Input data logic error: two few points "
             << "for dial " << getTitle() );

  ////////////////////////////////////////////////////////////////
  // Check if the spline slope calculation should be updated.  The slopes for
  // not-a-knot and natural splines are calculated by FillFromGraph and
  // FillFromSpoline using ROOT code.  That means we need to fill in the
  // slopes for the other types ("catmull-rom", "akima")
  if (splType == "catmull-rom") {
    // Fill the slopes according to the Catmull-Rom prescription.
    factory.FillCatmullRomSlopes(_xPointListBuffer_,
                         _yPointListBuffer_,
                         _slopeListBuffer_);
  }
  else if (splType == "akima") {
    // Fill the slopes according to the Akima prescription.
    factory.FillAkimaSlopes(_xPointListBuffer_, _yPointListBuffer_, _slopeListBuffer_);
  }

  ////////////////////////////////////////////////////////////////
  // Check if the spline can be treated as having uniformly spaced knots.
  ////////////////////////////////////////////////////////////////
  bool isUniform = true;
  for (int i=0; i<_xPointListBuffer_.size()-1; ++i) {
    // Could be precalculated, but this only gets run once per dial so go for
    // clarity instead.  The compiler probably optimizes it out of the loop.
    const double avgSpace = (_xPointListBuffer_.back()-_xPointListBuffer_.front())/(_xPointListBuffer_.size()-1.0);
    // Find out how far the point is from the expected lattice point
    const double delta = std::abs(_xPointListBuffer_[i] - _xPointListBuffer_[0] - i*avgSpace)/avgSpace;
    if (delta < uniformityTolerance) continue;
    // Point isn't in the right place so this is not uniform and break out of
    // the loop.
    isUniform = false;
    break;
  }

  ////////////////////////////////////////////////////////////////
  // Check if the spline is suppose to be monotonic and condition the slopes
  // if necessary.  This is ignored by "ROOT" splines.  The Catmull-Rom
  // splines have a special implementation for monotonic splines, so save a
  // flag that can be checked later.
  ////////////////////////////////////////////////////////////////
  bool isMonotonic = ( _globalDialSubType_.find("monotonic") != std::string::npos );
  if ( isMonotonic ) { ::util::MakeMonotonicSpline(_xPointListBuffer_, _yPointListBuffer_, _slopeListBuffer_); }

  // If there are only two points, then force a catmull-rom.  This could be
  // handled using a graph, but Catmull-Rom is fast, and works better with the
  // GPU.  The isMonotonic is forced to false so that this uses CompactSpline
  // instead of MonotonicSpline.
  if (_xPointListBuffer_.size() < 3) {
    splType = "catmull-rom";
    isMonotonic = false;
  }

  ///////////////////////////////////////////////////////////
  // Create the right kind low level spline class base on all of the previous
  // queries.  This is pushing the if-elseif-elseif limit so watch for when it
  // should change to the do-while-false idiom.  Make sure the individual
  // conditionals are less than 10 lines.
  ///////////////////////////////////////////////////////////
  std::unique_ptr<DialBase> dialBase;
  if (splType == "ROOT") {
    // The ROOT implementation of the spline has been explicitly requested, so
    // use it.
    dialBase = std::make_unique<Spline>();
  }
  else if (splType == "catmull-rom" and isMonotonic) {
    // Catmull-Rom is handled as a special case because it ignores the slopes,
    // and has an explicit monotonic implementatino.  It also must have
    // uniformly spaced knots.
    if (not isUniform) {
      LogError << "Monotonic Catmull-rom splines need a uniformly spaced points"
               << " Dial: " << getTitle()
               << std::endl;
      double step = (_xPointListBuffer_.back()-_xPointListBuffer_.front())/(_xPointListBuffer_.size()-1);
      for (int i = 0; i<_xPointListBuffer_.size()-1; ++i) {
        LogError << i << " --  X: " << _xPointListBuffer_[i]
                 << " X+1: " << _xPointListBuffer_[i+1]
                 << " step: " << step
                 << " error: " << _xPointListBuffer_[i+1] - _xPointListBuffer_[i] - step
                 << std::endl;
      }
      // If the user specified a tolerance then crash, otherwise trust the
      // user knows that it's not uniform and continue.
      LogThrowIf(uniformityTolerance != defUniformityTolerance,
                 "Invalid catmull-rom inputs -- Nonuniform spacing");
    }
    dialBase = std::make_unique<MonotonicSpline>();
  }
  else if (splType == "catmull-rom") {
    // Catmull-Rom is handled as a special case because it ignores the slopes.
    // This is the version when the spline doesn't need to be monotonic.
    if (not isUniform) {
      LogError << "Catmull-rom splines need a uniformly spaced points"
               << " Dial: " << getTitle()
               << std::endl;
      double step = (_xPointListBuffer_.back()-_xPointListBuffer_.front())/(_xPointListBuffer_.size()-1);
      for (int i = 0; i<_xPointListBuffer_.size()-1; ++i) {
        LogError << i << " --  X: " << _xPointListBuffer_[i]
                 << " X+1: " << _xPointListBuffer_[i+1]
                 << " step: " << step
                 << " error: " << _xPointListBuffer_[i+1] - _xPointListBuffer_[i] - step
                 << std::endl;
      }
      // If the user specified a tolerance then crash, otherwise trust the
      // user knows that it's not uniform and continue.
      LogThrowIf(uniformityTolerance != defUniformityTolerance,
                 "Invalid catmull-rom inputs -- Nonuniform spacing");
    }
    dialBase = std::make_unique<CompactSpline>();
  }
  else if (isUniform) {
    // Haven't matched a specific special case, but we have uniformly spaced
    // knots so we can use the faster UniformSpline implementation.
    dialBase = std::make_unique<UniformSpline>();
  }
  else {
    // Haven't matched a specific special case, and the knots are not
    // uniformly spaced, so we have to use the GeneralSpline implemenatation
    // which can handle any kind of cubic spline.
    dialBase = std::make_unique<GeneralSpline>();
  }

  // Initialize the spline from the slopes
  dialBase->buildDial(_xPointListBuffer_, _yPointListBuffer_, _slopeListBuffer_);

  // Pass the ownership without any constraints!
  return dialBase;

}
std::unique_ptr<DialBase> DialCollection::makeSurfaceDial(const TObject* src_) const{
  // always returns an invalid ptr if
  if( src_ == nullptr ){ return nullptr; }

  auto* srcObject = dynamic_cast<const TH2*>(src_);

  LogThrowIf(srcObject == nullptr, "Surface dial initializers must be a TH2");

  // Stuff the created dial into a unique_ptr, so it will be properly deleted
  // in the event of an exception.
  std::unique_ptr<DialBase> dialBase;

  if (_globalDialSubType_ == "Bilinear") {
    // Basic coding: Give a hint to the reader and put likely branch "first".
    // Do we really expect the cached version more than the uncached?
    dialBase = std::make_unique<Bilinear>();
  }
  else if (_globalDialSubType_ == "Bicubic") {
    dialBase = std::make_unique<Bicubic>();
  }

  if (not dialBase) {
    LogError << "Invalid dialSubType value: " << _globalDialSubType_ << std::endl;
    LogError << "Valid dialSubType values are: Bilinear, Bicubic" << std::endl;
    LogThrow("Invalid Surface dialSubType");
  }

  dialBase->buildDial(*srcObject);

  // Pass the ownership without any constraints!
  return dialBase;
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
