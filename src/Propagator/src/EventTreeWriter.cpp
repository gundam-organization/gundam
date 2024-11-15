//
// Created by Adrien BLANCHET on 19/11/2021.
//

#include "EventTreeWriter.h"
#include "Propagator.h"

#include "ConfigUtils.h"

#include "Logger.h"
#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Root.h"


void EventTreeWriter::configureImpl() {

  GenericToolbox::Json::fillValue(_config_, _isEnabled_, "isEnabled");
  LogReturnIf(not _isEnabled_, "Disabled EventTreeWriter.");

  GenericToolbox::Json::fillValue(_config_, _writeDials_, "writeDials");
  GenericToolbox::Json::fillValue(_config_, _nPointsPerDial_, "nPointsPerDial");

  if( _writeDials_ ){
    LogInfo << "EventTreeWriter configured as:" << std::endl;
    {
      LogScopeIndent;
      LogInfo << GET_VAR_NAME_VALUE(_writeDials_) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(_nPointsPerDial_) << std::endl;
    }
  }

  _threadPool_.setNThreads(GundamGlobals::getNbCpuThreads() );
}


void EventTreeWriter::writeEvents(const GenericToolbox::TFilePath& saveDir_, const std::vector<Event> & eventList_) const {
  this->writeEventsTemplate(saveDir_, eventList_);
}
void EventTreeWriter::writeEvents(const GenericToolbox::TFilePath& saveDir_, const std::vector<const EventDialCache::CacheEntry*>& cacheSampleList_) const{
  LogReturnIf(not _isEnabled_, "Disabled EventTreeWriter. Skipping writeEvents.");
  this->writeEventsTemplate(saveDir_, cacheSampleList_);
}

template<typename T> void EventTreeWriter::writeEventsTemplate(const GenericToolbox::TFilePath& saveDir_, const T& eventList_) const {
  LogReturnIf(eventList_.empty(), "No event to be written. Leaving...");

  const std::vector<EventDialCache::DialResponseCache>* dialElements{getDialElementsPtr(eventList_[0])};
  bool writeDials{dialElements != nullptr};

  auto* oldDir = GenericToolbox::getCurrentTDirectory();
  saveDir_.getDir()->cd();
  auto* tree = new TTree("events", "events");

  GenericToolbox::RawDataArray privateMemberArr;
  std::map<std::string, std::function<void(GenericToolbox::RawDataArray&, const Event&)>> leafDictionary;
  leafDictionary["eventWeight/D"] =   [](GenericToolbox::RawDataArray& arr_, const Event& ev_){ arr_.writeRawData(ev_.getWeights().current); };
  leafDictionary["treeWeight/D"] =    [](GenericToolbox::RawDataArray& arr_, const Event& ev_){ arr_.writeRawData(ev_.getWeights().base); };
  leafDictionary["sampleBinIndex/I"]= [](GenericToolbox::RawDataArray& arr_, const Event& ev_){ arr_.writeRawData(ev_.getIndices().bin); };
  leafDictionary["dataSetIndex/I"] =  [](GenericToolbox::RawDataArray& arr_, const Event& ev_){ arr_.writeRawData(ev_.getIndices().dataset); };
  leafDictionary["entryIndex/L"] =    [](GenericToolbox::RawDataArray& arr_, const Event& ev_){ arr_.writeRawData(ev_.getIndices().entry); };
  std::string branchDefStr;
  for( auto& leafDef : leafDictionary ){
    if( not branchDefStr.empty() ) branchDefStr += ":";
    branchDefStr += leafDef.first;
    leafDef.second(privateMemberArr, *EventTreeWriter::getEventPtr(eventList_[0])); // resize buffer
  }
  privateMemberArr.lockArraySize();
  tree->Branch("Event", &privateMemberArr.getRawDataArray()[0], branchDefStr.c_str());

  GenericToolbox::RawDataArray loadedLeavesArr;

  struct LeavesDictionary{
    std::string leafDefinitionStr{};
    bool disableArray{false};

    void dropData(GenericToolbox::RawDataArray& arr_, const GenericToolbox::AnyType& var_){
      arr_.writeMemoryContent(
          var_.getPlaceHolderPtr()->getVariableAddress(),
          var_.getPlaceHolderPtr()->getVariableSize()
      );
      if( disableArray ){ return; }
    }
  };
  std::vector<LeavesDictionary> lDict;


  auto* evPtr = EventTreeWriter::getEventPtr(eventList_[0]);
  if( evPtr != nullptr and evPtr->getVariables().getNameListPtr() != nullptr ){
    for( auto& varName : *EventTreeWriter::getEventPtr(eventList_[0])->getVariables().getNameListPtr() ){
      lDict.emplace_back();
      lDict.back().disableArray = true;

      auto& var = evPtr->getVariables().fetchVariable( varName ).get();
      char typeTag = GenericToolbox::findOriginalVariableType(var);
      LogThrowIf( typeTag == 0 or typeTag == char(0xFF), varName << " has an invalid leaf type." );

      std::string leafDefStr{ varName };
//      if(not disableArrays_ and lH.size() > 1){ leafDefStr += "[" + std::to_string(lH.size()) + "]"; }
      leafDefStr += "/";
      leafDefStr += typeTag;
      lDict.back().leafDefinitionStr = leafDefStr;
    }
  }

  std::vector<std::string> leafNamesList;
  if( not lDict.empty() ){
    branchDefStr = "";
    for( int iLeaf = 0 ; iLeaf < lDict.size() ; iLeaf++ ){
      if( not branchDefStr.empty() ) branchDefStr += ":";
      branchDefStr += lDict[iLeaf].leafDefinitionStr;
      leafNamesList.emplace_back(
          lDict[iLeaf].leafDefinitionStr.substr(0,lDict[iLeaf].leafDefinitionStr.find("[")).substr(0, lDict[iLeaf].leafDefinitionStr.find("/")));
      lDict[iLeaf].dropData(loadedLeavesArr, EventTreeWriter::getEventPtr(eventList_[0])->getVariables().getVarList()[iLeaf].get()); // resize buffer
    }
    loadedLeavesArr.lockArraySize();
    tree->Branch("Leaves", &loadedLeavesArr.getRawDataArray()[0], branchDefStr.c_str());
  }


  // dial writing part (used if the correct template is set)
  std::vector<std::vector<double>> parameterXvalues{};
  std::vector<TGraph*> graphParResponse{};
  std::vector<std::pair<size_t,size_t>> parIndexList{};

  if( writeDials ){
    LogThrowIf(parSetListPtr == nullptr);

    // how many pars?
    size_t nPars = 0;
    for( auto& parSet : *parSetListPtr ){
      nPars += parSet.getNbParameters();
    }

    // reserve max potential size
    parameterXvalues.reserve(nPars);
    graphParResponse.reserve(nPars);
    parIndexList.reserve(nPars);

    // create branches
    int iParSet{-1};
    for( auto& parSet : *parSetListPtr ){
      iParSet++;
      if( not parSet.isEnabled() ) continue;
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ) continue;

        // entry
        parameterXvalues.emplace_back(_nPointsPerDial_, par.getParameterValue());
        parIndexList.emplace_back(int(iParSet), par.getParameterIndex());
        graphParResponse.emplace_back( new TGraph(_nPointsPerDial_) );

        // determining eval points
        double min = par.getParameterValue() - 5*par.getStdDevValue();
        double max = par.getParameterValue() + 5*par.getStdDevValue();
        for( int iPt = 0 ; iPt < _nPointsPerDial_ ; iPt++ ){
          parameterXvalues.back()[iPt] = min + (max - min)*iPt/(_nPointsPerDial_-1);
          graphParResponse.back()->SetPointX(iPt, parameterXvalues.back()[iPt]);
        }

        graphParResponse.back()->SetMarkerStyle( kFullDotLarge );
        graphParResponse.back()->SetTitle( par.getFullTitle().c_str() );

        std::string brName{GenericToolbox::generateCleanBranchName(par.getFullTitle())};
        tree->Branch(brName.c_str(), &graphParResponse.back());
        // address of the ptr whould not change since we pre-reserved memory
      }
    }

    // parallel job to write dials
    _threadPool_.addJob("buildResponseGraph", [&](int iThread_){
      TGraph* grPtr{nullptr};
      for( size_t iGlobalPar = 0 ; iGlobalPar < parIndexList.size() ; iGlobalPar++ ){
        if( iThread_ != -1 and iGlobalPar % GundamGlobals::getNbCpuThreads() != iThread_ ) continue;

        // reset the corresponding graph
        grPtr = graphParResponse[iGlobalPar];

        // lighten the graph
        while( grPtr->GetN() != 1 ){ grPtr->RemovePoint(0); }
        grPtr->SetPointX(0, parameterXvalues[iGlobalPar][_nPointsPerDial_/2+1]);
        grPtr->SetPointY(0, 1);

        // fetch corresponding dial if it exists
        for( auto& dial : *dialElements ){
          if( dial.dialInterface->getInputBufferRef()->getInputParameterIndicesList()[0].parSetIndex == parIndexList[iGlobalPar].first
              and dial.dialInterface->getInputBufferRef()->getInputParameterIndicesList()[0].parIndex == parIndexList[iGlobalPar].second ){

            DialInputBuffer inputBuf{*dial.dialInterface->getInputBufferRef()};
            grPtr->RemovePoint(0); // remove the first and recreate the whole thing
            for( double xPoint : parameterXvalues[iGlobalPar] ){
              inputBuf.getInputBuffer()[0] = xPoint;
              grPtr->AddPoint(
                  xPoint,
                  DialInterface::evalResponse(
                      &inputBuf,
                      dial.dialInterface->getDialBaseRef(),
                      dial.dialInterface->getResponseSupervisorRef()
                  )
              );
            }

            break;
          }
        }
      }
    });
  }

  size_t iEvent{0}; size_t nEvents = (eventList_.size());
  for( auto& cacheEntry : eventList_ ){

    privateMemberArr.resetCurrentByteOffset();
    for( auto& leafDef : leafDictionary ){ leafDef.second(privateMemberArr, *EventTreeWriter::getEventPtr(cacheEntry)); }

    loadedLeavesArr.resetCurrentByteOffset();
    for( int iLeaf = 0 ; iLeaf < lDict.size() ; iLeaf++ ){
      lDict[iLeaf].dropData(
          loadedLeavesArr,
          EventTreeWriter::getEventPtr( cacheEntry )->getVariables().getVarList()[iLeaf].get()
      );
    }

    if( writeDials ){
      dialElements = getDialElementsPtr(cacheEntry);
      _threadPool_.runJob("buildResponseGraph");
    }

    tree->Fill();
  }

  if( writeDials ) {
    _threadPool_.removeJob("buildResponseGraph");
  }

  GenericToolbox::writeInTFile( saveDir_.getDir(), tree );
  delete tree;

  if(oldDir != nullptr){ oldDir->cd(); }
}
