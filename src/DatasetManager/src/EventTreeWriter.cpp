//
// Created by Adrien BLANCHET on 19/11/2021.
//

#include "EventTreeWriter.h"

#include "ConfigUtils.h"

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Json.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[TreeWriter]");
});

EventTreeWriter::EventTreeWriter() = default;
EventTreeWriter::~EventTreeWriter() = default;

void EventTreeWriter::readConfigImpl() {
  ConfigUtils::forwardConfig( _config_ );

  _writeDials_ = GenericToolbox::Json::fetchValue(_config_, "writeDials", _writeDials_);
  _nPointsPerDial_ = GenericToolbox::Json::fetchValue(_config_, "nPointsPerDial", _nPointsPerDial_);

  LogInfo << "EventTreeWriter configured as:" << std::endl;
  {
    LogScopeIndent;
    LogInfo << GET_VAR_NAME_VALUE(_writeDials_) << std::endl;
    LogInfo << GET_VAR_NAME_VALUE(_nPointsPerDial_) << std::endl;
  }
}

void EventTreeWriter::setFitSampleSetPtr(const FitSampleSet *fitSampleSetPtr) {
  _fitSampleSetPtr_ = fitSampleSetPtr;
}
void EventTreeWriter::setParSetListPtr(const std::vector<FitParameterSet> *parSetListPtr) {
  _parSetListPtr_ = parSetListPtr;
}
void EventTreeWriter::setEventDialCachePtr(const EventDialCache *eventDialCachePtr_){
  _eventDialCachePtr_ = eventDialCachePtr_;
}

void EventTreeWriter::writeSamples(TDirectory* saveDir_) const{
  LogInfo << "Writing sample data in TTrees..." << std::endl;

  for( const auto& sample : _fitSampleSetPtr_->getFitSampleList() ){
    LogScopeIndent;
    LogInfo << "Writing sample: " << sample.getName() << std::endl;

    for( bool isData : {false, true} ) {
      const auto *evListPtr = (isData ? &sample.getDataContainer().eventList : &sample.getMcContainer().eventList);
      if (evListPtr->empty()) continue;

      if( not _writeDials_ or isData ){
        this->writeEvents(GenericToolbox::mkdirTFile(saveDir_, sample.getName()), (isData ? "Data" : "MC"), *evListPtr);
      }
      else{
        LogThrowIf(_eventDialCachePtr_ == nullptr, "Can't write dials if event dial cache is not set.");

        std::vector<const EventDialCache::CacheElem_t*> cacheSampleList{};
        cacheSampleList.reserve( _eventDialCachePtr_->getCache().size() );
        for( auto& cacheEntry : _eventDialCachePtr_->getCache() ){
          if( cacheEntry.event->getSampleIndex() == sample.getIndex() ){
            cacheSampleList.emplace_back( &cacheEntry );
          }
        }
        cacheSampleList.shrink_to_fit();

        this->writeEvents(GenericToolbox::mkdirTFile(saveDir_, sample.getName()), "MC", cacheSampleList);
      }
    } // isData
  } // sample

}
void EventTreeWriter::writeEvents(TDirectory *saveDir_, const std::string& treeName_, const std::vector<PhysicsEvent> & eventList_) const {
  this->writeEventsTemplate(saveDir_, treeName_, eventList_);
}
void EventTreeWriter::writeEvents(TDirectory* saveDir_, const std::string& treeName_, const std::vector<const EventDialCache::CacheElem_t*>& cacheSampleList_) const{
  this->writeEventsTemplate(saveDir_, treeName_, cacheSampleList_);
}

template<typename T> void EventTreeWriter::writeEventsTemplate(TDirectory* saveDir_, const std::string& treeName_, const T& eventList_) const {
  LogThrowIf(saveDir_ == nullptr, "Save TDirectory is not set.");
  LogThrowIf(treeName_.empty(), "TTree name no set.");

  LogReturnIf(eventList_.empty(), "No event to be written. Leaving...");

  const std::vector<EventDialCache::DialsElem_t>* dialElements{getDialElementsPtr(eventList_[0])};
  bool writeDials{dialElements != nullptr};

  LogInfo << "Writing " << eventList_.size() << " events " << (writeDials? "with response dials ": "without response dials") << "in TTree " << treeName_ << std::endl;

  auto* oldDir = GenericToolbox::getCurrentTDirectory();

  saveDir_->cd();
  auto* tree = new TTree(treeName_.c_str(), treeName_.c_str());

  GenericToolbox::RawDataArray privateMemberArr;
  std::map<std::string, std::function<void(GenericToolbox::RawDataArray&, const PhysicsEvent&)>> leafDictionary;
  leafDictionary["eventWeight/D"] =   [](GenericToolbox::RawDataArray& arr_, const PhysicsEvent& ev_){ arr_.writeRawData(ev_.getEventWeight()); };
  leafDictionary["nominalWeight/D"] = [](GenericToolbox::RawDataArray& arr_, const PhysicsEvent& ev_){ arr_.writeRawData(ev_.getNominalWeight()); };
  leafDictionary["treeWeight/D"] =    [](GenericToolbox::RawDataArray& arr_, const PhysicsEvent& ev_){ arr_.writeRawData(ev_.getTreeWeight()); };
  leafDictionary["sampleBinIndex/I"]= [](GenericToolbox::RawDataArray& arr_, const PhysicsEvent& ev_){ arr_.writeRawData(ev_.getSampleBinIndex()); };
  leafDictionary["dataSetIndex/I"] =  [](GenericToolbox::RawDataArray& arr_, const PhysicsEvent& ev_){ arr_.writeRawData(ev_.getDataSetIndex()); };
  leafDictionary["entryIndex/L"] =    [](GenericToolbox::RawDataArray& arr_, const PhysicsEvent& ev_){ arr_.writeRawData(ev_.getEntryIndex()); };
  std::string leavesDefStr;
  for( auto& leafDef : leafDictionary ){
    if( not leavesDefStr.empty() ) leavesDefStr += ":";
    leavesDefStr += leafDef.first;
    leafDef.second(privateMemberArr, *EventTreeWriter::getEventPtr(eventList_[0])); // resize buffer
  }
  privateMemberArr.lockArraySize();
  tree->Branch("Event", &privateMemberArr.getRawDataArray()[0], leavesDefStr.c_str());

  GenericToolbox::RawDataArray loadedLeavesArr;
  auto loadedLeavesDict = EventTreeWriter::getEventPtr(eventList_[0])->generateLeavesDictionary(true);
  std::vector<std::string> leafNamesList;
  if( not loadedLeavesDict.empty() ){
    leavesDefStr = "";
    for( auto& leafDef : loadedLeavesDict ){
      if( not leavesDefStr.empty() ) leavesDefStr += ":";
      leavesDefStr += leafDef.first;
      leafNamesList.emplace_back(leafDef.first.substr(0,leafDef.first.find("[")).substr(0, leafDef.first.find("/")));
      leafDef.second(loadedLeavesArr, EventTreeWriter::getEventPtr(eventList_[0])->getLeafHolder(leafNamesList.back())); // resize buffer
    }
    loadedLeavesArr.lockArraySize();
    tree->Branch("Leaves", &loadedLeavesArr.getRawDataArray()[0], leavesDefStr.c_str());
  }


  // dial writing part (used if the correct template is set)
  std::vector<std::vector<double>> parameterXvalues{};
  std::vector<TGraph*> graphParResponse{};
  std::vector<std::pair<size_t,size_t>> parIndexList{};


  if( writeDials ){

    LogThrowIf(_parSetListPtr_ == nullptr, "Not parSet list provided.");

    // how many pars?
    size_t nPars = 0;
    for( auto& parSet : *_parSetListPtr_ ){
      nPars += parSet.getNbParameters();
    }

    // reserve max potential size
    parameterXvalues.reserve(nPars);
    graphParResponse.reserve(nPars);
    parIndexList.reserve(nPars);

    // create branches
    int iParSet{-1};
    for( auto& parSet : *_parSetListPtr_ ){
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
    GlobalVariables::getParallelWorker().addJob("buildResponseGraph", [&](int iThread_){
      TGraph* grPtr{nullptr};
      for( size_t iGlobalPar = 0 ; iGlobalPar < parIndexList.size() ; iGlobalPar++ ){
        if( iThread_ != -1 and iGlobalPar % GlobalVariables::getNbThreads() != iThread_ ) continue;

        // reset the corresponding graph
        grPtr = graphParResponse[iGlobalPar];

        // lighten the graph
        while( grPtr->GetN() != 1 ){ grPtr->RemovePoint(0); }
        grPtr->SetPointX(0, parameterXvalues[iGlobalPar][_nPointsPerDial_/2+1]);
        grPtr->SetPointY(0, 1);

        // fetch corresponding dial if it exists
        for( auto& dial : *dialElements ){
          if( dial.interface->getInputBufferRef()->getInputParameterIndicesList()[0] == parIndexList[iGlobalPar] ){

            DialInputBuffer inputBuf{*dial.interface->getInputBufferRef()};
            grPtr->RemovePoint(0); // remove the first and recreate the whole thing
            for( double xPoint : parameterXvalues[iGlobalPar] ){
              inputBuf.getBufferVector()[0] = xPoint;
              grPtr->AddPoint(
                  xPoint,
                  DialInterface::evalResponse(
                      &inputBuf,
                      dial.interface->getDialBaseRef(),
                      dial.interface->getResponseSupervisorRef()
                  )
              );
            }

            break;
          }
        }
      }
    });
  }

  int iLeaf;
  std::string progressTitle = LogInfo.getPrefixString() + Logger::getIndentStr() + "Writing " + treeName_;
  size_t iEvent{0}; size_t nEvents = (eventList_.size());
  for( auto& cacheEntry : eventList_ ){
    GenericToolbox::displayProgressBar(iEvent++,nEvents,progressTitle);

    privateMemberArr.resetCurrentByteOffset();
    for( auto& leafDef : leafDictionary ){ leafDef.second(privateMemberArr, *EventTreeWriter::getEventPtr(cacheEntry)); }

    iLeaf = 0;
    loadedLeavesArr.resetCurrentByteOffset();
    for( auto& leafDef : loadedLeavesDict ){ leafDef.second(loadedLeavesArr, EventTreeWriter::getEventPtr(cacheEntry)->getLeafHolder(leafNamesList[iLeaf++])); }

    if( writeDials ){
      dialElements = getDialElementsPtr(cacheEntry);
      GlobalVariables::getParallelWorker().runJob("buildResponseGraph");
    }

    tree->Fill();
  }

  if( writeDials ) {
    GlobalVariables::getParallelWorker().removeJob("buildResponseGraph");
  }


//  tree->Write();
  GenericToolbox::writeInTFile( saveDir_, tree );
  delete tree;

  if(oldDir != nullptr) oldDir->cd();
}
