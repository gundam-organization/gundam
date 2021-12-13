//
// Created by Adrien BLANCHET on 19/11/2021.
//

#include "EventTreeWriter.h"

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"



LoggerInit([]{ Logger::setUserHeaderStr("[TreeWriter]"); })

EventTreeWriter::EventTreeWriter() = default;
EventTreeWriter::~EventTreeWriter() = default;

void EventTreeWriter::setFitSampleSetPtr(const FitSampleSet *fitSampleSetPtr) {
  _fitSampleSetPtr_ = fitSampleSetPtr;
}
void EventTreeWriter::setParSetListPtr(const std::vector<FitParameterSet> *parSetListPtr) {
  _parSetListPtr_ = parSetListPtr;
}

void EventTreeWriter::writeSamples(TDirectory* saveDir_) const{

  for( const auto& sample : _fitSampleSetPtr_->getFitSampleList() ){
    LogInfo << "Writing sample: " << sample.getName() << std::endl;

    for( bool isData : {false, true} ) {

      const auto *evListPtr = (isData ? &sample.getDataContainer().eventList : &sample.getMcContainer().eventList);
      if (evListPtr->empty()) continue;

      auto* saveDir = GenericToolbox::mkdirTFile(saveDir_, sample.getName());

      std::string treeName = (isData ? "Data_TTree" : "MC_TTree");
      this->writeEvents(saveDir, treeName, *evListPtr);

    }
  }

}
void EventTreeWriter::writeEvents(TDirectory *saveDir_, const std::string& treeName_, const std::vector<PhysicsEvent> & eventList_) const {
  LogThrowIf(saveDir_ == nullptr, "Save TDirectory is not set.");
  LogThrowIf(treeName_.empty(), "TTree name no set.");

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
    leafDef.second(privateMemberArr, eventList_[0]); // resize buffer
  }
  privateMemberArr.lockArraySize();
  tree->Branch("Event", &privateMemberArr.getRawDataArray()[0], leavesDefStr.c_str());

  GenericToolbox::RawDataArray loadedLeavesArr;
  auto loadedLeavesDict = eventList_[0].generateLeavesDictionary(true);
  std::vector<std::string> leafNamesList;
  leavesDefStr = "";
  for( auto& leafDef : loadedLeavesDict ){
    if( not leavesDefStr.empty() ) leavesDefStr += ":";
    leavesDefStr += leafDef.first;
    leafNamesList.emplace_back(leafDef.first.substr(0,leafDef.first.find("[")).substr(0, leafDef.first.find("/")));
    leafDef.second(loadedLeavesArr, eventList_[0].getLeafHolder(leafNamesList.back())); // resize buffer
  }
  loadedLeavesArr.lockArraySize();
  tree->Branch("Leaves", &loadedLeavesArr.getRawDataArray()[0], leavesDefStr.c_str());

  std::vector<void*> parReferences;
  std::vector<TSpline3*> responseSplineList;
  std::vector<TSpline3> flatSplinesList; // flat splines for event not affected by parameter (1 spline per parameter)
  if( _writeDials_ ){
    std::vector<double> yFlat{1,1}; // 2 points at 1 weight
    if( _parSetListPtr_ != nullptr ){

      size_t nPars = 0;
      for( auto& parSet : *_parSetListPtr_ ){
        nPars += parSet.getNbParameters();
      }
      parReferences.resize(nPars, nullptr);
      responseSplineList.resize(nPars, nullptr);
      flatSplinesList.resize(nPars);

      int iPar{-1};
      for( auto& parSet : *_parSetListPtr_ ){
        for( auto& par : parSet.getParameterList() ){
          iPar++;
          std::string branchName = Form(
              "%s.%s_TSpline3",
              GenericToolbox::replaceSubstringInString(parSet.getName(), " ", "_").c_str(),
              GenericToolbox::replaceSubstringInString(par.getTitle(), " ", "_").c_str()
          );
          responseSplineList[iPar] = new TSpline3();
          parReferences[iPar] = ((void*)&par);
          tree->Branch(branchName.c_str(), &responseSplineList[iPar]);

          double min = par.getMinValue();
          double max = par.getMaxValue();
          if( min != min ) min = par.getPriorValue() - 5*par.getStdDevValue();
          if( max != max ) min = par.getPriorValue() + 5*par.getStdDevValue();
          flatSplinesList[iPar] = TSpline3(Form("flatSplinePar%i", iPar), min, max, &yFlat[0], 2);
        }

      }
    }
  }


  int iLeaf;
  int iPar{0};
  std::string progressTitle = LogInfo.getPrefixString() + "Writing " + treeName_;
  size_t iEvent{0}; size_t nEvents = (eventList_.size());
  for( auto& event : eventList_ ){
    GenericToolbox::displayProgressBar(iEvent++,nEvents,progressTitle);

    privateMemberArr.resetCurrentByteOffset();
    for( auto& leafDef : leafDictionary ){ leafDef.second(privateMemberArr, event); }

    iLeaf = 0;
    loadedLeavesArr.resetCurrentByteOffset();
    for( auto& leafDef : loadedLeavesDict ){ leafDef.second(loadedLeavesArr, event.getLeafHolder(leafNamesList[iLeaf++])); }

    if( _writeDials_ ){
      for( auto& spline : responseSplineList ){ *spline = flatSplinesList[iPar]; } // by default
      for( auto* dialPtr : event.getRawDialPtrList() ){
        iPar = GenericToolbox::findElementIndex(dialPtr->getAssociatedParameterReference(), parReferences);
        dialPtr->copySplineCache(*responseSplineList[iPar]);
      }
    }

    tree->Fill();
  }
  tree->Write();
  delete tree;

  if(oldDir != nullptr) oldDir->cd();
}
