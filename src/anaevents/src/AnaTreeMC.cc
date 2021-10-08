// This is the code that actually reads in the MC tree and fills the event info.
// The tree should be produced by feeding a HL2 microtree into the treeconvert macro.

#include "AnaTreeMC.hh"
#include "GenericToolbox.h"
#include "Logger.h"
#include "TTreeFormula.h"
#include <future>
#include "GlobalVariables.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[AnaTreeMC]");
} )

AnaTreeMC::AnaTreeMC(const std::string& file_name, const std::string& tree_name, bool extra_var)
  : read_extra_var(extra_var)
{
  fChain = new TChain(tree_name.c_str());
  fChain->Add(file_name.c_str());
  _file_name_ = file_name;
  SetBranches();
}

AnaTreeMC::~AnaTreeMC()
{
  if(fChain != nullptr)
    delete fChain->GetCurrentFile();
}

long int AnaTreeMC::GetEntry(long int entry) const
{
  // Read contents of entry.
  if(fChain == nullptr)
    return -1;
  else
    return fChain->GetEntry(entry);
}

void AnaTreeMC::SetBranches()
{
//    // Set branch addresses and branch pointers
//    fChain->SetBranchAddress("nutype", &nutype);
//    fChain->SetBranchAddress("beammode", &beammode);
//    fChain->SetBranchAddress("cut_branch", &sample);
//    fChain->SetBranchAddress("topology", &topology);
//    fChain->SetBranchAddress("reaction", &reaction);
//    fChain->SetBranchAddress("target", &target);
//    fChain->SetBranchAddress("D1True", &D1True);
//    fChain->SetBranchAddress("D1Reco", &D1Reco);
//    fChain->SetBranchAddress("D2True", &D2True);
//    fChain->SetBranchAddress("D2Reco", &D2Reco);
//    fChain->SetBranchAddress("q2_true", &q2_true);
//    fChain->SetBranchAddress("q2_reco", &q2_reco);
//    fChain->SetBranchAddress("enu_true", &enu_true);
//    fChain->SetBranchAddress("enu_reco", &enu_reco);
//    fChain->SetBranchAddress("weight", &weight);
//
//    if(read_extra_var)
//    {
//        //Put extra variables here.
//    }
}

void AnaTreeMC::GetEvents(std::vector<AnaSample*>& ana_samples,
                          const std::vector<SignalDef>& v_signal,
                          const bool evt_type)
{
  if(fChain == nullptr || ana_samples.empty()){
    return;
  }


  AnaEvent eventHolder;
  auto* enabledIntLeafs = new std::vector<std::string>(*eventHolder.GetIntVarNameListPtr());
  auto* enabledFloatLeafs = new std::vector<std::string>(*eventHolder.GetFloatVarNameListPtr());

  std::vector<TTreeFormula*> additionalCutsList;
  LogInfo << "Scanning additional cuts..." << std::endl;
  for(size_t iSample = 0 ; iSample < ana_samples.size() ; iSample++){

    if(ana_samples[iSample]->GetAdditionalCuts().empty()){
      additionalCutsList.emplace_back(nullptr);
      continue; // no additional cuts
    }

    LogInfo << "\"" << ana_samples[iSample]->GetAdditionalCuts() << "\" -> \""
            << ana_samples[iSample]->GetName() << "\""
            << std::endl;
    additionalCutsList.emplace_back(
      new TTreeFormula(
        Form("additional_cuts_%i", int(iSample)),
        ana_samples[iSample]->GetAdditionalCuts().c_str(),
        fChain
      )
    );

    for(int iPar = 0 ; iPar < additionalCutsList.back()->GetNcodes() ; iPar++){
      if(std::string(additionalCutsList.back()->GetLeaf(iPar)->GetTypeName()) == "Int_t"){
        if(not GenericToolbox::doesElementIsInVector(additionalCutsList.back()->GetLeaf(iPar)->GetName(), *enabledIntLeafs)){
          enabledIntLeafs->emplace_back(additionalCutsList.back()->GetLeaf(iPar)->GetName());
        }
      }
      else if(std::string(additionalCutsList.back()->GetLeaf(iPar)->GetTypeName()) == "Float_t"){
        if(not GenericToolbox::doesElementIsInVector(additionalCutsList.back()->GetLeaf(iPar)->GetName(), *enabledFloatLeafs)){
          enabledFloatLeafs->emplace_back(additionalCutsList.back()->GetLeaf(iPar)->GetName());
        }
      }
    }
    additionalCutsList.back()->SetTree(fChain);
  }

  LogDebug << "The following ints leafs will be stored in memory:" << std::endl;
  GenericToolbox::printVector(*enabledIntLeafs);
  eventHolder.SetIntVarNameListPtr(enabledIntLeafs);

  LogDebug << "The following floats leafs will be stored in memory:" << std::endl;
  GenericToolbox::printVector(*enabledFloatLeafs);
  eventHolder.SetFloatVarNameListPtr(enabledFloatLeafs);

  // Claiming memory and mapping events
  LogInfo << "Performing topology cuts on samples and claiming memory..." << std::endl;
  std::string progressTitle = LogWarning.getPrefixString() + "Claiming memory and mapping events...";
  int totalNbEventsToLoad = 0;
  for( int jEntry = 0 ; jEntry < fChain->GetEntries() ; jEntry++ ) {

    GenericToolbox::displayProgressBar(jEntry, fChain->GetEntries(), progressTitle);
    fChain->GetEntry(jEntry);

    for(size_t iSample = 0 ; iSample < ana_samples.size() ; iSample++){

      if(ana_samples[iSample]->GetSampleID() == fChain->GetLeaf("cut_branch")->GetValue(0)){

        fChain->SetNotify(additionalCutsList[iSample]);
        bool doEventPassCut = true;
        for(int jInstance = 0; jInstance < additionalCutsList[iSample]->GetNdata(); jInstance++) {
          if ( additionalCutsList[iSample]->EvalInstance(jInstance) == 0 ) {
            doEventPassCut = false;
            break;
          }
        }

        if(doEventPassCut){
          eventHolder.SetEventId(jEntry);
          ana_samples[iSample]->AddEvent(eventHolder); // copy constructor of eventHolder
          // Re-hook to make the member ptr point toward the copied data container addresses
//                    ana_samples[iSample]->GetEventList().back().HookIntMembers();
//                    ana_samples[iSample]->GetEventList().back().HookFloatMembers();
          totalNbEventsToLoad++;
        }

      }
    }
  } // jEntry

  // Will be used externally:
  GlobalVariables::getChainList().resize(GlobalVariables::getNbThreads());
  if(GlobalVariables::getNbThreads() > 1){
    for( int iThread = 0 ; iThread < GlobalVariables::getNbThreads() ; iThread++ ){
      GlobalVariables::getChainList().at(iThread) = new TChain(fChain->GetName());
      GlobalVariables::getChainList().at(iThread)->Add(_file_name_.c_str());
    }
  }
  else{
    GlobalVariables::getChainList().at(0) = fChain;
  }

  std::vector<TChain*>* chainListPtr = &GlobalVariables::getChainList();

  std::function<void(int)> fillEvents = [chainListPtr, ana_samples, additionalCutsList, this](int iThread_){

    GlobalVariables::getThreadMutex().lock();
    bool isMultiThreaded = (iThread_ != -1);
    TChain* threadChain = fChain; // Single thread
    if(isMultiThreaded){
      threadChain = chainListPtr->at(iThread_);
    }
    std::string progressBarPrefix;
    AnaEvent* eventPtr;
    GlobalVariables::getThreadMutex().unlock();

    for(size_t iSample = 0 ; iSample < ana_samples.size() ; iSample++){

      progressBarPrefix = LogWarning.getPrefixString() + "Fill sample: " + ana_samples.at(iSample)->GetName();

      for(int iEvent = 0 ; iEvent < ana_samples.at(iSample)->GetN() ; iEvent++){

        if( iEvent%GlobalVariables::getNbThreads() != iThread_ ){
          continue;
        }

        if( iThread_+1 == GlobalVariables::getNbThreads() ){
          GenericToolbox::displayProgressBar(iEvent, ana_samples.at(iSample)->GetN(), progressBarPrefix);
        }

        eventPtr = ana_samples.at(iSample)->GetEvent(iEvent);

        while(eventPtr->GetIsBeingEdited());
        if(eventPtr->GetTreeEventHasBeenDumped()) continue;

        eventPtr->SetIsBeingEdited(true);
        threadChain->GetEntry(eventPtr->GetEvId());
        eventPtr->DumpTreeEntryContent(threadChain); // Here is the most time consuming part
        eventPtr->SetIsBeingEdited(false);
      }

      if( iThread_+1 == GlobalVariables::getNbThreads() ){
        GenericToolbox::displayProgressBar(ana_samples.at(iSample)->GetN(), ana_samples.at(iSample)->GetN(), progressBarPrefix);
      }

    }

  };

  LogInfo << "Reading events from input file (" << fChain->GetEntries() << ")..." << std::endl;
  GlobalVariables::getParallelWorker().addJob("fillMCEvents", fillEvents);
  GlobalVariables::getParallelWorker().runJob("fillMCEvents");
  GlobalVariables::getParallelWorker().removeJob("fillMCEvents");

  LogDebug << "Reading input tree ended." << std::endl;


  LogInfo << "Process signals type..." << std::endl;
  for(size_t iSample = 0 ; iSample < ana_samples.size() ; iSample++){
    for(int iEvent = 0 ; iEvent < ana_samples[iSample]->GetN() ; iEvent++){
      int signal_type = 0;

      // Loop over all sets of temlate parameters as defined in the .json config file in the "template_par" entry for this detector:
      for(const auto& sd : v_signal)
      {
        bool sig_passed = true;

        // Loop over all the different signal definitions for this set of template parameters (e.g. topology, target, nutype, etc.):
        for(const auto& kv : sd.definition)
        {
          bool var_passed = false;

          // Loop over all the values for the current signal definition (e.g. all the different topology integers):
          for(const auto& val : kv.second)
          {
            if(ana_samples[iSample]->GetEvent(iEvent)->GetEventVar(kv.first) == val)
              var_passed = true;
          }
          sig_passed = sig_passed && var_passed;
        }
        if(sig_passed)
        {
          ana_samples[iSample]->GetEvent(iEvent)->SetSignalType(signal_type);
          ana_samples[iSample]->GetEvent(iEvent)->SetSignalEvent();
          break;
        }
        signal_type++;
      }
    }
  }

  LogInfo << "Printing sample stats..." << std::endl;
  for(auto& sample : ana_samples)
    sample->PrintStats();

  for(auto& additionalCuts : additionalCutsList){
    delete additionalCuts;
  }

}
