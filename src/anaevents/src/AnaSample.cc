#include <utility>

#include "../include/AnaSample.hh"
#include "GlobalVariables.h"
#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolboxRootExt.h"
#include "TTreeFormula.h"

using xsllh::FitBin;

LoggerInit([](){
  Logger::setUserHeaderStr("[AnaSample]");
} )

// ctor
AnaSample::AnaSample(int sample_id, std::string  name, std::string  detector,
                     std::string  binning, TTree* t_data)
  : m_sample_id(sample_id)
  , m_name(std::move(name))
  , m_detector(std::move(detector))
  , m_binning(std::move(binning))
  , m_data_tree(t_data)
  , m_norm(1.0)
{

  m_hpred    = nullptr;
  m_hmc      = nullptr;
  m_hmc_true = nullptr;
  m_hsig     = nullptr;
  m_hdata    = nullptr;

  Reset();
}

AnaSample::AnaSample(const SampleOpt& sample, TTree* t_data){

  m_sample_id = sample.cut_branch;
  m_name = sample.name;
  m_detector = sample.detector;
  m_binning = sample.binning;
  m_additional_cuts = sample.additional_cuts;
  m_fit_phase_space = sample.fit_phase_space;
  m_data_POT = sample.data_POT;
  m_mc_POT = sample.mc_POT;
  m_norm = 1.0;

  m_hpred    = nullptr;
  m_hmc      = nullptr;
  m_hmc_true = nullptr;
  m_hsig     = nullptr;
  m_hdata    = nullptr;

  m_data_tree = t_data;

  this->Reset();

}

// Private constructor
void AnaSample::Reset() {

  if(m_data_tree != nullptr){
    m_additional_cuts_formulae = new TTreeFormula(
      "additional_cuts", m_additional_cuts.c_str(), m_data_tree
    );
  }

  TH1::SetDefaultSumw2(true);
  SetBinning(m_binning);

  if(m_data_POT != 0 and m_mc_POT != 0){
    SetNorm(m_data_POT/m_mc_POT);
  }

  LogInfo << m_name << ", ID " << m_sample_id << std::endl
          << "Detector: " << m_detector << std::endl
          << "Bin edges: " << std::endl;

  m_llh = new PoissonLLH;

  MakeHistos(); // with default binning

  LogInfo << "MakeHistos called." << std::endl;

}

AnaSample::~AnaSample()
{
  delete m_hpred;
  delete m_hmc;
  delete m_hmc_true;
  delete m_hsig;
  delete m_hdata;
}

void AnaSample::SetBinning(const std::string& binning)
{
  m_binning = binning;
  m_nbins   = 0;

  std::ifstream fin(m_binning, std::ios::in);
  if(!fin.is_open())
  {
    LogError << "In AnaSample::SetBinning().\n"
             << "Failed to open binning file: " << m_binning << std::endl;
  }
  else
  {
    std::string line;
    while(std::getline(fin, line))
    {
      std::stringstream ss(line);
      std::vector<double> lowEdges;
      std::vector<double> highEdges;

      double lowEdge, highEdge;
      while( ss >> lowEdge >> highEdge ){
        lowEdges.emplace_back( lowEdge );
        highEdges.emplace_back( highEdge );
      }

      if( m_fit_phase_space.size() != lowEdges.size() ){
        LogWarning << "Bad bin: \"" << line << "\"" << std::endl;
        continue;
      }

      m_bin_edges.emplace_back(GeneralizedFitBin(lowEdges, highEdges));
    }
    m_nbins = m_bin_edges.size();
    if( m_nbins == 0 ){
      LogError << "No bin has been defined for the sample \"" << m_name << "\"." << std::endl;
      throw std::runtime_error("No bin has been defined for the sample.");
    }
  }
}

// Mapping the Highland topology codes to consecutive integers:
void AnaSample::SetTopologyHLCode(const std::vector<int>& HLTopologyCodes)
{
  for(std::size_t i=0; i < HLTopologyCodes.size(); ++i)
  {
    topology_HL_code[i] = HLTopologyCodes[i];
  }
}

// ClearEvents -- clears all events from event vector
void AnaSample::ClearEvents() { m_mc_events.clear(); }

// GetN -- get number of events stored
int AnaSample::GetN() const { return m_mc_events.size(); }

AnaEvent* AnaSample::GetEvent(int evnum)
{
  if(m_mc_events.empty())
  {
    LogError << "In AnaSample::GetEvent()" << std::endl;
    LogError << "No events are found in " << m_name << " sample." << std::endl;
    return nullptr;
  }
  else if(evnum >= m_mc_events.size())
  {
    LogError << "In AnaSample::GetEvent()" << std::endl;
    LogError << "Event number out of bounds in " << m_name << " sample." << std::endl;
    return nullptr;
  }

  return &m_mc_events.at(evnum);
}

std::vector<AnaEvent>& AnaSample::GetEventList(){
  return m_mc_events;
}

void AnaSample::AddEvent(AnaEvent& event) {
  m_mc_events.emplace_back(event);
  // since default constructor doesn't do it by itself
  m_mc_events.back().HookIntMembers();
  m_mc_events.back().HookFloatMembers();
}

void AnaSample::ResetWeights()
{
  for(auto& event : m_mc_events)
    event.SetEvWght(1.0);
}

void AnaSample::PrintStats() const
{
  double mem_kb = double(sizeof(m_mc_events) * m_mc_events.size()) / 1000.0;
  LogInfo << "Sample " << m_name << " ID = " << m_sample_id << std::endl;
  LogInfo << "Num of events = " << m_mc_events.size() << std::endl;
  LogInfo << "Memory used   = " << mem_kb << " kB." << std::endl;
}

void AnaSample::MakeHistos()
{
  delete m_hpred;
  delete m_hmc;
  delete m_hmc_true;
  delete m_hsig;

  std::string nameBuffer;

  nameBuffer = Form("%s_pred_recD1D2", m_name.c_str());
  m_hpred     = new TH1D(nameBuffer.c_str(), nameBuffer.c_str(), m_nbins, 0, m_nbins);
  nameBuffer = Form("%s_mc_recD1D2", m_name.c_str());
  m_hmc       = new TH1D(nameBuffer.c_str(), nameBuffer.c_str(), m_nbins, 0, m_nbins);
  nameBuffer = Form("%s_mc_trueD1D2", m_name.c_str());
  m_hmc_true  = new TH1D(nameBuffer.c_str(), nameBuffer.c_str(), m_nbins, 0, m_nbins);
  nameBuffer = Form("%s_mc_trueSignal", m_name.c_str());
  m_hsig      = new TH1D(nameBuffer.c_str(), nameBuffer.c_str(), m_nbins, 0, m_nbins);

  for(auto& histThread: _histThreadHandlers_){
    // histThread is a map<TH1D*, TH1D*> -> for example  histThread[m_hpred] = m_hpred for a given thread.
    for(auto& histList : histThread){
      delete histList;
    }
  }

  _histThreadHandlers_.clear();
  _histThreadHandlers_.resize(0);
  debugThreadBools.clear();
  debugThreadBools.resize(0);

  if(GlobalVariables::getNbThreads() > 1){
    for( int iThread = 0 ; iThread < GlobalVariables::getNbThreads() ; iThread++ ){
      _histThreadHandlers_.emplace_back();
      debugThreadBools.push_back(false);
      _histThreadHandlers_.back().resize(4);
      _histThreadHandlers_.back()[0] = (TH1D*) m_hpred->Clone();
      _histThreadHandlers_.back()[1] = (TH1D*) m_hmc->Clone();
      _histThreadHandlers_.back()[2] = (TH1D*) m_hmc_true->Clone();
      _histThreadHandlers_.back()[3] = (TH1D*) m_hsig->Clone();
    }
  }


  m_hpred->SetDirectory(nullptr);
  m_hmc->SetDirectory(nullptr);
  m_hmc_true->SetDirectory(nullptr);
  m_hsig->SetDirectory(nullptr);

  LogInfo << m_nbins << " bins inside MakeHistos()." << std::endl;
}

void AnaSample::SetData(TObject* hdata)
{
  delete m_hdata;
  m_hdata = (TH1D*) hdata->Clone(Form("%s_data", m_name.c_str()));
  m_hdata->SetDirectory(nullptr);
}

int AnaSample::GetBinIndex(const std::vector<double>& eventVarList_) const
{
  if( eventVarList_.size() !=  m_fit_phase_space.size() ){
    LogError << "The size of the event var list does not match the dimension of the fit binning." << std::endl;
    throw std::logic_error("The size of the event var list does not match the dimension of the fit binning.");
  }

  for( size_t iBin = 0 ; iBin < m_bin_edges.size() ; iBin++ ){
    if( m_bin_edges.at(iBin).isInBin(eventVarList_) ){
      return iBin;
    }
  }
  return -1;
}
int AnaSample::GetBinIndex(AnaEvent* event_) const
{
  std::vector<double> eventVarBuffer(m_fit_phase_space.size(),0);
  for( size_t iVar = 0 ; iVar < m_fit_phase_space.size() ; iVar++ ){
    eventVarBuffer[iVar] = double(event_->GetEventVarFloat(m_fit_phase_space[iVar]));
  }
  return GetBinIndex(eventVarBuffer);
}

void AnaSample::FillEventHist(int datatype, bool stat_fluc){

  this->FillEventHist(DataType(datatype), stat_fluc);

}

void AnaSample::FillEventHist(DataType datatype, bool stat_fluc){

  LogWarning << "Filling histograms of sample: " << this->GetName() << std::endl;

  this->FillMcHistograms(); // single thread

  // OLD FILL METHOD
//    for(std::size_t iEvent = 0; iEvent < m_mc_events.size(); ++iEvent){
//
//        double D1_rec  = m_mc_events[iEvent].GetRecoD1();
//        double D2_rec  = m_mc_events[iEvent].GetRecoD2();
//        double D1_true = m_mc_events[iEvent].GetTrueD1();
//        double D2_true = m_mc_events[iEvent].GetTrueD2();
////        double wght    = datatype >= 0 ? m_mc_events[iEvent].GetEvWght() : m_mc_events[iEvent].GetEvWghtMC();
//        double wght    = m_mc_events[iEvent].GetEvWght();
//
//        int anybin_index_rec  = GetBinIndex(D1_rec, D2_rec);
//        int anybin_index_true = GetBinIndex(D1_true, D2_true);
//
//        m_hpred->Fill(anybin_index_rec + 0.5, wght);
//        m_hmc->Fill(anybin_index_rec + 0.5, wght);
//        m_hmc_true->Fill(anybin_index_true + 0.5, wght);
//
//        if(m_mc_events[iEvent].isSignalEvent())
//            m_hsig->Fill(anybin_index_true + 0.5, wght);
//    }

  if(datatype == kMC or datatype == kReset) {
    return;
  }
  else if( datatype == kAsimov or datatype == kExternal or datatype == kData ) {

    SetDataType(datatype);
    SetData(m_hpred);
    m_hdata->Reset();

    if(m_data_events.empty()){
      FillDataEventList();
    }

    int kinematicBinIndex;
    for( size_t iEvent = 0 ; iEvent < m_data_events.size() ; iEvent++ ){
      kinematicBinIndex = this->GetBinIndex( &m_data_events[iEvent] );
      if(kinematicBinIndex != -1) {
        m_hdata->Fill(kinematicBinIndex + 0.5, m_data_events[iEvent].GetEvWght());
      }
    }

    if( datatype == kAsimov ){
      m_hdata->Scale( m_norm ); // should be normalized the same way as MC
    }

    if( (datatype == kAsimov or datatype == kExternal) and stat_fluc ) {
      LogInfo << "Applying statistical fluctuations on data..." << std::endl;
      for(unsigned int i = 1; i <= m_hdata->GetNbinsX(); ++i) {
        double val = gRandom->Poisson(m_hdata->GetBinContent(i));
        m_hdata->SetBinContent(i, val);
      }
    }

  }
  else {
    LogWarning << "In AnaSample::FillEventHist()\n"
               << "Invalid data type to fill histograms!\n";
  }

  LogInfo << GET_VAR_NAME_VALUE(this->CalcLLH()) << std::endl;

}

void AnaSample::FillMcHistograms(int iThread_){

  bool isMultiThreaded = (iThread_ != -1);
  bool foundNan = false;
  TH1D* histPredPtr = nullptr;
  TH1D* histMcPtr = nullptr;
  TH1D* histMcTruePtr = nullptr;
  TH1D* histSigPtr = nullptr;
  AnaEvent* anaEventPtr = nullptr;

  if(isMultiThreaded){
    histPredPtr   = _histThreadHandlers_[iThread_][0];
    histMcPtr     = _histThreadHandlers_[iThread_][1];
    histMcTruePtr = _histThreadHandlers_[iThread_][2];
    histSigPtr    = _histThreadHandlers_[iThread_][3];
  }
  else{
    histPredPtr   = m_hpred;
    histMcPtr     = m_hmc;
    histMcTruePtr = m_hmc_true;
    histSigPtr    = m_hsig;
  }

  histPredPtr->Reset("ICESM");
  histMcPtr->Reset("ICESM");
  histMcTruePtr->Reset("ICESM");
  histSigPtr->Reset("ICESM");

//    GenericToolbox::resetHistogram(histPredPtr);
//    GenericToolbox::resetHistogram(histMcPtr);
//    GenericToolbox::resetHistogram(histMcTruePtr);
//    GenericToolbox::resetHistogram(histSigPtr);

  for( size_t iMcEvent = 0 ; iMcEvent < m_mc_events.size() ; iMcEvent++ ){

    if(isMultiThreaded){
      if( iMcEvent % GlobalVariables::getNbThreads() != iThread_){
        continue;
      }
    }

    anaEventPtr = &m_mc_events.at(iMcEvent);

    // Events are not supposed to move for one bin to another with the current implementation
    // So the bin index shall be computed once

    // Wait if another thread is editing the binning
    while(anaEventPtr->GetIsBeingEdited());
    anaEventPtr->SetIsBeingEdited(true);
    if(anaEventPtr->GetTrueBinIndex() == -1){
      anaEventPtr->SetTrueBinIndex(this->GetBinIndex(anaEventPtr) );
    }
    if(anaEventPtr->GetRecoBinIndex() == -1){
      anaEventPtr->SetRecoBinIndex(this->GetBinIndex(anaEventPtr) );
    }
    anaEventPtr->SetIsBeingEdited(false);

    if(anaEventPtr->GetEvWght() != anaEventPtr->GetEvWght()){
      GlobalVariables::getThreadMutex().lock();
      if(not foundNan) foundNan = true;
      LogError << this->GetName() << ": Event#" << anaEventPtr->GetEvId()
               << " -> " << GET_VAR_NAME_VALUE(anaEventPtr->GetEvWght()) << std::endl;
      GlobalVariables::getThreadMutex().unlock();
    }

    histPredPtr->Fill(anaEventPtr->GetRecoBinIndex() + 0.5, anaEventPtr->GetEvWght());
    histMcPtr->Fill(anaEventPtr->GetRecoBinIndex() + 0.5, anaEventPtr->GetEvWght());
    histMcTruePtr->Fill(anaEventPtr->GetTrueBinIndex() + 0.5, anaEventPtr->GetEvWght());

    if(anaEventPtr->isSignalEvent()){
      histSigPtr->Fill(anaEventPtr->GetTrueBinIndex() + 0.5, anaEventPtr->GetEvWght());
    }

  }

  if(foundNan){
    throw std::runtime_error("foundNan");
  }

  if(not isMultiThreaded){
    // DONT APPLY NORMALISATION ON THREAD HISTOGRAMS
    // It leaves computational noise after stacking already normalized histograms
    // Computational noise appears in the chi2 at approx 1E-13
    m_hpred->Scale(m_norm);
    m_hmc->Scale(m_norm);
    m_hsig->Scale(m_norm);
  }

}
void AnaSample::MergeMcHistogramsThread()
{

//    LogDebug << "Merging histogram threads of sample: " << this->GetName() << std::endl;

  for(size_t iThread = 0; iThread < _histThreadHandlers_.size(); iThread++){
    debugThreadBools[iThread] = false;
  }

  m_hpred->Reset("ICESM");
  m_hmc->Reset("ICESM");
  m_hmc_true->Reset("ICESM");
  m_hsig->Reset("ICESM");

//    GenericToolbox::resetHistogram(m_hpred);
//    GenericToolbox::resetHistogram(m_hmc);
//    GenericToolbox::resetHistogram(m_hmc_true);
//    GenericToolbox::resetHistogram(m_hsig);

  for(size_t iThread = 0; iThread < _histThreadHandlers_.size(); iThread++)
  {
//        LogAlert << "FILLING " << GET_VAR_NAME_VALUE(iThread) << std::endl;
//        for(auto& histPair: _histThreadHandlers_[iThread]){
//            for( int iBin = 0 ; iBin <= histPair.first->GetNbinsX()+1 ; iBin++ ){
//                histPair.first->SetBinContent(
//                    iBin,
//                    histPair.first->GetBinContent(iBin)
//                    + histPair.second->GetBinContent(iBin)
//                    );
//            }
//        }

    for( int iBin = 0 ; iBin <= m_hpred->GetNbinsX()+1 ; iBin++ ){
      m_hpred->SetBinContent(iBin, m_hpred->GetBinContent(iBin) + _histThreadHandlers_[iThread][0]->GetBinContent(iBin));
    }
    for( int iBin = 0 ; iBin <= m_hmc->GetNbinsX()+1 ; iBin++ ){
      m_hmc->SetBinContent(iBin, m_hmc->GetBinContent(iBin) + _histThreadHandlers_[iThread][1]->GetBinContent(iBin));
    }
    for( int iBin = 0 ; iBin <= m_hmc_true->GetNbinsX()+1 ; iBin++ ){
      m_hmc_true->SetBinContent(iBin, m_hmc_true->GetBinContent(iBin) + _histThreadHandlers_[iThread][2]->GetBinContent(iBin));
    }
    for( int iBin = 0 ; iBin <= m_hsig->GetNbinsX()+1 ; iBin++ ){
      m_hsig->SetBinContent(iBin, m_hsig->GetBinContent(iBin) + _histThreadHandlers_[iThread][3]->GetBinContent(iBin));
    }

//        m_hpred->Add(_histThreadHandlers_[iThread][0]);
//        m_hmc->Add(_histThreadHandlers_[iThread][1]);
//        m_hmc_true->Add(_histThreadHandlers_[iThread][2]);
//        m_hsig->Add(_histThreadHandlers_[iThread][3]);

    debugThreadBools[iThread] = true;
  }

  m_hpred->Scale(m_norm);
  m_hmc->Scale(m_norm);
  m_hsig->Scale(m_norm);

}

void AnaSample::FillDataEventList(){

  if(not m_data_events.empty()){
    LogFatal << "Data event list not empty." << std::endl;
    throw std::logic_error("Data event list not empty.");
  }

  if(m_hdata_type == kReset){
    LogFatal << "m_hdata_type not set." << std::endl;
    throw std::logic_error("m_hdata_type not set.");
  }

  if(m_hdata_type == kAsimov){
    LogDebug << "Filling data events (Asimov)..." << std::endl;
    m_data_events.clear();
    m_data_events.resize(m_mc_events.size());
    for( size_t iEvent = 0 ; iEvent < m_mc_events.size() ; iEvent++ ){
      m_data_events.at(iEvent) = (m_mc_events[iEvent]); // Copy
      m_data_events.at(iEvent).HookIntMembers();
      m_data_events.at(iEvent).HookFloatMembers();
      m_data_events.at(iEvent).SetAnaEventType(AnaEventType::AnaEventType::DATA);
    }
  }
  else{
    if( m_data_tree == nullptr ){
      LogError << "Can't fill data events since TTree is not specified" << std::endl;
      throw std::logic_error("m_data_tree == nullptr");
    }

    LogDebug << "Filling data events..." << std::endl;
    Long64_t nbEntries = m_data_tree->GetEntries();

    int cut_branch;
    m_data_tree->SetBranchAddress("cut_branch", &cut_branch);

    m_additional_cuts_formulae->SetTree(m_data_tree);
    m_data_tree->SetNotify(m_additional_cuts_formulae); // This is needed only for TChain.

    bool skipEvent;
    for(Long64_t iEntry = 0 ; iEntry < nbEntries; iEntry++){

      m_data_tree->GetEntry(iEntry);
      skipEvent = false;

      if(cut_branch != m_sample_id){
        skipEvent = true;
      }
      if(skipEvent) continue;

      for( Int_t iInstance = 0 ; iInstance < m_additional_cuts_formulae->GetNdata() ; iInstance++ ) {
        if ( m_additional_cuts_formulae->EvalInstance(iInstance) == 0 ) {
          skipEvent = true;
          break;
        }
      }
      if(skipEvent) continue;

      m_data_events.emplace_back(AnaEvent(AnaEventType::AnaEventType::DATA));

      m_data_events.back().SetEventId(iEntry);
      m_data_events.back().DumpTreeEntryContent(m_data_tree);

    }
  }

}

void AnaSample::PropagateMcWeightsToAsimovDataEvents(){

  if(m_hdata_type != DataType::kAsimov){
    LogFatal << __METHOD_NAME__ << " can only be used with DataType::kAsimov." << std::endl;
    throw std::logic_error("m_hdata_type != DataType::kAsimov");
  }
  if(m_data_events.empty()){
    LogFatal << "Data event list empty." << std::endl;
    throw std::logic_error("Data event list empty.");
  }

  for( size_t iEvent = 0 ; iEvent < m_mc_events.size() ; iEvent++ ){
    m_data_events.at(iEvent).SetEvWght( m_mc_events.at(iEvent).GetEvWght() );
  }

}

void AnaSample::SetLLHFunction(const std::string& func_name)
{
  if(m_llh != nullptr)
    delete m_llh;

  if(func_name.empty())
  {
    LogInfo << "Likelihood function name empty. Setting to Poisson by default." << std::endl;
    m_llh = new PoissonLLH;
  }
  else if(func_name == "Poisson")
  {
    LogInfo << "Setting likelihood function to Poisson." << std::endl;
    m_llh = new PoissonLLH;
  }
  else if(func_name == "Effective")
  {
    LogInfo << "Setting likelihood function to Tianlu's effective likelihood." << std::endl;
    m_llh = new EffLLH;
  }
  else if(func_name == "Barlow")
  {
    LogInfo << "Setting likelihood function to Barlow-Beeston." << std::endl;
    m_llh = new BarlowLLH;
  }
}
void AnaSample::SetDataType(DataType dataType_){
  if(dataType_ == DataType::kReset or dataType_ == DataType::kMC){
    LogFatal << "Can't set data type to either kReset or kMC" << std::endl;
    throw std::logic_error("Invalid data type.");
  }
  m_hdata_type = dataType_;
}

// Compute the statistical chi2 contribution from this sample based on the current m_hpred and m_hdata histograms:
double AnaSample::CalcLLH() const
{
  if(m_hdata == nullptr)
  {
    LogError << "In AnaSample::CalcLLH()\n"
             << "Need to define data histogram." << std::endl;
    throw std::runtime_error("m_hdata is a nullptr.");
  }

  // Number of reco bins as specified in binning file:
  const unsigned int nbins = m_hpred->GetNbinsX();

  // Array of the number of expected events in each bin:
  double* exp_w  = m_hpred->GetArray();

  // Array of sum of squares of weights in each bin for the expected events:
  double* exp_w2 = m_hpred->GetSumw2()->GetArray();

  // Array of the number of measured events in data in each bin:
  double* data   = m_hdata->GetArray();

  // generateFormula chi2 variable which will be updated below and then returned:
  double chi2 = 0.;

  // Loop over all bins:
  double buff;
  for(unsigned int i = 1; i <= nbins; ++i)
  {
    // Compute chi2 contribution from current bin (done in Likelihoods.hh):
    buff = (*m_llh)(exp_w[i], exp_w2[i], data[i]);
    chi2 += buff;
  }

  // Sum of the chi2 contributions for each bin is returned:
  return chi2;
}
double AnaSample::CalcChi2() const
{
  if(m_hdata == nullptr)
  {
    LogError << "In AnaSample::CalcChi2()\n"
             << "Need to define data histogram." << std::endl;
    return 0.0;
  }

  int nbins = m_hpred->GetNbinsX();
  if(nbins != m_hdata->GetNbinsX())
  {
    LogError << "In AnaSample::CalcChi2()\n"
             << "Binning mismatch between data and mc.\n"
             << "MC bins: " << nbins << ", Data bins: " << m_hdata->GetNbinsX()
             << std::endl;
    return 0.0;
  }

  double chi2 = 0.0;
  for(int j = 1; j <= nbins; ++j)
  {
    double obs = m_hdata->GetBinContent(j);
    double exp = m_hpred->GetBinContent(j);
    if(exp > 0.0)
    {
      // added when external fake datasets (you cannot reweight when simply 0)
      // this didn't happen when all from same MC since if exp=0 then obs =0

      chi2 += 2 * (exp - obs);
      if(obs > 0.0)
        chi2 += 2 * obs * TMath::Log(obs / exp);

      if(chi2 < 0.0)
      {
#ifndef NDEBUG
        LogWarning << "In AnaSample::CalcChi2()\n"
                   << "Stat chi2 is less than 0: " << chi2 << ", setting to 0."
                   << std::endl;
        LogWarning << "exp and obs is: " << exp << " and " << obs << "."
                   << std::endl;
#endif
        chi2 = 0.0;
      }
    }
  }

  if(chi2 != chi2)
  {
    LogWarning << "In AnaSample::CalcChi2()\n"
               << "Stat chi2 is nan, setting to 0." << std::endl;
    chi2 = 0.0;
  }

  return chi2;
}
double AnaSample::CalcEffLLH() const
{
  const unsigned int nbins = m_hpred->GetNbinsX();
  double* exp_w  = m_hpred->GetArray();
  double* exp_w2 = m_hpred->GetSumw2()->GetArray();
  double* data   = m_hdata->GetArray();

  //std::cout << m_name << std::endl;

  double llh_eff = 0.0;
  for(unsigned int i = 1; i <= nbins; ++i)
  {
    if(exp_w[i] <= 0.0)
      continue;

    const double b = exp_w[i] / exp_w2[i];
    const double a = (exp_w[i] * b) + 1.0;
    const double k = data[i];

    //std::cout << "--------------" << std::endl;
    //std::cout << "i  : " << i << std::endl;
    //std::cout << "w  : " << exp_w[i] << std::endl;
    //std::cout << "w2 : " << exp_w2[i] << std::endl;
    //std::cout << "a  : " << a << std::endl;
    //std::cout << "b  : " << b << std::endl;
    //std::cout << "k  : " << data[i] << std::endl;

    llh_eff += a * std::log(b) + std::lgamma(k+a) - std::lgamma(k+1) - ((k+a) * std::log1p(b)) - std::lgamma(a);
  }

  return -2 * llh_eff;
}

void AnaSample::Write(TDirectory* dirout, const std::string& bsname, int fititer)
{
  dirout->cd();
  m_hpred->Write(Form("%s_pred", bsname.c_str()));
  m_hmc_true->Write(Form("%s_true", bsname.c_str()));
  if(fititer == 0)
  {
    m_hmc->Write(Form("%s_mc", bsname.c_str()));
    if(m_hdata != nullptr)
      m_hdata->Write(Form("%s_data", bsname.c_str()));
  }
}

void AnaSample::GetSampleBreakdown(TDirectory* dirout, const std::string& tag,
                                   const std::vector<std::string>& topology, bool save)
{
  const int ntopology = topology.size();
  int compos[ntopology];
  std::vector<TH1D> hAnybin_rec;
  std::vector<TH1D> hAnybin_true;

  for(int i = 0; i < ntopology; ++i)
  {
    compos[i] = 0;
    hAnybin_rec.emplace_back(
      TH1D(Form("%s_Anybins_rec_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
           Form("%s_Anybins_rec_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
           m_nbins, 0, m_nbins));
    hAnybin_rec[i].SetDirectory(0);
    hAnybin_rec[i].GetXaxis()->SetTitle("Bin Index");

    hAnybin_true.emplace_back(
      TH1D(Form("%s_Anybins_true_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
           Form("%s_Anybins_true_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
           m_nbins, 0, m_nbins));
    hAnybin_true[i].SetDirectory(0);
    hAnybin_true[i].GetXaxis()->SetTitle("Bin Index");
  }

  int Ntot = GetN();

  // Loop over all events:
  for(std::size_t i = 0; i < m_mc_events.size(); ++i)
  {
    double D1_rec    = m_mc_events[i].GetRecoD1();
    double D2_rec    = m_mc_events[i].GetRecoD2();
    double D1_true   = m_mc_events[i].GetTrueD1();
    double D2_true   = m_mc_events[i].GetTrueD2();
    double wght      = m_mc_events[i].GetEvWght();
    int evt_topology = m_mc_events[i].GetTopology();

    compos[evt_topology]++;
    int anybin_index_rec  = GetBinIndex(&m_mc_events[i]);
    int anybin_index_true = GetBinIndex(&m_mc_events[i]);

    // Fill histogram for this topolgy with the current event:
    hAnybin_rec[topology_HL_code[evt_topology]].Fill(anybin_index_rec + 0.5, wght);
    hAnybin_true[topology_HL_code[evt_topology]].Fill(anybin_index_true + 0.5, wght);
  }

  dirout->cd();
  for(int i = 0; i < ntopology; ++i)
  {
    hAnybin_true[i].Scale(m_norm);
    hAnybin_rec[i].Scale(m_norm);

    if(save == true)
    {
      hAnybin_true[i].Write();
      hAnybin_rec[i].Write();
    }
  }

  LogInfo << "GetSampleBreakdown()\n"
          << "============ Sample " << m_name << " ===========" << std::endl;

  for(int j = 0; j < ntopology; ++j)
  {
    std::cout << std::setw(10) << topology[j] << std::setw(5) << j << std::setw(5) << compos[j]
              << std::setw(10) << ((1.0 * compos[j]) / Ntot) * 100.0 << "%" << std::endl;
  }

  std::cout << std::setw(10) << "Total" << std::setw(5) << " " << std::setw(5) << Ntot
            << std::setw(10) << "100.00%" << std::endl;
}


void AnaSample::SaveMcEventsSnapshot() {

//    LogWarning << "Saving MC events in a snapshot..." << std::endl;

  m_mc_events_snap.clear();
  for( size_t iEvent = 0 ; iEvent < m_mc_events.size() ; iEvent++ ){
    m_mc_events_snap.emplace_back(m_mc_events.at(iEvent)); // COPY
    m_mc_events_snap.back().HookIntMembers();
    m_mc_events_snap.back().HookFloatMembers();
  }

//    LogInfo << "MC events saved." << std::endl;

}
void AnaSample::SaveHistogramsSnapshot(){

//    LogWarning << "Saving MC histograms in a snapshot..." << std::endl;

  m_hmc_true_snap = (TH1D*) m_hmc_true->Clone();
  m_hmc_snap = (TH1D*) m_hmc->Clone();
  m_hpred_snap = (TH1D*) m_hpred->Clone();
  m_hdata_snap = (TH1D*) m_hdata->Clone();
  m_hsig_snap = (TH1D*) m_hsig->Clone();

  _histThreadHandlersSnap_.clear();
  _histThreadHandlersSnap_.resize(0);
  for( size_t iThread = 0 ; iThread < _histThreadHandlers_.size() ; iThread++){
    _histThreadHandlersSnap_.emplace_back();
    _histThreadHandlersSnap_[iThread].resize(_histThreadHandlers_[iThread].size());
    _histThreadHandlersSnap_[iThread][0] = (TH1D*) _histThreadHandlers_[iThread][0]->Clone();
    _histThreadHandlersSnap_[iThread][1] = (TH1D*) _histThreadHandlers_[iThread][1]->Clone();
    _histThreadHandlersSnap_[iThread][2] = (TH1D*) _histThreadHandlers_[iThread][2]->Clone();
    _histThreadHandlersSnap_[iThread][3] = (TH1D*) _histThreadHandlers_[iThread][3]->Clone();
  }

//    LogInfo << "MC histograms saved." << std::endl;

}

void AnaSample::CompareHistogramsWithSnapshot(){

  LogWarning << "Comparing histograms with the snapshot..." << std::endl;

  std::function<bool(TH1D*, TH1D*, bool)> histAreSame = [](TH1D* hSnap_, TH1D* hCurrent_, bool isSilent_ = false){

    bool isSame = true;
    for( int iBin = 0 ; iBin <= hCurrent_->GetNbinsX()+1 ; iBin++ ){
      if(hSnap_->GetBinContent(iBin) != hCurrent_->GetBinContent(iBin)){
        if(not isSilent_){
          LogAlert << hCurrent_->GetName() << " (" << GET_VAR_NAME_VALUE(iBin) << "): ";
          LogAlert << hSnap_->GetBinContent(iBin);
          LogAlert << (hSnap_->GetBinContent(iBin) > hCurrent_->GetBinContent(iBin) ? " > ": " < ");
          LogAlert << hCurrent_->GetBinContent(iBin) << std::endl;
        }
        isSame = false;
      }
    }
    return isSame;

  };

  histAreSame(m_hmc_true_snap, m_hmc_true, false);
  histAreSame(m_hmc_snap, m_hmc, false);
  histAreSame(m_hpred_snap, m_hpred, false);
  histAreSame(m_hdata_snap, m_hdata, false);
  histAreSame(m_hsig_snap, m_hsig, false);

  bool threadHistAreSame = true;
  for( size_t iThread = 0 ; iThread < _histThreadHandlers_.size() ; iThread++){
    LogInfo << GET_VAR_NAME_VALUE(iThread) << std::endl;
    if(
      not histAreSame(_histThreadHandlersSnap_[iThread][0], _histThreadHandlers_[iThread][0], false)
      or not histAreSame(_histThreadHandlersSnap_[iThread][1], _histThreadHandlers_[iThread][1], false)
      or not histAreSame(_histThreadHandlersSnap_[iThread][2], _histThreadHandlers_[iThread][2], false)
      or not histAreSame(_histThreadHandlersSnap_[iThread][3], _histThreadHandlers_[iThread][3], false)
      ){
      threadHistAreSame = false;
    };
  }

  if(threadHistAreSame){
    // Checking who's missing?
    TH1D* temp_pred = (TH1D*) m_hpred->Clone();
    TH1D* temp_mc = (TH1D*) m_hmc->Clone();
    TH1D* temp_mc_true = (TH1D*) m_hmc_true->Clone();
    TH1D* temp_sig = (TH1D*) m_hsig->Clone();


    int missingThreadId = -1;
    for( size_t iMissingThread = 0 ; iMissingThread < _histThreadHandlers_.size() ; iMissingThread++){

      temp_pred->Reset("ICESM");
      temp_mc->Reset("ICESM");
      temp_mc_true->Reset("ICESM");
      temp_sig->Reset("ICESM");

      for( size_t iThread = 0 ; iThread < _histThreadHandlers_.size() ; iThread++){

        if(iMissingThread == iThread){
          continue;
        }

        for( int iBin = 0 ; iBin <= temp_pred->GetNbinsX()+1 ; iBin++ ){
          temp_pred->SetBinContent(
            iBin,
            temp_pred->GetBinContent(iBin)
            + _histThreadHandlers_[iThread][0]->GetBinContent(iBin)
          );
        }

        for( int iBin = 0 ; iBin <= temp_mc->GetNbinsX()+1 ; iBin++ ){
          temp_mc->SetBinContent(
            iBin,
            temp_mc->GetBinContent(iBin)
            + _histThreadHandlers_[iThread][1]->GetBinContent(iBin)
          );
        }

        for( int iBin = 0 ; iBin <= temp_mc_true->GetNbinsX()+1 ; iBin++ ){
          temp_mc_true->SetBinContent(
            iBin,
            temp_mc_true->GetBinContent(iBin)
            + _histThreadHandlers_[iThread][2]->GetBinContent(iBin)
          );
        }

        for( int iBin = 0 ; iBin <= temp_sig->GetNbinsX()+1 ; iBin++ ){
          temp_sig->SetBinContent(
            iBin,
            temp_sig->GetBinContent(iBin)
            + _histThreadHandlers_[iThread][3]->GetBinContent(iBin)
          );
        }

      }

      temp_pred->Scale(m_norm);
      temp_mc->Scale(m_norm);
      temp_sig->Scale(m_norm);

      // Check if the current hist are the same as the one we just built when intentionnaly skipping one...
      if(    histAreSame(temp_mc_true, m_hmc_true, true)
             or histAreSame(temp_mc, m_hmc, true)
             or histAreSame(temp_pred, m_hpred, true)
             or histAreSame(temp_sig, m_hsig, true)
        ){
        LogError << GET_VAR_NAME_VALUE(iMissingThread) << std::endl;
        break;
      }

    }

  }

  GenericToolbox::printVector(debugThreadBools);


  LogInfo << "Format: snapshot -> current" << std::endl;
  LogInfo << "Comparison with the snapshot ended." << std::endl;

}
void AnaSample::CompareMcEventsWeightsWithSnapshot()
{

  LogWarning << "Comparing MC event weights with the snapshot..." << std::endl;
  bool isIdentical = true;
  for( size_t iEvent = 0 ; iEvent < m_mc_events.size() ; iEvent++ ){
    if(m_mc_events_snap[iEvent].GetEvWght() != m_mc_events[iEvent].GetEvWght()){
      LogAlert << GET_VAR_NAME_VALUE(iEvent) << ": "
               << m_mc_events_snap[iEvent].GetEvWght() << " -> "
               << m_mc_events[iEvent].GetEvWght() << std::endl;
    }
  }
  LogInfo << "Format: snapshot-weight -> current-weight" << std::endl;
  LogInfo << "Comparison with the snapshot ended." << std::endl;

}

