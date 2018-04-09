//////////////////////////////////////////////////////////
//
//  A class for event samples for for Any analysis
//
//
//
//  Created: Nov 17 2015   
//
//////////////////////////////////////////////////////////
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <assert.h>

#include "AnySample.hh"

using namespace std;

// ctor
AnySample::AnySample(int sample_id, string name, 
           std::vector<std::pair <double,double> > v_d1edges, 
           std::vector<std::pair <double,double> > v_d2edges, 
           TTree* data, bool isBuffer, bool isEmpty, bool isIngrid)
{  
  m_sampleid = sample_id; //unique id
  m_name     = name;      //some comprehensible name
  //cout<<"NEW SAMPLE with name "<<name<<" sample id "<<sample_id<<endl;
  m_data_tree = data;
  m_D1edges = v_d1edges;
  m_D2edges = v_d2edges;
  m_empty = isEmpty;
  m_BufferBin = isBuffer;
  m_ingrid = isIngrid;

  for(int i=0;i<v_d1edges.size(); i++){
    cout<<v_d2edges[i].first<<"  "<<v_d2edges[i].second<<"  "<<v_d1edges[i].first<<"  "<<v_d1edges[i].second<<endl;
  }

  //Default binning choices
  nbins_enu = 28;
  bins_enu = new double[nbins_enu + 1];
  for(int i=0;i<=nbins_enu;i++)
    {
      if(i<10) bins_enu[i] = i*0.1;
      else if(i>=10 && i<18) bins_enu[i] = 1.0 + (i-10)*0.2;
      else if(i==18) bins_enu[i] = 2.7;
      else if(i>=19 && i<23) bins_enu[i] = 3.0 + (i-19)*0.5;
      else if(i>=23) bins_enu[i] = 5.0 + (i-23)*1.0;
    }

  // New way of choosing default binning based on input binning, 
  // relies on later bins always having larger values than earlier bins!

  int nbins_temp = v_d1edges.size();
  vector<double> bins_d1_vector;
  for(int i=0;i<nbins_temp;i++){
    if (bins_d1_vector.size()==0) bins_d1_vector.push_back(v_d1edges[i].first);
    else{
      if (bins_d1_vector.back()!=v_d1edges[i].first){
        if (!m_BufferBin && i>0 && v_d1edges[i].first<v_d1edges[i-1].first) bins_d1_vector.push_back(v_d1edges[i-1].second);
        bins_d1_vector.push_back(v_d1edges[i].first);
      }
    }
  }
  bins_d1_vector.push_back(v_d1edges[nbins_temp-1].second);
//
  nbins_D1 = bins_d1_vector.size()-1;
  cout << "There are " << nbins_D1 << " d1 bins" << endl;

  bins_D1 = new double[nbins_D1 + 1];
  bool dqPlotBinsSet=false;
  for(int i=0;i<=nbins_D1;i++){
    if(i!=0){
      if(bins_d1_vector[i-1]>bins_d1_vector[i] && !dqPlotBinsSet){
        nbinsD1_toPlot=i-1;
        dqPlotBinsSet=true;
      }
    }
    bins_D1[i]  =  bins_d1_vector[i];
    cout << "bins_D1 " << i << " is " << bins_d1_vector[i] << endl;
  }
  if(!dqPlotBinsSet) nbinsD1_toPlot=nbins_D1;
  bins_D1toPlot = new double[nbinsD1_toPlot + 1];
  cout << "There are " << nbinsD1_toPlot << " d1 bins that will be used for plotting" << endl;

  for(int i=0;i<=nbinsD1_toPlot;i++){
    bins_D1toPlot[i]  =  bins_d1_vector[i];
    cout << "bins_D1toPlot " << i << " is " << bins_d1_vector[i] << endl;
  }

  nbins_temp = v_d1edges.size();
  vector<double> bins_d2_vector;
  for(int i=0;i<nbins_temp;i++){
    if (bins_d2_vector.size()==0) bins_d2_vector.push_back(v_d2edges[i].first);
    else{
      if (bins_d2_vector.back()!=v_d2edges[i].first) bins_d2_vector.push_back(v_d2edges[i].first);
    }
  }
  bins_d2_vector.push_back(v_d2edges[nbins_temp-1].second);

  nbins_D2 = bins_d2_vector.size()-1;
  cout << "There are " << nbins_D2 << " d2 bins" << endl;

  bins_D2 = new double[nbins_D2 + 1];

  for(int i=0;i<=nbins_D2;i++){
    bins_D2[i]  =  bins_d2_vector[i];
    cout << "bins_D2 " << i << " is " << bins_d2_vector[i] << endl;
  }


  nAnybins=m_D1edges.size();
  bins_Any = new double[nAnybins+1];
  for (int i=0; i<=nAnybins; i++){
    bins_Any[i]=i;
  }
  cout<<"Any bins defined"<<endl;
  //event distribution histo
  m_hpred = NULL;
  m_hmc   = NULL;
  m_hmc_true = NULL;

  m_sig   = NULL;
  MakeHistos(); //with default binning
  
  cout<<"MakeHistos called"<<endl;
  //data (or toy) histo
  m_hdata = NULL;

  m_norm  = 1.0;
}

// dtor
AnySample::~AnySample()
{
  m_hpred->Delete();
  m_hmc->Delete();
  if(m_hdata != NULL) m_hdata->Delete();
  delete [] bins_D1;
  delete [] bins_D2;
  delete [] bins_enu;
}

// MakeEventHisto
void AnySample::MakeHistos()
{
  if(m_hpred != NULL) m_hpred->Delete();
  m_hpred = new TH1D(Form("%s_pred_recD1D2", m_name.c_str()),
                     Form("%s_pred_recD1D2", m_name.c_str()),
                     nAnybins, bins_Any);
  m_hpred->SetDirectory(0);

  if(m_hmc != NULL) m_hmc->Delete();
  m_hmc = new TH1D(Form("%s_mc_recD1D2", m_name.c_str()),
                   Form("%s_mc_recD1D2", m_name.c_str()),
                   nAnybins, bins_Any);
  m_hmc->SetDirectory(0);

  if(m_hmc_true != NULL) m_hmc_true->Delete();
  m_hmc_true = new TH1D(Form("%s_mc_TrueD1D2", m_name.c_str()),
                   Form("%s_mc_TrueD1D2", m_name.c_str()),
                   nAnybins, bins_Any);
  m_hmc_true->SetDirectory(0);
  cout<<nAnybins<<" bins inside MakeHistos"<<endl;
}

void AnySample::SetData(TObject *hdata)
{
  //clone the data histogram internally
  if(m_hdata != NULL) m_hdata->Delete();
  m_hdata = (TH1D*)hdata->Clone(Form("%s_data", m_name.c_str()));
  m_hdata->SetDirectory(0);
}

// SetD1Binning
void AnySample::SetD1Binning(int nbins, double *bins)
{
  nbins_D1 = nbins;
  delete [] bins_D1;
  bins_D1 = new double[nbins_D1 + 1];
  for(int i=0;i<=nbins_D1;i++) bins_D1[i] = bins[i];
}

// SetD2Binning
void AnySample::SetD2Binning(int nbins, double *bins)
{
  nbins_D2 = nbins;
  delete [] bins_D2;
  bins_D2 = new double[nbins_D2 + 1];
  for(int i=0;i<=nbins_D2;i++) bins_D2[i] = bins[i];
}

// SetEnuBinning
void AnySample::SetEnuBinning(int nbins, double *bins)
{
  nbins_enu = nbins;
  delete [] bins_enu;
  bins_enu = new double[nbins_enu + 1];
  for(int i=0;i<=nbins_enu;i++) bins_enu[i] = bins[i];
}

// FillEventHist
void AnySample::FillEventHisto(int datatype)
{
  if(m_hpred) m_hpred->Reset();
  if(m_hmc) m_hmc->Reset();
  if(m_empty) return; // This sample will have no events
  for(size_t i=0;i<m_events.size();i++)
  {
    double D1_rec   = m_events[i].GetRecD1trk();
    double D2_rec = m_events[i].GetRecD2trk();
    double D1_true   = m_events[i].GetTrueD1trk();
    double D2_true = m_events[i].GetTrueD2trk();
    double wght      = m_events[i].GetEvWght();
    for(int j=0; j<nAnybins; j++){
      if((D1_rec > m_D1edges[j].first) && (D1_rec  < m_D1edges[j].second)  &&
         (D2_rec  > m_D2edges[j].first) && (D2_rec  < m_D2edges[j].second)){
        m_hpred->Fill(j+0.5,wght);
        m_hmc->Fill(j+0.5,wght);
        break;
      }
    }
    for(int j=0; j<nAnybins; j++){
      if((D1_true > m_D1edges[j].first) && (D1_true  < m_D1edges[j].second)  &&
         (D2_true  > m_D2edges[j].first) && (D2_true  < m_D2edges[j].second)){
        m_hmc_true->Fill(j+0.5,wght);
        break;
      }
    }
  }
  m_hpred->Scale(m_norm);
  m_hmc->Scale(m_norm);
  
  //data without stat variation: useful when nuisance parameters 
  //varied in the toys
  if(datatype==1) 
  {
    SetData(m_hpred);
    m_hdata->Reset();
    for(int j=1;j<=m_hpred->GetNbinsX();j++)
    {
      double val = m_hpred->GetBinContent(j);
      //cout<<"bin "<<j<<" entry "<<val<<endl;
      if(val == 0.0) {
        cout<<"AnySample:"<<m_sampleid<<" bin "<<j<<" with 0 entries may cause proble on chi2 computations"<<endl;
        continue;
      }
      m_hdata->SetBinContent(j,val);  //without statistical fluctuations
    }
  }

  //data with statistical variation 
  //(used when no nuisance sampling but nuisances are fitted)
  else if(datatype==3) 
  {
    SetData(m_hpred);
    m_hdata->Reset();
    for(int j=1;j<=m_hpred->GetNbinsX();j++)
    {
      double val = m_hpred->GetBinContent(j);
      //cout<<"bin "<<j<<" entry "<<val<<endl;
      if(val == 0.0) {
        cout<<"AnySample:"<<m_sampleid<<" bin "<<j<<" with 0 entries may cause proble on chi2 computations"<<endl;
        continue;
      }
      double binc = gRandom->Poisson(val);
      m_hdata->SetBinContent(j,binc); //with statistical fluctuations
    }
  }

  //data from external (fake) dataset 
  else if(datatype==2 || datatype==4) {
    SetData(m_hpred);
    m_hdata->Reset();
    //double potD = 57.34;   //in units of 10^19
    //double potMC_genie=384.762;
    //double potMC_genie=389.5; //neut
    //double potMC_genie=380.0; //nuwro
    //double potMC_genie = 57.34; 

    Float_t D1_rec_tree,D2_rec_tree,wght; 
    Int_t topology;
    m_data_tree->SetBranchAddress("cutBranch",&topology);        
    m_data_tree->SetBranchAddress("weight",&wght); 
    m_data_tree->SetBranchAddress("D1Rec",&D1_rec_tree);
    m_data_tree->SetBranchAddress("D2Rec",&D2_rec_tree);
   
    for(size_t i=0;i<m_data_tree->GetEntries();i++){
      m_data_tree->GetEntry(i);
      if(topology != m_sampleid) continue;
      for(int j=0; j<nAnybins; j++){
        if((D1_rec_tree > m_D1edges[j].first) && (D1_rec_tree < m_D1edges[j].second)  &&
           (D2_rec_tree  > m_D2edges[j].first) && (D2_rec_tree  < m_D2edges[j].second)){
          m_hdata->Fill(j+0.5,wght);
          // cout << "Filling data event:" << endl;
          // cout << "D1_rec_tree: " << D1_rec_tree << endl;
          // cout << "D2_rec_tree: " << D2_rec_tree << endl;
          // cout << "Branch: " << topology << endl;
          break;
        }
        else if(j==(nAnybins-1)){
          cout << "Warning: no bin for current data event!" << endl;
          cout << "D1_rec_tree: " << D1_rec_tree << endl;
          cout << "D2_rec_tree: " << D2_rec_tree << endl;
          cout << "Branch: " << topology << endl;
        }
      }  
    }

    cout << "Data histo filled: " << endl;
    m_hdata->Print();

    if(datatype==4) {  //Reweight fake data set
      //m_hdata->Scale(potD/potMC_genie);
      //add MC or data (!!!!) statistical variations also to genie dataset to evaluate genie MC stat uncert
      //DON'T USE FOR REAL DATA!!!!!!!!!!!!
  
      cout << "Warning, REWEIGHTING DATA!" << endl;
      for(int j=1;j<=m_hdata->GetNbinsX();j++)
      {    
        double val = m_hdata->GetBinContent(j);
        if(val == 0.0) {
          cout<<"AnySample:"<<m_sampleid<<" bin "<<j<<" with 0 entries may cause problem on chi2 computations"<<endl;
          continue;
        }
        double binc = gRandom->Poisson(val);
        m_hdata->SetBinContent(j,binc);  //add statistical fluctuations
      }
      //m_hdata->Scale(potD/potMC_genie);
    }    
  }

}

double AnySample::CalcChi2()
{
  if(m_empty == true) return 0.0;
  
  if(m_hdata == NULL) 
  {
    cerr<<"ERROR: need to define data histogram"<<endl;
    return 0.0;
  }
  
  int nx = m_hpred->GetNbinsX();
  //int ny = m_hpred->GetNbinsY();
  
  if(nx != m_hdata->GetNbinsX())// || ny != m_hdata->GetNbinsY())
  {
    cerr<<"ERROR: binning mismatch between data and mc"<<endl;
    return 0.0;
  }

  double chi2 = 0.0;
  for(int j=1;j<=nx;j++)
  {
    double obs = m_hdata->GetBinContent(j);
    double exp = m_hpred->GetBinContent(j);
    if(exp>0.0){  //added when external fake datasets (you cannot reweight when simply 0)
                  // this didn't happen when all from same MC since if exp=0 then obs =0
      chi2 += 2*(exp - obs);
      if(obs>0.0) chi2 += 2*obs*TMath::Log(obs/exp);

      if(chi2 < 0)
      {
        cerr<<"WARTNING: stat chi2 is less than 0: " << chi2 << ", setting to 0"<<endl;
        cerr<<"exp and obs is: " << exp << " and " << obs <<endl;
        chi2 = 0.0;
      }
    }
      //DEBUG Time:
      //cout << "obs / exp / chi2: " <<obs<<"/"<<exp<<"/"<<chi2<<endl;
    }
  
  if(chi2 != chi2)
  {
    cerr<<"ERROR: stat chi2 is nan"<<endl;
    chi2 = 0.0;
  }

  /*
    // CHI2 DEBUG
    cout << endl << "*** CHI2 DEBUG ***" << endl;
    cout << "data histo is:" << endl;
    m_hdata->Print("all");
    cout << "MC histo is:" << endl;
    m_hpred->Print("all");
    cout << "chi2 is:" << chi2;
    cout << endl << "*** CHI2 DEBUG ***" << endl;
  */
 
  return chi2;
}

// GetSampleBreakdown 
void AnySample::GetSampleBreakdown(TDirectory *dirout, string tag, bool save)
{
  int nreac = 8;
  const char *names[] = {"cc0pi0p", "cc0pi1p", "cc0pinp", "cc1pi+", "ccother", 
       "backg", "Null", "OOFV"};
  TH1D *henu_rec[nreac];
  TH1D *hD1_rec[nreac];
  TH1D *hD2_rec[nreac];
  TH2D *hD1D2_rec[nreac];
  TH1D *hD1_true[nreac];
  TH1D *hD2_true[nreac];
  TH1D *hAnybin_true[nreac];
  TH1D *hAnybin_rec[nreac];
  int compos[nreac];
  
  //cout<<"AnySample::GetSampleBreakdown - Inializing histos of reactions" << endl;

  for(int i=0;i<nreac;i++)
  {
    compos[i] = 0;
    henu_rec[i] = new TH1D(Form("%s_RecEnu_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                           Form("%s_RecEnu_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                           nbins_enu, bins_enu);
    henu_rec[i]->SetDirectory(0);
    henu_rec[i]->GetXaxis()->SetTitle("Recon E_{#nu} (GeV)");

    //cout << "RecD1:" << endl;

    hD1_rec[i] = new TH1D(Form("%s_RecD1_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                           Form("%s_RecD1_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                           nbinsD1_toPlot, bins_D1toPlot);
    hD1_rec[i]->SetDirectory(0);
    hD1_rec[i]->GetXaxis()->SetTitle("Recon D1 (GeV/c)");

    //cout << "RecD2:" << endl;

    hD2_rec[i] = new TH1D(Form("%s_RecD2_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                          Form("%s_RecD2_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                          nbins_D2, bins_D2);
    hD2_rec[i]->SetDirectory(0);
    hD2_rec[i]->GetXaxis()->SetTitle("Recon D2");

    //cout << "RecD1D2:" << endl;
    
    hD1D2_rec[i] = new TH2D(Form("%s_RecD1D2_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                             Form("%s_RecD1D2_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                             nbinsD1_toPlot, bins_D1toPlot, nbins_D2, bins_D2);
    hD1D2_rec[i]->SetDirectory(0);
    hD1D2_rec[i]->GetXaxis()->SetTitle("Recon D1 (GeV/c)");
    hD1D2_rec[i]->GetXaxis()->SetTitle("Recon D2");

    //cout << "TrueD1" << endl;

    hD1_true[i] = new TH1D(Form("%s_TrueD1_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                            Form("%s_TrueD1_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                            nbinsD1_toPlot, bins_D1toPlot);
    hD1_true[i]->SetDirectory(0);
    hD1_true[i]->GetXaxis()->SetTitle("True D1 (GeV/c)");

    //cout << "TrueD2" << endl;

    hD2_true[i] = new TH1D(Form("%s_TrueD2_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                           Form("%s_TrueD2_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                           nbins_D2, bins_D2);
    hD2_true[i]->SetDirectory(0);
    hD2_true[i]->GetXaxis()->SetTitle("True D2");

    //cout << "TrueAny" << endl;

    hAnybin_true[i] = new TH1D(Form("%s_Anybins_true_%s_%s",   m_name.c_str(),names[i],tag.c_str()),
                               Form("%s_Anybins_true_%s_%s",   m_name.c_str(),names[i],tag.c_str()),
                               nAnybins, bins_Any);
    hAnybin_true[i]->SetDirectory(0);
    hAnybin_true[i]->GetXaxis()->SetTitle("Any bins");

    //cout << "RecAny" << endl;

    hAnybin_rec[i] = new TH1D(Form("%s_Anybins_rec_%s_%s",  m_name.c_str(),names[i],tag.c_str()),
                              Form("%s_Anybins_rec_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                              nAnybins, bins_Any);
    hAnybin_rec[i]->SetDirectory(0);
    hAnybin_rec[i]->GetXaxis()->SetTitle("Any bins");
  }

  if(m_sig != NULL) m_sig->Delete();
  m_sig = new TH1D(Form("%s_signalOnly_%s", m_name.c_str(),tag.c_str()),
                   Form("%s_signalOnly_%s", m_name.c_str(),tag.c_str()),
                   nAnybins, bins_Any);
  m_sig->SetDirectory(0);

  //loop over the events

  //cout<<"AnySample::GetSampleBreakdown - Collecting events" << endl;
  int Ntot = GetN();
  for(size_t i=0;i<m_events.size();i++)
  {
    //cout<<"AnySample::GetSampleBreakdown - In event loop iteration " << i << " out of " << m_events.size() << endl;
    double enu_rec, D1_rec, D2_rec, D1_true, D2_true, wght;
    enu_rec = m_events[i].GetRecEnu();
    D1_rec = m_events[i].GetRecD1trk();
    D2_rec = m_events[i].GetRecD2trk();
    D1_true = m_events[i].GetTrueD1trk();
    D2_true = m_events[i].GetTrueD2trk();
    wght    = m_events[i].GetEvWght();
    int rtype = m_events[i].GetReaction();
    //cout<< "AnySample::GetSampleBreakdown - rtype is: " << rtype << endl;

    // Warning - hard code hack warning ahead:
    // My reaction variable is mectopology from CC0Pi HL2 v1r15 so:
    // 0 - cc0pi0p
    // 1 - cc0pi1p
    // 2 - cc0pinp
    // 3 - cc1pi
    // 4 - ccother
    // 5 - BKG (not numuCC)
    // 6 - Nothing at all (WHY!?!?!?)
    // 7 - OOFGDFV
    // So I hack in a fix to stop a NULL reaction cat.


    //if((rtype==7)) rtype=6; //BKG is 5 then OOFV is 7, 6 is skipped causing array to overrun

    //cout<< "AnySample::GetSampleBreakdown - Event breakdown:" << endl;
    //m_events[i].Print();

    
    //cout<< "AnySample::GetSampleBreakdown - Filling histos" << endl;

    compos[rtype]++;
    henu_rec[rtype]->Fill(enu_rec, wght);
    hD1_rec[rtype]->Fill(D1_rec, wght);
    hD2_rec[rtype]->Fill(D2_rec, wght);
    hD1_true[rtype]->Fill(D1_true, wght);
    hD2_true[rtype]->Fill(D2_true, wght);
    hD1D2_rec[rtype]->Fill(D1_rec, D2_rec, wght);


    //cout<< "AnySample::GetSampleBreakdown - Filling histos with analysis binning" << endl;

    for(int j=0; j<nAnybins; j++){
      if((D1_true > m_D1edges[j].first) && (D1_true < m_D1edges[j].second)  &&
         (D2_true > m_D2edges[j].first) && (D2_true < m_D2edges[j].second)){
        hAnybin_true[rtype]->Fill(j+0.5,wght);
        break;
      }
    }
    for(int j=0; j<nAnybins; j++){
      if((D1_rec > m_D1edges[j].first) && (D1_rec < m_D1edges[j].second)  &&
         (D2_rec > m_D2edges[j].first) && (D2_rec < m_D2edges[j].second)){
        hAnybin_rec[rtype]->Fill(j+0.5,wght);
        break;
      }
    } 
    //********************************************
    // Warning: Hardcoded signal definition below:
    //********************************************
    for(int j=0; j<nAnybins; j++){
      if( (D1_true > m_D1edges[j].first) && (D1_true < m_D1edges[j].second)  &&
          (D2_true > m_D2edges[j].first) && (D2_true < m_D2edges[j].second)  &&
          ( (rtype==1) || (rtype==2) ) ) {
      //if( (D1_true > m_D1edges[j].first) && (D1_true < m_D1edges[j].second)  &&
      //    (D2_true > -0.5) && (D2_true < 0.5)  && (rtype==1 || rtype == 2) ) {
        m_sig->Fill(j+0.5,wght);
        break;
      }
    }
  }
  
  //cout<<"AnySample::GetSampleBreakdown - Wrapping up" << endl;

  dirout->cd();
  //tree->Write();
  //cout << "Scale Factor Is: " << m_norm << endl;
  //m_sig->Print("all");
  m_sig->Scale(m_norm);
  //m_sig->Print("all");

  for(int i=0;i<nreac;i++)
  {
    henu_rec[i]->Scale(m_norm);
    hD1_rec[i]->Scale(m_norm);
    hD2_rec[i]->Scale(m_norm);
    hD1_true[i]->Scale(m_norm);
    hD2_true[i]->Scale(m_norm);
    hD1D2_rec[i]->Scale(m_norm);
    hAnybin_true[i]->Scale(m_norm);
    hAnybin_rec[i]->Scale(m_norm);

    if(save){
      henu_rec[i]->Write();
      hD1_rec[i]->Write();
      hD2_rec[i]->Write();
      hD1_true[i]->Write();
      hD2_true[i]->Write();
      hD1D2_rec[i]->Write();
      hAnybin_true[i]->Write();
      hAnybin_rec[i]->Write();
    }

    henu_rec[i]->Delete();
    hD1_true[i]->Delete();
    hD2_true[i]->Delete();
    hD1_rec[i]->Delete();
    hD2_rec[i]->Delete();
    hD1D2_rec[i]->Delete();
    hAnybin_true[i]->Delete();
    hAnybin_rec[i]->Delete();
  }
  if(save){
    cout<<"============> Sample "<<m_name<<" <============"<<endl;
    for(int j=0;j<nreac;j++)
      cout<<setw(10)<<names[j]<<setw(5)<<j<<setw(10)<<compos[j]
          <<setw(10)<<(float)(compos[j])/Ntot*100.0<<"%"<<endl;
  }
}

// Write
void AnySample::Write(TDirectory *dirout, const char *bsname, int fititer)
{
  dirout->cd();
  m_hpred->Write(Form("%s_pred", bsname));
  m_hmc_true->Write(Form("%s_true", bsname));
  if(fititer==0){
    m_hmc->Write(Form("%s_mc", bsname));
    if(m_hdata != NULL) m_hdata->Write(Form("%s_data", bsname));
  }
}
