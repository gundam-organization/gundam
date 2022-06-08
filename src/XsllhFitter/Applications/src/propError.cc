/******************************************************

This is the main for the fitter. In the long run I hope
to make it work like the xsTool macros: you set all your
inputs here and then can avoid delving into the fitters
backend. For the moment this isn't completely the case
for example a lot of the systematics specifics are in
the appropriate systematic parameters methods.


******************************************************/


#include <iostream> 
#include <iomanip>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <assert.h>

#include <TCanvas.h>
#include <TH1F.h>

#include "FluxParameters.h"
#include "FluxParameters_norm.hh"
#include "DetParameters.h"
#include "FSIParameters.hh"
#include "NuclFSIParameters.hh"
#include "XsecParameters.h"
#include "FitParameters.h"
#include "XsecFitter.hh"
#include "anyTreeMC.hh"
#include "AnySample.hh"


using namespace std;

double calcChi2(TH1D*, TH1D*, TMatrixD); // chi2 function prototype

int main(int argc, char *argv[])
{

  char const * xslf_env = getenv("XSLLHFITTER");
  if(!xslf_env){
    std::cerr << "[ERROR]: environment variable \"XSLLHFITTER\" not set. "
    "Cannot determine source tree location." << std::endl;
    return 1;
  }
  string inputDir = std::string(xslf_env) + "/inputs";
  string fsel     = inputDir + "/NeutAir5_2DV2.root";
  string fakeData     = fsel;
  string nomResult;
  string fnameout = "propErrorOut.root";
  string ffluxcov     = inputDir + "/flux_covariance_banff_13av1.1.root";
  string fxsecbinning = inputDir + "/dptbinning2DPS_shortest_inclusive.txt";
  string fxsecbinning_coarse = inputDir + "/dptbinning2DPS_shortest_inclusive_coarse.txt";
  string fitOutputFile = inputDir + "/fitOutputFile.txt";
  //double potD     = 372.67;   //in units of 10^19 GENIE Air
  //double potD     = 349.15;   //in units of 10^19 NEUT Water
  //double potD     = 47.7323; //first pass data 

  //Optional binnings to improve GoF in sidebands: 
  string onePiAltBinning = inputDir + "/dptbinning2DPS_shortest_inclusive_1piFix.txt";
  string disAltBinning = inputDir + "/dptbinning2DPS_shortest_inclusive_disFix.txt";

  TH1D* combSigHist;
  TH1D* nomSigHist;
  TH1D* integralHist = NULL; 
  TH1D* sigdist[200];
  TH1D* sigdistInt = new TH1D("sigdistInt", "sigdistInt", 100000, 0, 100000);
  TH1D* sigdistPint = new TH1D("sigdistPint", "sigdistPint", 100000, 0, 100000);
  TH1D* sigdistRoll[200];
  TH1D* chi2RecoHist_data = new TH1D("chi2RecoHist_data", "chi2RecoHist_data", 1000, 0, 1000);
  TH1D* chi2RecoHist_stat = new TH1D("chi2RecoHist_stat", "chi2RecoHist_stat", 1000, 0, 1000);
  TH1D* chi2RecoHist_comp = new TH1D("chi2RecoHist_comp", "chi2RecoHist_comp", 1000, -500, 500);
  TH1D* chi2RecoHist_bf = new TH1D("chi2RecoHist_bf", "chi2RecoHist_bf", 1000, 0, 1000);
  TH1D* chi2RecoHist_bf_data = new TH1D("chi2RecoHist_bf_data", "chi2RecoHist_bf_data", 1000, 0, 1000);

  double potD     = 331.6; //in units 10^19 Neut Air
  double potMC    = 331.6; //in units 10^19 Neut Air
  int seed        = 1019;
  int ntoys = 300;
  int isPullStudy = 0;
  int isShapeOnly = 0;
  int ntemplatefixed = 0;
  int firstFixed = 99;
  int USESYST = 1;
  int useControlRegions=1;
  int STATFLUCTONLY = 0;
  int rmFineDet = 0;
  int useAltBinning=0;

  bool isBuffer=true;


  //get command line options
  char cc;
  while((cc = getopt(argc, argv, "i:d:e:b:B:o:f:n:N:s:t:p:r:y:S:F:G:c:a:R:A:P:D:")) != -1)
  {
    switch(cc)
    {
      case 'i': //selected events
        fsel = optarg;
        break; 
      case 'd': //selected events
        fakeData = optarg;
        break; 
      case 'e': //flux covariance matrix
        ffluxcov = optarg;
        break;
      case 'b': //binning for xsec weights
        fxsecbinning = optarg;
        break;
      case 'B': //binning for xsec weights
        fxsecbinning_coarse = optarg;
        break;
      case 'o': //output file
        fnameout = optarg;
        break;
      case 'f': //output file
        fitOutputFile = optarg;
        break;
      case 'n': //data POT
        potD = atof(optarg);
        break;
      case 'N': //MC POT
        potMC = atof(optarg);
        break;
      case 's': //random seed
        seed = atoi(optarg);
        break;
      case 't': //number of toys
        ntoys = atoi(optarg);
        break;
      case 'p': //number of toys
        isPullStudy = atoi(optarg);
        break;
      case 'r': //number of toys
        nomResult = optarg;
        break;
      case 'y': //use syst option
        USESYST = atoi(optarg);
        break;
      case 'S': //shape only mode
        isShapeOnly = atoi(optarg);
        break;
      case 'F': //How many ci's did you fix
        ntemplatefixed = atoi(optarg);
        break;     
      case 'G': //What was the first ci that you fixed
        firstFixed = atoi(optarg);
        break;
      case 'c': //template or model param fit?
        useControlRegions = atoi(optarg);
        break;
      case 'a': //template or model param fit?
        STATFLUCTONLY = atoi(optarg);
        break;
      case 'R': //template or model param fit?
        rmFineDet = atoi(optarg);
        break;
      case 'A': //binning for xsec weights
        useAltBinning = atoi(optarg);
        break;
      case 'P': //binning for xsec weights
        onePiAltBinning = optarg;
        break;
      case 'D': //binning for xsec weights
        disAltBinning = optarg;
        break;
      case 'h': //help 
        std::cout << "USAGE: " 
                  << argv[0] << " OPTIONS:" << std::endl
                  << "-i : \tset MC location" << std::endl
                  << "-e : \tset flux covariances location" << std::endl
                  << "-d : \tset detector covariances location" << std::endl
                  << "-b : \tset xec bin definitions" << std::endl
                  << "-o : \tset name of output file" << std::endl
                  << "-n : \tset POT for data in units 10**19" << std::endl
                  << "-N : \tset POT for MC in units 10**19" << std::endl
                  << "-s : \tset random seed" << std::endl
                  << "-c : \tset whether to use conrol regions" << std::endl
                  << "-s : \tset if only want to apply stat. fluct" << std::endl
                  << "-A : \tset wether to use alt SB binnings" << std::endl
                  << "-D : \tset DIS SB alt binning" << std::endl
                  << "-P : \tset 1Pi SB alt binning" << std::endl
                  << "-R : \tset whether to expect \"fine\" det params in vavar mat" << std::endl;
        return 0;
        break;
      default:
        return 1;
    }
  }

  if(ntemplatefixed!=0 && firstFixed==99){
    cout << "ERROR: you've told me that you've fixed parameters but haven't said which ones, need to specify G" << endl;
    return 0;
  }


  /******************* Print Chosen Options to Screen ***********************************/
  cout << endl;
  cout << "MC location is: " << fsel << endl;
  cout << "fakeDataData location is: " << fakeData << endl;
  cout << "Output file is: " << fnameout << endl;
  cout << "MC POT is: " << potMC << endl;
  cout << "Fake Data POT is: " << potD << endl;
  cout << "Is pulls study? " << isPullStudy << endl;
  cout << "Is shape only? " << isShapeOnly << endl;
  cout << "Number of cis fixed " << ntemplatefixed << endl;
  cout << "First ci fixed " << firstFixed << endl;
  cout << "Binning to be used is " << fxsecbinning << endl;
  cout << "used syst mode is " << USESYST << endl;
  cout << "Using control regions? " << useControlRegions << endl;
  cout << "Stat study only? " << STATFLUCTONLY << endl;
  cout << "Remove det fine params? " << rmFineDet << endl;
  cout << endl;

  /******************* Setup data trees and binning ***********************************/

  // Setup data trees

  TFile *fdata = new TFile(TString(fakeData)); 
  TTree *tdata = (TTree*)(fdata->Get("selectedEvents"));
  if(!tdata){
    cout << "ERROR: Couldn't find \"selectedEvents\" tree in file: " << fakeData << endl;
    return 0;
  }

  TFile *finput= new TFile(fitOutputFile.c_str(),"READ");
  TTree *tfresult = (TTree*)(finput->Get("selectedEvents"));
  if(!tfresult){
    cout << "ERROR: Couldn't find \"selectedEvents\" tree in file: " << nomResult << endl;
    return 0;
  }

  TFile *nomResultFile = new TFile(TString(nomResult)); 



  //Set up bin edges for fine binning

  int nbins = 0;

  std::vector<std::pair<double, double> > v_D1edges;
  std::vector<std::pair<double, double> > v_D2edges;
  std::vector<std::pair<double, double> > v_D1edges_Dummy;
  std::vector<std::pair<double, double> > v_D2edges_Dummy;
  ifstream fin(fxsecbinning.c_str());
  assert(fin.is_open());
  string line;
  while (getline(fin, line))
  {
   nbins++;
   stringstream ss(line);
   double D1_1, D1_2, D2_1, D2_2;
   if(!(ss>>D2_1>>D2_2>>D1_1>>D1_2))
   {
     cerr<<"Bad line format: "<<endl
         <<"     "<<line<<endl;
     continue;
   }
     v_D1edges.push_back(make_pair(D1_1,D1_2));
     v_D2edges.push_back(make_pair(D2_1,D2_2));
     v_D1edges_Dummy.push_back(make_pair(D1_1-9999.0,D1_2-9999.0));
     v_D2edges_Dummy.push_back(make_pair(D2_1-9999.0,D2_2-9999.0));
  }
  fin.close();

  cout << "Found "  << nbins << " bins" << endl;

  // Set up edges for any (combined 2D->1D) binning

  int nAnybins=v_D1edges.size();
  double *bins_Any = new double[nAnybins+1];
  for (int i=0; i<=nAnybins; i++){
    bins_Any[i]=i;
  }
  combSigHist = new TH1D("combSigHist","combSigHist",nAnybins,bins_Any);
  nomSigHist = new TH1D("nomSigHist","nomSigHist",nAnybins,bins_Any);

  //Set up bin edges for coarse binning

  int nbins_coarse = 0;


  std::vector<std::pair<double, double> > v_D1edges_coarse;
  std::vector<std::pair<double, double> > v_D2edges_coarse;
  ifstream fin_coarse(fxsecbinning_coarse.c_str());
  if(!fin_coarse.is_open()) cout << "WARNING: Binning file \"" << fxsecbinning_coarse << "\" could not be opened" << endl;
  assert(fin_coarse.is_open());
  string line_coarse;
  while (getline(fin_coarse, line_coarse))
  {
    nbins_coarse++;
    stringstream ss(line_coarse);
    double D1_1, D1_2, D2_1, D2_2;
    if(!(ss>>D2_1>>D2_2>>D1_1>>D1_2))
    {
      cerr<<"Bad line format: "<<endl
          <<"     "<<line_coarse<<endl;
      continue;
    }
      v_D1edges_coarse.push_back(make_pair(D1_1,D1_2));
      v_D2edges_coarse.push_back(make_pair(D2_1,D2_2));
  }
  fin_coarse.close();

  cout << "Found "  << nbins_coarse << " bins for coarse binning" << endl;

  //Set up alt binning if requested
  std::vector<std::pair<double, double> > v_D1edges_1pi;
  std::vector<std::pair<double, double> > v_D2edges_1pi;
  std::vector<std::pair<double, double> > v_D1edges_dis;
  std::vector<std::pair<double, double> > v_D2edges_dis;
  int nbins_1pi = 0;
  int nbins_dis = 0;

  if(useAltBinning==1){

    //Set up bin edges for 1pi alt binning

    ifstream fin_1pi(onePiAltBinning.c_str());
    if(!fin_1pi.is_open()) cout << "WARNING: Binning file \"" << onePiAltBinning << "\" could not be opened" << endl;
    assert(fin_1pi.is_open());
    string line_1pi;
    while (getline(fin_1pi, line_1pi))
    {
      nbins_1pi++;
      stringstream ss(line_1pi);
      double D1_1, D1_2, D2_1, D2_2;
      if(!(ss>>D2_1>>D2_2>>D1_1>>D1_2))
      {
        cerr<<"Bad line format: "<<endl
            <<"     "<<line_1pi<<endl;
        continue;
      }
        v_D1edges_1pi.push_back(make_pair(D1_1,D1_2));
        v_D2edges_1pi.push_back(make_pair(D2_1,D2_2));
    }
    fin_1pi.close();
    cout << "Found "  << nbins_1pi << " bins for 1pi alt binning" << endl;

    //Set up bin edges for dis alt binning

    ifstream fin_dis(disAltBinning.c_str());
    if(!fin_dis.is_open()) cout << "WARNING: Binning file \"" << disAltBinning << "\" could not be opened" << endl;
    assert(fin_dis.is_open());
    string line_dis;
    while (getline(fin_dis, line_dis))
    {
      nbins_dis++;
      stringstream ss(line_dis);
      double D1_1, D1_2, D2_1, D2_2;
      if(!(ss>>D2_1>>D2_2>>D1_1>>D1_2))
      {
        cerr<<"Bad line format: "<<endl
            <<"     "<<line_dis<<endl;
        continue;
      }
        v_D1edges_dis.push_back(make_pair(D1_1,D1_2));
        v_D2edges_dis.push_back(make_pair(D2_1,D2_2));
    }
    fin_dis.close();
    cout << "Found "  << nbins_dis << " bins for dis alt binning" << endl;

  }
  else{
    v_D1edges_1pi=v_D1edges;
    v_D2edges_1pi=v_D2edges;
    v_D1edges_dis=v_D1edges;
    v_D2edges_dis=v_D2edges;
  }

  /*************************************** FLUX *****************************************/

  //setup enu bins for flux 

  TFile *finfluxbins = TFile::Open(ffluxcov.c_str()); 
  TAxis *nd_numu_bins = (TAxis*)finfluxbins->Get("nd5_numode_numu_bins");

  vector<double> enubins;
  enubins.push_back(nd_numu_bins->GetBinLowEdge(1));
  for(int i=0;i<nd_numu_bins->GetNbins();i++)
  {
    enubins.push_back(nd_numu_bins->GetBinUpEdge(i+1));
  }


  /********************************************************************************************/

  /******************************** DETECTOR SYSTEMATICS STARTING WEIGHTS *********************/
  //setup D1,D2 bins, starting param values and covm for det syst --------------

  //TVectorD* det_weights = (TVectorD*)findetcov->Get("det_weights");

  // Currently some rather horrible hard coding to make a vector of 1s of the correct size
  // This should really be an input to the fitter, WIP! ***

  const int ndetcovmatele = 54;//30;

  double arr[ndetcovmatele];
  for(int i=0; i<ndetcovmatele; i++){ arr[i]=(1.0);}

  TVectorD* det_weights = new TVectorD(ndetcovmatele, arr);

  const int ndetcovmatele_fine = 48;
  
  double arr_fine[ndetcovmatele_fine];
  for(int i=0; i<ndetcovmatele_fine; i++){ arr_fine[i]=(1.0);}
  
  TVectorD* det_weights_fine = new TVectorD(ndetcovmatele_fine, arr_fine);
  
  det_weights_fine->Print();
  


  
  /***********************************************************************************/

  /********************************XSEC RESPONSE FUNCTIONS****************************/

  // Currently only deals with BG xsec model params

  vector<TFile*> responsefunctions;
      
  TFile* CAResrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NXSec_CA5RES_allVariation.root").c_str());
  responsefunctions.push_back(CAResrespfunc);
  TFile* MAResrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NXSec_MaNFFRES_allVariation.root").c_str());
  responsefunctions.push_back(MAResrespfunc);
  TFile* BgResrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NXSec_BgSclRES_allVariation.root").c_str());
  responsefunctions.push_back(BgResrespfunc);
  TFile* CCNuErespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWG2012a_ccnueE0_allVariation.root").c_str());
  responsefunctions.push_back(CCNuErespfunc);
  TFile* dismpirespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWG2012a_dismpishp_allVariation.root").c_str());
  responsefunctions.push_back(dismpirespfunc);
  TFile* CCCohrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWG2012a_cccohE0_allVariation.root").c_str());
  responsefunctions.push_back(CCCohrespfunc);
  TFile* NCCohrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWG2012a_nccohE0_allVariation.root").c_str());
  responsefunctions.push_back(NCCohrespfunc);
  TFile* NCOthrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWG2012a_ncotherE0_allVariation.root").c_str());
  responsefunctions.push_back(NCOthrespfunc);
  TFile* EbCrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWG2014a_Eb_C12_allVariation.root").c_str());
  responsefunctions.push_back(EbCrespfunc);
  // TFile* MAQErespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NXSec_MaCCQE_allVariation.root").c_str());
  // responsefunctions.push_back(MAQErespfunc);
  // TFile* pFrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWG2014a_pF_C12_allVariation.root").c_str());
  // responsefunctions.push_back(pFrespfunc);
  // TFile* MECrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWGMEC_Norm_C12_allVariation.root").c_str());
  // responsefunctions.push_back(MECrespfunc);
  TFile* MAQErespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NXSec_MaCCQE_allVariation.rootDummy").c_str());
  responsefunctions.push_back(MAQErespfunc);
  TFile* pFrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWG2014a_pF_C12_allVariation.rootDummy").c_str());
  responsefunctions.push_back(pFrespfunc);
  TFile* MECrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWGMEC_Norm_C12_allVariation.rootDummy").c_str());
  responsefunctions.push_back(MECrespfunc);


          
  /*****************************************************************************************/


  /************************FSI RESPONSE FUNCTIONS*****************************************/

  vector<TFile*> responsefunctions_FSI;

  TFile* FrInelLowrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NCasc_FrCExLow_pi_allVariation.root").c_str());
  responsefunctions_FSI.push_back(FrInelLowrespfunc);
  TFile* FrInelHighrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NCasc_FrInelHigh_pi_allVariation.root").c_str());
  responsefunctions_FSI.push_back(FrInelHighrespfunc);
  TFile* FrPiProdrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NCasc_FrPiProd_pi_allVariation.root").c_str());
  responsefunctions_FSI.push_back(FrPiProdrespfunc);
  TFile* FrAbsrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NCasc_FrAbs_pi_allVariation.root").c_str());
  responsefunctions_FSI.push_back(FrAbsrespfunc);
  TFile* FrCExLowrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NCasc_FrCExLow_pi_allVariation.root").c_str());
  responsefunctions_FSI.push_back(FrCExLowrespfunc);
  TFile* FrCExHighrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NCasc_FrInelHigh_pi_allVariation.root").c_str());
  responsefunctions_FSI.push_back(FrCExHighrespfunc);

  
  /*********************************************************************************/
  

  TFile *fout = TFile::Open(fnameout.c_str(), "RECREATE");
  cout<<"output file open"<<endl;

  /*****************************************************************************************/


  /************************ ADD ANALYSIS SAMPLES *****************************************/


  /************************** Add analysis samples from the data: **************************/
  
  vector<AnaSample*> samples;

  cout << "Data / MC scale factor is: " << potD/potMC << endl;

  // The sample ID (first arg) should match the cutBranch corresponding to it

  // Primary Signal Samples:

  /*
  AnySample sam1(0, "MuTPC",v_D1edges, v_D2edges,tdata);
  sam1.SetNorm(potD/potMC);
  samples.push_back(&sam1);
  */

  AnySample sam2(1, "MuTPCpTPC",v_D1edges, v_D2edges, tdata, isBuffer);
  sam2.SetNorm(potD/potMC);
  samples.push_back(&sam2);
  
  AnySample sam3(2, "MuTPCpFGD",v_D1edges, v_D2edges, tdata, isBuffer);
  sam3.SetNorm(potD/potMC);
  samples.push_back(&sam3);
  
  AnySample sam4(3, "MuFGDPTPC",v_D1edges, v_D2edges, tdata, isBuffer);
  sam4.SetNorm(potD/potMC);
  samples.push_back(&sam4);

  
  // AnySample sam5(4, "MuFGD",v_D1edges, v_D2edges,tdata);
  // sam5.SetNorm(potD/potMC);
  // samples.push_back(&sam5);

  //Control Samples:

  bool emptyCR;
  if(useControlRegions==1) emptyCR=false;
  else if (useControlRegions==0) emptyCR=true;
  
  AnySample sam6(5, "CC1pi",v_D1edges_1pi, v_D2edges_1pi, tdata, isBuffer, emptyCR);
  sam6.SetNorm(potD/potMC);
  samples.push_back(&sam6);
    
  AnySample sam7(6, "DIS",v_D1edges_dis, v_D2edges_dis, tdata, isBuffer, emptyCR);
  sam7.SetNorm(potD/potMC);
  samples.push_back(&sam7);
  
  //Extra Signal Samples:
  
  AnySample sam8(7, "muTPC_Np",v_D1edges, v_D2edges, tdata, isBuffer);
  sam8.SetNorm(potD/potMC);
  samples.push_back(&sam8);


  // AnySample sam9(8, "muFGDpTPC_Np",v_D1edges, v_D2edges,tdata);
  // sam9.SetNorm(potD/potMC);
  // samples.push_back(&sam9);

  // AnySample sam10(9, "muFGD_Np",v_D1edges, v_D2edges,tdata);
  // sam10.SetNorm(potD/potMC);
  // samples.push_back(&sam10);

  int nsamples = samples.size();

  //read MC events
  anyTreeMC selTree(fsel.c_str());
  cout << "Reading and collecting events" << endl;
  selTree.GetEvents(samples);
  //get brakdown by reaction
  cout << "Getting sample breakdown by reaction" << endl;
  for(size_t s=0;s<samples.size();s++){
    ((AnySample*)(samples[s]))->GetSampleBreakdown(fout,"nominal",false);
  }


  /************************** Add analysis samples from the fit result: **************************/
  
  vector<AnaSample*> samples_fresult;

  // The sample ID (first arg) should match the cutBranch corresponding to it

  // Primary Signal Samples:

  /*
  AnySample sam1(0, "MuTPC",v_D1edges, v_D2edges,tfresult);
  sam1.SetNorm(potD/potMC);
  samples_fresult.push_back(&sam1);
  */

  AnySample sam2_fr(1, "MuTPCpTPC",v_D1edges, v_D2edges, tfresult, isBuffer);
  sam2_fr.SetNorm(potD/potMC);
  samples_fresult.push_back(&sam2_fr);
  
  AnySample sam3_fr(2, "MuTPCpFGD",v_D1edges, v_D2edges, tfresult, isBuffer);
  sam3_fr.SetNorm(potD/potMC);
  samples_fresult.push_back(&sam3_fr);
  
  AnySample sam4_fr(3, "MuFGDPTPC",v_D1edges, v_D2edges, tfresult, isBuffer);
  sam4_fr.SetNorm(potD/potMC);
  samples_fresult.push_back(&sam4_fr);

  
  // AnySample sam5_fr(4, "MuFGD",v_D1edges, v_D2edges,tfresult);
  // sam5_fr.SetNorm(potD/potMC);
  // samples_fresult.push_back(&sam5_fr);

  //Control Samples:
  
  AnySample sam6_fr(5, "CC1pi",v_D1edges_1pi, v_D2edges_1pi, tfresult, isBuffer, emptyCR);
  sam6_fr.SetNorm(potD/potMC);
  samples_fresult.push_back(&sam6_fr);
    
  AnySample sam7_fr(6, "DIS",v_D1edges_dis, v_D2edges_dis, tfresult, isBuffer, emptyCR);
  sam7_fr.SetNorm(potD/potMC);
  samples_fresult.push_back(&sam7_fr);
  
  //Extra Signal Samples:
  
  AnySample sam8_fr(7, "muTPC_Np",v_D1edges, v_D2edges, tfresult, isBuffer);
  sam8_fr.SetNorm(potD/potMC);
  samples_fresult.push_back(&sam8_fr);
  
  // AnySample sam9_fr(8, "muFGDpTPC_Np",v_D1edges, v_D2edges,tfresult);
  // sam9_fr.SetNorm(potD/potMC);
  // samples_fresult.push_back(&sam9_fr);

  // AnySample sam10_fr(9, "muFGD_Np",v_D1edges, v_D2edges,tfresult);
  // sam10_fr.SetNorm(potD/potMC);
  // samples_fresult.push_back(&sam10_fr);

  int nsamples_fresult = samples_fresult.size();

  /*****************************************************************************************/


  /************************ ADD FIT PARAMETERS *****************************************/

  //define fit param classes
  vector<AnaFitParameters*> fitpara;
  //fitpara.SetFluxHisto(h_flux);

  // When filling the fitparas note that there are some assumptions later in the code
  // that the fit (i.e. template) parameters are stored at index 0. For this reason 
  // always fill the fit parameters first. 
  
  //Fit parameters
  FitParameters sigfitpara(fxsecbinning, "par_fit");
  sigfitpara.InitEventMap(samples, 0);
  fitpara.push_back(&sigfitpara);
  
  //Flux parameters
  FluxParameters fluxpara(enubins);
  fluxpara.InitEventMap(samples, 0);
  if(USESYST==1 || USESYST==2) fitpara.push_back(&fluxpara);
  else fluxpara.SetNpar(0);

  //Det parameters
  DetParameters detpara((inputDir + "/dptbinning2DPS_shortest_inclusive_det.txt").c_str(), det_weights, samples, "par_detAve");
  detpara.InitEventMap(samples, 0);
  if(USESYST==1 || USESYST==3) fitpara.push_back(&detpara);
  else detpara.SetNpar(0);

  DetParameters detpara_fine((inputDir + "/dptDetCovBinning.txt").c_str(),det_weights_fine, samples, "par_detFine");
  detpara_fine.InitEventMap(samples, 0);
  detpara_fine.SetNpar(0);
  //if(USESYST==1) fitpara.push_back(&detpara_fine);
  //else detpara_fine.SetNpar(0);

  //Xsec parameters
  XsecParameters xsecpara;
  xsecpara.StoreResponseFunctions(responsefunctions, v_D1edges, v_D2edges);
  xsecpara.InitEventMap(samples, 0);
  if(USESYST==1 || USESYST==4) fitpara.push_back(&xsecpara);
  else xsecpara.SetNpar(0);

  //FSI parameters
  FSIParameters fsipara;
  fsipara.StoreResponseFunctions(responsefunctions_FSI, v_D1edges, v_D2edges);
  fsipara.InitEventMap(samples, 0);
  if(USESYST==1 || USESYST==4) fitpara.push_back(&fsipara);
  else fsipara.SetNpar(0);


  /*****************************************************************************************/


  //*********************** PROP ERROR Starts Here *********************

  int nsigparam = sigfitpara.Npar - ntemplatefixed;
  vector<double> allparams;

  TMatrixDSym *covariance   = (TMatrixDSym*)finput->Get("res_cov_matrix");
  TVectorD *priorVec = ((TVectorD*)finput->Get("res_vector"));
  TH1D *prefitParams = ((TH1D*)finput->Get("prefitParams"));

  double minuitchi2 = ( ((TGraph*)finput->Get("final_chi2_fromMinuit"))->GetY())[0];

  // The covariance matrix from the fit doesn't include fixed parameters. To account for this we first have to
  // remove the fixed parameters from the fit output parameter vector so that we can make toys. Then, for each
  // toy we need to add the parameters back in (at their fixed value) so that we can reweight. 

  int ndetParaFineParams = 0;
  if(ntemplatefixed!=0 || rmFineDet!=0){
    cout << "Prior vector was: " << endl;
    priorVec->Print();

    const int nparamsnow = priorVec->GetNrows() - ntemplatefixed;
    double arr[nparamsnow];
    for(int i=0;i<nparamsnow;i++){
      if(i<firstFixed) arr[i] = (*priorVec)(i);
      else arr[i] = (*priorVec)(i+ntemplatefixed);
    } 
    // Relevent if no fine det params in covar matrix (need to remove them from the vector):
    const int nparamsnow_rmdetfine = priorVec->GetNrows() - ntemplatefixed - detpara_fine.Npar;
    double arr_rmdetfine[nparamsnow_rmdetfine];
    for(int i=0;(i<nparamsnow_rmdetfine) && (rmFineDet==1);i++){
      if(i<firstFixed && i < sigfitpara.Npar + fluxpara.Npar + detpara.Npar) arr_rmdetfine[i] = (*priorVec)(i);
      else if (i < sigfitpara.Npar + fluxpara.Npar + detpara.Npar) arr_rmdetfine[i] = (*priorVec)(i+ntemplatefixed);
      else arr_rmdetfine[i] = (*priorVec)(i+ntemplatefixed+detpara_fine.Npar);
    } 

    delete priorVec;
    if(rmFineDet==0) priorVec = new TVectorD(nparamsnow, arr);
    if(rmFineDet==1){
      priorVec = new TVectorD(nparamsnow_rmdetfine, arr_rmdetfine);
      ndetParaFineParams = detpara_fine.Npar; // remember the actual size incase we need it later
      detpara_fine.SetNpar(0);
    }
    cout << "Prior now is: " << endl;
    priorVec->Print();
  }

  //Check the dimensionality of everything:

  if(covariance->GetNrows() != (priorVec->GetNrows()) ){
    cout<<"Error dimensions of prior vector ("<<priorVec->GetNrows()<<")vs does not match covariance matrix ("<<covariance->GetNrows()<<")"<<endl;
    abort();
  }
  cout << endl << "Param vector read from file: " << endl;
  for(uint i=0; i<covariance->GetNrows(); i++){
    allparams.push_back((*priorVec)[i]);
    cout<<i<<" "<<((*priorVec)[i])<<endl;
  }
  if((nsigparam + fluxpara.Npar + detpara.Npar + detpara_fine.Npar + xsecpara.Npar + fsipara.Npar) != priorVec->GetNrows()){
    cout << "ERROR: Mismatch between parameters expected and parameters found ..." << endl;
    cout << "Expected parameters do not seem to match found parameters, expected " << 
           nsigparam + fluxpara.Npar + detpara.Npar + detpara_fine.Npar + xsecpara.Npar + fsipara.Npar << ", found " <<
           priorVec->GetNrows() << endl;
    return 0;
  }
  else cout << "Expected parameters seem to match found parameters, expected " << 
       nsigparam + fluxpara.Npar + detpara.Npar + xsecpara.Npar + fsipara.Npar << ", found " <<
       priorVec->GetNrows() << endl;

  //End of the dimensionality checks

  fout->cd();
  covariance->Write("res_cov_matrix");
  priorVec->Write("res_vector");

  ThrowParms *throwParms = new ThrowParms((*priorVec),(*covariance));
  TRandom3 *rand = new TRandom3(seed);
  //set gRandom to our rand
  gRandom = rand;

  Double_t rollcomb;
  Double_t integral, pIntegral; 
  for(int i=0; i<200; i++){sigdist[i]=NULL; sigdistRoll[i]=NULL;} 


  double evt_bin_toy[100][10001];// = {0}; // bins, toys
  if(ntoys>10000){
    cout << "WARNING: for over 10000 toys need to change hard coded array index" << endl; 
    return 0;
  }
  TMatrixDSym covar(nbins); 
  TMatrixDSym covar_norm(nbins); 

  //Split up the prior vector into consituant parameters

  //First convert TVectorD into a vector
  vector<double> priorVecV;
  cout << "converted prior vec is:" << endl;
  for(uint i=0; i<priorVec->GetNrows(); i++){
    priorVecV.push_back((*priorVec)[i]);
    cout << i << " " << priorVecV[i] << endl;
  }
  //Now do the splitting:
  vector<double>::const_iterator first = priorVecV.begin() + 0;
  vector<double>::const_iterator last =  priorVecV.begin() + nsigparam;
  vector <double> sigfitparams_prior (first,last);
  first = priorVecV.begin() + nsigparam;
  last =  priorVecV.begin() + nsigparam + fluxpara.Npar;
  vector <double> fluxparams_prior (first,last);
  first = priorVecV.begin() + nsigparam + fluxpara.Npar;
  last =  priorVecV.begin() + nsigparam + fluxpara.Npar + detpara.Npar;
  vector <double> detparams_prior (first,last);
  first = priorVecV.begin() + nsigparam + fluxpara.Npar + detpara.Npar + detpara_fine.Npar;
  last =  priorVecV.begin() + nsigparam + fluxpara.Npar + detpara.Npar + detpara_fine.Npar + xsecpara.Npar;
  vector <double> xsecparams_prior (first,last);
  first = priorVecV.begin() + nsigparam + fluxpara.Npar + detpara.Npar + detpara_fine.Npar + xsecpara.Npar;
  last =  priorVecV.begin() + nsigparam + fluxpara.Npar + detpara.Npar + detpara_fine.Npar + xsecpara.Npar + fsipara.Npar;
  vector <double> fsiparams_prior (first,last);
  //Finished splitting

  // Deal with fixed params:

  for(int p=0;p<ntemplatefixed;p++){
   vector<double>::iterator it = sigfitparams_prior.begin() + firstFixed;
   sigfitparams_prior.insert(it, 1.0);
  }

  //Find reco level chi2 between prior vector and data (should be the best fit chi2)
  //First Reweight samples:
  for(size_t s=0;s<samples.size();s++)
  {
    //loop over events
    for(int i=0;i<samples[s]->GetN();i++)
    {
      AnaEvent* ev = samples[s]->GetEvent(i);
      ev->SetEvWght(ev->GetEvWghtMC()); 
      //do weights for each AnaFitParameters obj
      sigfitpara.ReWeight(ev, s, i, sigfitparams_prior);
      if(USESYST==1 || USESYST==2) fluxpara.ReWeight(ev, s, i, fluxparams_prior);
      if(USESYST==1 || USESYST==3) detpara.ReWeight(ev, s, i, detparams_prior);
      if(USESYST==1 || USESYST==4) xsecpara.ReWeight(ev, s, i, xsecparams_prior);
      if(USESYST==1 || USESYST==4) fsipara.ReWeight(ev, s, i, fsiparams_prior);

      // Fill histo with reweighted event:
      double D1_true = ev->GetTrueD1trk();
      double D2_true = ev->GetTrueD2trk();
      double wght    = ev->GetEvWght();
      int    rtype   = ev->GetReaction();
      //********************************************
      // Warning: Hardcoded signal definition below:
      //********************************************
      for(int j=0; j<nAnybins; j++){
        if( (D1_true > v_D1edges[j].first) && (D1_true < v_D1edges[j].second)  &&
            (D2_true > v_D2edges[j].first) && (D2_true < v_D2edges[j].second)  &&
            ( (rtype==1) || (rtype==2) ) ) {
          nomSigHist->Fill(j+0.5,wght);
          break;
        }
      }
    }
  }
  nomSigHist->Scale(potD/potMC);

  // Now calc the chi2:
  // Now calculate the chi2 of the best fit to the data:
  double chi2_rec_bf_data=0;
  for(size_t s=0;s<samples.size();s++)
  {
    cout << "Evaluating chi2 of data to best fit MC in sample " << s << endl;
    ((AnySample*)(samples[s]))->FillEventHisto(2); 
    //calculate chi2 for each sample:
    chi2_rec_bf_data += ((AnySample*)(samples[s]))->CalcChi2();
    cout << "chi2 of best fit to best fit MC in sample " << s << " is: " <<  ((AnySample*)(samples[s]))->CalcChi2() << endl;;
  }
  chi2RecoHist_bf_data->Fill(chi2_rec_bf_data, ntoys/10);


  // As a sanity check the chi2 of the bf wrt the MC rw to the MC should be 0:
  double chi2_rec_bf=0;
  for(size_t s=0;s<samples_fresult.size();s++)
  {
    cout << "Evaluating chi2 of data to best fit MC in sample " << s << endl;
    ((AnySample*)(samples_fresult[s]))->FillEventHisto(2); 
    //calculate chi2 for each sample:
    chi2_rec_bf += ((AnySample*)(samples_fresult[s]))->CalcChi2();
    cout << "chi2 of data to best fit MC in sample " << s << " is: " <<  ((AnySample*)(samples_fresult[s]))->CalcChi2() << endl;;
  }
  chi2RecoHist_bf->Fill(chi2_rec_bf, ntoys/10);


  //Finished

  // Now take throws of the MC according to the postfit covariance matrix:

  cout << "Will throw  " << ntoys << " toys" << endl;
  //to be redone for each toy!////////////////////////////////////////
  for(int t=0; t<ntoys; t++){
    cout << "Processing toy " << t << endl;
    //vector< vector<double> > par_throws;

    vector<double> allparams_throw;
    throwParms->ThrowSet(allparams_throw);
    vector<double>::const_iterator first = allparams_throw.begin() + 0;
    vector<double>::const_iterator last =  allparams_throw.begin() + nsigparam;
    vector <double> sigfitparams_throw (first,last);
    first = allparams_throw.begin() + nsigparam;
    last =  allparams_throw.begin() + nsigparam + fluxpara.Npar;
    vector <double> fluxparams_throw (first,last);
    first = allparams_throw.begin() + nsigparam + fluxpara.Npar;
    last =  allparams_throw.begin() + nsigparam + fluxpara.Npar + detpara.Npar;
    vector <double> detparams_throw (first,last);
    first = allparams_throw.begin() + nsigparam + fluxpara.Npar + detpara.Npar + detpara_fine.Npar;
    last =  allparams_throw.begin() + nsigparam + fluxpara.Npar + detpara.Npar + detpara_fine.Npar + xsecpara.Npar;
    vector <double> xsecparams_throw (first,last);
    first = allparams_throw.begin() + nsigparam + fluxpara.Npar + detpara.Npar + detpara_fine.Npar + xsecpara.Npar;
    last =  allparams_throw.begin() + nsigparam + fluxpara.Npar + detpara.Npar + detpara_fine.Npar + xsecpara.Npar + fsipara.Npar;
    vector <double> fsiparams_throw (first,last);

    for(int p=0;p<ntemplatefixed;p++){
     vector<double>::iterator it = sigfitparams_throw.begin() + firstFixed;
     sigfitparams_throw.insert(it, 1.0);
    }

    if(t<100){
      cout<<"CHECK "<<endl;
      cout<<"ccqe size "<<sigfitparams_throw.size()<<endl;
      for(int j=0; j<sigfitparams_throw.size(); j++){
        cout<<j<<" "<<sigfitparams_throw[j]<<endl;
        //par_throws.push_back(sigfitparams_throw[j]);
      }
      cout<<"flux size "<<fluxparams_throw.size()<<endl;
      for(int j=0; j<fluxparams_throw.size(); j++){
        cout<<fluxparams_throw[j]<<endl;
        //par_throws.push_back(fluxparams_throw[j]);
      }
      cout<<"det size "<<detparams_throw.size()<<endl;
      for(int j=0; j<detparams_throw.size(); j++){
        cout<<detparams_throw[j]<<endl;
        //par_throws.push_back(detparams_throw[j]);
      }
      cout<<"xsec size "<<xsecparams_throw.size()<<endl;
      for(int j=0; j<xsecparams_throw.size(); j++){
        cout<<xsecparams_throw[j]<<endl;
        //par_throws.push_back(xsecparams_throw[j]);
      }
      cout<<"fsi size "<<fsiparams_throw.size()<<endl;
      for(int j=0; j<fsiparams_throw.size(); j++){
        cout<<fsiparams_throw[j]<<endl;
        //par_throws.push_back(fsiparams_throw[j]);
      }
    }

    combSigHist->Reset();
    for(size_t s=0;s<samples.size();s++)
    {
      //loop over events
      for(int i=0;i<samples[s]->GetN();i++)
      {
        AnaEvent* ev = samples[s]->GetEvent(i);
        ev->SetEvWght(ev->GetEvWghtMC()); 
        //do weights for each AnaFitParameters obj
        if(STATFLUCTONLY==0){
          if(i==0 && s==0 && t<100) {cout<<"On the following event:: " <<endl; ev->Print();}
          sigfitpara.ReWeight(ev, s, i, sigfitparams_throw);
          if(i==0 && s==0 && t<100) cout<<"Weight after sig param RW: "<< ev->GetEvWght() <<endl;
          if(USESYST==1 || USESYST==2) fluxpara.ReWeight(ev, s, i, fluxparams_throw);
          if(i==0 && s==0 && t<100) cout<<"Weight after flux param RW: "<< ev->GetEvWght() <<endl;
          if(USESYST==1 || USESYST==3) detpara.ReWeight(ev, s, i, detparams_throw);
          if(i==0 && s==0 && t<100) cout<<"Weight after det param RW: "<< ev->GetEvWght() <<endl;
          if(USESYST==1 || USESYST==4) xsecpara.ReWeight(ev, s, i, xsecparams_throw);
          if(i==0 && s==0 && t<100) cout<<"Weight after xsec param RW: "<< ev->GetEvWght() <<endl;
          if(USESYST==1 || USESYST==4) fsipara.ReWeight(ev, s, i, fsiparams_throw);
          if(i==0 && s==0 && t<100) cout<<"Weight after fsi param RW: "<< ev->GetEvWght() <<endl;
        }
        else{
          sigfitpara.ReWeight(ev, s, i, sigfitparams_prior);
          if(USESYST==1 || USESYST==2) fluxpara.ReWeight(ev, s, i, fluxparams_prior);
          if(USESYST==1 || USESYST==3) detpara.ReWeight(ev, s, i, detparams_prior);
          if(USESYST==1 || USESYST==4) xsecpara.ReWeight(ev, s, i, xsecparams_prior);
          if(USESYST==1 || USESYST==4) fsipara.ReWeight(ev, s, i, fsiparams_prior);
        }
        // Fill histo with reweighted event:
        double D1_true = ev->GetTrueD1trk();
        double D2_true = ev->GetTrueD2trk();
        double wght    = ev->GetEvWght();
        int    rtype   = ev->GetReaction();
        //********************************************
        // Warning: Hardcoded signal definition below:
        //********************************************
        for(int j=0; j<nAnybins; j++){
          if( (D1_true > v_D1edges[j].first) && (D1_true < v_D1edges[j].second)  &&
              (D2_true > v_D2edges[j].first) && (D2_true < v_D2edges[j].second)  &&
              ( (rtype==1) || (rtype==2) ) ) {
            combSigHist->Fill(j+0.5,wght);
            break;
          }
        }
      }
    }
    combSigHist->Scale(potD/potMC);


    // Get new histos following the reweight, new version lives in the loop above:

    //Old Version (relys on binning being the same across all samples)
    // combSigHist = NULL;
    // for(size_t s=0;s<samples.size();s++)
    // {
    //   ((AnySample*)(samples[s]))->GetSampleBreakdown(fout,Form("toy_%d",t), false);
    //   if(s==0) combSigHist = ((AnySample*)(samples[s]))->GetSignalHisto();
    //   else combSigHist->Add(((AnySample*)(samples[s]))->GetSignalHisto());
    // }


    // Find reco-level chi2 between stat fluct of throw and throw itself: 
    double chi2_rec_stat=0;
    for(size_t s=0;s<samples.size();s++)
    {
      ((AnySample*)(samples[s]))->FillEventHisto(3); // apply stat fluct to the throw (which itself is just a syst fluct)
      //((AnySample*)(samples[s]))->FillEventHisto(1); 
      //calculate chi2 for each sample:
      chi2_rec_stat += ((AnySample*)(samples[s]))->CalcChi2();
      cout << "chi2 of throw and stat fluxt of throw, for toy  " <<  t << " and sample " << s << ", is: " <<  ((AnySample*)(samples[s]))->CalcChi2() << endl;;
    }
    cout << "Final chi2 of throw and stat fluxt of throw, for toy  " <<  t << ", is " << chi2_rec_stat << endl;
    chi2RecoHist_stat->Fill(chi2_rec_stat);

    // Find reco-level chi2 between data and throw: 
    double chi2_rec_data=0;
    for(size_t s=0;s<samples.size();s++)
    {
      ((AnySample*)(samples[s]))->FillEventHisto(2); // apply stat fluct to the throw (which itself is just a syst fluct)
      //calculate chi2 for each sample:
      chi2_rec_data += ((AnySample*)(samples[s]))->CalcChi2();
      cout << "chi2 of throw and stat fluxt of throw, for toy  " <<  t << " and sample " << s << ", is: " <<  ((AnySample*)(samples[s]))->CalcChi2() << endl;;
    }
    cout << "Final chi2 of throw and stat fluxt of throw, for toy  " <<  t << ", is " << chi2_rec_data << endl;
    chi2RecoHist_data->Fill(chi2_rec_data);

    // Compare the two to get a p-value
    chi2RecoHist_comp->Fill(chi2_rec_stat-chi2_rec_data);


    //Next section forms a histogram for each throw of each bin in the MC to propogate an error to the xsec


    integral = combSigHist->Integral();
    pIntegral = combSigHist->Integral(1,8);
    //cout << "Integral is: " << integral << endl;

    // Find total xsec
    sigdistInt->Fill(integral);
    sigdistPint->Fill(pIntegral);

    // Find binned xsec
    if(integralHist==NULL) integralHist = new TH1D("integralHist","integralHist", 10000, integral/2, integral*2);
    integralHist->Fill(integral);
    if(isShapeOnly!=0){
      if(isShapeOnly==1)combSigHist->Scale(1/integral);
      if(isShapeOnly==2)combSigHist->Scale(1/integral,"width");
      for(int b=0;b<(combSigHist->GetNbinsX());b++){
        if(sigdist[b]==NULL) sigdist[b] = new TH1D(Form("signalBin_%d_distribution", b),
                                                   Form("signalBin_%d_distribution", b),
                                                   11000, -0.1, 1); 
        if(sigdistRoll[b]==NULL) sigdistRoll[b] = new TH1D(Form("signalBinRoll_%d_distribution", b),
                                                           Form("signalBinRoll_%d_distribution", b),
                                                           11000, -0.1, 1); 
      }
    }
    for(int b=0;b<(combSigHist->GetNbinsX());b++){
      if(sigdist[b]==NULL) sigdist[b] = new TH1D(Form("signalBin_%d_distribution", b),
                                                 Form("signalBin_%d_distribution", b),
                                                 11000, -1000, 10000); 
      if(sigdistRoll[b]==NULL) sigdistRoll[b] = new TH1D(Form("signalBinRoll_%d_distribution", b),
                                                         Form("signalBinRoll_%d_distribution", b),
                                                         11000, -1000, 10000); 

      sigdist[b]->Fill(combSigHist->GetBinContent(b+1));
      if( b<( (combSigHist->GetNbinsX()) - 3 ) ){
        rollcomb = combSigHist->GetBinContent(b+1) + combSigHist->GetBinContent(b+2) + combSigHist->GetBinContent(b+3);
        sigdistRoll[b]->Fill(rollcomb);
      }
      evt_bin_toy[b][t] = combSigHist->GetBinContent(b+1);
    }

    fout->cd();

    combSigHist->SetNameTitle(Form("AllSampleSignalHisto_toy_%d", t), Form("AllSampleSignalHisto_toy_%d", t));
    //combSigHist->Write();

    // TVectorD throwVec(sigfitparams_throw.size());
    // for(uint i=0; i<sigfitparams_throw.size(); i++)
    //   throwVec(i) = sigfitparams_throw[i];
    // throwVec.Write(Form("sigfitparam_toy_%d",t));

    // throwVec.ResizeTo(detparams_throw.size());
    // for(uint i=0; i<detparams_throw.size(); i++)
    //   throwVec(i) = detparams_throw[i];
    // throwVec.Write(Form("detparam_toy_%d",t));

    // throwVec.ResizeTo(xsecparams_throw.size());
    // for(uint i=0; i<xsecparams_throw.size(); i++)
    //   throwVec(i) = xsecparams_throw[i];
    // throwVec.Write(Form("xsecparam_toy_%d",t));

    // throwVec.ResizeTo(fsiparams_throw.size());
    // for(uint i=0; i<fsiparams_throw.size(); i++)
    //   throwVec(i) = fsiparams_throw[i];
    // throwVec.Write(Form("fsiparam_toy_%d",t));

  }
   ///////////////////////////////////////////////
  

  nomSigHist->Write();
  for(int i=0; i<200; i++){
   if(sigdist[i]!=NULL) sigdist[i]->Write(); 
   if(sigdistRoll[i]!=NULL) sigdistRoll[i]->Write(); 
  }
  sigdistInt->Write();
  sigdistPint->Write();
  integralHist->Write();

  TH1D* SigHistoFinal = new TH1D("SigHistoFinal","SigHistoFinal",nbins,0,nbins); 
  TH1D* SigHistoFinalRelError = new TH1D("SigHistoFinalRelError","SigHistoFinalRelError",nbins,0,nbins); 
  TH1D* SigHistoRollFinal = new TH1D("SigHistoRollFinal","SigHistoRollFinal",nbins-3,0,nbins-3); 
  TH1D* SigHistoIntFinal = new TH1D("SigHistoIntFinal","SigHistoIntFinal",1,0,1); 
  TH1D* SigHistoPintFinal = new TH1D("SigHistoPintFinal","SigHistoPintFinal",1,0,1); 

  SigHistoIntFinal->SetBinContent(1, sigdistInt->GetMean());
  SigHistoIntFinal->SetBinError(1, sigdistInt->GetRMS());
  SigHistoPintFinal->SetBinContent(1, sigdistPint->GetMean());
  SigHistoPintFinal->SetBinError(1, sigdistPint->GetRMS());
  for(int i=0;i<nbins;i++){
    SigHistoFinal->SetBinContent(i+1, sigdist[i]->GetMean());
    SigHistoFinal->SetBinError(i+1, sigdist[i]->GetRMS());
    SigHistoFinalRelError->SetBinContent(i+1, sigdist[i]->GetRMS()/sigdist[i]->GetMean());
    if(i<nbins-3){
      SigHistoRollFinal->SetBinContent(i+1, sigdistRoll[i]->GetMean());
      SigHistoRollFinal->SetBinError(i+1, sigdistRoll[i]->GetRMS());
    }
  }
  SigHistoFinal->Write();
  SigHistoFinalRelError->Write();
  SigHistoRollFinal->Write();
  SigHistoIntFinal->Write();
  SigHistoPintFinal->Write();
  for(int t=0;t<ntoys;t++){
    for(int i=0;i<nbins;i++){
      for(int j=0;j<nbins;j++){
        covar[i][j] += (1/(double)ntoys) * ( evt_bin_toy[i][t] - (sigdist[i]->GetMean()) ) * ( evt_bin_toy[j][t] - (sigdist[j]->GetMean()) );
        covar_norm[i][j] += covar[i][j]/( (sigdist[i]->GetMean()) * (sigdist[j]->GetMean()) );
       //cout << "evt_bin_toy: " << evt_bin_toy[i][t] << endl; 
       //cout << "sigdist[i]->GetMean(): " << sigdist[i]->GetMean() << endl;
       //cout << "covar: " << covar[i][j] << " calc as " << (1/(double)ntoys) << " * " << ( evt_bin_toy[i][t] - (sigdist[i]->GetMean()) ) << " * " << ( evt_bin_toy[j][t] - (sigdist[j]->GetMean()) ) << endl;
      }
    }
  }

  //Calculate Corrolation Matrix
  TMatrixD cormatrix(nbins,nbins);
  for(int r=0;r<nbins;r++){
    for(int c=0;c<nbins;c++){
      cormatrix[r][c]= covar[r][c]/sqrt((covar[r][r]*covar[c][c]));
    }
  }

  covar.Write("covar_xsec");
  covar_norm.Write("covarnorm_xsec");
  cormatrix.Write("corr_xsec");
  chi2RecoHist_data->Write("chi2RecoHist_data");
  chi2RecoHist_stat->Write("chi2RecoHist_stat");
  chi2RecoHist_comp->Write("chi2RecoHist_comp");
  chi2RecoHist_bf->Write("chi2RecoHist_bf");
  chi2RecoHist_bf_data->Write("chi2RecoHist_bf_data");

  TH1D* freqP = new TH1D("freqP", "freqP", 1000, 0, 1);
  TH1D* systWeightedP = new TH1D("systWeightedP", "systWeightedP", 1000, 0, 1);
  double freqP_val = chi2RecoHist_stat->Integral(chi2RecoHist_stat->FindBin(chi2_rec_bf_data), chi2RecoHist_stat->GetNbinsX())/chi2RecoHist_stat->Integral();
  double systWeightedP_val = chi2RecoHist_comp->Integral((chi2RecoHist_comp->FindBin(0)), (chi2RecoHist_comp->GetNbinsX()))/chi2RecoHist_comp->Integral();
  freqP->Fill(freqP_val, ntoys/10);
  systWeightedP->Fill(systWeightedP_val, ntoys/10);
  freqP->Write();
  systWeightedP->Write();

  TH1D* paramResultHisto = new TH1D("paramResultHisto","paramResultHisto",priorVec->GetNrows(),0,priorVec->GetNrows());
  TH1D* paramResultHisto_red = new TH1D("paramResultHisto","paramResultHisto",priorVec->GetNrows()-3,0,priorVec->GetNrows()-3);
  TH1D* prefitParams_red = new TH1D("paramResultHisto","paramResultHisto",priorVec->GetNrows()-3,0,priorVec->GetNrows()-3);
  for(int p=0; p<priorVec->GetNrows(); p++){
    paramResultHisto->SetBinContent(p+1, (*priorVec)(p));
    paramResultHisto->SetBinError(p+1, sqrt((*(covariance))[p][p]));
    if(p<8){ // Accounts for fixed dpt buffer bin
      paramResultHisto_red->SetBinContent(p+1, (*priorVec)(p));
      paramResultHisto_red->SetBinError(p+1, sqrt((*(covariance))[p][p]));
      prefitParams_red->SetBinContent(p+1, prefitParams->GetBinContent(p+1));
      prefitParams_red->SetBinError(p+1, prefitParams->GetBinError(p+1));
    }
    else if(p<83){
      paramResultHisto_red->SetBinContent(p+1, (*priorVec)(p));
      paramResultHisto_red->SetBinError(p+1, sqrt((*(covariance))[p][p]));
      prefitParams_red->SetBinContent(p+1, prefitParams->GetBinContent(p+2));
      prefitParams_red->SetBinError(p+1, prefitParams->GetBinError(p+2));
    }
    else if(p<priorVec->GetNrows()-3){
      paramResultHisto_red->SetBinContent(p+1, (*priorVec)(p+3));
      paramResultHisto_red->SetBinError(p+1, sqrt((*(covariance))[p+3][p+3]));
      prefitParams_red->SetBinContent(p+1, prefitParams->GetBinContent(p+5));
      prefitParams_red->SetBinError(p+1, prefitParams->GetBinError(p+5));
    }
  }
  paramResultHisto->SetMarkerStyle(20);
  paramResultHisto->SetMarkerSize(0.5);
  paramResultHisto_red->SetMarkerStyle(20);
  paramResultHisto_red->SetMarkerSize(0.5);
  paramResultHisto->Write("paramResultHisto");
  paramResultHisto_red->Write("paramResultHisto_red");
  prefitParams->Write("prefitParams");
  prefitParams_red->Write("prefitParams_red");

  //*****************************************
  //*********** XSEC CHI2 SECTION ***********
  //*****************************************

  int nbins_Xsec = 8;

  TVectorD xsecPriorVec(nAnybins);
  for(int i=0; i<nAnybins; i++){ xsecPriorVec(i)=nomSigHist->GetBinContent(i+1); }

  ThrowParms *xsecThrowParms = new ThrowParms(xsecPriorVec,covar);
  TRandom3 *rand_xsec = new TRandom3(seed+1);
  gRandom = rand_xsec;

  TH1D* xsecInToy = new TH1D("xsecInToy", "xsecInToy", nbins_Xsec, 0, nbins_Xsec);
  TH1D* chi2XsecToy = new TH1D("chi2XsecToy", "chi2XsecToy", 40, 0, 40);

  for(int t=0; t<ntoys; t++){
    vector<double> xsec_throw;
    xsecThrowParms->ThrowSet(xsec_throw);
    for(int i=0; i<nbins_Xsec; i++){ xsecInToy->SetBinContent(i+1, xsec_throw[i]);}
    double chi2_xsecToy = calcChi2(xsecInToy, nomSigHist, covar.GetSub(0,nbins_Xsec-1,0,nbins_Xsec-1));
    chi2XsecToy->Fill(chi2_xsecToy);
  }

  chi2XsecToy->Write();
  xsecInToy->Write("xsecInToyExample");


  //Eff+Flux Error hard code:

  TH1D* effFluxErr = new TH1D("effFluxErr", "effFluxErr", nbins_Xsec, 0, nbins_Xsec);
  effFluxErr->SetBinContent(1, 0.086);
  effFluxErr->SetBinContent(2, 0.086);
  effFluxErr->SetBinContent(3, 0.086);
  effFluxErr->SetBinContent(4, 0.086);
  effFluxErr->SetBinContent(5, 0.091);
  effFluxErr->SetBinContent(6, 0.087);
  effFluxErr->SetBinContent(7, 0.088);
  effFluxErr->SetBinContent(8, 0.119);

  effFluxErr->Write("effFluxErr");


  //*****************************************
  //*********** PULLS STUDY SECTION *********
  //*****************************************

  if(isPullStudy!=0){
    cout << "Running pulls study section" << endl;
    // Need to reset nsigparam since we may have had to mess with it to deal with fixed parmaeters:
    nsigparam = sigfitpara.Npar;
    TVectorD *TTrueParamVec;
    TVectorD *FitParamVec  = ((TVectorD*)finput->Get("res_vector"));
    if(isPullStudy==1) TTrueParamVec= ((TVectorD*)finput->Get("paramVector_iter0")); //syst fluct
    else if(isPullStudy==2) { // Stat fluct asimov
      TTrueParamVec = ((TVectorD*)finput->Get("res_vector"));
      for(int j=0; j<(priorVec->GetNrows()); j++) (*TTrueParamVec)[j]=1.000;
    }
    else if(isPullStudy==3) { // Stat fluct fake data
      TTrueParamVec = ((TVectorD*)nomResultFile->Get("res_vector"));
    }
    else cout << "ERROR: invalid -p value!" << endl;
    cout << "True Parameters: "<< endl;
    if(!TTrueParamVec) cout << "WARNING: Failed to read true param vector from fake data file, is the defo a pulls study?" << endl;
    TTrueParamVec->Print();
    vector<double> trueParamVec;
    int npars = TTrueParamVec->GetNrows();
    for(int i=0; i<npars; i++) trueParamVec.push_back((*TTrueParamVec)(i));
    vector<double>::const_iterator first = trueParamVec.begin() + 0;
    vector<double>::const_iterator last =  trueParamVec.begin() + nsigparam;
    vector <double> sigfitparams_true (first,last);
    first = trueParamVec.begin() + nsigparam;
    last =  trueParamVec.begin() + nsigparam + fluxpara.Npar;
    vector <double> fluxparams_true (first,last);
    first = trueParamVec.begin() + nsigparam + fluxpara.Npar;
    last =  trueParamVec.begin() + nsigparam + fluxpara.Npar + detpara.Npar;
    vector <double> detparams_true (first,last);
    first = trueParamVec.begin() + nsigparam + fluxpara.Npar + detpara.Npar;
    last =  trueParamVec.begin() + nsigparam + fluxpara.Npar + detpara.Npar + detpara_fine.Npar;
    vector <double> detparamsfine_true (first,last);
    first = trueParamVec.begin() + nsigparam + fluxpara.Npar + detpara.Npar +detpara_fine.Npar;
    last =  trueParamVec.begin() + nsigparam + fluxpara.Npar + detpara.Npar +detpara_fine.Npar + xsecpara.Npar;
    vector <double> xsecparams_true (first,last);
    first = trueParamVec.begin() + nsigparam + fluxpara.Npar + detpara.Npar +detpara_fine.Npar + xsecpara.Npar;
    last =  trueParamVec.begin() + nsigparam + fluxpara.Npar + detpara.Npar +detpara_fine.Npar + xsecpara.Npar + fsipara.Npar;
    vector <double> fsiparams_true (first,last);
    
    cout<<"CHECK Pulls Study "<<endl;
    cout<<"ccqe size "<<sigfitparams_true.size()<<endl;
    for(int j=0; j<sigfitparams_true.size(); j++){
      cout<<j<<" "<<sigfitparams_true[j]<<endl;
    }
    cout<<"flux size "<<fluxparams_true.size()<<endl;
    for(int j=0; j<fluxparams_true.size(); j++){
      cout<<fluxparams_true[j]<<endl;
    }
    cout<<"det size "<<detparams_true.size()<<endl;
    for(int j=0; j<detparams_true.size(); j++){
      cout<<detparams_true[j]<<endl;
    }
    cout<<"det fine size "<<detparamsfine_true.size()<<endl;
    for(int j=0; j<detparamsfine_true.size(); j++){
      cout<<detparamsfine_true[j]<<endl;
    }
    cout<<"xsec size "<<xsecparams_true.size()<<endl;
    for(int j=0; j<xsecparams_true.size(); j++){
      cout<<xsecparams_true[j]<<endl;
    }
    cout<<"fsi size "<<fsiparams_true.size()<<endl;
    for(int j=0; j<fsiparams_true.size(); j++){
      cout<<fsiparams_true[j]<<endl;
    }

    combSigHist->Reset();
    for(size_t s=0;s<samples.size();s++)
    {
      //loop over events
      for(int i=0;i<samples[s]->GetN();i++)
      {
        AnaEvent* ev = samples[s]->GetEvent(i);
        ev->SetEvWght(ev->GetEvWghtMC()); 
        //do weights for each AnaFitParameters obj
        sigfitpara.ReWeight(ev, s, i, sigfitparams_true);
        fluxpara.ReWeight(ev, s, i, fluxparams_true);
        detpara.ReWeight(ev, s, i, detparams_true);
        //detpara_fine.ReWeight(ev, s, i, detparamsfine_true);
        xsecpara.ReWeight(ev, s, i, xsecparams_true);
        fsipara.ReWeight(ev, s, i, fsiparams_true);

        // Fill histo with reweighted event:
        double D1_true = ev->GetTrueD1trk();
        double D2_true = ev->GetTrueD2trk();
        double wght    = ev->GetEvWght();
        int    rtype   = ev->GetReaction();
        //********************************************
        // Warning: Hardcoded signal definition below:
        //********************************************
        for(int j=0; j<nAnybins; j++){
          if( (D1_true > v_D1edges[j].first) && (D1_true < v_D1edges[j].second)  &&
              (D2_true > v_D2edges[j].first) && (D2_true < v_D2edges[j].second)  &&
              ( (rtype==1) || (rtype==2) ) ) {
            combSigHist->Fill(j+0.5,wght);
            break;
          }
        }
      }
    }
    combSigHist->Scale(potD/potMC);


    // Get new histos following the reweight, new version lives in the loop above:

    //Old Version (relys on binning being the same across all samples)
    // combSigHist = NULL;
    // for(size_t s=0;s<samples.size();s++){
    //   ((AnySample*)(samples[s]))->GetSampleBreakdown(fout,"true", false);
    //   if(s==0) combSigHist = ((AnySample*)(samples[s]))->GetSignalHisto();
    //   else combSigHist->Add(((AnySample*)(samples[s]))->GetSignalHisto());
    //   combSigHist->Print("all");
    // }
    if(isShapeOnly==1){
      combSigHist->Scale(1/(combSigHist->Integral()));
    }

    fout->cd();

    combSigHist->SetNameTitle("AllSampleSignalHisto_true", "AllSampleSignalHisto_true");
    combSigHist->Write();

    TVectorD trueVec(sigfitparams_true.size());
    for(uint i=0; i<sigfitparams_true.size(); i++)
      trueVec(i) = sigfitparams_true[i];
    trueVec.Write("sigfitparam_true");

    trueVec.ResizeTo(detparamsfine_true.size());
    for(uint i=0; i<detparamsfine_true.size(); i++)
      trueVec(i) = detparamsfine_true[i];
    trueVec.Write("detparamfine_true");

    trueVec.ResizeTo(xsecparams_true.size());
    for(uint i=0; i<xsecparams_true.size(); i++)
      trueVec(i) = xsecparams_true[i];
    trueVec.Write("xsecparam_true");

    trueVec.ResizeTo(fsiparams_true.size());
    for(uint i=0; i<fsiparams_true.size(); i++)
      trueVec(i) = fsiparams_true[i];
    trueVec.Write("fsiparam_true");

    TH1D* pullHisto = new TH1D("pullHisto","pullHisto",nbins,0,nbins);
    TH1D* biasHisto = new TH1D("biasHisto","biasHisto",nbins,0,nbins);
    TH1D* pullHistoInt = new TH1D("pullHistoInt","pullHistoInt",1,0,1);
    TH1D* pullHistoPint = new TH1D("pullHistoInt","pullHistoInt",1,0,1);
    TH1D* pullHistoRoll = new TH1D("pullHistoRoll","pullHistoRoll",nbins-3,0,nbins-3);
    Double_t x[100], y[100], yb[100];
    Double_t xRoll[100], yRoll[100];
    Double_t xInt[100], yInt[100];
    Double_t xPint[100], yPint[100];

    pullHistoInt->SetBinContent(1,( ( (SigHistoIntFinal->GetBinContent(1))-(combSigHist->Integral()) )/((SigHistoIntFinal->GetBinError(1))) ));
    xInt[0]=1;
    yInt[0]= ((SigHistoIntFinal->GetBinContent(1))-(combSigHist->Integral()) )/(SigHistoIntFinal->GetBinError(1));

    pullHistoPint->SetBinContent(1,( ( (SigHistoPintFinal->GetBinContent(1))-(combSigHist->Integral(1,8)) )/((SigHistoPintFinal->GetBinError(1))) ));
    xPint[0]=1;
    yPint[0]= ((SigHistoPintFinal->GetBinContent(1))-(combSigHist->Integral(1,8)) )/(SigHistoPintFinal->GetBinError(1));

    for(int i=0;i<nbins;i++){
      pullHisto->SetBinContent(i+1,( ( (SigHistoFinal->GetBinContent(i+1))-(combSigHist->GetBinContent(i+1)) )/((SigHistoFinal->GetBinError(i+1)  )) ));
      biasHisto->SetBinContent(i+1,( ( (SigHistoFinal->GetBinContent(i+1))-(combSigHist->GetBinContent(i+1)) )/((SigHistoFinal->GetBinContent(i+1))) ));
      x[i]=i+1;
      y[i]=  ( (SigHistoFinal->GetBinContent(i+1))-(combSigHist->GetBinContent(i+1)) )/((SigHistoFinal->GetBinError(i+1)));
      yb[i]=  ( (SigHistoFinal->GetBinContent(i+1))-(combSigHist->GetBinContent(i+1)) )/((SigHistoFinal->GetBinContent(i+1)));
      if(i<nbins-3){
        rollcomb = combSigHist->GetBinContent(i+1)+combSigHist->GetBinContent(i+2)+combSigHist->GetBinContent(i+3);
        pullHistoRoll->SetBinContent(i+1,( ( (SigHistoRollFinal->GetBinContent(i+1))-rollcomb)/((SigHistoRollFinal->GetBinError(i+1))) ));
        xRoll[i]=i+1;
        yRoll[i]=( (SigHistoRollFinal->GetBinContent(i+1))-rollcomb)/((SigHistoRollFinal->GetBinError(i+1)));
      }
    }
    TGraph* pullGr= new TGraph(nbins,x,y);
    TGraph* biasGr= new TGraph(nbins,x,yb);
    TGraph* pullRollGr= new TGraph(nbins-3,xRoll,yRoll);
    TGraph* pullIntGr= new TGraph(1,xInt,yInt);
    TGraph* pullPintGr= new TGraph(1,xPint,yPint);

    /*******************************************/
    // Calc chi2:
    /*******************************************/

    // Calc chi2 from xsec comp
    TH1D* chi2Histo = new TH1D("chi2Histo","chi2Histo",1,0,1);
    TH1D* chi2Histo_xsecBins = new TH1D("chi2Histo_xsecBins","chi2Histo_xsecBins",1,0,1);
    TH1D* SigHistoFinal_xsecBins = new TH1D("SigHistoFinal_xsecBins", "SigHistoFinal_xsecBins", nbins_Xsec, 0, nbins_Xsec);
    for(int i=0; i<nbins_Xsec; i++) {SigHistoFinal_xsecBins->SetBinContent(i+1, SigHistoFinal->GetBinContent(i+1));}

    chi2Histo->SetBinContent(1, calcChi2(SigHistoFinal,combSigHist,covar) );
    chi2Histo_xsecBins->SetBinContent(1, calcChi2(SigHistoFinal_xsecBins,combSigHist,covar.GetSub(0,nbins_Xsec-1,0,nbins_Xsec-1)) );
    chi2Histo->Write("chi2");
    chi2Histo_xsecBins->Write("chi2_xsecBins");

    // Calc chi2 from param comp

    //Remove Fixed Parameters:
    cout << "Altering param vector to account for fixed parameters ... " << endl;
    if(ntemplatefixed!=0 || rmFineDet!=0){
      const int nparamsnow = FitParamVec->GetNrows() - ntemplatefixed;
      double arrTrue[nparamsnow];
      double arrFit[nparamsnow];
      for(int i=0;i<nparamsnow;i++){
        if(i<firstFixed){
          arrTrue[i] = (*TTrueParamVec)(i);
          arrFit[i] = (*FitParamVec)(i);
        }
        else{
          arrTrue[i] = (*TTrueParamVec)(i+ntemplatefixed);
          arrFit[i] = (*FitParamVec)(i+ntemplatefixed);
        }
      } 
      // Relevent if no fine det params in covar matrix (need to remove them from the vector):
      const int nparamsnow_rmdetfine = FitParamVec->GetNrows() - ntemplatefixed - ndetParaFineParams;
      double arr_rmdetfine_true[nparamsnow_rmdetfine];
      double arr_rmdetfine_fit[nparamsnow_rmdetfine];
      for(int i=0;(i<nparamsnow_rmdetfine) && (rmFineDet==1);i++){
        if(i<firstFixed && i < sigfitpara.Npar + fluxpara.Npar + detpara.Npar){
          arr_rmdetfine_true[i] = (*TTrueParamVec)(i);
          arr_rmdetfine_fit[i] = (*FitParamVec)(i);
        }
        else if (i < sigfitpara.Npar + fluxpara.Npar + detpara.Npar){
          arr_rmdetfine_true[i] = (*TTrueParamVec)(i+ntemplatefixed);
          arr_rmdetfine_fit[i] = (*FitParamVec)(i+ntemplatefixed);
        }
        else{
          arr_rmdetfine_true[i] = (*TTrueParamVec)(i+ntemplatefixed+ndetParaFineParams);
          arr_rmdetfine_fit[i] = (*FitParamVec)(i+ntemplatefixed+ndetParaFineParams);
        } 
      }
      delete TTrueParamVec;
      delete FitParamVec;
      if(rmFineDet==0){
        TTrueParamVec = new TVectorD(nparamsnow, arrTrue);
        FitParamVec = new TVectorD(nparamsnow, arrFit);
      }
      if(rmFineDet==1){
        cout << "removed fine det params, size of param vector is now: " << nparamsnow_rmdetfine << endl;
        TTrueParamVec = new TVectorD(nparamsnow_rmdetfine, arr_rmdetfine_true);
        FitParamVec = new TVectorD(nparamsnow_rmdetfine, arr_rmdetfine_fit);
      }
    }
    // End removal of fixed params

    //Dimensionality check:
    if(FitParamVec->GetNrows() != covariance->GetNrows()){
      cout << "ERROR: param vector and covariance matrix have different dimensions ..." << endl;
      cout << "DEBUG info: " << endl;
      cout << "  param vector: " << FitParamVec->GetNrows() << endl;
      cout << "  covar : " << covariance->GetNrows() << endl;
      cout << "  rmFineDet is " << rmFineDet << endl;
      return 0;
    }

    TH1D* TTrueParamHisto = new TH1D("TTrueParamHisto","TTrueParamHisto",TTrueParamVec->GetNrows(),0,TTrueParamVec->GetNrows());
    TH1D* FitParamHisto = new TH1D("FitParamHisto","FitParamHisto",FitParamVec->GetNrows(),0,FitParamVec->GetNrows());
    for(int p=0; p<FitParamVec->GetNrows(); p++){
      TTrueParamHisto->SetBinContent(p+1, (*TTrueParamVec)(p));
      TTrueParamHisto->SetBinError(p+1, 0);
      FitParamHisto->SetBinContent(p+1, (*FitParamVec)(p));
      FitParamHisto->SetBinError(p+1, (*(covariance))[p][p]);
    }

    TTrueParamHisto->Write("TTrueParamHisto");
    FitParamHisto->Write("FitParamHisto");

    TMatrixD* covarianceTMD = (TMatrixD*)covariance;
    TH1D* chi2Histo_param = new TH1D("chi2Histo_param","chi2Histo_param",1,0,1);
    chi2Histo_param->SetBinContent(1, calcChi2(TTrueParamHisto,FitParamHisto,(*covarianceTMD)) );
    chi2Histo_param->Write("chi2_param");

    TH1D* chi2Histo_minuit = new TH1D("chi2Histo_minuit","chi2Histo_minuit",1,0,1);
    chi2Histo_minuit->SetBinContent(1, minuitchi2);
    chi2Histo_minuit->Write("chi2Histo_minuit");

    TH1D* chi2Histo_minuitParamComp = new TH1D("chi2Histo_minuitParamComp","chi2Histo_minuitParamComp",1,0,1);
    chi2Histo_minuitParamComp->SetBinContent(1, chi2Histo_minuit->GetBinContent(1) - chi2Histo_param->GetBinContent(1));
    chi2Histo_minuitParamComp->Write("chi2Histo_minuitParamComp");   



    /*******************************************/
    // End Calc chi2:
    /*******************************************/
    pullGr->Write("pullGr");
    biasGr->Write("biasGr");
    pullRollGr->Write("pullRollGr");
    pullIntGr->Write("pullIntGr");
    pullPintGr->Write("pullPintGr");

    pullHisto->Write("pullHisto");
    biasHisto->Write("biasHisto");
    pullHistoRoll->Write("pullHistoRoll");
    pullHistoInt->Write("pullHistoInt");
    pullHistoPint->Write("pullHistoPint");

  }

  fout->Close();

  cout << "**********************" << endl;
  cout << "PropError all finished" << endl;
  cout << "**********************" << endl;

  return 0;
}

double calcChi2(TH1D* h1, TH1D* h2, TMatrixD covar){
  double chi2=0;
  //cout << "Determinant of covar is: " << covar.Determinant() << endl;
  //covar.Print();
  covar.SetTol(1e-200);
  covar.Invert();
  //covar.Print();
  for(int i=0; i<h1->GetNbinsX(); i++){
    for(int j=0; j<h1->GetNbinsX(); j++){
      chi2+= ((h1->GetBinContent(i+1)) - (h2->GetBinContent(i+1)))*covar[i][j]*((h1->GetBinContent(j+1)) - (h2->GetBinContent(j+1)));
    }
  }
  return chi2;
}

