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

#include "../fitparam/include/FluxParameters.hh"
#include "FluxParameters_norm.hh"
#include "../fitparam/include/DetParameters.hh"
#include "FSIParameters.hh"
#include "NuclFSIParameters.hh"
#include "../fitparam/include/XsecParameters.hh"
#include "../fitparam/include/FitParameters.hh"
#include "../xsecfit/include/XsecFitter.hh"
#include "anyTreeMC.hh"
#include "AnySample.hh"


using namespace std;

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
  //string fakeData     = inputDir + "/GenieAir2_2DV2.root";
  string fakeData     = inputDir + "/NeutAir5_2DV2.root";
  string ffluxcov     = inputDir + "/flux_covariance_banff_13av1.1.root";
  string fdetcov_fine = inputDir + "/dptCovMat500Toys.root";
  //string fdetcov = inputDir + "/dptCovMat500Toys.root";
  string fdetcov = inputDir + "/newFitsFeb17/dpt/covMatFromParaOut.root";
  string fxsecbinning = inputDir + "/dptbinning2DPS_shortest_inclusive.txt";
  string fxsecbinning_coarse = inputDir + "/dptbinning2DPS_shortest_inclusive_coarse.txt";
  string fnameout = "fitresults.root";
  string paramVectorFname = "fitresults.root";

  //Optional binnings to improve GoF in sidebands: 
  string onePiAltBinning = inputDir + "/dptbinning2DPS_shortest_inclusive_1piFix.txt";
  string disAltBinning = inputDir + "/dptbinning2DPS_shortest_inclusive_disFix.txt";

  //double potD     = 372.67;   //in units of 10^19 GENIE Air
  //double potD     = 349.15;   //in units of 10^19 NEUT Water
  //double potD     = 58.11;   //in units of 10^19 runs data
  //double potD     = 61.17;   //in units of 10^19 runs data done badly
  //double potD     = 331.6; //in units 10^19 Neut Air
  //double potD     = 66.8151 // FHC V1 data
  //double potD     = 47.7323; //first pass data 
  double potD     = 331.6; //in units 10^19 Neut Air
  double potMC    = 331.6; //in units 10^19 Neut Air
  //Genie Neut air ratio is 1.12385
  //Neut water air ratio is 1.05293
  //Neut air data ratio is 0.1752
  //Genie data ratio is 0.1559
  //Neut air bad data ratio is 0.1844
  //Neut air first pass ratio is 0.1439
  //Neut air rdp FHC v1 ratio is 0.2015
  //Genie air rdp FHC v1 ratio is 0.17929
  int seed        = 1019;
  double regparam = 0.001;
  double regparam2 = 0.001;
  int fitmode = 2;
  int USESYST = 1;
  int fitMethod = 2;
  int statFluct = 1;
  int nbins = 0;
  int nbins_coarse = 0;
  int nbins_1pi = 0;
  int nbins_dis = 0;
  int pmode = 0;
  int templateFit=1;
  int useControlRegions=1;
  int useAltBinning=0;

  bool isAltPrior=false;

  bool isBuffer=true;

  //get command line options
  char cc;
  while((cc = getopt(argc, argv, "i:e:b:B:o:n:N:s:S:h:f:r:R:m:y:M:p:v:t:c:P:D:A:")) != -1)
  {
    switch(cc)
    {
      case 'i': //selected events
        fsel = optarg;
        break; 
      case 'e': //flux covariance matrix
        ffluxcov = optarg;
        break;
      case 'd': //det covariance matrix
        fdetcov = optarg;
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
      case 'n': //data POT
        potD = atof(optarg);
        break;
      case 'N': //MC POT
        potMC = atof(optarg);
        break;
      case 's': //random seed
        seed = atoi(optarg);
        break;
      case 'S': //random seed
        statFluct = atoi(optarg);
        break;
      case 'f': //fake data file
        fakeData = optarg;
        break;
      case 'r': //regularisation param
        regparam = atof(optarg);
        break;
      case 'R': //regularisation param
        regparam2 = atof(optarg);
        break;
      case 'm': //regularisation param
        fitmode = atoi(optarg);
        break;
      case 'y': //regularisation param
        USESYST = atoi(optarg);
        break;
      case 'M': //regularisation param
        fitMethod = atoi(optarg);
        break;
      case 'p': //regularisation param
        pmode = atoi(optarg);
        break;
      case 'v': //regularisation param
        paramVectorFname = optarg;
        break;
      case 't': //template or model param fit?
        templateFit = atoi(optarg);
        break;
      case 'c': //template or model param fit?
        useControlRegions = atoi(optarg);
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
                  << "-b : \tset xec bin definitions (fine)" << std::endl
                  << "-B : \tset xec bin definitions (coarse)" << std::endl
                  << "-o : \tset name of output file" << std::endl
                  << "-n : \tset POT for data in units 10**19" << std::endl
                  << "-N : \tset POT for MC in units 10**19" << std::endl
                  << "-s : \tset random seed" << std::endl
                  << "-f : \tset (fake) data location" << std::endl
                  << "-r : \tset regularisation parameter for IPS bins" << std::endl
                  << "-R : \tset regularisation parameter for OOPS bins" << std::endl
                  << "-m : \tset fitter mode" << std::endl
                  << "-y : \tset USESYST mode (1 for syt+stat, 0 for stats only)" << std::endl
                  << "-M : \tset fitter method (MIGRAD/HESSE/MINOS)" << std::endl
                  << "-A : \tset wether to use alt SB binnings" << std::endl
                  << "-D : \tset DIS SB alt binning" << std::endl
                  << "-P : \tset 1Pi SB alt binning" << std::endl
                  << "-p : \tset whether nuisance parameters can alter the signal (default is they can, mode 0)" << std::endl;
        return 0;
        break;
      default:
        return 1;
    }
  }

  if(fitmode==10) isAltPrior = true;

  //get info on the priors 
  TFile *finfluxcov = TFile::Open(ffluxcov.c_str()); //contains flux and det. systematics info
  TFile *findetcov = TFile::Open(fdetcov.c_str()); //contains flux and det. systematics info
  TFile *findetcov_fine = TFile::Open(fdetcov_fine.c_str()); //contains flux and det. systematics info

  if(!finfluxcov){
    cout << "WARNING: Could not find flux covariance matrix file, press any key to continue" << endl;
    getchar();
  }
  if(!findetcov){
    cout << "WARNING: Could not find det covariance matrix file, press any key to continue" << endl;
    getchar();
  }
  if(!findetcov_fine){
    cout << "WARNING: Could not find det_fine covariance matrix file, press any key to continue" << endl;
    getchar();
  }

  /******************* Print Chosen Options to Screen ***********************************/

    cout << "MC location is: " << fsel << endl;
    cout << "Fake Data location is: " << fakeData << endl;
    cout << "MC POT is: " << potMC << endl;
    cout << "Fake Data POT is: " << potD << endl;
    cout << "Regularisation Strength (IPS) is: " << regparam << endl;
    cout << "Regularisation Strength (OOPS) is: " << regparam2 << endl;
    cout << "Output file is: " << fnameout << endl;
    cout << "Fitter mode is: " << fitmode << endl;
    cout << "Fitter method is: " << fitMethod << endl << endl;

    /******************* Setup data trees and binning ***********************************/

    // Setup data trees

    cout << "*********************************" << endl;
    cout << " Setup data trees and binning " << endl;
    cout << "*********************************" << endl << endl;

    TFile *fdata = new TFile(TString(fakeData)); 
    if(!finfluxcov){
      cout << "ERROR: Could not find fake data file (you can just provide the MC file again if you want to make fake data on the fly)" << endl;
      return 0;
    }
    TTree *tdata = (TTree*)(fdata->Get("selectedEvents"));
    if(!tdata){
      cout << "ERROR: Could not find selectedEvents tree in fake data file " << endl;
      return 0;
    }

    //Set up bin edges for fine binning
    std::vector<std::pair<double, double> > v_D1edges;
    std::vector<std::pair<double, double> > v_D2edges;
    ifstream fin(fxsecbinning.c_str());
    if(!fin.is_open()) cout << "WARNING: Binning file \"" << fxsecbinning << "\" could not be opened" << endl;
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
    }
    fin.close();

    cout << "Found "  << nbins << " bins for fine binning" << endl;

    //Set up bin edges for coarse binning
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

    cout << "*********************************" << endl;
    cout << " Setup Flux " << endl;
    cout << "*********************************" << endl << endl;
  
  //setup enu bins and covm for flux 
  TAxis *nd_numu_bins = (TAxis*)finfluxcov->Get("nd5_numode_numu_bins");
  TMatrixDSym *cov_flux_in   = (TMatrixDSym*)finfluxcov->Get("total_flux_cov");
  TMatrixDSym cov_flux(nd_numu_bins->GetNbins());
  vector<double> enubins;
  enubins.push_back(nd_numu_bins->GetBinLowEdge(1));
  for(int i=0;i<nd_numu_bins->GetNbins();i++)
  {
    enubins.push_back(nd_numu_bins->GetBinUpEdge(i+1));
    for(int j=0;j<nd_numu_bins->GetNbins();j++)
    {
      cov_flux(i, j) = (*cov_flux_in)(i,j);
    }
  }
  
  finfluxcov->Close();
  /*****************************************************************************************/

  /********************************COV.MATRIX FOR DETECTOR SYSTEMATICS*********************/

  cout << "*********************************" << endl;
  cout << " Setup Cov Matrix for Det Syst " << endl;
  cout << "*********************************" << endl << endl;

  //setup D1,D2 bins, starting param values and covm for det syst --------------

  //TVectorD* det_weights = (TVectorD*)findetcov->Get("det_weights");

  // Currently some rather horrible hard coding to make a vector of 1s of the correct size
  // This should really be an input to the fitter, WIP! ***

  cout << " Coarse " << endl;

  const int ndetcovmatele = 54;

  double arr[ndetcovmatele];
  for(int i=0; i<ndetcovmatele; i++){ arr[i]=(1.0);}

  TVectorD* det_weights = new TVectorD(ndetcovmatele, arr);

  cout << " Input det weights " << endl;
  det_weights->Print();

  // ***
 
  TMatrixDSym *cov_det_in   = (TMatrixDSym*)findetcov->Get("covMat_mean_norm");

  //Turn absolute covar matrix into a relative error covar matrix:
  //TMatrixDSym *cov_det   = (TMatrixDSym*)findetcov->Get("covMat");
  //TMatrixDSym *cov_det_in   = new TMatrixDSym(ndetcovmatele);
  //for(size_t m=0; m<cov_det->GetNrows(); m++){
  //  for(size_t k=0; k<cov_det->GetNrows(); k++){


  //Temp hack to remove samples from matrix, more hard coding for my CC0Pi ***
    //TMatrixDSym *cov_det_now   = (TMatrixDSym*)findetcov->Get("covMat_norm");
    //TMatrixDSym *cov_det_in   = new TMatrixDSym(ndetcovmatele);
    //cov_det_now->GetSub(0,(ndetcovmatele-1),*cov_det_in);
    //cout << " Input det cov matrix from file " << endl;
    //cov_det_in->Print();
  // ***


  if(!cov_det_in) cout << "Warning! Problem opening detector cov matrix" << endl;
  TMatrixDSym cov_det(cov_det_in->GetNrows());
  for(size_t m=0; m<cov_det_in->GetNrows(); m++){
    for(size_t k=0; k<cov_det_in->GetNrows(); k++){
      cov_det(m, k) = (*cov_det_in)(m,k);
      if((m==k) && cov_det(m, k) < 0.000001) cov_det(m, k)=1.0; // Remove 0 from bins that don't do anything
      //if(m==k) cov_det(m, k)= 1.2*(cov_det(m, k))+0.0001; // Add small terms to the diag to make matrix invertible
      //if((m==k) && cov_det(m, k) < 0.0001) cov_det(m, k)=0.0001; // Add small terms to the diag to make matrix invertible
      //May need to add small terms to the last sample to make thh matrix invertable:
      //if((m>19)&&(k>19)&&(m==k)) cov_det(m, k) = 1.1*((*cov_det_in)(m,k));
    }
  }

  cov_det.SetTol(1e-160);
  cout << " Input det cov matrix " << endl;
  cov_det.Print();
  double det = cov_det.Determinant();

  //cout << " Inverted cov mat: " << endl;
  //cov_det.Invert();
  //cov_det.Print();
  //return 0;   

  if(abs(det) < 1e-160){
    cout << "Warning, det cov matrix is non invertable. Det is:" << endl;
    cout << det << endl;
    return 0;   
  }  


  findetcov->Close();

  // ***** Fine binning: *****

  cout << " Fine " << endl;

  //TVectorD* det_weights_fine = (TVectorD*)findetcov_fine->Get("det_weights");

  // Currently some rather horrible hard coding to make a vector of 1s of the correct size
  // This should really be an input to the fitter, WIP! ***

  const int ndetcovmatele_fine = 30; //48

  double arr_fine[ndetcovmatele_fine];
  for(int i=0; i<ndetcovmatele_fine; i++){ arr_fine[i]=(1.0);}

  TVectorD* det_weights_fine = new TVectorD(ndetcovmatele_fine, arr_fine);

  cout << " Input det weights " << endl;
  det_weights_fine->Print();

  //***

  //TMatrixDSym *cov_det_in_fine   = (TMatrixDSym*)findetcov_fine->Get("covMat_norm");


  //Temp hack to remove samples from matrix ***
  TMatrixDSym *cov_det_now_fine   = (TMatrixDSym*)findetcov_fine->Get("covMat_norm");
  TMatrixDSym *cov_det_in_fine   = new TMatrixDSym(ndetcovmatele_fine);
  cov_det_now_fine->GetSub(0,(ndetcovmatele_fine-1),*cov_det_in_fine);
  cout << " Input det cov matrix from file " << endl;
  cov_det_in_fine->Print();
  // ***

  //TMatrixDSym *cov_det_in_fine   = (TMatrixDSym*)findetcov_fine->Get("cov_det");

  TMatrixDSym cov_det_fine(cov_det_in_fine->GetNrows());
  for(size_t m=0; m<cov_det_in_fine->GetNrows(); m++){
    for(size_t k=0; k<cov_det_in_fine->GetNrows(); k++){
      cov_det_fine(m, k) = (*cov_det_in_fine)(m,k);
      if((m==k) && cov_det_fine(m, k) < 0.0001) cov_det_fine(m, k)=0.0005; // Add small terms to the diag to make matrix invertible
    }
  }

  cov_det_fine.SetTol(1e-100);
  //cout << " Input det cov matrix " << endl;
  //cov_det_fine.Print();
  double det_fine = cov_det_fine.Determinant();

  if(abs(det_fine) < 1e-100){
    cout << "Warning, det cov matrix with fine binning is non invertable. Det is:" << endl;
    cout << det_fine << endl;
    return 0;   
  }  

  findetcov_fine->Close();

  
  /***********************************************************************************/

  /********************************XSEC RESPONSE FUNCTIONS****************************/

  // Currently only deals with BG xsec model params


  cout << "*********************************" << endl;
  cout << " Setup Xsec Resp Functions " << endl;
  cout << "*********************************" << endl << endl;

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

  if(templateFit!=1){ // If fitting model params use CCQE params, if using tempalte fit use dummy params
    TFile* MAQErespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NXSec_MaCCQE_allVariation.root").c_str());
    responsefunctions.push_back(MAQErespfunc);
    TFile* pFrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWG2014a_pF_C12_allVariation.root").c_str());
    responsefunctions.push_back(pFrespfunc);
    TFile* MECrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWGMEC_Norm_C12_allVariation.root").c_str());
    responsefunctions.push_back(MECrespfunc);
  }
  else{
    TFile* MAQErespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NXSec_MaCCQE_allVariation.rootDummy").c_str());
    responsefunctions.push_back(MAQErespfunc);
    TFile* pFrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWG2014a_pF_C12_allVariation.rootDummy").c_str());
    responsefunctions.push_back(pFrespfunc);
    TFile* MECrespfunc = new TFile((inputDir + "/responsefuncSD/NoRecBinning/NIWGMEC_Norm_C12_allVariation.rootDummy").c_str());
    responsefunctions.push_back(MECrespfunc);
  }

  TMatrixDSym cov_xsec(12);

  // Cov mat for xsec should be an input, not hard coded, WIP!

  cov_xsec(0,0) = 0.01412; //CA5Res
  //cov_xsec(0,0) = 0.0625; //CA5Res Boosted to allow extra freedom in sidebands
  cov_xsec(0,1) = 0.0;
  cov_xsec(0,2) = 0.0;
  cov_xsec(0,3) = 0.0;
  cov_xsec(0,4) = 0.0;
  cov_xsec(0,5) = 0.0;
  cov_xsec(0,6) = 0.0;
  cov_xsec(0,7) = 0.0;
  cov_xsec(0,8) = 0.0;
  cov_xsec(0,9) = 0.0;
  cov_xsec(0,10) = 0.0;
  cov_xsec(0,11) = 0.0;

  cov_xsec(1,1) = 0.02493; //MARES
  //cov_xsec(1,1) = 0.03905; //MARES Boosted to account for CA5 degeneracy
  //cov_xsec(1,1) = 0.117; //MARES Boosted to allow extra freedom in sidebands
  cov_xsec(1,2) = 0;
  cov_xsec(1,3) = 0;
  cov_xsec(1,4) = 0;
  cov_xsec(1,5) = 0;
  cov_xsec(1,6) = 0;
  cov_xsec(1,7) = 0;
  cov_xsec(1,8) = 0;
  cov_xsec(1,9) = 0;
  cov_xsec(1,10) = 0;
  cov_xsec(1,11) = 0;

  cov_xsec(2,2) = 0.02367; //BgRES
  //cov_xsec(2,2) = 0.09468; //BgRES boosted to allow extra freedom in sidebands
  cov_xsec(2,3) = 0;
  cov_xsec(2,4) = 0;
  cov_xsec(2,5) = 0;
  cov_xsec(2,6) = 0;
  cov_xsec(2,7) = 0;
  cov_xsec(2,8) = 0;
  cov_xsec(2,9) = 0;
  cov_xsec(2,10) = 0;
  cov_xsec(2,11) = 0;

  cov_xsec(3,3) = 0.0004; //CCNUE_0
  cov_xsec(3,4) = 0;
  cov_xsec(3,5) = 0;
  cov_xsec(3,6) = 0;
  cov_xsec(3,7) = 0; 
  cov_xsec(3,8) = 0;
  cov_xsec(3,9) = 0;
  cov_xsec(3,10) = 0;
  cov_xsec(3,11) = 0;

  cov_xsec(4,4) = 0.16; //dismpishp
  //cov_xsec(4,4) = 0.64; //dismpishp  boosted to allow extra freedom in sidebands
  cov_xsec(4,5) = 0;
  cov_xsec(4,6) = 0;
  cov_xsec(4,7) = 0;
  cov_xsec(4,8) = 0;
  cov_xsec(4,9) = 0;
  cov_xsec(4,10) = 0;
  cov_xsec(4,11) = 0;

  //cov_xsec(5,5) = 1.0; //CCCOH
  cov_xsec(5,5) = 5.0; //CCCOH  boosted to allow extra freedom in sidebands
  cov_xsec(5,6) = 0;
  cov_xsec(5,7) = 0;
  cov_xsec(5,8) = 0;
  cov_xsec(5,9) = 0;
  cov_xsec(5,10) = 0;
  cov_xsec(5,11) = 0;

  cov_xsec(6,6) = 0.09; //NCCOH
  cov_xsec(6,7) = 0;
  cov_xsec(6,8) = 0;
  cov_xsec(6,9) = 0;
  cov_xsec(6,10) = 0;
  cov_xsec(6,11) = 0;
  
  cov_xsec(7,7) = 0.09; //NCOTHER
  cov_xsec(7,8) = 0;
  cov_xsec(7,9) = 0;
  cov_xsec(7,10) = 0;
  cov_xsec(7,11) = 0;

  cov_xsec(8,8) = 0.1296; //Eb_C12 
  cov_xsec(8,9) = 0;
  cov_xsec(8,10) = 0;
  cov_xsec(8,11) = 0;

  cov_xsec(9,9) = 0.0214069; //MAQE 
  cov_xsec(9,10) = 0;  
  cov_xsec(9,11) = 0.05314; 

  cov_xsec(10,10) = 0.0204446; //PF 
  cov_xsec(10,11) = 0; 
  
  cov_xsec(11,11) = 0.5373; //MEC_C

  if(templateFit!=1){
    cov_xsec(9,9) = 3.0; //MAQE 
    cov_xsec(9,11) = 0; 
    cov_xsec(10,10) = 3.0; //PF 
    cov_xsec(11,11) = 3.0; //MEC_C
  }

  cov_xsec.SetTol(1e-200);
  cov_xsec.Print();
  double det_cov_xsec = cov_xsec.Determinant();

  if(abs(det_cov_xsec) < 1e-200){
    cout << "Warning, xsec cov matrix is non invertable. Det is:" << endl;
    cout << det_cov_xsec << endl;
    return 0;   
  }  

          
  /*****************************************************************************************/


  /************************FSI RESPONSE FUNCTIONS*****************************************/

  cout << "*********************************" << endl;
  cout << " Setup Pion FSI Resp Functions " << endl;
  cout << "*********************************" << endl << endl;

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
  // Cov mat for fsi should be an input, not hard coded, WIP!
  
  // Taken from BANFF prefit matrix
  TMatrixDSym cov_fsi(6);
  cov_fsi(0,0)= 0.17;
  cov_fsi(0,1)= -0.002778;
  cov_fsi(0,2)= 0;
  cov_fsi(0,3)= 0.02273;
  cov_fsi(0,4)= 0.005;
  cov_fsi(0,5)= 0;
  cov_fsi(1,1)= 0.1142;
  cov_fsi(1,2)= -0.1667;
  cov_fsi(1,3)= -0.001263;
  cov_fsi(1,4)= -0.002083;
  cov_fsi(1,5)=  -0.09259;
  cov_fsi(2,2)= 0.25;
  cov_fsi(2,3)= -5.204e-18;
  cov_fsi(2,4)= 0;
  cov_fsi(2,5)= 0.1389;
  cov_fsi(3,3)= 0.1694;
  cov_fsi(3,4)= -0.002273;
  cov_fsi(3,5)= -3.469e-18;
  cov_fsi(4,4)= 0.3213;
  cov_fsi(4,5)= 1.735e-18;
  cov_fsi(5,5)= 0.07716;

  // //Need to add 0.001 to the diagonal, otherwise not positive definite
   cov_fsi(0,0)= cov_fsi(0,0)+0.001;
   cov_fsi(1,1)= cov_fsi(1,1)+0.001;
   cov_fsi(2,2)= cov_fsi(2,2)+0.001;
   cov_fsi(3,3)= cov_fsi(3,3)+0.001;
   cov_fsi(4,4)= cov_fsi(4,4)+0.001;
   cov_fsi(5,5)= cov_fsi(5,5)+0.001;

   cov_fsi.SetTol(1e-200);
   cov_fsi.Print();
   double det_cov_fsi = cov_fsi.Determinant();

   if(abs(det_cov_fsi) < 1e-200){
     cout << "Warning, pion FSI cov matrix is non invertable. Det is:" << endl;
     cout << det_cov_fsi << endl;
     return 0;   
   }  
           
  /*********************************************************************************/
  
  /************************Nucleon FSI RESPONSE FUNCTIONS*****************************/

  /************************Currently don't work, WIP ********************************/

  vector<TFile*> responsefunctions_NuclFSI;
  /*TFile* NMFPrespfunc = new TFile("../inputs/responsefunc/N_MFP_all_variation.root");
  responsefunctions_NuclFSI.push_back(NMFPrespfunc);
  TFile* NFrElasrespfunc = new TFile("../inputs/responsefunc/N_FrElas_all_variation.root");
  responsefunctions_NuclFSI.push_back(NFrElasrespfunc);
  TFile* NFrAbsrespfunc = new TFile("../inputs/responsefunc/N_FrAbs_all_variation.root");
  responsefunctions_NuclFSI.push_back(NFrAbsrespfunc);*/

  //NuclFSI parameters are not correlated for now 
  TMatrixDSym cov_nuclfsi(3);
  cov_nuclfsi(0,0)= 0.20*0.20;
  cov_nuclfsi(0,1)= 0;
  cov_nuclfsi(0,2)= 0;
  cov_nuclfsi(1,1)= 0.3*0.3;
  cov_nuclfsi(1,2)= 0;
  cov_nuclfsi(2,2)= 0.20*0.20;

  cov_nuclfsi.SetTol(1e-200);
  cov_nuclfsi.Print();
  double det_cov_nuclfsi = cov_nuclfsi.Determinant();

  if(abs(det_cov_nuclfsi) < 1e-200){
    cout << "Warning, nucleon FSI cov matrix is non invertable. Det is:" << endl;
    cout << det_cov_nuclfsi << endl;
    return 0;   
  } 

  /*********************************************************************************/
  
  /************************Regularisation Cov Matrix********************************/

  /*********************************************************************************/ 

  cout << "*********************************" << endl;
  cout << " Setup Regularisation Cov Matrix " << endl;
  cout << "*********************************" << endl << endl;

  //const int nabins = 17;
  const int nabins = nbins;
  const int nprimbins = 8;
  //const int nprimbins = nbins;
  TMatrixDSym cov_reg(nabins);
  for(int j=0; j<nabins; j++){
    for(int i=0; i<nabins; i++){
      if(i>nprimbins || j>nprimbins){
        if(j==i) cov_reg(i,j)=1;
        else cov_reg(i,j)=0;
      }
      else{
        if(j==i) cov_reg(i,j)=2*regparam;
        else if(j==i+1) cov_reg(i,j)=-1*regparam;
        else if(j==i-1) cov_reg(i,j)=-1*regparam;
        else cov_reg(i,j)=0;
      }
    }
  }
  cov_reg(0,0) = 1*regparam;



  cov_reg.SetTol(1e-100);
  cout << "Cov Reg is: " << endl;
  cov_reg.Print();
  double det_reg = cov_reg.Determinant();

  if(abs(det_reg) < 1e-100){
    cout << "Warning, reg cov matrix is non invertable. Det is:" << endl;
    cout << det_reg << endl;
    return 0;   
  }  

  TMatrixDSym cov_reg_inv = (*((TMatrixDSym *)(cov_reg.Clone())));
  cov_reg_inv.Invert();

  cout << "Cov Reg Inv: " << endl;
  cov_reg_inv.Print();



  
  /*********************************************************************************/

  /************************ Initiate Samples ********************************/

  /*********************************************************************************/ 

  cout << "*********************************" << endl;
  cout << " Initiate Samples " << endl;
  cout << "*********************************" << endl << endl;
  

  TFile *fout = TFile::Open(fnameout.c_str(), "RECREATE");
  cout<<"output file open"<<endl;


  // Add analysis samples:
  
  vector<AnaSample*> samples;

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
    
  AnySample sam7(6, "DIS",v_D1edges_dis, v_D2edges_dis,tdata, isBuffer, emptyCR);
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
  
  //--
  //read MC events
  anyTreeMC selTree(fsel.c_str());
  cout << "Reading and collecting events" << endl;
  selTree.GetEvents(samples);
  //get brakdown by reaction
  cout << "Getting sample breakdown by reaction" << endl;
  for(size_t s=0;s<samples.size();s++){
    ((AnySample*)(samples[s]))->GetSampleBreakdown(fout,"nominal",true);
  }

  cout<<"nominal data done"<<endl;


  //*************** FITTER SETTINGS **************************
  //In the bit below we choose which params are used in the fit
  //For stats only just use fit params
  //**********************************************************

  cout << "*********************************" << endl;
  cout << " Initiate Fitting Parameters " << endl;
  cout << "*********************************" << endl << endl;

  //define fit param classes
  vector<AnaFitParameters*> fitpara;
  //fitpara.SetFluxHisto(h_flux);

  // When filling the fitparas note that there are some assumptions later in the code
  // that the fit parameters are stored at index 0. For this reason always fill the 
  // fit parameters first. 
  
  //Fit parameters
  FitParameters sigfitpara(fxsecbinning, "par_fit", isAltPrior);
  if(fitmode==5 || fitmode==7) sigfitpara.SetRegCovarianceMatrix(&cov_reg);
  if(fitmode!=7) sigfitpara.SetCovarianceMatrix(&cov_reg_inv);
  sigfitpara.InitEventMap(samples, 0);
  if(templateFit==1) fitpara.push_back(&sigfitpara);
  
  cout<<"fit parameters done"<<endl;

  // WIP WIP WIP ***


  //Flux shape
  // FluxParameters fluxpara(enubins,"par_flux_shape");
  // fluxpara.SetCovarianceMatrix(&cov_flux);
  // TFile *fflux= new TFile("../inputs/c_flux.root","READ");
  // TH1F *flux=(TH1F*)(((TCanvas*)(fflux->Get("c_flux")))->GetPrimitive("flux"));
  // fluxpara.SetFluxHisto(flux);
  // fluxpara.InitEventMap(samples);
  // fitpara.push_back(&fluxpara);
  
  //Flux parameters normalization only
  // vector<double> enubins_norm;
  // enubins_norm.push_back(0);
  // enubins_norm.push_back(9999999);
  // FluxParameters fluxpara_norm(enubins_norm,"par_flux_norm");
  // TMatrixDSym cov_flux_norm(1);
  // (cov_flux_norm)(0,0) = (0.11*0.11);
  // fluxpara_norm.SetCovarianceMatrix(&cov_flux_norm);
  // fluxpara_norm.InitEventMap(samples);
  // fitpara.push_back(&fluxpara_norm);

  //Nucleon FSI parameters
  // NuclFSIParameters nuclfsipara;
  // nuclfsipara.SetCovarianceMatrix(&cov_nuclfsi);
  // nuclfsipara.StoreResponseFunctions(responsefunctions_NuclFSI, v_D1edges, v_D2edges);
  // nuclfsipara.InitEventMap(samples);
  // fitpara.push_back(&nuclfsipara);

  

  //Flux parameters
  FluxParameters fluxpara(enubins);
  fluxpara.SetCovarianceMatrix(&cov_flux);
  fluxpara.InitEventMap(samples, 0);
  if(USESYST==1 || USESYST==2) fitpara.push_back(&fluxpara);


  //Det parameters
  DetParameters detpara((inputDir + "/dptbinning2DPS_shortest_inclusive_det.txt").c_str(), det_weights, samples, "par_detAve");
  detpara.SetCovarianceMatrix(&cov_det);
  detpara.InitEventMap(samples, 0);
  if(USESYST==1 || USESYST==3) fitpara.push_back(&detpara);


  //DetParameters detpara_fine((inputDir + "/dptDetCovBinning.txt").c_str(),det_weights_fine, samples, "par_detFine");
  //detpara_fine.SetCovarianceMatrix(&cov_det_fine);
  //detpara_fine.InitEventMap(samples, pmode);
  //if(USESYST==1 && templateFit==1) fitpara.push_back(&detpara_fine);
  //if(USESYST==1 && fitmode!=2 && fitmode!=3 && fitmode!=8) fitpara.push_back(&detpara_fine);


  //Xsec parameters
  XsecParameters xsecpara;
  xsecpara.SetCovarianceMatrix(&cov_xsec);
  xsecpara.StoreResponseFunctions(responsefunctions, v_D1edges, v_D2edges);
  xsecpara.InitEventMap(samples, pmode);
  if(USESYST==1 || USESYST==4) fitpara.push_back(&xsecpara);

  //FSI parameters
  FSIParameters fsipara;
  fsipara.SetCovarianceMatrix(&cov_fsi);
  fsipara.StoreResponseFunctions(responsefunctions_FSI, v_D1edges, v_D2edges);
  fsipara.InitEventMap(samples, pmode);
  if(USESYST==1 || USESYST==4) fitpara.push_back(&fsipara);


  //Instantiate fitter obj
  XsecFitter xsecfit(seed);
  //init w/ para vector
  std::cout << "initialising fitter with regularisaion param " << regparam << std::endl;
  xsecfit.InitFitter(fitpara, regparam, regparam2, nprimbins, paramVectorFname);
  xsecfit.SetPOTRatio(potD/potMC);
  std::cout << "fitter initialised " << regparam << std::endl;
  

   
  //fix parameters
  /*
  xsecfit.FixParameter("flux_norm",1);
  xsecfit.FixParameter("flux_shape",1);
  xsecfit.FixParameter("flux",1);
  xsecfit.FixParameter("detFine",1);
  xsecfit.FixParameter("MAres",1.41);
  xsecfit.FixParameter("CCoth",0.0);
  xsecfit.FixParameter("Piless",0.0);
  xsecfit.FixParameter("CC1piE0",1.1);
  xsecfit.FixParameter("CC1piE1",1.0);
  xsecfit.FixParameter("CCCoh",1.0);
  xsecfit.FixParameter("NCoth",1.0);
  xsecfit.FixParameter("NC1pi0E0",0.96);
  xsecfit.FixParameter("NC1piE0",1.0);
  xsecfit.FixParameter("PionFSI",1);
  */

  xsecfit.FixParameter("par_ccqe8",1);
  // xsecfit.FixParameter("par_ccqe9",1);
  // xsecfit.FixParameter("par_ccqe10",1);
  // xsecfit.FixParameter("par_ccqe11",1);
  // xsecfit.FixParameter("par_ccqe12",1);
  // xsecfit.FixParameter("par_ccqe13",1);

  if(templateFit==-1) xsecfit.FixParameter("MEC_C",-0.5); 



  for(int i=0; i<48; i++) xsecfit.FixParameter(Form("par_detFine%d",i),1);


  //set frequency to save output
  xsecfit.SetSaveMode(fout, 1);

  // N.B Fit mode 1 currently does not work, if you want to throw nuisances but fix the detector parameters then 
  // set a ngligable regularisation parameter. 

  //fitmode: 1 = generate toy dataset from nuisances (WITH stat fluct)
  //         2 = fake data from MC or real data
  //         3 = no nuisance sampling only stat fluctuation 
  //         4 = fake data from MC or real data with statistical fluctuations applied to that data
  //         5 = generate toy dataset from nuisances and regularised c_i (WITH stat fluct)
  //         6 = generate toy dataset from nuisances and random c_i (WITH stat fluct)
  //         7 = generate toy dataset from nuisances and regularised c_i (WITH stat fluct) but fit without reg
  //         8 = Asimov (Make fake data that == MC)
  //         9 = fake data from param vector (need to sepecify paramVectorFname)
  //         10 = altPriorsTest

  //fitmethod: 1 = MIGRAD only
  //           2 = MIGRAD + HESSE
  //           3 = MINOS

  //statFluct (only relevent if fitmode is set to gen fake data with nuisances):
  //           0 = Do not apply Stat Fluct to fake data 
  //           1 = Apply Stat Fluct to fake data 

  cout << "*********************************" << endl;
  cout << " Running Fitter ... " << endl;
  cout << "*********************************" << endl << endl;

  xsecfit.Fit(samples, fitmode, fitMethod, statFluct);
  
  fout->Close();

  cout << "*********************************" << endl;
  cout << " Fit Complete, see MINUIT output above for status " << endl;
  cout << "*********************************" << endl << endl;

  return 0;
}
