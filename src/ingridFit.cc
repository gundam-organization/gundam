/******************************************************

Bare bones of what needs to be set up to fit some set of
parameters defined in /fitparams/src/FitParameters


******************************************************/


#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <string>

#include <TCanvas.h>
#include <TH1F.h>

#include "FitParameters.hh"
#include "XsecFitter.hh"
#include "anyTreeMC.hh"
#include "AnySample.hh"

#include "FluxParameters.hh"


using namespace std;

int main(int argc, char *argv[])
{

  char const * xslf_env = getenv("XSLLHFITTER");
  if(!xslf_env){
    std::cerr << "[ERROR]: environment variable \"XSLLHFITTER\" not set. "
    "Cannot determine source tree location." << std::endl;
    return 1;
  }
  string inputDir = std::string(xslf_env) + "/inputs/ingridFit";
  string fsel     = inputDir + "/fixed_weight.root"; //"/NeutAir5_2DV2.root";
  string fakeData     = inputDir + "/fixed_nominal.root"; //"/NeutAir5_2DV2.root";
  string fxsecbinning = inputDir + "/ingrid_binning.txt"; //"/dptbinning2DPS_shortest_inclusive.txt";
  int nprimbins = 8;
  string paramVectorFname = "fitresults.root";

  double potD     = 331.6; //in units 10^19 Neut Air
  double potMC    = 331.6; //in units 10^19 Neut Air
  int seed        = 1019;
  string fnameout = "ingridFit.root";
  int nbins;

  int isBuffer=false; // Is the final bin just for including events that go beyond xsec binning
                      // e.g. events with over 5GeV pmu if binning in pmu


  // Setup data trees

  TFile *fdata = new TFile(TString(fakeData));
  TTree *tdata = (TTree*)(fdata->Get("selectedEvents"));


  //Set up bin edges

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

  //Set up systematics:


  /*************************************** FLUX *****************************************/
  cout << "*********************************" << endl;
  cout << " Setup Flux " << endl;
  cout << "*********************************" << endl << endl;


  //input File
  string finfluxcovFN = inputDir + "/flux_numu_cov.root";
  TFile *finfluxcov = TFile::Open(TString(finfluxcovFN)); //contains flux systematics info


  //setup enu bins and covm for flux
  TH1D *nd_numu_bins_hist = (TH1D*)finfluxcov->Get("flux_binning");
  TAxis *nd_numu_bins = nd_numu_bins_hist->GetXaxis();
  vector<double> enubins;
  enubins.push_back(nd_numu_bins->GetBinLowEdge(1));
  for(int i=0;i<nd_numu_bins->GetNbins();i++)
  {
    enubins.push_back(nd_numu_bins->GetBinUpEdge(i+1));
  }
  //Cov mat stuff:
  TMatrixDSym *cov_flux_in   = (TMatrixDSym*)finfluxcov->Get("flux_numu_cov");
  TMatrixDSym cov_flux(cov_flux_in->GetNrows());
  for(int i=0;i<cov_flux_in->GetNrows();i++){
    for(int j=0;j<cov_flux_in->GetNrows();j++)
    {
      cov_flux(i, j) = (*cov_flux_in)(i,j);
    }
  }
  finfluxcov->Close();

  /*************************************** FLUX END *************************************/


  TFile *fout = TFile::Open(fnameout.c_str(), "RECREATE");
  cout<<"output file open"<<endl;

  // Add analysis samples:

  vector<AnaSample*> samples;

  // The sample ID (first arg) should match the cutBranch corresponding to it

  AnySample sam2(1, "MuTPCpTPC",v_D1edges, v_D2edges, tdata, isBuffer, false);
  sam2.SetNorm(potD/potMC);
  samples.push_back(&sam2);

  AnySample sam6(5, "CC1pi",v_D1edges, v_D2edges, tdata, isBuffer, true);
  sam6.SetNorm(potD/potMC);
  samples.push_back(&sam6);

  AnySample sam7(6, "DIS",v_D1edges, v_D2edges, tdata, isBuffer, true);
  sam7.SetNorm(potD/potMC);
  samples.push_back(&sam7);

  AnySample sam8(10, "Ingrid",v_D1edges, v_D2edges, tdata, isBuffer, false, true);
  sam8.SetNorm(potD/potMC);
  samples.push_back(&sam8);

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


  //*************** FITTER SETTINGS **************************
  //In the bit below we choose which params are used in the fit
  //For stats only just use fit params
  //**********************************************************

  //define fit param classes
  vector<AnaFitParameters*> fitpara;

  // When filling the fitparas note that there are some assumptions later in the code
  // that the fit parameters are stored at index 0. For this reason always fill the
  // fit parameters first.

  //Fit parameters
  FitParameters sigfitpara(fxsecbinning.c_str());
  sigfitpara.InitEventMap(samples, 0);
  fitpara.push_back(&sigfitpara);

  //Flux parameters
  FluxParameters fluxpara(enubins, "par_flux", true);
  fluxpara.SetCovarianceMatrix(&cov_flux);
  fluxpara.InitEventMap(samples, 0);
  fitpara.push_back(&fluxpara);

  //Instantiate fitter obj
  XsecFitter xsecfit(seed);

  //init w/ para vector
  xsecfit.InitFitter(fitpara, 0, 0, nprimbins, paramVectorFname);
  std::cout << "fitter initialised " << std::endl;


  xsecfit.FixParameter("par_ccqe0", 1.0);
  xsecfit.FixParameter("par_ccqe1", 1.0);
  xsecfit.FixParameter("par_ccqe2", 1.0);
  xsecfit.FixParameter("par_ccqe3", 1.0);
  xsecfit.FixParameter("par_ccqe4", 1.0);
  xsecfit.FixParameter("par_ccqe5", 1.0);
  xsecfit.FixParameter("par_ccqe6", 1.0);
  xsecfit.FixParameter("par_ccqe7", 1.0);
  xsecfit.FixParameter("par_ccqe8", 1.0);
  xsecfit.FixParameter("par_ccqe9", 1.0);
  xsecfit.FixParameter("par_ccqe10", 1.0);
  xsecfit.FixParameter("par_ccqe11", 1.0);
  xsecfit.FixParameter("par_ccqe12", 1.0);
  xsecfit.FixParameter("par_ccqe13", 1.0);
  xsecfit.FixParameter("par_ccqe14", 1.0);
  xsecfit.FixParameter("par_ccqe15", 1.0);

  //set frequency to save output
  xsecfit.SetSaveMode(fout, 1);

  //fitmode: 1 = generate toy dataset from nuisances (WITH stat fluct)
  //         2 = fake data from MC or real data
  //         3 = no nuisance sampling only stat fluctuation
  //         4 = fake data from MC or real data with statistical fluctuations applied to that data
  //         5 = generate toy dataset from nuisances and regularised c_i (WITH stat fluct)
  //         6 = generate toy dataset from nuisances and random c_i (WITH stat fluct)
  //         7 = generate toy dataset from nuisances and regularised c_i (WITH stat fluct) but fit without reg
  //         8 = Asimov (Make fake data that == MC)


  //fitmethod: 1 = MIGRAD only
  //           2 = MIGRAD + HESSE
  //           3 = MINOS

  //statFluct (only relevent if fitmode is set to gen fake data with nuisances):
  //           0 = Do not apply Stat Fluct to fake data
  //           1 = Apply Stat Fluct to fake data

  xsecfit.Fit(samples, 2, 2, 0);

  fout->Close();

  return 0;
}
