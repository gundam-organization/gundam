/******************************************************

Code to take output of the fitter and produce a corrected
Nevents spectrum compared to the MC and fake data truth

Now takes result from propError code to include full errors
on the fit result. 

Author: Stephen Dolan
Date Created: Jun 2016

******************************************************/


#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <assert.h>

#include <TCanvas.h>
#include <TLegend.h>
#include <TH1F.h>
#include <THStack.h>
#include <TTree.h>
#include <TString.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TMath.h>

#include <TMatrixD.h>
#include <TVectorD.h>
#include <TMatrixDSym.h>


// Stuff for ThrowParams header

#include <algorithm>
#include <math.h>
#include <TRandom3.h>
#include <TMatrixT.h>
#include <TMatrixTSym.h>
#include <TVectorT.h>
#include <TDecompChol.h>
#include <TDecompBK.h>
#include <TDecompSVD.h>
#include <TLatex.h>
#include <vector>


//#include "/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/utils/src/ThrowParms.hh"
//#include "/data/t2k/dolan/software/xsLLhFitter/xsLLhFitter/utils/src/ThrowParms.cc"

using namespace std;

char* plotsDir = "/data/t2k/dolan/fitting/feb17_refit/summaryPlots/plotsOut/";
bool gibuumode = false;

class ThrowParms 
{
 private:
  typedef TVectorT<double> TVectorD;
  typedef TMatrixTSym<double> TMatrixDSym;
  typedef TMatrixT<double> TMatrixD;

  TVectorD    *pvals;
  TMatrixDSym *covar; 
  TMatrixD    *chel_dec;
  int         npars;

public:
  ThrowParms(TVectorD &parms, TMatrixDSym &covm);
  ~ThrowParms();
  int GetSize() {return npars;};
  void ThrowSet(std::vector<double> &parms);

private:
  void CheloskyDecomp(TMatrixD &chel_mat);
  void StdNormRand(double *z);
};


double calcChi2(TH1D* h1, TH1D* h2, TMatrixDSym covar_in, bool isShapeOnly=false){
  double chi2=0;
  TMatrixDSym covar(covar_in);
  TMatrixDSym covarForShapeOnly(covar_in);
  //To avoid working with tiny numbers (and maybe running into double precision issues)
  if(!isShapeOnly){
    for(int i=0; i<h1->GetNbinsX(); i++){
      for(int j=0; j<h1->GetNbinsX(); j++){
        covar[i][j]=covar_in[i][j]*1E80;
        covarForShapeOnly[i][j]=covar_in[i][j]*1E80;
      }
    }
  }

  //Matrix inversion section:
  bool inversionError = false;
  TMatrixDSym* covar_inv = new TMatrixDSym();
  int count = 0;
  while(true){
    inversionError = false;
    //For an accurate matrix inversion (and guaranteed symmetric inverse):  
    //TDecompBK covar_bk(covar);
    //TMatrixDSym covar_inv = covar_bk.Invert();
    //Alternative method (the one used in NUISANCE):
    TDecompSVD LU = TDecompSVD(covar);
    covar_inv = new TMatrixDSym(covar.GetNrows(), LU.Invert().GetMatrixArray(), "");
  
    //Check that the matrix inversion worked 
    TMatrixD test = covar*(*covar_inv);
    for(int i=0; i<h1->GetNbinsX(); i++){
      for(int j=0; j<h1->GetNbinsX(); j++){
        if(i==j && (test[i][j]>1.00001 || test[i][j]<0.99999)){
          if(!inversionError) cout << "****** WARNING: Issue with matrix inversion in chi2 calculation. See below!" << endl; 
          inversionError=true;
        }
        else if(i!=j && (test[i][j]>0.0000001)) {
          if(!inversionError) cout << "****** WARNING: Issue with matrix inversion in chi2 calculation. See below!" << endl; 
          inversionError=true;
        }
      }
    }
    // Section below still WIP
    //inversionError=false; 
    if(inversionError){ 
      cout << "DEBUG info after interation " << count << endl;
      cout << "  Is shape only: " << isShapeOnly << endl;
      //cout << "  Input matrix:" << endl;
      //covar.Print();
      //cout << "  Inverted matrix:" << endl;
      //covar_inv->Print();
      //cout << "  Test matrix:" << endl;
      //test.Print();
      cout << "  Input matrix: deritminent is: " << covar.Determinant() <<  endl;
      cout << "Will try adding 0.000001pc to the diagonal ..." << endl;
      for(int i=0; i<h1->GetNbinsX(); i++){
          covar[i][i]=covar[i][i]*1.00000001;
      }
      count++;
      //if(count>10)getchar();
    }
    else{
      cout << "chi2 calculated successfully after interaction " << count << endl;
      //if(count>0) test.Print();
      break;
    }
  }

  for(int i=0; i<h1->GetNbinsX(); i++){
    for(int j=0; j<h1->GetNbinsX(); j++){
      if(!isShapeOnly) chi2+= 1E80*((h1->GetBinContent(i+1)) - (h2->GetBinContent(i+1)))*((*covar_inv)[i][j])*((h1->GetBinContent(j+1)) - (h2->GetBinContent(j+1)));
      else chi2+= ((h1->GetBinContent(i+1)) - (h2->GetBinContent(i+1)))*((*covar_inv)[i][j])*((h1->GetBinContent(j+1)) - (h2->GetBinContent(j+1)));
    }
  }

  //Calc quick shape chi2: 
  if(!isShapeOnly){
    double shapeOnlyChi2=0;
    double normRelError=9.58976791458074074e-02; // hard coded, sorry

    TH1D* h2Norm = new TH1D(*h2);
    h2Norm->Scale(h1->Integral()/h2->Integral());

    if(abs(h2Norm->Integral()-h1->Integral())>1e-40) cout << "Problem with histo scaling in shape only chi2, integrals are: " << h2Norm->Integral() << " and " << h1->Integral() <<endl;

    for(int i=0; i<h1->GetNbinsX(); i++){
      for(int j=0; j<h1->GetNbinsX(); j++){
        covarForShapeOnly[i][j]=(covarForShapeOnly[i][j]-(normRelError*normRelError*h1->GetBinContent(i+1)*h1->GetBinContent(j+1)));
      }
    }
    TDecompSVD LU_2 = TDecompSVD(covarForShapeOnly);
    covar_inv = new TMatrixDSym(covarForShapeOnly.GetNrows(), LU_2.Invert().GetMatrixArray(), "");

    for(int i=0; i<h1->GetNbinsX(); i++){
      for(int j=0; j<h1->GetNbinsX(); j++){
        shapeOnlyChi2+= 1E80*((h1->GetBinContent(i+1)) - (h2Norm->GetBinContent(i+1)))*((*covar_inv)[i][j])*((h1->GetBinContent(j+1)) - (h2Norm->GetBinContent(j+1)));
      }
    }
    cout << "Quick shape only chi2 calcualted successfully: " << shapeOnlyChi2 << endl;  
  }

  if(!inversionError) cout << "Chi2 calcualted successfully: " << chi2 << endl;
  return chi2;
}

TH1D* makeShapeOnly(TH1D* h1){
  double integral = h1->Integral();
  for(int i=0;i<h1->GetNbinsX();i++){
    h1->SetBinContent(i+1, h1->GetBinContent(i+1)/integral);
  }
  return h1;
}

void labelHist(TH1D* h1, string var, bool isShapeOnly){
  h1->UseCurrentStyle();
  if(isShapeOnly==false){
    if(var=="dpt"){
      h1->SetXTitle("#deltap_{T} (GeV)");
      h1->SetYTitle("#frac{d#sigma}{d#deltap_{T}} (cm^{2} Nucleon^{-1} GeV^{-1})");
      h1->GetYaxis()->SetRangeUser(0,12E-39);
      if(gibuumode) h1->GetYaxis()->SetRangeUser(0,8E-39);
    }
    else if(var=="dphit"){
      h1->SetXTitle("#delta#phi_{T} (radians)");
      h1->SetYTitle("#frac{d#sigma}{d#delta#phi_{T}} (cm^{2} Nucleon^{-1} radian^{-1})");
      h1->GetYaxis()->SetRangeUser(0,6E-39);
      if(gibuumode) h1->GetYaxis()->SetRangeUser(0,4E-39);
    }
    else if(var=="dat"){
      h1->SetXTitle("#delta#alpha_{T} (radians)");
      h1->SetYTitle("#frac{d#sigma}{d#delta#alpha_{T}} (cm^{2} Nucleon^{-1} radian^{-1})");
      h1->GetYaxis()->SetRangeUser(0,1.2E-39);
    }
    else cout << "Unrecognised variable: " << var << endl;
  }
  else if(isShapeOnly==true){
    if(var=="dpt"){
      h1->SetXTitle("#deltap_{T} (GeV)");
      h1->SetYTitle("#frac{1}{#sigma} #frac{d#sigma}{d#deltap_{T}}");
      h1->GetYaxis()->SetRangeUser(0,0.3999);
    }
    else if(var=="dphit"){
      h1->SetXTitle("#delta#phi_{T} (radians)");
      h1->SetYTitle("#frac{1}{#sigma} #frac{d#sigma}{d#delta#phi_{T}}");
      h1->GetYaxis()->SetRangeUser(0,0.44999);
    }
    else if(var=="dat"){
      h1->SetXTitle("#delta#alpha_{T} (radians)");
      h1->SetYTitle("#frac{1}{#sigma} #frac{d#sigma}{d#delta#alpha_{T}}");
      h1->GetYaxis()->SetRangeUser(0,0.3999);
    }
    else cout << "Unrecognised variable: " << var << endl;
  }
  return;
}

//Example running:


void calcShapeXsecWithErrors(TString xsecFilename, TString outFilename, const int ntoys = 1000, char* var="dpt", TString nuisanceFilename="", bool newNuise=false){
  TFile* xsecFile = new TFile(xsecFilename);
  TMatrixDSym* covariance = (TMatrixDSym*) xsecFile->Get("covarXsec");
  TH1D* xsecResult = (TH1D*) xsecFile->Get("dif_xSecFit_allError");
  const int nXsecBins = xsecResult->GetNbinsX();
  TVectorD xsecVec(nXsecBins);

  for(int i=0; i<nXsecBins; i++){
    xsecVec[i] = xsecResult->GetBinContent(i+1);
  }


  TH1D* xsecBinDist[200];
  for(int b=0;b<nXsecBins;b++){
    xsecBinDist[b] = new TH1D(Form("bin_%d_toys", b), Form("bin_%d_toys", b), 11000, -0.1, 1); 
  }

  double shapeXsecNom_bin[nXsecBins]; // bin
  double shapeXsec_bin_toy[nXsecBins][ntoys]; // bins, toys
  double fullXsec_bin_toy[nXsecBins][ntoys]; // bins, toys
  //if(ntoys>10000){
  //  cout << "WARNING: for over 10000 toys need to change hard coded array index" << endl; 
  //  return;
  //}

  TH1D* shapeXSecResult = new TH1D(*xsecResult);
  TH1D* totalXsecHist = new TH1D("totalXsec", "totalXsec", 10000, 0, 1e-37);
  for(int b=0; b<(nXsecBins); b++){
    shapeXSecResult->SetBinContent(b+1, (xsecResult->GetBinContent(b+1))/xsecResult->Integral());
    shapeXsecNom_bin[b] = shapeXSecResult->GetBinContent(b+1);
  }

  ThrowParms *throwParms = new ThrowParms(xsecVec,(*covariance));
  TRandom3 *rand = new TRandom3(0);

  cout << "Will throw  " << ntoys << " toys" << endl;
  for(int t=0; t<ntoys; t++){
    vector<double> allparams_throw;
    throwParms->ThrowSet(allparams_throw);

    double totalXsec = 0;
    //Find total xsec
    for(int b=0; b<nXsecBins; b++){ 
      totalXsec += allparams_throw[b];
    }

    //Find shape only xsec
    for(int b=0; b<nXsecBins; b++){
      fullXsec_bin_toy[b][t]=allparams_throw[b];
      shapeXsec_bin_toy[b][t]=allparams_throw[b]/totalXsec;
      xsecBinDist[b]->Fill(shapeXsec_bin_toy[b][t]);
    }
    totalXsecHist->Fill(totalXsec);
    if(t<2){
      cout << "Toy " << t << " complete, total xsec is " << totalXsec << endl;
      cout << "Toy bins are as follows: " << endl;
      for(int b=0; b<nXsecBins; b++){
        cout << "Bin " << b << " is " << shapeXsec_bin_toy[b][t] << " (nominal is " << shapeXsecNom_bin[b] << ")" << endl;
      }
    }
  }

  double normRelError = (totalXsecHist->GetRMS()/totalXsecHist->GetMean());

  TMatrixDSym covar(nXsecBins); 
  TMatrixDSym cormatrix(nXsecBins);
  TMatrixDSym covar_alt(nXsecBins); 
  TMatrixDSym cormatrix_alt(nXsecBins);
  TMatrixDSym covar_sanityTest(nXsecBins); 
  TMatrixDSym covar_avg(nXsecBins); 
  TMatrixDSym covar_norm(nXsecBins); 
  TMatrixDSym covar_norm_xsec(nXsecBins); 
  for(int t=0;t<ntoys;t++){
    for(int i=0;i<nXsecBins;i++){
      for(int j=0;j<nXsecBins;j++){
        covar_sanityTest[i][j] += (1/(double)ntoys) * ( fullXsec_bin_toy[i][t] - (xsecResult->GetBinContent(i+1)) ) * ( fullXsec_bin_toy[j][t] - (xsecResult->GetBinContent(j+1)) );
        covar[i][j] += (1/(double)ntoys) * ( shapeXsec_bin_toy[i][t] - (shapeXsecNom_bin[i]) ) * ( shapeXsec_bin_toy[j][t] - (shapeXsecNom_bin[j]) );
      }
    }
  }

  //Calculate Corrolation Matrix and shape only covar size on average xsec
  for(int r=0;r<nXsecBins;r++){
    for(int c=0;c<nXsecBins;c++){
      cormatrix[r][c] = covar[r][c]/sqrt((covar[r][r]*covar[c][c]));
      covar_avg[r][c] = covar[r][c] * totalXsecHist->GetMean() * totalXsecHist->GetMean();
      covar_norm[r][c] = covar[r][c]/( (shapeXsecNom_bin[r]) * (shapeXsecNom_bin[c]) );
      covar_norm_xsec[r][c] = ((*covariance)[r][c])/( (xsecResult->GetBinContent(r+1)) * (xsecResult->GetBinContent(c+1)) );
      //Alternative: subtract norm relative error to decompose covar into shape part and norm part
      covar_alt[r][c] = (covar_norm_xsec[r][c]-(normRelError*normRelError))*( (xsecResult->GetBinContent(r+1)) * (xsecResult->GetBinContent(c+1)) );
    }
  }
  for(int r=0;r<nXsecBins;r++){
    for(int c=0;c<nXsecBins;c++){
      cormatrix_alt[r][c] = covar_alt[r][c]/sqrt((covar_alt[r][r]*covar_alt[c][c]));
    }
  }

  //Put errors on result histo
  for(int b=0; b<(nXsecBins); b++){
    shapeXSecResult->SetBinError(b+1, sqrt(covar[b][b]));
  }

  //Collect relevent info from xsecfile

  TMatrixDSym* covar_xsec = (TMatrixDSym*)xsecFile->Get("covarXsec");
  TMatrixDSym* corr_xsec  = (TMatrixDSym*)xsecFile->Get("cormatrix");

  TH1D* NuWro  = (TH1D*)xsecFile->Get(Form("NuWro_%s",var));
  TH1D* NEUT6D = (TH1D*)xsecFile->Get(Form("NEUT6D_%s",var));
  TH1D* NEUT6B = (TH1D*)xsecFile->Get(Form("NEUT6B_%s",var));
  TH1D* GENIE  = (TH1D*)xsecFile->Get(Form("GENIE_%s",var));

  //Make shape only versions

  TH1D* NuWro_shapeOnly  = makeShapeOnly( (TH1D*)xsecFile->Get(Form("NuWro_%s",var))  ); NuWro_shapeOnly->SetNameTitle("NuWro_shapeOnly", "NuWro_shapeOnly"); 
  TH1D* NEUT6D_shapeOnly = makeShapeOnly( (TH1D*)xsecFile->Get(Form("NEUT6D_%s",var)) ); NEUT6D_shapeOnly->SetNameTitle("NEUT6D_shapeOnly", "NEUT6D_shapeOnly");
  TH1D* NEUT6B_shapeOnly = makeShapeOnly( (TH1D*)xsecFile->Get(Form("NEUT6B_%s",var)) ); NEUT6B_shapeOnly->SetNameTitle("NEUT6B_shapeOnly", "NEUT6B_shapeOnly");
  TH1D* GENIE_shapeOnly  = makeShapeOnly( (TH1D*)xsecFile->Get(Form("GENIE_%s",var))  ); GENIE_shapeOnly->SetNameTitle("GENIE_shapeOnly", "GENIE_shapeOnly");

  NuWro_shapeOnly->GetYaxis()->SetRangeUser(0, 0.5);
  NEUT6D_shapeOnly->GetYaxis()->SetRangeUser(0, 0.5);
  NEUT6B_shapeOnly->GetYaxis()->SetRangeUser(0, 0.5);
  GENIE_shapeOnly->GetYaxis()->SetRangeUser(0, 0.5);

  //Compare shape only and full chi2s

  double NuWroXsec_chi2 = calcChi2(xsecResult,NuWro,*(covar_xsec));
  double NEUT6DXsec_chi2 = calcChi2(xsecResult,NEUT6D,*(covar_xsec));
  double NEUT6BXsec_chi2 = calcChi2(xsecResult,NEUT6B,*(covar_xsec));
  double GENIEXsec_chi2 = calcChi2(xsecResult,GENIE,*(covar_xsec));

  double NuWroShapeOnly_chi2 = calcChi2(shapeXSecResult,NuWro_shapeOnly,covar,true);
  double NEUT6DShapeOnly_chi2 = calcChi2(shapeXSecResult,NEUT6D_shapeOnly,covar,true);
  double NEUT6BShapeOnly_chi2 = calcChi2(shapeXSecResult,NEUT6B_shapeOnly,covar,true);
  double GENIEShapeOnly_chi2 = calcChi2(shapeXSecResult,GENIE_shapeOnly,covar,true);

  cout << "NuWro  shape only and full chi2 are " << NuWroShapeOnly_chi2  << " and " << NuWroXsec_chi2  << endl;
  cout << "NEUT6D shape only and full chi2 are " << NEUT6DShapeOnly_chi2 << " and " << NEUT6DXsec_chi2 << endl;
  cout << "NEUT6B shape only and full chi2 are " << NEUT6BShapeOnly_chi2 << " and " << NEUT6BXsec_chi2 << endl;
  cout << "GENIE  shape only and full chi2 are " << GENIEShapeOnly_chi2  << " and " << GENIEXsec_chi2  << endl;

  //Get extra MC from nuisance file (if specified)

  TFile* nuisanceFile = new TFile(nuisanceFilename);

  TH1D* nuisMC = new TH1D();
  TH1D* nuisQE = new TH1D();
  TH1D* nuis2p2h = new TH1D();
  TH1D* nuisrespi = new TH1D();
  TH1D* nuisother = new TH1D();
  TH1D* nuisMC_shapeOnly = new TH1D();
  TH1D* nuisQE_shapeOnly = new TH1D();
  TH1D* nuis2p2h_shapeOnly = new TH1D();
  TH1D* nuisrespi_shapeOnly = new TH1D();
  TH1D* nuisother_shapeOnly = new TH1D();
  TH1D* chi2Hist = new TH1D("chi2Hist", "chi2Hist", 5, 0, 5);
  TH1D* chi2Hist_shapeOnly = new TH1D("chi2Hist_shapeOnly", "chi2Hist_shapeOnly", 5, 0, 5);

  double nuisXsec_chi2 = -9.99;
  double nuisShapeOnly_chi2 = -9.99;

  if(nuisanceFile){

    if(!newNuise){
      // Get stack by PDG
      cout << "Getting nuisPdgCanv and list of primitives" << endl;
      TCanvas* nuisPdgCanv = (TCanvas*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_PDG_CANV",var));
      cout << "Getting nuisPdgCanv primitives ... ";
      nuisMC = (TH1D*) nuisPdgCanv->GetPrimitive(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MC",var)); cout << " nuisMC, "; 

      THStack* nuisStack = (THStack*) (nuisPdgCanv->GetListOfPrimitives())->At(2); 
      nuisQE      = (TH1D*) (nuisStack->GetHists())->At(1);  cout << " nuisQE, ";
      nuis2p2h    = (TH1D*) (nuisStack->GetHists())->At(2);  cout << " nuis2p2h, ";
      TH1D* nuisdppp    = (TH1D*) (nuisStack->GetHists())->At(11); cout << " nuisdppp, ";
      TH1D* nuisdppz    = (TH1D*) (nuisStack->GetHists())->At(12); cout << " nuisdppz, ";
      TH1D* nuisdnpp    = (TH1D*) (nuisStack->GetHists())->At(13); cout << " nuisdnpp, ";
      TH1D* nuisddpg    = (TH1D*) (nuisStack->GetHists())->At(17); cout << " nuisddpg, ";
      TH1D* nuismultipi = (TH1D*) (nuisStack->GetHists())->At(21); cout << " nuismultipi, ";
      TH1D* nuisdeta    = (TH1D*) (nuisStack->GetHists())->At(22); cout << " nuisdeta, ";
      TH1D* nuisdis     = (TH1D*) (nuisStack->GetHists())->At(26); cout << " nuisdis, ";
      
      TH1D* nuisdgibuu  = (TH1D*) (nuisStack->GetHists())->At(10); cout << " nuisdgibuu, ";
      TH1D* gibuuex1     = (TH1D*) (nuisStack->GetHists())->At(6); cout << " gibuuex1, ";
      TH1D* gibuuex2     = (TH1D*) (nuisStack->GetHists())->At(7); cout << " gibuuex2, ";
      cout << "" << endl;

      nuisrespi = (TH1D*) nuisdppp->Clone();
      nuisrespi->Add(nuisdppz); nuisrespi->Add(nuisdnpp); cout << " added to make nuisrespi, ";  
      if(nuisdgibuu) nuisrespi->Add(nuisdgibuu);

      nuisother = (TH1D*) nuisddpg->Clone();
      nuisother->Add(nuismultipi); nuisother->Add(nuisdeta); nuisother->Add(nuisdis); cout << " added to make nuisother, ";
      if(gibuuex1 && gibuuex2){
        nuisother->Add(gibuuex1);
        nuisother->Add(gibuuex2);
      } 
      cout << "" << endl;
    }
    else{

      cout << "Getting nuis output ...." << endl;
      nuisMC = (TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MC",var)); cout << " nuisMC, "; 

      nuisrespi=(TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MODES_CC1piponp",var));
      nuisother=(TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MODES_CCDIS",var));

      nuisQE = (TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MODES_CCQE",var)); cout << " nuisQE, ";
      nuis2p2h = (TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MODES_CC2p2h",var)); cout << " nuis2p2h, ";
      if(!nuis2p2h){
        nuis2p2h = (TH1D*) nuisQE->Clone();
        nuis2p2h->Reset();
      }

      TH1D* nuispiponp  = new TH1D();  nuispiponp = (TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MODES_CC1piponp",var)); cout << " nuispiponp, ";
      TH1D* nuispi0onn  = new TH1D();  nuispi0onn = (TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MODES_CC1pi0onn",var)); cout << " nuispi0onn, ";
      TH1D* nuispiponn  = new TH1D();  nuispiponn = (TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MODES_CC1piponn",var)); cout << " nuispiponn, ";
      TH1D* nuiscccoh   = new TH1D();   nuiscccoh = (TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MODES_CCcoh",var)); cout << " nuiscccoh, ";
      TH1D* nuiscc1gam  = new TH1D();  nuiscc1gam = (TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MODES_CC1gamma",var)); cout << " nuiscc1gam, ";
      TH1D* nuismultipi = new TH1D(); nuismultipi = (TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MODES_CCMultipi",var)); cout << " nuismultipi, ";
      TH1D* nuiseta     = new TH1D();     nuiseta = (TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MODES_CC1eta",var)); cout << " nuiseta, ";
      TH1D* nuislamkp   = new TH1D();   nuislamkp = (TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MODES_CC1lamkp",var)); cout << " nuislamkp, ";
      TH1D* nuisdis     = new TH1D();     nuisdis = (TH1D*) nuisanceFile->Get(Form("T2K_CC0pinp_STV_XSec_1D%s_nu_MODES_CCDIS",var)); cout << " nuisdis, ";
      cout << " got histos!" << endl;

      nuispiponp->Print("all");
      nuispi0onn->Print("all");
      nuispiponn->Print("all");      

      if(!nuisrespi){
        cout << "There is no nuispiponp contribution, this seems surprising ..." << endl;
        cout << "We rely on this exisiting, cannot continue without it" << endl;
        return;
      } 
      if(!nuisother){
        cout << "There is no nuisdis contribution, this seems surprising ..." << endl;
        cout << "We rely on this exisiting, cannot continue without it" << endl;
        return;
      } 

      //nuisrespi = (TH1D*) nuispiponp->Clone(); cout << " Building nuisrespi, "; 
      nuisrespi->Add(nuispi0onn); nuisrespi->Add(nuispiponn); cout << " added to make nuisrespi, ";  

      //nuisother = (TH1D*) nuisdis->Clone(); cout << " Building nuisother, "; 
      nuisother->Add(nuiscc1gam); nuisother->Add(nuismultipi); nuisother->Add(nuiseta); nuisother->Add(nuislamkp); nuisother->Add(nuiscccoh);  cout << " added to make nuisother, ";
      cout << "" << endl;

    }

    // Make shape only versions of histos

    nuisMC->SetMarkerStyle(0);
    nuisrespi->SetMarkerStyle(0);
    nuisother->SetMarkerStyle(0);
    nuisQE->SetMarkerStyle(0);
    nuis2p2h->SetMarkerStyle(0);

    nuisMC_shapeOnly = (TH1D*)nuisMC->Clone();
    nuisMC_shapeOnly = makeShapeOnly(nuisMC_shapeOnly);

    nuisQE_shapeOnly    = (TH1D*)nuisQE->Clone();    nuisQE_shapeOnly->Scale(1/nuisMC->Integral());
    nuis2p2h_shapeOnly  = (TH1D*)nuis2p2h->Clone();  nuis2p2h_shapeOnly->Scale(1/nuisMC->Integral());
    nuisrespi_shapeOnly = (TH1D*)nuisrespi->Clone(); nuisrespi_shapeOnly->Scale(1/nuisMC->Integral());
    nuisother_shapeOnly = (TH1D*)nuisother->Clone(); nuisother_shapeOnly->Scale(1/nuisMC->Integral());

    // Calculate chi2s

    nuisXsec_chi2 = calcChi2(xsecResult,nuisMC,*(covar_xsec));
    nuisShapeOnly_chi2 = calcChi2(shapeXSecResult,nuisMC_shapeOnly,covar,true);

    cout << "Nuisance shape only and full chi2 are " << nuisShapeOnly_chi2 << " and " << nuisXsec_chi2  << endl;

    chi2Hist->SetBinContent(1, NuWroXsec_chi2);
    chi2Hist->SetBinContent(2, NEUT6DXsec_chi2);
    chi2Hist->SetBinContent(3, NEUT6BXsec_chi2);
    chi2Hist->SetBinContent(4, GENIEXsec_chi2);
    chi2Hist->SetBinContent(5, nuisXsec_chi2);
    chi2Hist_shapeOnly->SetBinContent(1, NuWroShapeOnly_chi2);
    chi2Hist_shapeOnly->SetBinContent(2, NEUT6DShapeOnly_chi2);
    chi2Hist_shapeOnly->SetBinContent(3, NEUT6BShapeOnly_chi2);
    chi2Hist_shapeOnly->SetBinContent(4, GENIEShapeOnly_chi2);
    chi2Hist_shapeOnly->SetBinContent(5, nuisShapeOnly_chi2);

    chi2Hist->GetXaxis()->SetBinLabel(1, "NuWro");
    chi2Hist->GetXaxis()->SetBinLabel(2, "NEUT 6D");
    chi2Hist->GetXaxis()->SetBinLabel(3, "NEUT 6B");
    chi2Hist->GetXaxis()->SetBinLabel(4, "GENIE");
    chi2Hist->GetXaxis()->SetBinLabel(5, nuisanceFilename);
    chi2Hist_shapeOnly->GetXaxis()->SetBinLabel(1, "NuWro");
    chi2Hist_shapeOnly->GetXaxis()->SetBinLabel(2, "NEUT 6D");
    chi2Hist_shapeOnly->GetXaxis()->SetBinLabel(3, "NEUT 6B");
    chi2Hist_shapeOnly->GetXaxis()->SetBinLabel(4, "GENIE");
    chi2Hist_shapeOnly->GetXaxis()->SetBinLabel(5, nuisanceFilename);


  }
  else cout << "No valid nuisance input found at " << nuisanceFilename << endl;

  cout << "Writting output file" << endl;
  
  TFile* outFile = new TFile(outFilename, "RECREATE");

  TString outPrefex = nuisanceFilename;
  Int_t dot = outPrefex.First('.');
  Int_t len = outPrefex.Length();
  outPrefex.Remove(dot,len-dot);
  Int_t slash = outPrefex.First('/');
  len = outPrefex.Length();
  outPrefex.Remove(0,slash+1);
  slash = outPrefex.First('/');
  len = outPrefex.Length();
  outPrefex.Remove(0,slash+1);

  for(int b=0; b<(nXsecBins); b++){
    xsecBinDist[b]->Write();
  }
  totalXsecHist->Write();

  if(newNuise)  plotsDir = "/data/t2k/dolan/fitting/feb17_refit/summaryPlots/plotsOut_oct17Nuis/";

  // Set titles and ranges:

  labelHist(xsecResult , var, false);
  labelHist(shapeXSecResult , var, true);

  xsecResult->Write("Result_Xsec");
  shapeXSecResult->Write("Result_shapeOnly");

  //Quick sanity check that throws were okay":
  TMatrixDSym sanityTest(covar_sanityTest);
  for(int i=0;i<nXsecBins;i++){
    for(int j=0;j<nXsecBins;j++){
      sanityTest[i][j]=(covar_sanityTest[i][j]-((*covar_xsec)[i][j]))/((*covar_xsec)[i][j]);
    }
  }

  cout << "The matrix below should be full of small entries (relative to 1)" << endl; 

  sanityTest.Print();

  // ***************************************

  sanityTest.Write("sanityTest");
  covar_sanityTest.Write("covar_sanityTest");

  covar_xsec->Write("covar_Xsec");
  covar_norm_xsec.Write("covarnorm_Xsec");
  corr_xsec->Write("cor_Xsec");

  covar.Write("covar_shapeOnly");
  covar_avg.Write("covar_shapeOnly_avgXsec");
  covar_norm.Write("covarnorm_shapeOnly");
  cormatrix.Write("cor_shapeOnly");

  covar_alt.Write("covar_shapeOnly_alt");
  cormatrix_alt.Write("cor_shapeOnly_alt");

  labelHist(NuWro , var, false);
  labelHist(NEUT6D , var, false);
  labelHist(NEUT6B , var, false);
  labelHist(GENIE , var, false);
  labelHist(NuWro_shapeOnly , var, true);
  labelHist(NEUT6D_shapeOnly , var, true);
  labelHist(NEUT6B_shapeOnly , var, true);
  labelHist(GENIE_shapeOnly , var, true);

  NuWro->SetMarkerStyle(0); NuWro->Write("NuWro");
  NEUT6D->SetMarkerStyle(0); NEUT6D->Write("NEUT6D");
  NEUT6B->SetMarkerStyle(0); NEUT6B->Write("NEUT6B");
  GENIE->SetMarkerStyle(0); GENIE->Write("GENIE");
  NuWro_shapeOnly->SetMarkerStyle(0); NuWro_shapeOnly->Write("NuWro_shapeOnly");
  NEUT6D_shapeOnly->SetMarkerStyle(0); NEUT6D_shapeOnly->Write("NEUT6D_shapeOnly");
  NEUT6B_shapeOnly->SetMarkerStyle(0); NEUT6B_shapeOnly->Write("NEUT6B_shapeOnly");
  GENIE_shapeOnly->SetMarkerStyle(0); GENIE_shapeOnly->Write("GENIE_shapeOnly");

  if(nuisanceFile){
    labelHist(nuisMC, var, false);
    labelHist(nuisQE, var, false);
    labelHist(nuis2p2h, var, false);
    labelHist(nuisrespi, var, false);
    labelHist(nuisother, var, false);
    labelHist(nuisMC_shapeOnly, var, true);
    labelHist(nuisQE_shapeOnly, var, true);
    labelHist(nuis2p2h_shapeOnly, var, true);
    labelHist(nuisrespi_shapeOnly, var, true);
    labelHist(nuisother_shapeOnly, var, true);

    nuisMC->Write("nuisMC");
    nuisQE->Write("nuisQE");
    nuis2p2h->Write("nuis2p2h");
    nuisrespi->Write("nuisrespi");
    nuisother->Write("nuisother");
    nuisMC_shapeOnly->Write("nuisMC_shapeOnly");
    nuisQE_shapeOnly->Write("nuisQE_shapeOnly");
    nuis2p2h_shapeOnly->Write("nuis2p2h_shapeOnly");
    nuisrespi_shapeOnly->Write("nuisrespi_shapeOnly");
    nuisother_shapeOnly->Write("nuisother_shapeOnly");

    // chi2 hists

    chi2Hist->Write();
    chi2Hist_shapeOnly->Write();

    // Reaction Canvases

    xsecResult->SetLineColor(kBlack);
    shapeXSecResult->SetLineColor(kBlack);

/* 
    //original version

    nuisQE->SetLineColor(kRed);
    nuis2p2h->SetLineColor(kViolet+10);
    nuisrespi->SetLineColor(kGreen);
    nuisother->SetLineColor(kCyan);

    nuisQE->SetFillColor(kRed-7);
    nuis2p2h->SetFillColor(kViolet-7);
    nuisrespi->SetFillColor(kGreen-7);
    nuisother->SetFillColor(kCyan-7);

    nuisQE_shapeOnly->SetLineColor(kRed);
    nuis2p2h_shapeOnly->SetLineColor(kViolet+10);
    nuisrespi_shapeOnly->SetLineColor(kGreen);
    nuisother_shapeOnly->SetLineColor(kCyan);

    nuisQE_shapeOnly->SetFillColor(kRed-7);
    nuis2p2h_shapeOnly->SetFillColor(kViolet-7);
    nuisrespi_shapeOnly->SetFillColor(kGreen-7);
    nuisother_shapeOnly->SetFillColor(kCyan-7);

*/

    // paper version

    nuisQE->SetLineColor(kRed);
    nuis2p2h->SetLineColor(kBlue+3);
    nuisrespi->SetLineColor(kGreen);
    nuisother->SetLineColor(kCyan);

    nuisQE->SetFillColor(kRed-7);
    nuis2p2h->SetFillColor(kBlue-4);
    nuisrespi->SetFillColor(kGreen-7);
    nuisother->SetFillColor(kCyan-7);

    nuisQE->SetFillStyle(3004);
    nuis2p2h->SetFillStyle(3005);
    nuisrespi->SetFillStyle(3001);
    nuisother->SetFillStyle(3002);

    nuisQE_shapeOnly->SetLineColor(kRed);
    nuis2p2h_shapeOnly->SetLineColor(kBlue+3);
    nuisrespi_shapeOnly->SetLineColor(kGreen);
    nuisother_shapeOnly->SetLineColor(kCyan);

    nuisQE_shapeOnly->SetFillColor(kRed-7);
    nuis2p2h_shapeOnly->SetFillColor(kBlue-4);
    nuisrespi_shapeOnly->SetFillColor(kGreen-7);
    nuisother_shapeOnly->SetFillColor(kCyan-7);

    nuisQE_shapeOnly->SetFillStyle(3004);
    nuis2p2h_shapeOnly->SetFillStyle(3005);
    nuisrespi_shapeOnly->SetFillStyle(3001);
    nuisother_shapeOnly->SetFillStyle(3002);

    TLegend* ReacLeg;
    TLegend* ReacLeg_shapeOnly;
    TLegend* gibuuLeg;

    if((string)var=="dat") ReacLeg = new TLegend(0.2,0.5,0.4,0.85);
    else ReacLeg = new TLegend(0.65,0.5,0.85,0.85);
    ReacLeg->AddEntry(xsecResult,"T2K","ep");
    ReacLeg->AddEntry(nuisQE,"CCQE","lf");
    if     (strcasestr(nuisanceFilename.Data(),"genie")) ReacLeg->AddEntry(nuis2p2h,"2p2h_{E}","lf");
    else if(strcasestr(nuisanceFilename.Data(),"gibuu")) ReacLeg->AddEntry(nuis2p2h,"2p2h_{G}","lf");
    else ReacLeg->AddEntry(nuis2p2h,"2p2h_{N}","lf");
    ReacLeg->AddEntry(nuisrespi,"RES(#pi prod.)","lf");
    ReacLeg->AddEntry(nuisother,"Other","lf");
    ReacLeg->AddEntry((TObject*)0, Form("#chi^{2}=%.2f",nuisXsec_chi2), "");
    ReacLeg->SetFillColor(kWhite);
    ReacLeg->SetLineColor(kWhite);

    if((string)var=="dat") ReacLeg_shapeOnly = new TLegend(0.2,0.5,0.4,0.85);
    else ReacLeg_shapeOnly = new TLegend(0.65,0.5,0.85,0.85);
    ReacLeg_shapeOnly->AddEntry(xsecResult,"T2K","ep");
    ReacLeg_shapeOnly->AddEntry(nuisQE,"CCQE","lf");
    if     (strcasestr(nuisanceFilename.Data(),"genie")) ReacLeg_shapeOnly->AddEntry(nuis2p2h,"2p2h_{E}","lf");
    else if(strcasestr(nuisanceFilename.Data(),"gibuu")) ReacLeg_shapeOnly->AddEntry(nuis2p2h,"2p2h_{G}","lf");
    else ReacLeg_shapeOnly->AddEntry(nuis2p2h,"2p2h_{N}","lf");
    ReacLeg_shapeOnly->AddEntry(nuisrespi,"RES(#pi prod.)","lf");
    ReacLeg_shapeOnly->AddEntry(nuisother,"Other","lf");
    ReacLeg_shapeOnly->AddEntry((TObject*)0, Form("#chi^{2}=%.2f",nuisShapeOnly_chi2), "");
    ReacLeg_shapeOnly->SetFillColor(kWhite);
    ReacLeg_shapeOnly->SetLineColor(kWhite);

    if((string)var=="dat") gibuuLeg = new TLegend(0.4,0.81,0.85,0.87);
    else gibuuLeg = new TLegend(0.2,0.81,0.65,0.87);
    gibuuLeg->SetHeader("GiBUU 2016, CH"); gibuuLeg->AddEntry((TObject*)0, "(ANL Pi-Prod., Oset Delta in-medium width broadening)", "h");
    gibuuLeg->SetFillColor(kWhite);
    gibuuLeg->SetLineColor(kWhite);


    TLatex latex;
    latex.SetTextSize(0.05);

    nuisMC->SetMarkerStyle(0);
    nuisMC_shapeOnly->SetMarkerStyle(0);
    nuisMC_shapeOnly->GetYaxis()->SetTitleOffset(1.05);

    TCanvas* canv_allSelReacComp = new TCanvas("canv_reacStackXsec","canv_reacStackXsec");
    THStack *allSelReacStack = new THStack("reacStackXsec","reacStackXsec");
    //xsecResult->Draw("E1");
    nuisMC->Draw("HIST");
    allSelReacStack->Add(nuisother);
    allSelReacStack->Add(nuisrespi);
    allSelReacStack->Add(nuis2p2h);
    allSelReacStack->Add(nuisQE);
    allSelReacStack->Draw("sameHIST");
    xsecResult->Draw("sameE1");
    ReacLeg->Draw();
    if(gibuumode) gibuuLeg->Draw();
    canv_allSelReacComp->Update();
    //latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
    canv_allSelReacComp->Write();
    canv_allSelReacComp->SaveAs(Form("%s%s_%s_reacStackXsec.png",plotsDir,outPrefex.Data(),var));
    canv_allSelReacComp->SaveAs(Form("%s/rootfiles/%s_%s_reacStackXsec.pdf",plotsDir,outPrefex.Data(),var));

    TCanvas* canv_allSelReacComp_SO = new TCanvas("canv_reacStackShapeOnly","canv_reacStackShapeOnly");
    THStack *allSelReacStack_SO = new THStack("reacStackShapeOnly","reacStackShapeOnly");
    //shapeXSecResult->Draw("E1");
    nuisMC_shapeOnly->Draw("HIST");
    allSelReacStack_SO->Add(nuisother_shapeOnly);
    allSelReacStack_SO->Add(nuisrespi_shapeOnly);
    allSelReacStack_SO->Add(nuis2p2h_shapeOnly);
    allSelReacStack_SO->Add(nuisQE_shapeOnly);
    allSelReacStack_SO->Draw("sameHIST");
    shapeXSecResult->Draw("sameE1");
    ReacLeg_shapeOnly->Draw();
    if(gibuumode) gibuuLeg->Draw();
    canv_allSelReacComp_SO->Update();
    //latex.DrawLatex(gPad->GetUxmax()*0.65,gPad->GetUymax()*1.02,"#bf{T2K Preliminary}");
    canv_allSelReacComp_SO->Write();
    canv_allSelReacComp_SO->SaveAs(Form("%s%s_%s_reacStackShapeOnly.png",plotsDir,outPrefex.Data(),var));
    canv_allSelReacComp_SO->SaveAs(Form("%s/rootfiles/%s_%s_reacStackShapeOnly.pdf",plotsDir,outPrefex.Data(),var));

    if((string)var!="dat"){
      ReacLeg->Delete();
      ReacLeg_shapeOnly->Delete();
      gibuuLeg->Delete();

      TLegend* ReacLeg_inlay = new TLegend(0.45,0.3,0.84,0.85);
      ReacLeg_inlay->AddEntry(xsecResult,"T2K","ep");
      ReacLeg_inlay->AddEntry(nuisQE,"CCQE","lf");
      if     (strcasestr(nuisanceFilename.Data(),"genie")) ReacLeg_inlay->AddEntry(nuis2p2h,"2p2h_{E}","lf");
      else if(strcasestr(nuisanceFilename.Data(),"gibuu")) ReacLeg_inlay->AddEntry(nuis2p2h,"2p2h_{G}","lf");
      else ReacLeg_inlay->AddEntry(nuis2p2h,"2p2h_{N}","lf");
      ReacLeg_inlay->AddEntry(nuisrespi,"RES(#pi prod.)","lf");
      ReacLeg_inlay->AddEntry(nuisother,"Other","lf");
      ReacLeg_inlay->AddEntry((TObject*)0, Form("#chi^{2}=%.2f",nuisXsec_chi2), "");
      ReacLeg_inlay->SetFillColor(kWhite);
      ReacLeg_inlay->SetLineColor(kWhite);

      TLegend* ReacLeg_shapeOnly_inlay = new TLegend(0.45,0.3,0.84,0.85);
      ReacLeg_shapeOnly_inlay->AddEntry(xsecResult,"T2K","ep");
      ReacLeg_shapeOnly_inlay->AddEntry(nuisQE,"CCQE","lf");
      if     (strcasestr(nuisanceFilename.Data(),"genie")) ReacLeg_shapeOnly_inlay->AddEntry(nuis2p2h,"2p2h_{E}","lf");
      else if(strcasestr(nuisanceFilename.Data(),"gibuu")) ReacLeg_shapeOnly_inlay->AddEntry(nuis2p2h,"2p2h_{G}","lf");
      else ReacLeg_shapeOnly_inlay->AddEntry(nuis2p2h,"2p2h_{N}","lf");
      ReacLeg_shapeOnly_inlay->AddEntry(nuisrespi,"RES(#pi prod.)","lf");
      ReacLeg_shapeOnly_inlay->AddEntry(nuisother,"Other","lf");
      ReacLeg_shapeOnly_inlay->AddEntry((TObject*)0, Form("#chi^{2}=%.2f",nuisShapeOnly_chi2), "");
      ReacLeg_shapeOnly_inlay->SetFillColor(kWhite);
      ReacLeg_shapeOnly_inlay->SetLineColor(kWhite);

      TPad* inlay = new TPad("inlay","inlay",0.30,0.25,0.92,0.90);
      inlay->cd();
      THStack *allSelReacStack_inlay = new THStack("reacStackXsec","reacStackXsec");
      TH1D* nuisMC_clone =  new TH1D(*nuisMC);
      nuisMC_clone->SetXTitle("");
      nuisMC_clone->SetYTitle("");
      if((string)var=="dpt"){ 
        nuisMC_clone->GetXaxis()->SetRangeUser(0.26,1.1);
        nuisMC_clone->GetYaxis()->SetRangeUser(0.0,2.0E-39);
      }
      if((string)var=="dphit"){
        nuisMC_clone->GetXaxis()->SetRangeUser(0.52,3.1416);
        nuisMC_clone->GetYaxis()->SetRangeUser(0.0,0.7E-39);
      }
      nuisMC_clone->Draw("HIST");
      allSelReacStack_inlay->Add(nuisother);
      allSelReacStack_inlay->Add(nuisrespi);
      allSelReacStack_inlay->Add(nuis2p2h);
      allSelReacStack_inlay->Add(nuisQE);
      allSelReacStack_inlay->Draw("sameHIST");
      xsecResult->Draw("sameE1");
      ReacLeg_inlay->Draw();
      if(gibuumode) gibuuLeg->Draw();
      inlay->Update();
      canv_allSelReacComp->cd();
      inlay->SetFillStyle(0);
      inlay->Draw();
      canv_allSelReacComp->Write("canv_allSelReacComp_inlay");
      canv_allSelReacComp->SaveAs(Form("%s%s_%s_reacStackXsec_inlay.png",plotsDir,outPrefex.Data(),var));
      canv_allSelReacComp->SaveAs(Form("%s/rootfiles/%s_%s_reacStackXsec_inlay.pdf",plotsDir,outPrefex.Data(),var));
  
      TPad* inlay_SO = new TPad("inlay","inlay",0.30,0.25,0.92,0.90);
      inlay_SO->cd();
      THStack *allSelReacStack_SO_inlay = new THStack("reacStackShapeOnly","reacStackShapeOnly");
      //shapeXSecResult->Draw("E1");
      TH1D* nuisMC_shapeOnly_clone = new TH1D(*nuisMC_shapeOnly);
      nuisMC_shapeOnly_clone->SetXTitle("");
      nuisMC_shapeOnly_clone->SetYTitle("");
      if((string)var=="dpt"){ 
        nuisMC_shapeOnly_clone->GetXaxis()->SetRangeUser(0.26,1.1);
        nuisMC_shapeOnly_clone->GetYaxis()->SetRangeUser(0.0,0.08);
      }
      if((string)var=="dphit"){
        nuisMC_shapeOnly_clone->GetXaxis()->SetRangeUser(0.52,3.1416);
        nuisMC_shapeOnly_clone->GetYaxis()->SetRangeUser(0.0,0.05);
      }
      nuisMC_shapeOnly_clone->Draw("HIST");
      allSelReacStack_SO_inlay->Add(nuisother_shapeOnly);
      allSelReacStack_SO_inlay->Add(nuisrespi_shapeOnly);
      allSelReacStack_SO_inlay->Add(nuis2p2h_shapeOnly);
      allSelReacStack_SO_inlay->Add(nuisQE_shapeOnly);
      allSelReacStack_SO_inlay->Draw("sameHIST");
      shapeXSecResult->Draw("sameE1");
      ReacLeg_shapeOnly_inlay->Draw();
      if(gibuumode) gibuuLeg->Draw();
      inlay_SO->Update();
      canv_allSelReacComp_SO->cd();
      inlay_SO->SetFillStyle(0);
      inlay_SO->Draw();
      canv_allSelReacComp_SO->Write("canv_allSelReacComp_SO_inlay");
      canv_allSelReacComp_SO->SaveAs(Form("%s%s_%s_reacStackShapeOnly_inlay.png",plotsDir,outPrefex.Data(),var));
      canv_allSelReacComp_SO->SaveAs(Form("%s/rootfiles/%s_%s_reacStackShapeOnly_inlay.pdf",plotsDir,outPrefex.Data(),var));
    }
  }
  cout << "Finished :-)" << endl;
}



  
ThrowParms::ThrowParms(TVectorD &parms, TMatrixDSym &covm)
{  
  npars = parms.GetNrows();
  //std::cout << "Number of parameters " << npars << std::endl;
  pvals = new TVectorD(npars);
  covar = new TMatrixDSym(npars);
  //parms.Print();
  (*pvals) = parms;
  (*covar) = covm;
  TDecompChol chdcmp(*covar);
  if(!chdcmp.Decompose())
  {
    std::cerr<<"ERROR: Cholesky decomposition failed"<<std::endl;
    exit(-1);
  }
  chel_dec = new TMatrixD(chdcmp.GetU());
  CheloskyDecomp((*chel_dec));
}
   
ThrowParms::~ThrowParms()
{
  if(pvals!=NULL)    pvals->Delete();
  if(covar!=NULL)    covar->Delete();
  if(chel_dec!=NULL) chel_dec->Delete();
}
   
void ThrowParms::ThrowSet(std::vector<double> &parms)
{
  if(!parms.empty()) parms.clear();
  parms.resize(npars);

  if(npars>1){
    int half_pars = npars/2+npars%2;
    TVectorD std_rand(npars);
    for(int j=0; j<half_pars; j++){
      double z[2];
      StdNormRand(z);
      std_rand(j) = z[0];
      if(npars%2==0 || j!=half_pars-1)
      std_rand(j+half_pars) = z[1];
    }
    TVectorD prod = (*chel_dec)*std_rand;
    for(int i=0;i<npars;i++) parms[i] = prod(i) + (*pvals)(i);
  }
  else{
    parms[0]=gRandom->Gaus(0,sqrt((*covar)(0,0))) + (*pvals)(0);
  }
}

void ThrowParms::StdNormRand(double *z)
{
  
  double u = 2.*gRandom->Rndm()-1.;
  double v = 2.*gRandom->Rndm()-1.;
  
  double s = u*u+v*v;
  
  while(s==0 || s>=1.){
    u = 2.*gRandom->Rndm()-1.;
    v = 2.*gRandom->Rndm()-1.;
    s = u*u+v*v;
  }
  
  z[0] = u*sqrt(-2.*TMath::Log(s)/s);
  z[1] = v*sqrt(-2.*TMath::Log(s)/s);
}

void ThrowParms::CheloskyDecomp(TMatrixD &chel_mat)
{
  for(int i=0; i<npars; i++)
  {
    for(int j=0; j<npars; j++)
    {
      //if diagonal element
      if(i==j)
      {
        chel_mat(i,i) = (*covar)(i,i);
        for(int k=0; k<=i-1; k++) chel_mat(i,i) = chel_mat(i,i)-pow(chel_mat(i,k),2);
        chel_mat(i,i) = sqrt(chel_mat(i,i));
        //if lower half
      } 
      else if(j<i) 
      {
        chel_mat(i,j) = (*covar)(i,j);
        for(int k=0; k<=j-1; k++) chel_mat(i,j) = chel_mat(i,j)-chel_mat(i,k)*chel_mat(j,k);
        chel_mat(i,j) = chel_mat(i,j)/chel_mat(j,j);
      } else chel_mat(i,j) = 0.;
    }
  }
}
 
