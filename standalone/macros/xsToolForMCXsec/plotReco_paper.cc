/******************************************************

Produces diagnostic plots of the fit input/output trees

Author: Stephen Dolan
Date Created: October 2016

******************************************************/

#include <iostream> 
#include <iomanip>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <assert.h>

#include <TCanvas.h>
#include <TLegend.h>
#include <TH1F.h>
#include <THStack.h>
#include <TGraph.h>
#include <TTree.h>
#include <TString.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TMath.h>


using namespace std;

bool makeFinalWM = true;

int plotReco(TString inFilenameMC, TString inFilenameData, TString outFileName, TString varName="dpt", Float_t potRatio=1.0 /*0.2015*/, bool useFineBins=true,  bool scaleByBinWidth=true, bool weightMatching=false)
{
  TFile *infileMC = new TFile(inFilenameMC);
  TTree *intreeMC = (TTree*)infileMC->Get("selectedEvents");

  TFile *infileWM = new TFile("/data/t2k/dolan/fitting/feb17_refit/summaryPlots/dptResults/quickFtX_fitOut.root");
  TTree *intreeWM = (TTree*)infileWM->Get("selectedEvents");

  TFile *infileData = new TFile(inFilenameData);
  TTree *intreeData = (TTree*)infileData->Get("selectedEvents");

  TFile *outfile = new TFile(outFileName,"recreate");

  Int_t          reaction;
  Int_t          cutBranch=-999;
  Int_t          mectopology;
  Float_t        D1True;
  Float_t        D2True;
  Float_t        D1Rec;
  Float_t        D2Rec;
  Float_t        pMomRec;
  Float_t        pThetaRec;
  Float_t        pMomTrue;
  Float_t        pThetaTrue;
  Float_t        muMomRec;
  Float_t        muThetaRec;
  Float_t        muMomTrue;
  Float_t        muThetaTrue;
  Float_t        muCosThetaRec;
  Float_t        muCosThetaTrue;
  Float_t        pCosThetaRec;
  Float_t        pCosThetaTrue;
  Float_t        hmpMomRec;
  Float_t        hmpMomTrue;
  Float_t        hmpCosThetaRec;
  Float_t        hmpCosThetaTrue;
  Float_t        infpMMomDif;
  Float_t        infpAnglDif;
  Float_t        infp3MomDif;

  Float_t        Enureco=0;
  Float_t        Enutrue=0;
  Float_t        weight;

  Int_t          cutBranch_data;
  Float_t        D1Rec_data;
  Float_t        D2Rec_data;
  Float_t        muMomRec_data;
  Float_t        muCosThetaRec_data;
  Float_t        pMomRec_data;
  Float_t        pCosThetaRec_data;
  Float_t        weight_data;
  Float_t        hmpMomRec_data;
  Float_t        hmpCosThetaRec_data;
  Float_t        infpMMomDif_data;
  Float_t        infpAnglDif_data;
  Float_t        infp3MomDif_data;

  Int_t          reactionWM;
  Int_t          cutBranchWM=-999;
  Int_t          mectopologyWM;
  Float_t        muMomRecWM;
  Float_t        muMomTrueWM;
  Float_t        muCosThetaRecWM;
  Float_t        muCosThetaTrueWM;
  Float_t        weightWM;


  if(weightMatching){
    intreeWM->SetBranchAddress("reaction", &reactionWM);
    intreeWM->SetBranchAddress("cutBranch", &cutBranchWM);
    intreeWM->SetBranchAddress("mectopology", &mectopologyWM);
    intreeWM->SetBranchAddress("muMomRec", &muMomRecWM);
    intreeWM->SetBranchAddress("muMomTrue", &muMomTrueWM);
    intreeWM->SetBranchAddress("muCosThetaRec" , &muCosThetaRecWM);
    intreeWM->SetBranchAddress("muCosThetaTrue" , &muCosThetaTrueWM);
    intreeWM->SetBranchAddress("weight", &weightWM);
  }


  intreeMC->SetBranchAddress("reaction", &reaction);
  intreeMC->SetBranchAddress("cutBranch", &cutBranch);
  intreeMC->SetBranchAddress("mectopology", &mectopology);
  intreeMC->SetBranchAddress("D1True", &D1True);
  intreeMC->SetBranchAddress("D1Rec", &D1Rec);
  intreeMC->SetBranchAddress("D2True", &D2True);
  intreeMC->SetBranchAddress("D2Rec", &D2Rec);
  intreeMC->SetBranchAddress("muMomRec", &muMomRec);
  intreeMC->SetBranchAddress("muMomTrue", &muMomTrue);
  intreeMC->SetBranchAddress("muCosThetaRec" , &muCosThetaRec);
  intreeMC->SetBranchAddress("muCosThetaTrue" , &muCosThetaTrue);
  intreeMC->SetBranchAddress("pMomRec" , &pMomRec);
  intreeMC->SetBranchAddress("pMomTrue", &pMomTrue);
  intreeMC->SetBranchAddress("pCosThetaRec", &pCosThetaRec);
  intreeMC->SetBranchAddress("pCosThetaTrue", &pCosThetaTrue);
  intreeMC->SetBranchAddress("hmpMomRec" , &hmpMomRec);
  intreeMC->SetBranchAddress("hmpMomTrue", &hmpMomTrue);
  intreeMC->SetBranchAddress("hmpCosThetaRec", &hmpCosThetaRec);
  intreeMC->SetBranchAddress("hmpCosThetaTrue", &hmpCosThetaTrue);
  intreeMC->SetBranchAddress("infpMomDif" , &infpMMomDif);
  intreeMC->SetBranchAddress("infAngleDif" , &infpAnglDif);
  intreeMC->SetBranchAddress("inf3MomDif" , &infp3MomDif);
  intreeMC->SetBranchAddress("Enureco", &Enureco); 
  intreeMC->SetBranchAddress("Enutrue", &Enutrue);
  intreeMC->SetBranchAddress("weight", &weight);

  intreeData->SetBranchAddress("cutBranch", &cutBranch_data);
  intreeData->SetBranchAddress("D1Rec", &D1Rec_data);
  intreeData->SetBranchAddress("D2Rec", &D2Rec_data);
  intreeData->SetBranchAddress("muMomRec", &muMomRec_data);
  intreeData->SetBranchAddress("muCosThetaRec" , &muCosThetaRec_data);
  intreeData->SetBranchAddress("pMomRec" , &pMomRec_data);
  intreeData->SetBranchAddress("pCosThetaRec", &pCosThetaRec_data);
  intreeData->SetBranchAddress("hmpMomRec" , &hmpMomRec_data);
  intreeData->SetBranchAddress("hmpCosThetaRec", &hmpCosThetaRec_data);
  intreeData->SetBranchAddress("infpMomDif" , &infpMMomDif_data);
  intreeData->SetBranchAddress("infAngleDif" , &infpAnglDif_data);
  intreeData->SetBranchAddress("inf3MomDif" , &infp3MomDif_data);
  intreeData->SetBranchAddress("weight", &weight_data);


  Float_t *bins, *pmuBins, *cthmuBins, *ppBins, *cthpBins;
  Int_t nbinsn , pmuNBins, cthmuNBins, ppNBins, cthpNBins;
  if(varName=="dpt" && useFineBins){
    cout << "Using dpT"<< endl;
    nbinsn = 14;
    Float_t binsn[15] = { 0.00001, 0.04, 0.08, 0.12, 0.15, 0.175, 0.2, 0.23, 0.26, 0.31, 0.36, 0.43, 0.51, 0.75, 1.1};
    bins = new Float_t[15];
    for(int n=0;n<=nbinsn;n++) { bins[n]=binsn[n]; }
  }
  else if((varName=="dphiT" || varName=="dphit") && useFineBins){
    cout << "Using dphiT"<< endl;
    nbinsn = 15;
    Float_t binsn[16] = { 0.00001, 0.035, 0.07, 0.12, 0.17, 0.23, 0.28, 0.34, 0.43, 0.52, 0.65, 0.85, 1.1, 1.5, 2.0, 3.14159};
    bins = new Float_t[16];
    for(int n=0;n<=nbinsn;n++) { bins[n]=binsn[n]; }
  }
  else if((varName=="dalphaT" || varName=="dalphat" || varName=="dat") && useFineBins){
    cout << "Using dalphaT"<< endl;
    nbinsn = 16;
    Float_t binsn[17] = { 0.00001, 0.24, 0.47, 0.78, 1.02, 1.26, 1.54, 1.75, 2.0, 2.17, 2.34, 2.5, 2.64, 2.75, 2.89, 3.05, 3.14159};
    bins = new Float_t[17];
    for(int n=0;n<=nbinsn;n++) { bins[n]=binsn[n]; }
  }
  else if(varName=="dpt" && !useFineBins){
    cout << "Using dpT"<< endl;
    nbinsn = 8;
    Float_t binsn[9] = { 0.00001, 0.08, 0.12, 0.155, 0.2, 0.26, 0.36, 0.51, 1.1};
    bins = new Float_t[9];
    for(int n=0;n<=nbinsn;n++) { bins[n]=binsn[n]; }
  }
  else if((varName=="dphiT" || varName=="dphit") && !useFineBins){
    cout << "Using dphiT"<< endl;
    nbinsn = 8;
    Float_t binsn[9] = { 0.00001, 0.067, 0.14, 0.225, 0.34, 0.52, 0.85, 1.5, 3.14159};
    bins = new Float_t[9];
    for(int n=0;n<=nbinsn;n++) { bins[n]=binsn[n]; }
  }
  else if((varName=="dalphaT" || varName=="dalphat" || varName=="dat") && !useFineBins){
    cout << "Using dalphaT"<< endl;
    nbinsn = 8;
    Float_t binsn[9] = { 0.00001, 0.47, 1.02, 1.54, 1.98, 2.34, 2.64, 2.89, 3.14159};
    bins = new Float_t[9];
    for(int n=0;n<=nbinsn;n++) { bins[n]=binsn[n]; }
  }
  else{
    cout << "ERROR: Unrecognised variable: " << varName << endl;
    return 0;
  }
  if(useFineBins){
    //pmu
    pmuNBins = 20;
    Float_t pmuBinsn[21] = { 0.0000001, 200, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000, 1125, 1250, 1375, 1500, 1750, 2000, 2500, 3000};
    pmuBins = new Float_t[21];
    for(int n=0;n<=pmuNBins;n++) { pmuBins[n]=pmuBinsn[n]; }
    //cthmu
    cthmuNBins = 20;
    Float_t cthmuBinsn[21] = { -1.0, -0.5, -0.3, 0.0, 0.3, 0.45, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0};
    cthmuBins = new Float_t[21];
    for(int n=0;n<=cthmuNBins;n++) { cthmuBins[n]=cthmuBinsn[n]; }
    //pp
    ppNBins = 16;
    Float_t ppBinsn[17] = { 0.0000001, 300, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1500};
    ppBins = new Float_t[17];
    for(int n=0;n<=ppNBins;n++) { ppBins[n]=ppBinsn[n]; }
    //cthp
    cthpNBins = 13;
    Float_t cthpBinsn[14] = {-1.0, -0.3, 0.0, 0.3, 0.45, 0.6, 0.7, 0.75, 0.8, 0.85, 0.89, 0.92, 0.95, 1.0};
    cthpBins = new Float_t[14];
    for(int n=0;n<=cthpNBins;n++) { cthpBins[n]=cthpBinsn[n]; }
  }
  else {
    //pmu
    pmuNBins = 10;
    Float_t pmuBinsn[11] = { 0.0000001, 400, 500, 600, 700, 800, 1000, 1250, 1500, 2000, 3000};
    pmuBins = new Float_t[11];
    for(int n=0;n<=pmuNBins;n++) { pmuBins[n]=pmuBinsn[n]; }
    //cthmu
    cthmuNBins = 10;
    Float_t cthmuBinsn[11] = { -1.0, -0.3, 0.3, 0.6, 0.7, 0.8, 0.85, 0.9, 0.94, 0.98, 1.0};
    cthmuBins = new Float_t[11];
    for(int n=0;n<=cthmuNBins;n++) { cthmuBins[n]=cthmuBinsn[n]; }
    //pp
    ppNBins = 6;
    Float_t ppBinsn[7] = { 0.0000001, 500, 600, 700, 800, 900, 1100};
    ppBins = new Float_t[7];
    for(int n=0;n<=ppNBins;n++) { ppBins[n]=ppBinsn[n]; }
    //cthp
    cthpNBins = 7;
    Float_t cthpBinsn[8] = {-1.0, -0.3, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0};
    cthpBins = new Float_t[8];
    for(int n=0;n<=cthpNBins;n++) { cthpBins[n]=cthpBinsn[n]; }
  }

  TObjArray *listOhist = new TObjArray(500);


  //**********************************************
  // Data
  //**********************************************

  TH1D* stvData = new TH1D("stvData","stvData",nbinsn,bins); listOhist->Add(stvData);

  TH1D* infpMMomDifData = new TH1D("infpMMomDifData","infpMMomDifData",16,-1400,1800); listOhist->Add(infpMMomDifData);
  TH1D* infpAnglDifData = new TH1D("infpAnglDifData","infpAnglDifData",16,-1.57,1.57); listOhist->Add(infpAnglDifData);
  TH1D* infp3MomDifData = new TH1D("infp3MomDifData","infp3MomDifData",16,0.00,3000); listOhist->Add(infp3MomDifData);

  TH1D* mommuData = new TH1D("mommuData","mommuData",pmuNBins,pmuBins); listOhist->Add(mommuData);
  TH1D* cthmuData = new TH1D("cthmuData","cthmuData",cthmuNBins,cthmuBins); listOhist->Add(cthmuData);
  TH1D* momprData = new TH1D("momprData","momprData",ppNBins,ppBins); listOhist->Add(momprData);
  TH1D* cthprData = new TH1D("cthprData","cthprData",cthpNBins,cthpBins); listOhist->Add(cthprData);

  TH1D* mommuData_1pi = new TH1D("mommuData_1pi","mommuData_1pi",pmuNBins,pmuBins); listOhist->Add(mommuData_1pi);
  TH1D* cthmuData_1pi = new TH1D("cthmuData_1pi","cthmuData_1pi",cthmuNBins,cthmuBins); listOhist->Add(cthmuData_1pi);
  TH1D* momprData_1pi = new TH1D("momprData_1pi","momprData_1pi",ppNBins,ppBins); listOhist->Add(momprData_1pi);
  TH1D* cthprData_1pi = new TH1D("cthprData_1pi","cthprData_1pi",cthpNBins,cthpBins); listOhist->Add(cthprData_1pi);

  TH1D* mommuData_dis = new TH1D("mommuData_dis","mommuData_dis",pmuNBins,pmuBins); listOhist->Add(mommuData_dis);
  TH1D* cthmuData_dis = new TH1D("cthmuData_dis","cthmuData_dis",cthmuNBins,cthmuBins); listOhist->Add(cthmuData_dis);
  TH1D* momprData_dis = new TH1D("momprData_dis","momprData_dis",ppNBins,ppBins); listOhist->Add(momprData_dis);
  TH1D* cthprData_dis = new TH1D("cthprData_dis","cthprData_dis",cthpNBins,cthpBins); listOhist->Add(cthprData_dis);


  //**********************************************
  // Topology
  //**********************************************

  //CC0piNpIPS

  TH1D* stvCC0piNpIPS = new TH1D("stvCC0piNpIPS","stvCC0piNpIPS",nbinsn,bins); listOhist->Add(stvCC0piNpIPS);
  TH1D* mommuCC0piNpIPS = new TH1D("mommuCC0piNpIPS","mommuCC0piNpIPS",pmuNBins,pmuBins); listOhist->Add(mommuCC0piNpIPS);
  TH1D* cthmuCC0piNpIPS = new TH1D("cthmuCC0piNpIPS","cthmuCC0piNpIPS",cthmuNBins,cthmuBins); listOhist->Add(cthmuCC0piNpIPS);
  TH1D* momprCC0piNpIPS = new TH1D("momprCC0piNpIPS","momprCC0piNpIPS",ppNBins,ppBins); listOhist->Add(momprCC0piNpIPS);
  TH1D* cthprCC0piNpIPS = new TH1D("cthprCC0piNpIPS","cthprCC0piNpIPS",cthpNBins,cthpBins); listOhist->Add(cthprCC0piNpIPS);
  TH1D* infpMMomDifCC0piNpIPS = new TH1D("infpMMomDifCC0piNpIPS","infpMMomDifCC0piNpIPS",16,-1400,1800); listOhist->Add(infpMMomDifCC0piNpIPS);
  TH1D* infpAnglDifCC0piNpIPS = new TH1D("infpAnglDifCC0piNpIPS","infpAnglDifCC0piNpIPS",16,-1.57,1.57); listOhist->Add(infpAnglDifCC0piNpIPS);
  TH1D* infp3MomDifCC0piNpIPS = new TH1D("infp3MomDifCC0piNpIPS","infp3MomDifCC0piNpIPS",16,0.00,3000); listOhist->Add(infp3MomDifCC0piNpIPS);

  //CC0piNpOOPS

  TH1D* stvCC0piNpOOPS = new TH1D("stvCC0piNpOOPS","stvCC0piNpOOPS",nbinsn,bins); listOhist->Add(stvCC0piNpOOPS);
  TH1D* mommuCC0piNpOOPS = new TH1D("mommuCC0piNpOOPS","mommuCC0piNpOOPS",pmuNBins,pmuBins); listOhist->Add(mommuCC0piNpOOPS);
  TH1D* cthmuCC0piNpOOPS = new TH1D("cthmuCC0piNpOOPS","cthmuCC0piNpOOPS",cthmuNBins,cthmuBins); listOhist->Add(cthmuCC0piNpOOPS);
  TH1D* momprCC0piNpOOPS = new TH1D("momprCC0piNpOOPS","momprCC0piNpOOPS",ppNBins,ppBins); listOhist->Add(momprCC0piNpOOPS);
  TH1D* cthprCC0piNpOOPS = new TH1D("cthprCC0piNpOOPS","cthprCC0piNpOOPS",cthpNBins,cthpBins); listOhist->Add(cthprCC0piNpOOPS);
  TH1D* infpMMomDifCC0piNpOOPS = new TH1D("infpMMomDifCC0piNpOOPS","infpMMomDifCC0piNpOOPS",16,-1400,1800); listOhist->Add(infpMMomDifCC0piNpOOPS);
  TH1D* infpAnglDifCC0piNpOOPS = new TH1D("infpAnglDifCC0piNpOOPS","infpAnglDifCC0piNpOOPS",16,-1.57,1.57); listOhist->Add(infpAnglDifCC0piNpOOPS);
  TH1D* infp3MomDifCC0piNpOOPS = new TH1D("infp3MomDifCC0piNpOOPS","infp3MomDifCC0piNpOOPS",16,0.00,3000); listOhist->Add(infp3MomDifCC0piNpOOPS);

  //CC1pi

  TH1D* stvCC1pi = new TH1D("stvCC1pi","stvCC1pi",nbinsn,bins); listOhist->Add(stvCC1pi);
  TH1D* mommuCC1pi = new TH1D("mommuCC1pi","mommuCC1pi",pmuNBins,pmuBins); listOhist->Add(mommuCC1pi);
  TH1D* cthmuCC1pi = new TH1D("cthmuCC1pi","cthmuCC1pi",cthmuNBins,cthmuBins); listOhist->Add(cthmuCC1pi);
  TH1D* momprCC1pi = new TH1D("momprCC1pi","momprCC1pi",ppNBins,ppBins); listOhist->Add(momprCC1pi);
  TH1D* cthprCC1pi = new TH1D("cthprCC1pi","cthprCC1pi",cthpNBins,cthpBins); listOhist->Add(cthprCC1pi);
  TH1D* infpMMomDifCC1pi = new TH1D("infpMMomDifCC1pi","infpMMomDifCC1pi",16,-1400,1800); listOhist->Add(infpMMomDifCC1pi);
  TH1D* infpAnglDifCC1pi = new TH1D("infpAnglDifCC1pi","infpAnglDifCC1pi",16,-1.57,1.57); listOhist->Add(infpAnglDifCC1pi);
  TH1D* infp3MomDifCC1pi = new TH1D("infp3MomDifCC1pi","infp3MomDifCC1pi",16,0.00,3000); listOhist->Add(infp3MomDifCC1pi);

  //CCOther

  TH1D* stvCCOther = new TH1D("stvCCOther","stvCCOther",nbinsn,bins); listOhist->Add(stvCCOther);
  TH1D* mommuCCOther = new TH1D("mommuCCOther","mommuCCOther",pmuNBins,pmuBins); listOhist->Add(mommuCCOther);
  TH1D* cthmuCCOther = new TH1D("cthmuCCOther","cthmuCCOther",cthmuNBins,cthmuBins); listOhist->Add(cthmuCCOther);
  TH1D* momprCCOther = new TH1D("momprCCOther","momprCCOther",ppNBins,ppBins); listOhist->Add(momprCCOther);
  TH1D* cthprCCOther = new TH1D("cthprCCOther","cthprCCOther",cthpNBins,cthpBins); listOhist->Add(cthprCCOther);
  TH1D* infpMMomDifCCOther = new TH1D("infpMMomDifCCOther","infpMMomDifCCOther",16,-1400,1800); listOhist->Add(infpMMomDifCCOther);
  TH1D* infpAnglDifCCOther = new TH1D("infpAnglDifCCOther","infpAnglDifCCOther",16,-1.57,1.57); listOhist->Add(infpAnglDifCCOther);
  TH1D* infp3MomDifCCOther = new TH1D("infp3MomDifCCOther","infp3MomDifCCOther",16,0.00,3000); listOhist->Add(infp3MomDifCCOther);

  //Other

  TH1D* stvOther = new TH1D("stvOther","stvOther",nbinsn,bins); listOhist->Add(stvOther);
  TH1D* mommuOther = new TH1D("mommuOther","mommuOther",pmuNBins,pmuBins); listOhist->Add(mommuOther);
  TH1D* cthmuOther = new TH1D("cthmuOther","cthmuOther",cthmuNBins,cthmuBins); listOhist->Add(cthmuOther);
  TH1D* momprOther = new TH1D("momprOther","momprOther",ppNBins,ppBins); listOhist->Add(momprOther);
  TH1D* cthprOther = new TH1D("cthprOther","cthprOther",cthpNBins,cthpBins); listOhist->Add(cthprOther);
  TH1D* infpMMomDifOther = new TH1D("infpMMomDifOther","infpMMomDifOther",16,-1400,1800); listOhist->Add(infpMMomDifOther);
  TH1D* infpAnglDifOther = new TH1D("infpAnglDifOther","infpAnglDifOther",16,-1.57,1.57); listOhist->Add(infpAnglDifOther);
  TH1D* infp3MomDifOther = new TH1D("infp3MomDifOther","infp3MomDifOther",16,0.00,3000); listOhist->Add(infp3MomDifOther);

  //1pi CR

  //CC0piNpIPS
  TH1D* mommuCC0piNpIPS_1pi = new TH1D("mommuCC0piNpIPS_1pi","mommuCC0piNpIPS_1pi",pmuNBins,pmuBins); listOhist->Add(mommuCC0piNpIPS_1pi);
  TH1D* cthmuCC0piNpIPS_1pi = new TH1D("cthmuCC0piNpIPS_1pi","cthmuCC0piNpIPS_1pi",cthmuNBins,cthmuBins); listOhist->Add(cthmuCC0piNpIPS_1pi);
  TH1D* momprCC0piNpIPS_1pi = new TH1D("momprCC0piNpIPS_1pi","momprCC0piNpIPS_1pi",ppNBins,ppBins); listOhist->Add(momprCC0piNpIPS_1pi);
  TH1D* cthprCC0piNpIPS_1pi = new TH1D("cthprCC0piNpIPS_1pi","cthprCC0piNpIPS_1pi",cthpNBins,cthpBins); listOhist->Add(cthprCC0piNpIPS_1pi);
  //CC0piNpOOPS
  TH1D* mommuCC0piNpOOPS_1pi = new TH1D("mommuCC0piNpOOPS_1pi","mommuCC0piNpOOPS_1pi",pmuNBins,pmuBins); listOhist->Add(mommuCC0piNpOOPS_1pi);
  TH1D* cthmuCC0piNpOOPS_1pi = new TH1D("cthmuCC0piNpOOPS_1pi","cthmuCC0piNpOOPS_1pi",cthmuNBins,cthmuBins); listOhist->Add(cthmuCC0piNpOOPS_1pi);
  TH1D* momprCC0piNpOOPS_1pi = new TH1D("momprCC0piNpOOPS_1pi","momprCC0piNpOOPS_1pi",ppNBins,ppBins); listOhist->Add(momprCC0piNpOOPS_1pi);
  TH1D* cthprCC0piNpOOPS_1pi = new TH1D("cthprCC0piNpOOPS_1pi","cthprCC0piNpOOPS_1pi",cthpNBins,cthpBins); listOhist->Add(cthprCC0piNpOOPS_1pi);
  //CC1pi
  TH1D* mommuCC1pi_1pi = new TH1D("mommuCC1pi_1pi","mommuCC1pi_1pi",pmuNBins,pmuBins); listOhist->Add(mommuCC1pi_1pi);
  TH1D* cthmuCC1pi_1pi = new TH1D("cthmuCC1pi_1pi","cthmuCC1pi_1pi",cthmuNBins,cthmuBins); listOhist->Add(cthmuCC1pi_1pi);
  TH1D* momprCC1pi_1pi = new TH1D("momprCC1pi_1pi","momprCC1pi_1pi",ppNBins,ppBins); listOhist->Add(momprCC1pi_1pi);
  TH1D* cthprCC1pi_1pi = new TH1D("cthprCC1pi_1pi","cthprCC1pi_1pi",cthpNBins,cthpBins); listOhist->Add(cthprCC1pi_1pi);
  //CCOther
  TH1D* mommuCCOther_1pi = new TH1D("mommuCCOther_1pi","mommuCCOther_1pi",pmuNBins,pmuBins); listOhist->Add(mommuCCOther_1pi);
  TH1D* cthmuCCOther_1pi = new TH1D("cthmuCCOther_1pi","cthmuCCOther_1pi",cthmuNBins,cthmuBins); listOhist->Add(cthmuCCOther_1pi);
  TH1D* momprCCOther_1pi = new TH1D("momprCCOther_1pi","momprCCOther_1pi",ppNBins,ppBins); listOhist->Add(momprCCOther_1pi);
  TH1D* cthprCCOther_1pi = new TH1D("cthprCCOther_1pi","cthprCCOther_1pi",cthpNBins,cthpBins); listOhist->Add(cthprCCOther_1pi);
  //Other
  TH1D* mommuOther_1pi = new TH1D("mommuOther_1pi","mommuOther_1pi",pmuNBins,pmuBins); listOhist->Add(mommuOther_1pi);
  TH1D* cthmuOther_1pi = new TH1D("cthmuOther_1pi","cthmuOther_1pi",cthmuNBins,cthmuBins); listOhist->Add(cthmuOther_1pi);
  TH1D* momprOther_1pi = new TH1D("momprOther_1pi","momprOther_1pi",ppNBins,ppBins); listOhist->Add(momprOther_1pi);
  TH1D* cthprOther_1pi = new TH1D("cthprOther_1pi","cthprOther_1pi",cthpNBins,cthpBins); listOhist->Add(cthprOther_1pi);
  //Total
  TH1D* mommuTotal_1pi = new TH1D("mommuTotal_1pi","mommuTotal_1pi",pmuNBins,pmuBins); listOhist->Add(mommuTotal_1pi);
  TH1D* cthmuTotal_1pi = new TH1D("cthmuTotal_1pi","cthmuTotal_1pi",cthmuNBins,cthmuBins); listOhist->Add(cthmuTotal_1pi);
  TH1D* momprTotal_1pi = new TH1D("momprTotal_1pi","momprTotal_1pi",ppNBins,ppBins); listOhist->Add(momprTotal_1pi);
  TH1D* cthprTotal_1pi = new TH1D("cthprTotal_1pi","cthprTotal_1pi",cthpNBins,cthpBins); listOhist->Add(cthprTotal_1pi);

  //DIS CR

  //CC0piNpIPS
  TH1D* mommuCC0piNpIPS_dis = new TH1D("mommuCC0piNpIPS_dis","mommuCC0piNpIPS_dis",pmuNBins,pmuBins); listOhist->Add(mommuCC0piNpIPS_dis);
  TH1D* cthmuCC0piNpIPS_dis = new TH1D("cthmuCC0piNpIPS_dis","cthmuCC0piNpIPS_dis",cthmuNBins,cthmuBins); listOhist->Add(cthmuCC0piNpIPS_dis);
  TH1D* momprCC0piNpIPS_dis = new TH1D("momprCC0piNpIPS_dis","momprCC0piNpIPS_dis",ppNBins,ppBins); listOhist->Add(momprCC0piNpIPS_dis);
  TH1D* cthprCC0piNpIPS_dis = new TH1D("cthprCC0piNpIPS_dis","cthprCC0piNpIPS_dis",cthpNBins,cthpBins); listOhist->Add(cthprCC0piNpIPS_dis);
  //CC0piNpOOPS
  TH1D* mommuCC0piNpOOPS_dis = new TH1D("mommuCC0piNpOOPS_dis","mommuCC0piNpOOPS_dis",pmuNBins,pmuBins); listOhist->Add(mommuCC0piNpOOPS_dis);
  TH1D* cthmuCC0piNpOOPS_dis = new TH1D("cthmuCC0piNpOOPS_dis","cthmuCC0piNpOOPS_dis",cthmuNBins,cthmuBins); listOhist->Add(cthmuCC0piNpOOPS_dis);
  TH1D* momprCC0piNpOOPS_dis = new TH1D("momprCC0piNpOOPS_dis","momprCC0piNpOOPS_dis",ppNBins,ppBins); listOhist->Add(momprCC0piNpOOPS_dis);
  TH1D* cthprCC0piNpOOPS_dis = new TH1D("cthprCC0piNpOOPS_dis","cthprCC0piNpOOPS_dis",cthpNBins,cthpBins); listOhist->Add(cthprCC0piNpOOPS_dis);
  //CC1pi
  TH1D* mommuCC1pi_dis = new TH1D("mommuCC1pi_dis","mommuCC1pi_dis",pmuNBins,pmuBins); listOhist->Add(mommuCC1pi_dis);
  TH1D* cthmuCC1pi_dis = new TH1D("cthmuCC1pi_dis","cthmuCC1pi_dis",cthmuNBins,cthmuBins); listOhist->Add(cthmuCC1pi_dis);
  TH1D* momprCC1pi_dis = new TH1D("momprCC1pi_dis","momprCC1pi_dis",ppNBins,ppBins); listOhist->Add(momprCC1pi_dis);
  TH1D* cthprCC1pi_dis = new TH1D("cthprCC1pi_dis","cthprCC1pi_dis",cthpNBins,cthpBins); listOhist->Add(cthprCC1pi_dis);
  //CCOther
  TH1D* mommuCCOther_dis = new TH1D("mommuCCOther_dis","mommuCCOther_dis",pmuNBins,pmuBins); listOhist->Add(mommuCCOther_dis);
  TH1D* cthmuCCOther_dis = new TH1D("cthmuCCOther_dis","cthmuCCOther_dis",cthmuNBins,cthmuBins); listOhist->Add(cthmuCCOther_dis);
  TH1D* momprCCOther_dis = new TH1D("momprCCOther_dis","momprCCOther_dis",ppNBins,ppBins); listOhist->Add(momprCCOther_dis);
  TH1D* cthprCCOther_dis = new TH1D("cthprCCOther_dis","cthprCCOther_dis",cthpNBins,cthpBins); listOhist->Add(cthprCCOther_dis);
  //Other
  TH1D* mommuOther_dis = new TH1D("mommuOther_dis","mommuOther_dis",pmuNBins,pmuBins); listOhist->Add(mommuOther_dis);
  TH1D* cthmuOther_dis = new TH1D("cthmuOther_dis","cthmuOther_dis",cthmuNBins,cthmuBins); listOhist->Add(cthmuOther_dis);
  TH1D* momprOther_dis = new TH1D("momprOther_dis","momprOther_dis",ppNBins,ppBins); listOhist->Add(momprOther_dis);
  TH1D* cthprOther_dis = new TH1D("cthprOther_dis","cthprOther_dis",cthpNBins,cthpBins); listOhist->Add(cthprOther_dis);
  //Total
  TH1D* mommuTotal_dis = new TH1D("mommuTotal_dis","mommuTotal_dis",pmuNBins,pmuBins); listOhist->Add(mommuTotal_dis);
  TH1D* cthmuTotal_dis = new TH1D("cthmuTotal_dis","cthmuTotal_dis",cthmuNBins,cthmuBins); listOhist->Add(cthmuTotal_dis);
  TH1D* momprTotal_dis = new TH1D("momprTotal_dis","momprTotal_dis",ppNBins,ppBins); listOhist->Add(momprTotal_dis);
  TH1D* cthprTotal_dis = new TH1D("cthprTotal_dis","cthprTotal_dis",cthpNBins,cthpBins); listOhist->Add(cthprTotal_dis);


  //**********************************************
  // Branch comp
  //**********************************************

  int nBranch = 6;
  int piSbNum = 5;
  int disSbNum = piSbNum+1;

  TH1D* allSelBranchCompData = new TH1D("allSelBranchCompData","allSelBranchCompData",nBranch,0,nBranch); 

  TH1D* allSelBranchCompCC0Pi1PMC  = new TH1D("allSelBranchCompCC0Pi1PMC","allSelBranchCompCC0Pi1PMC",nBranch,0,nBranch); 
  TH1D* allSelBranchCompCC0PiNPMC  = new TH1D("allSelBranchCompCC0PiNPMC","allSelBranchCompCC0PiNPMC",nBranch,0,nBranch); 
  TH1D* allSelBranchCompCC1PiMC    = new TH1D("allSelBranchCompCC1PiMC","allSelBranchCompCC1PiMC",nBranch,0,nBranch); 
  TH1D* allSelBranchCompCCOtherMC  = new TH1D("allSelBranchCompCCOtherMC","allSelBranchCompCCOtherMC",nBranch,0,nBranch); 
  TH1D* allSelBranchCompOthMC      = new TH1D("allSelBranchCompOthMC","allSelBranchCompOthMC",nBranch,0,nBranch); 

  allSelBranchCompData->SetMarkerSize(1.5);

  // MC Loop:
  Int_t cutBranchConv;
  Float_t globBinCentre;
  bool isSigBranch0piNp=false, isSigBranch0pi=false, isSTVIPS=false, isMultidimIPS=false, is1piCRBranch=false, isdisCRBranch=false, isIfkIPS=false;
  Long64_t nentries = intreeMC->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    nb = intreeMC->GetEntry(jentry); nbytes += nb;
    weight = weight*potRatio;

    if(weightMatching){
      Long64_t nentriesWM = intreeWM->GetEntriesFast();
      Long64_t nbytesWM = 0, nbWM = 0;
      for (Long64_t jentryWM=0; jentryWM<nentries;jentryWM++) {
        nbWM = intreeWM->GetEntry(jentryWM); nbytesWM += nbWM;
        if(muMomTrue==muMomTrueWM){
          if(muCosThetaTrue!=muCosThetaTrueWM) cout << "WARNING! MISMATCH IN TREES" << endl;
          weight=weightWM;
          //cout << "Found match! Weight is: " << weight << endl;
          break;
        }
      }
    }

    //Set boolians
    isSigBranch0piNp = (cutBranch==1 || cutBranch==2 || cutBranch==3 || cutBranch==7);
    isSigBranch0pi = (isSigBranch0piNp || cutBranch==0 || cutBranch==4);
    is1piCRBranch = cutBranch==piSbNum;
    isdisCRBranch = cutBranch==disSbNum;
    isSTVIPS = (D2True==0);
    isIfkIPS = (pMomTrue>450.0 && pCosThetaTrue>0.4);
    isMultidimIPS = (pMomTrue>500.0);

    // Events comp by branch
    if (cutBranch==piSbNum) cutBranchConv=99;
    else if (cutBranch==disSbNum) cutBranchConv=99;
    else if (cutBranch>disSbNum) cutBranchConv=cutBranch-2;
    else cutBranchConv=cutBranch;

    if     (mectopology==1) allSelBranchCompCC0Pi1PMC->Fill(cutBranchConv+0.5, weight); 
    else if(mectopology==2) allSelBranchCompCC0PiNPMC->Fill(cutBranchConv+0.5, weight); 
    else if(mectopology==3) allSelBranchCompCC1PiMC->Fill(cutBranchConv+0.5, weight); 
    else if(mectopology==4) allSelBranchCompCCOtherMC->Fill(cutBranchConv+0.5, weight); 
    else                    allSelBranchCompOthMC->Fill(cutBranchConv+0.5, weight); 

    //Fill STV
    if(isSigBranch0piNp && D2Rec==0){
      if( (mectopology==1 || mectopology==2) && isSTVIPS )       stvCC0piNpIPS->Fill(D1Rec, weight);
      else if( (mectopology==1 || mectopology==2) && !isSTVIPS ) stvCC0piNpOOPS->Fill(D1Rec, weight);
      else if(mectopology==3)                                    stvCC1pi->Fill(D1Rec, weight);
      else if(mectopology==4 || mectopology==0)                  stvCCOther->Fill(D1Rec, weight);
      else                                                       stvOther->Fill(D1Rec, weight);
    }

    //Fill infpMMomDif
    if(isSigBranch0piNp && pMomRec>450.0 && pCosThetaRec>0.4){
      if( (mectopology==1 || mectopology==2) && isIfkIPS )       infpMMomDifCC0piNpIPS->Fill(infpMMomDif, weight);
      else if( (mectopology==1 || mectopology==2) && !isIfkIPS ) infpMMomDifCC0piNpOOPS->Fill(infpMMomDif, weight);
      else if(mectopology==3)                                    infpMMomDifCC1pi->Fill(infpMMomDif, weight);
      else if(mectopology==4 || mectopology==0)                  infpMMomDifCCOther->Fill(infpMMomDif, weight);
      else                                                       infpMMomDifOther->Fill(infpMMomDif, weight);
    }

    //Fill infpAnglDif
    if(isSigBranch0piNp && pMomRec>450.0 && pCosThetaRec>0.4){
      if( (mectopology==1 || mectopology==2) && isIfkIPS )       infpAnglDifCC0piNpIPS->Fill(infpAnglDif, weight);
      else if( (mectopology==1 || mectopology==2) && !isIfkIPS ) infpAnglDifCC0piNpOOPS->Fill(infpAnglDif, weight);
      else if(mectopology==3)                                    infpAnglDifCC1pi->Fill(infpAnglDif, weight);
      else if(mectopology==4 || mectopology==0)                  infpAnglDifCCOther->Fill(infpAnglDif, weight);
      else                                                       infpAnglDifOther->Fill(infpAnglDif, weight);
    }

    //Fill infp3MomDif
    if(isSigBranch0piNp && pMomRec>450.0 && pCosThetaRec>0.4){
      if( (mectopology==1 || mectopology==2) && isIfkIPS )       infp3MomDifCC0piNpIPS->Fill(infp3MomDif, weight);
      else if( (mectopology==1 || mectopology==2) && !isIfkIPS ) infp3MomDifCC0piNpOOPS->Fill(infp3MomDif, weight);
      else if(mectopology==3)                                    infp3MomDifCC1pi->Fill(infp3MomDif, weight);
      else if(mectopology==4 || mectopology==0)                  infp3MomDifCCOther->Fill(infp3MomDif, weight);
      else                                                       infp3MomDifOther->Fill(infp3MomDif, weight);
    }

    //Fill pmu

    if(isSigBranch0pi){
      if( (mectopology==1 || mectopology==2) && isMultidimIPS )       mommuCC0piNpIPS->Fill(muMomRec, weight);
      else if( (mectopology==1 || mectopology==2) && !isMultidimIPS ) mommuCC0piNpOOPS->Fill(muMomRec, weight);
      else if(mectopology==3)                                         mommuCC1pi->Fill(muMomRec, weight);
      else if(mectopology==4 || mectopology==0)                       mommuCCOther->Fill(muMomRec, weight);
      else                                                            mommuOther->Fill(muMomRec, weight);
    }
    else if (is1piCRBranch){
      if( (mectopology==1 || mectopology==2) && isMultidimIPS )       mommuCC0piNpIPS_1pi->Fill(muMomRec, weight);
      else if( (mectopology==1 || mectopology==2) && !isMultidimIPS ) mommuCC0piNpOOPS_1pi->Fill(muMomRec, weight);
      else if(mectopology==3)                                         mommuCC1pi_1pi->Fill(muMomRec, weight);
      else if(mectopology==4 || mectopology==0)                       mommuCCOther_1pi->Fill(muMomRec, weight);
      else                                                            mommuOther_1pi->Fill(muMomRec, weight);
      mommuTotal_1pi->Fill(muMomRec, weight);
    }
    else if (isdisCRBranch){
      if( (mectopology==1 || mectopology==2) && isMultidimIPS )       mommuCC0piNpIPS_dis->Fill(muMomRec, weight);
      else if( (mectopology==1 || mectopology==2) && !isMultidimIPS ) mommuCC0piNpOOPS_dis->Fill(muMomRec, weight);
      else if(mectopology==3)                                         mommuCC1pi_dis->Fill(muMomRec, weight);
      else if(mectopology==4 || mectopology==0)                       mommuCCOther_dis->Fill(muMomRec, weight);
      else                                                            mommuOther_dis->Fill(muMomRec, weight);
      mommuTotal_dis->Fill(muMomRec, weight);
    }

    //Fill cthmu
  
    if(isSigBranch0pi){
      if( (mectopology==1 || mectopology==2) && isMultidimIPS )       cthmuCC0piNpIPS->Fill(muCosThetaRec, weight);
      else if( (mectopology==1 || mectopology==2) && !isMultidimIPS ) cthmuCC0piNpOOPS->Fill(muCosThetaRec, weight);
      else if(mectopology==3)                                         cthmuCC1pi->Fill(muCosThetaRec, weight);
      else if(mectopology==4 || mectopology==0)                       cthmuCCOther->Fill(muCosThetaRec, weight);
      else                                                            cthmuOther->Fill(muCosThetaRec, weight);
    }
    else if(is1piCRBranch){
      if( (mectopology==1 || mectopology==2) && isMultidimIPS )       cthmuCC0piNpIPS_1pi->Fill(muCosThetaRec, weight);
      else if( (mectopology==1 || mectopology==2) && !isMultidimIPS ) cthmuCC0piNpOOPS_1pi->Fill(muCosThetaRec, weight);
      else if(mectopology==3)                                         cthmuCC1pi_1pi->Fill(muCosThetaRec, weight);
      else if(mectopology==4 || mectopology==0)                       cthmuCCOther_1pi->Fill(muCosThetaRec, weight);
      else                                                            cthmuOther_1pi->Fill(muCosThetaRec, weight);
      cthmuTotal_1pi->Fill(muCosThetaRec, weight);
    }
    else if(isdisCRBranch){
      if( (mectopology==1 || mectopology==2) && isMultidimIPS )       cthmuCC0piNpIPS_dis->Fill(muCosThetaRec, weight);
      else if( (mectopology==1 || mectopology==2) && !isMultidimIPS ) cthmuCC0piNpOOPS_dis->Fill(muCosThetaRec, weight);
      else if(mectopology==3)                                         cthmuCC1pi_dis->Fill(muCosThetaRec, weight);
      else if(mectopology==4 || mectopology==0)                       cthmuCCOther_dis->Fill(muCosThetaRec, weight);
      else                                                            cthmuOther_dis->Fill(muCosThetaRec, weight);
      cthmuTotal_dis->Fill(muCosThetaRec, weight);
    }

    //Fill pp

    if(isSigBranch0piNp){
      if( (mectopology==1 || mectopology==2) && isMultidimIPS )       momprCC0piNpIPS->Fill(pMomRec, weight);
      else if( (mectopology==1 || mectopology==2) && !isMultidimIPS ) momprCC0piNpOOPS->Fill(pMomRec, weight);
      else if(mectopology==3)                                         momprCC1pi->Fill(pMomRec, weight);
      else if(mectopology==4 || mectopology==0)                       momprCCOther->Fill(pMomRec, weight);
      else                                                            momprOther->Fill(pMomRec, weight);
    }
    else if(is1piCRBranch){
      if( (mectopology==1 || mectopology==2) && isMultidimIPS )       momprCC0piNpIPS_1pi->Fill(hmpMomRec, weight);
      else if( (mectopology==1 || mectopology==2) && !isMultidimIPS ) momprCC0piNpOOPS_1pi->Fill(hmpMomRec, weight);
      else if(mectopology==3)                                         momprCC1pi_1pi->Fill(hmpMomRec, weight);
      else if(mectopology==4 || mectopology==0)                       momprCCOther_1pi->Fill(hmpMomRec, weight);
      else                                                            momprOther_1pi->Fill(hmpMomRec, weight);
      momprTotal_1pi->Fill(hmpMomRec, weight);
    }
    else if(isdisCRBranch){
      if( (mectopology==1 || mectopology==2) && isMultidimIPS )       momprCC0piNpIPS_dis->Fill(hmpMomRec, weight);
      else if( (mectopology==1 || mectopology==2) && !isMultidimIPS ) momprCC0piNpOOPS_dis->Fill(hmpMomRec, weight);
      else if(mectopology==3)                                         momprCC1pi_dis->Fill(hmpMomRec, weight);
      else if(mectopology==4 || mectopology==0)                       momprCCOther_dis->Fill(hmpMomRec, weight);
      else                                                            momprOther_dis->Fill(hmpMomRec, weight);
      momprTotal_dis->Fill(hmpMomRec, weight);
    }

    //Fill cthp
  
    if(isSigBranch0piNp && pMomRec>500){
      if( (mectopology==1 || mectopology==2) && isMultidimIPS )       cthprCC0piNpIPS->Fill(pCosThetaRec, weight);
      else if( (mectopology==1 || mectopology==2) && !isMultidimIPS ) cthprCC0piNpOOPS->Fill(pCosThetaRec, weight);
      else if(mectopology==3)                                         cthprCC1pi->Fill(pCosThetaRec, weight);
      else if(mectopology==4 || mectopology==0)                       cthprCCOther->Fill(pCosThetaRec, weight);
      else                                                            cthprOther->Fill(pCosThetaRec, weight);
    }
    else if(is1piCRBranch) {
      if( (mectopology==1 || mectopology==2) && isMultidimIPS )       cthprCC0piNpIPS_1pi->Fill(hmpCosThetaRec, weight);
      else if( (mectopology==1 || mectopology==2) && !isMultidimIPS ) cthprCC0piNpOOPS_1pi->Fill(hmpCosThetaRec, weight);
      else if(mectopology==3)                                         cthprCC1pi_1pi->Fill(hmpCosThetaRec, weight);
      else if(mectopology==4 || mectopology==0)                       cthprCCOther_1pi->Fill(hmpCosThetaRec, weight);
      else                                                            cthprOther_1pi->Fill(hmpCosThetaRec, weight);    
      cthprTotal_1pi->Fill(hmpCosThetaRec, weight);
    }
    else if(isdisCRBranch) {
      if( (mectopology==1 || mectopology==2) && isMultidimIPS )       cthprCC0piNpIPS_dis->Fill(hmpCosThetaRec, weight);
      else if( (mectopology==1 || mectopology==2) && !isMultidimIPS ) cthprCC0piNpOOPS_dis->Fill(hmpCosThetaRec, weight);
      else if(mectopology==3)                                         cthprCC1pi_dis->Fill(hmpCosThetaRec, weight);
      else if(mectopology==4 || mectopology==0)                       cthprCCOther_dis->Fill(hmpCosThetaRec, weight);
      else                                                            cthprOther_dis->Fill(hmpCosThetaRec, weight);  
      cthprTotal_dis->Fill(hmpCosThetaRec, weight);
    }
  }


  // Data Loop:
  bool  isSigBranch0piNp_data=false, isSigBranch0pi_data=false, is1piCRBranch_data=false, isdisCRBranch_data=false;;
  nentries = intreeData->GetEntriesFast();
  nbytes = 0;
  nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    nb = intreeData->GetEntry(jentry); nbytes += nb;

    //Set boolians
    isSigBranch0piNp_data = (cutBranch_data==1 || cutBranch_data==2 || cutBranch_data==3 || cutBranch_data==7);
    isSigBranch0pi_data = (isSigBranch0piNp_data || cutBranch_data==0 || cutBranch_data==4);
    is1piCRBranch_data = cutBranch_data==piSbNum;
    isdisCRBranch_data = cutBranch_data==disSbNum;

    // Events comp by branch
    if (cutBranch_data==piSbNum) cutBranchConv=99;
    else if (cutBranch_data==disSbNum) cutBranchConv=99;
    else if (cutBranch_data>disSbNum) cutBranchConv=cutBranch_data-2;
    else cutBranchConv=cutBranch_data;
    allSelBranchCompData->Fill(cutBranchConv+0.5, weight_data);

    //Fill STV
    if(isSigBranch0piNp_data && D2Rec_data==0){
      //cout << "Filling STV data with:" << D1Rec_data << endl;
      stvData->Fill(D1Rec_data, weight_data);
      //stvData->Print("all");
    }

    //Fill infpMMomDif
    if(isSigBranch0piNp_data &&  pMomRec_data>450.0 && pCosThetaRec_data>0.4){
      infpMMomDifData->Fill(infpMMomDif_data, weight_data);
    }
    //Fill infpAnglDif
    if(isSigBranch0piNp_data &&  pMomRec_data>450.0 && pCosThetaRec_data>0.4){
      infpAnglDifData->Fill(infpAnglDif_data, weight_data);
    }
    //Fill infp3MomDif
    if(isSigBranch0piNp_data &&  pMomRec_data>450.0 && pCosThetaRec_data>0.4){
      infp3MomDifData->Fill(infp3MomDif_data, weight_data);
    }

    //Fill pmu
    if(isSigBranch0pi_data){
      mommuData->Fill(muMomRec_data, weight_data);
    }
    else if(is1piCRBranch_data){
      mommuData_1pi->Fill(muMomRec_data, weight_data);
    }
    else if(isdisCRBranch_data){
      mommuData_dis->Fill(muMomRec_data, weight_data);
    }

    //Fill cthmu
    if(isSigBranch0pi_data){
      cthmuData->Fill(muCosThetaRec_data, weight_data);
    }
    else if(is1piCRBranch_data){
      cthmuData_1pi->Fill(muCosThetaRec_data, weight_data);
    }
    else if(isdisCRBranch_data){
      cthmuData_dis->Fill(muCosThetaRec_data, weight_data);
    }

    //Fill pp
    if(isSigBranch0piNp_data){
      momprData->Fill(pMomRec_data, weight_data);
    }
    else if(is1piCRBranch_data){
      momprData_1pi->Fill(hmpMomRec_data, weight_data);
    }
    else if(isdisCRBranch_data){
      momprData_dis->Fill(hmpMomRec_data, weight_data);
    }

    //Fill cthp
    if(isSigBranch0piNp_data){
      cthprData->Fill(pCosThetaRec_data, weight_data);
    }
    else if(is1piCRBranch_data){
      cthprData_1pi->Fill(hmpCosThetaRec_data, weight_data);
    }
    else if(isdisCRBranch_data){
      cthprData_dis->Fill(hmpCosThetaRec_data, weight_data);
    }

  }

  //stvCC0piNp
  //mommuCC0piNp
  //cthmuCC0piNp
  //momprCC0piNp
  //cthprCC0piNp
  //infpMMomDifCC0piNp
  //infpAnglDifCC0piNp
  //infp3MomDifCC0piNp

  for(int i=0; i<listOhist->GetEntries(); i++){
    if(scaleByBinWidth){
      ((TH1D*)(listOhist->At(i)))->Sumw2();
      ((TH1D*)(listOhist->At(i)))->Scale(1, "width");
      ((TH1D*)(listOhist->At(i)))->GetYaxis()->SetTitleOffset(1.3);
    }
    if ( strstr(((TH1D*)(listOhist->At(i)))->GetName(),"data") ){
      ((TH1D*)(listOhist->At(i)))->SetMarkerStyle(20);
      ((TH1D*)(listOhist->At(i)))->SetMarkerSize(1.5);
    }
    if ( strstr(((TH1D*)(listOhist->At(i)))->GetName(),"CC0piNpIPS") ){
      ((TH1D*)(listOhist->At(i)))->SetLineColor(kBlue+3);
      ((TH1D*)(listOhist->At(i)))->SetFillColor(kBlue-4);
      ((TH1D*)(listOhist->At(i)))->SetFillStyle(3004);
    }
    if ( strstr(((TH1D*)(listOhist->At(i)))->GetName(),"CC0piNpOOPS") ){
      ((TH1D*)(listOhist->At(i)))->SetLineColor(kCyan);
      ((TH1D*)(listOhist->At(i)))->SetFillColor(kCyan-7);
      ((TH1D*)(listOhist->At(i)))->SetFillStyle(3005);
    }
    if ( strstr(((TH1D*)(listOhist->At(i)))->GetName(),"CC1pi") ){
      ((TH1D*)(listOhist->At(i)))->SetLineColor(kGreen);
      ((TH1D*)(listOhist->At(i)))->SetFillColor(kGreen-7);
      ((TH1D*)(listOhist->At(i)))->SetFillStyle(3001);
    }
    if ( strstr(((TH1D*)(listOhist->At(i)))->GetName(),"CCOther") ){
      ((TH1D*)(listOhist->At(i)))->SetLineColor(kRed);
      ((TH1D*)(listOhist->At(i)))->SetFillColor(kRed-7);
      ((TH1D*)(listOhist->At(i)))->SetFillStyle(3002);
    }
    else if ( strstr(((TH1D*)(listOhist->At(i)))->GetName(),"Other") ){
      ((TH1D*)(listOhist->At(i)))->SetLineColor(kViolet);
      ((TH1D*)(listOhist->At(i)))->SetFillColor(kViolet-8);
      ((TH1D*)(listOhist->At(i)))->SetFillStyle(3006);
    }
    if ( strstr(((TH1D*)(listOhist->At(i)))->GetName(),"cth") ){
      //cout << "Setting cth title " << endl;
      ((TH1D*)(listOhist->At(i)))->GetYaxis()->SetTitle	("Events/bin width");
      if(strstr(((TH1D*)(listOhist->At(i)))->GetName(),"mu")) ((TH1D*)(listOhist->At(i)))->GetXaxis()->SetTitle("cos(#theta_{#mu})");
      if(strstr(((TH1D*)(listOhist->At(i)))->GetName(),"pr")){ 
        if(strstr(((TH1D*)(listOhist->At(i)))->GetName(),"_1pi") || strstr(((TH1D*)(listOhist->At(i)))->GetName(),"_dis")) ((TH1D*)(listOhist->At(i)))->GetXaxis()->SetTitle("cos(#theta_{HMP})");
        else ((TH1D*)(listOhist->At(i)))->GetXaxis()->SetTitle("cos(#theta_{p})");
      }
    }
    if ( strstr(((TH1D*)(listOhist->At(i)))->GetName(),"mom") ){
      //cout << "Setting mom title " << endl;
      ((TH1D*)(listOhist->At(i)))->GetYaxis()->SetTitle	("Events/MeV");
      if(strstr(((TH1D*)(listOhist->At(i)))->GetName(),"mu")) ((TH1D*)(listOhist->At(i)))->GetXaxis()->SetTitle("p^{reco}_{#mu} (MeV)");
      if(strstr(((TH1D*)(listOhist->At(i)))->GetName(),"pr")){
        if(strstr(((TH1D*)(listOhist->At(i)))->GetName(),"_1pi") || strstr(((TH1D*)(listOhist->At(i)))->GetName(),"_dis")) ((TH1D*)(listOhist->At(i)))->GetXaxis()->SetTitle("p^{reco}_{HMP} (MeV)");
        else ((TH1D*)(listOhist->At(i)))->GetXaxis()->SetTitle("p^{reco}_{p} (MeV)");
      }
    }
    if ( strstr(((TH1D*)(listOhist->At(i)))->GetName(),"infpMMomDif") ){
      ((TH1D*)(listOhist->At(i)))->GetYaxis()->SetTitle ("Events/MeV");
      ((TH1D*)(listOhist->At(i)))->GetXaxis()->SetTitle("#Delta p_{p}^{reco} (MeV)");
    }
    if ( strstr(((TH1D*)(listOhist->At(i)))->GetName(),"infpAnglDif") ){
      ((TH1D*)(listOhist->At(i)))->GetYaxis()->SetTitle ("Events/degree");
      ((TH1D*)(listOhist->At(i)))->GetXaxis()->SetTitle("#Delta #theta_{p}^{reco} (radians)");
    }
    if ( strstr(((TH1D*)(listOhist->At(i)))->GetName(),"infp3MomDifData") ){
      ((TH1D*)(listOhist->At(i)))->GetYaxis()->SetTitle ("Events/MeV");
      ((TH1D*)(listOhist->At(i)))->GetXaxis()->SetTitle("|#Delta p_{p}^{reco}| (MeV)");
    }
    if ( strstr(((TH1D*)(listOhist->At(i)))->GetName(),"stv") ){
      if(varName=="dpt"){
        //cout << "Setting dpt title " << endl;
        ((TH1D*)(listOhist->At(i)))->GetXaxis()->SetTitle("#deltap^{reco}_{T} (GeV)");
        ((TH1D*)(listOhist->At(i)))->GetYaxis()->SetTitle	("Events/GeV");
        //cout << "dpt x title is: " << ((TH1D*)(listOhist->At(i)))->GetXaxis()->GetTitle() << endl;
      }//
      else if(varName=="dphiT" || varName=="dphit"){
        ((TH1D*)(listOhist->At(i)))->GetXaxis()->SetTitle("#delta#phi^{reco}_{T} (radians)");
        ((TH1D*)(listOhist->At(i)))->GetYaxis()->SetTitle	("Events/radian");
      }
      else if(varName=="dalphaT" || varName=="dalphat" || varName=="dat"){
        ((TH1D*)(listOhist->At(i)))->GetXaxis()->SetTitle("#delta#alpha^{reco}_{T} (radians)");
        ((TH1D*)(listOhist->At(i)))->GetYaxis()->SetTitle	("Events/radian");
      }
    }
    if ( strstr(((TH1D*)(listOhist->At(i)))->GetName(),"Data") ){
      ((TH1D*)(listOhist->At(i)))->Write();
    }
  }

  //Branch Comp


  allSelBranchCompCC0Pi1PMC->SetLineColor(kRed);
  allSelBranchCompCC0PiNPMC->SetLineColor(kBlue+3);
  allSelBranchCompCC1PiMC->SetLineColor(kGreen);
  allSelBranchCompCCOtherMC->SetLineColor(kViolet);
  allSelBranchCompOthMC->SetLineColor(kCyan);

  allSelBranchCompCC0Pi1PMC->SetFillColor(kRed-7);
  allSelBranchCompCC0PiNPMC->SetFillColor(kBlue-4);
  allSelBranchCompCC1PiMC->SetFillColor(kGreen-7);
  allSelBranchCompCCOtherMC->SetFillColor(kViolet-8);
  allSelBranchCompOthMC->SetFillColor(kCyan-7);

  allSelBranchCompCC0Pi1PMC->SetFillStyle(3004);
  allSelBranchCompCC0PiNPMC->SetFillStyle(3005);
  allSelBranchCompCC1PiMC->SetFillStyle(3001);
  allSelBranchCompCCOtherMC->SetFillStyle(3002);
  allSelBranchCompOthMC->SetFillStyle(3006);

  stvData->SetMarkerSize(1.5);
  infpMMomDifData->SetMarkerSize(1.5);
  infpAnglDifData->SetMarkerSize(1.5);
  infp3MomDifData->SetMarkerSize(1.5);
  mommuData->SetMarkerSize(1.5);
  cthmuData->SetMarkerSize(1.5);
  momprData->SetMarkerSize(1.5);
  cthprData->SetMarkerSize(1.5);
  mommuData_1pi->SetMarkerSize(1.5);
  cthmuData_1pi->SetMarkerSize(1.5);
  momprData_1pi->SetMarkerSize(1.5);
  cthprData_1pi->SetMarkerSize(1.5);
  mommuData_dis->SetMarkerSize(1.5);
  cthmuData_dis->SetMarkerSize(1.5);
  momprData_dis->SetMarkerSize(1.5);
  cthprData_dis->SetMarkerSize(1.5);

  //Legends

  TLegend* TopoLeg = new TLegend(0.65,0.5,0.85,0.85);
  TopoLeg->AddEntry(stvData,"Data","ep");
  TopoLeg->AddEntry(stvCC0piNpIPS,"CC0#piNp IPS","lf");
  TopoLeg->AddEntry(stvCC0piNpOOPS,"CC0#piNp OOPS","lf");
  TopoLeg->AddEntry(stvCC1pi,"CC1#pi^{+}","lf");
  TopoLeg->AddEntry(stvCCOther,"CCOther","lf");
  TopoLeg->AddEntry(stvOther,"Other","lf");
  TopoLeg->SetFillColor(kWhite);
  TopoLeg->SetLineColor(kWhite);


  TLegend* TopoLegBr = new TLegend(0.65,0.5,0.85,0.85);
  TopoLegBr->AddEntry(stvData,"Data","ep");
  TopoLegBr->AddEntry(allSelBranchCompCC0Pi1PMC,"CC0#pi1p","lf");
  TopoLegBr->AddEntry(allSelBranchCompCC0PiNPMC,"CC0#piNp","lf");
  TopoLegBr->AddEntry(allSelBranchCompCC1PiMC,"CC1#pi^{+}","lf");
  TopoLegBr->AddEntry(allSelBranchCompCCOtherMC,"CCOther","lf");
  TopoLegBr->AddEntry(allSelBranchCompOthMC,"Other","lf");
  TopoLegBr->SetFillColor(kWhite);
  TopoLegBr->SetLineColor(kWhite);

  TLegend* TopoLegCth = new TLegend(0.25,0.5,0.45,0.85);
  TopoLegCth->AddEntry(stvData,"Data","ep");
  TopoLegCth->AddEntry(stvCC0piNpIPS,"CC0#piNp IPS","lf");
  TopoLegCth->AddEntry(stvCC0piNpOOPS,"CC0#piNp OOPS","lf");
  TopoLegCth->AddEntry(stvCC1pi,"CC1#pi^{+}","lf");
  TopoLegCth->AddEntry(stvCCOther,"CCOther","lf");
  TopoLegCth->AddEntry(stvOther,"Other","lf");
  TopoLegCth->SetFillColor(kWhite);
  TopoLegCth->SetLineColor(kWhite);

  TLegend* TopoLeg_md = new TLegend(0.65,0.5,0.85,0.85);
  TopoLeg_md->AddEntry(stvData,"Data","ep");
  TopoLeg_md->AddEntry(stvCC0piNpIPS,"CC0#piNp p_{p}<500 MeV","lf");
  TopoLeg_md->AddEntry(stvCC0piNpOOPS,"CC0#piNp p_{p}>500 MeV","lf");
  TopoLeg_md->AddEntry(stvCC1pi,"CC1#pi^{+}","lf");
  TopoLeg_md->AddEntry(stvCCOther,"CCOther","lf");
  TopoLeg_md->AddEntry(stvOther,"Other","lf");
  TopoLeg_md->SetFillColor(kWhite);
  TopoLeg_md->SetLineColor(kWhite);

  TLegend* TopoLegCth_md = new TLegend(0.25,0.5,0.45,0.85);
  TopoLegCth_md->AddEntry(stvData,"Data","ep");
  TopoLegCth_md->AddEntry(stvCC0piNpIPS,"CC0#piNp p_{p}<500 MeV","lf");
  TopoLegCth_md->AddEntry(stvCC0piNpOOPS,"CC0#piNp p_{p}>500 MeV","lf");
  TopoLegCth_md->AddEntry(stvCC1pi,"CC1#pi^{+}","lf");
  TopoLegCth_md->AddEntry(stvCCOther,"CCOther","lf");
  TopoLegCth_md->AddEntry(stvOther,"Other","lf");
  TopoLegCth_md->SetFillColor(kWhite);
  TopoLegCth_md->SetLineColor(kWhite);

  // Canvases:

  TCanvas* canv_allSelBranchComp = new TCanvas("canv_allSelBranchComp","canv_allSelBranchComp",1920,1080);
  THStack *allSelBranchStack = new THStack("allSelBranchStack","allSelBranchStack");
  allSelBranchStack->Add(allSelBranchCompOthMC);
  allSelBranchStack->Add(allSelBranchCompCCOtherMC);
  allSelBranchStack->Add(allSelBranchCompCC1PiMC);
  allSelBranchStack->Add(allSelBranchCompCC0PiNPMC);
  allSelBranchStack->Add(allSelBranchCompCC0Pi1PMC);
  allSelBranchStack->Draw("HIST");
  allSelBranchStack->GetYaxis()->SetTitle("Events");
  allSelBranchStack->GetXaxis()->SetBinLabel(1, "#mu TPC (1-track)");
  allSelBranchStack->GetXaxis()->SetBinLabel(2, "#mu TPC + p TPC");
  allSelBranchStack->GetXaxis()->SetBinLabel(3, "#mu TPC + p FGD");
  allSelBranchStack->GetXaxis()->SetBinLabel(4, "#mu FGD + p TPC");
  allSelBranchStack->GetXaxis()->SetBinLabel(5, "#mu FGD (1-track)");
  allSelBranchStack->GetXaxis()->SetBinLabel(6, "#mu TPC + multi p");
  allSelBranchStack->Draw("HIST");
  allSelBranchCompData->Draw("sameE1");
  TopoLegBr->Draw();
  canv_allSelBranchComp->Write();
  canv_allSelBranchComp->SaveAs("allSelBranchComp.pdf");
  canv_allSelBranchComp->SaveAs("allSelBranchComp.root");

  //STV
  TCanvas* canv_stvTopo = new TCanvas("canv_stvTopo","canv_stvTopo",1920,1080);
  stvData->Draw("E1");
  THStack *stvTopoStack = new THStack("stvTopoStack","stvTopoStack");
  stvTopoStack->Add(stvOther);
  stvTopoStack->Add(stvCCOther);
  stvTopoStack->Add(stvCC1pi);
  stvTopoStack->Add(stvCC0piNpOOPS);
  stvTopoStack->Add(stvCC0piNpIPS);
  stvTopoStack->Draw("HISTsame");
  stvData->Draw("sameE1");
  if((varName=="dalphaT" || varName=="dalphat" || varName=="dat")) TopoLegCth->Draw();
  else TopoLeg->Draw();
  canv_stvTopo->Write();
  if(varName=="dpt") { canv_stvTopo->SaveAs("dptTopo.pdf"); canv_stvTopo->SaveAs("dptTopo.root"); }
  else if((varName=="dphiT" || varName=="dphit")) { canv_stvTopo->SaveAs("dphitTopo.pdf"); canv_stvTopo->SaveAs("dphitTopo.root"); }
  else if((varName=="dalphaT" || varName=="dalphat" || varName=="dat")) { canv_stvTopo->SaveAs("datTopo.pdf"); canv_stvTopo->SaveAs("datTopo.root"); }


  //infpMMomDif
  TCanvas* canv_infpMMomDifTopo = new TCanvas("canv_infpMMomDifTopo","canv_infpMMomDifTopo",1920,1080);
  infpMMomDifData->Draw("E1");
  THStack *infpMMomDifTopoStack = new THStack("infpMMomDifTopoStack","infpMMomDifTopoStack");
  infpMMomDifTopoStack->Add(infpMMomDifOther);
  infpMMomDifTopoStack->Add(infpMMomDifCCOther);
  infpMMomDifTopoStack->Add(infpMMomDifCC1pi);
  infpMMomDifTopoStack->Add(infpMMomDifCC0piNpOOPS);
  infpMMomDifTopoStack->Add(infpMMomDifCC0piNpIPS);
  infpMMomDifTopoStack->Draw("HISTsame");
  infpMMomDifData->Draw("sameE1");
  TopoLeg->Draw();
  canv_infpMMomDifTopo->Write();
  canv_infpMMomDifTopo->SaveAs("infpMMomDifTopo.pdf");
  canv_infpMMomDifTopo->SaveAs("infpMMomDifTopo.root");


  //infpAnglDif
  TCanvas* canv_infpAnglDifTopo = new TCanvas("canv_infpAnglDifTopo","canv_infpAnglDifTopo",1920,1080);
  THStack *infpAnglDifTopoStack = new THStack("infpAnglDifTopoStack","infpAnglDifTopoStack");
  infpAnglDifData->Draw("E1");
  infpAnglDifTopoStack->Add(infpAnglDifOther);
  infpAnglDifTopoStack->Add(infpAnglDifCCOther);
  infpAnglDifTopoStack->Add(infpAnglDifCC1pi);
  infpAnglDifTopoStack->Add(infpAnglDifCC0piNpOOPS);
  infpAnglDifTopoStack->Add(infpAnglDifCC0piNpIPS);
  infpAnglDifTopoStack->Draw("HISTsame");
  infpAnglDifData->Draw("sameE1");
  TopoLegCth->Draw();
  canv_infpAnglDifTopo->Write();
  canv_infpAnglDifTopo->SaveAs("infpAnglDifTopo.pdf");
  canv_infpAnglDifTopo->SaveAs("infpAnglDifTopo.root");


  //infp3MomDif
  TCanvas* canv_infp3MomDifTopo = new TCanvas("canv_infp3MomDifTopo","canv_infp3MomDifTopo",1920,1080);
  THStack *infp3MomDifTopoStack = new THStack("infp3MomDifTopoStack","infp3MomDifTopoStack");
  infp3MomDifData->Draw("E1");
  infp3MomDifTopoStack->Add(infp3MomDifOther);
  infp3MomDifTopoStack->Add(infp3MomDifCCOther);
  infp3MomDifTopoStack->Add(infp3MomDifCC1pi);
  infp3MomDifTopoStack->Add(infp3MomDifCC0piNpOOPS);
  infp3MomDifTopoStack->Add(infp3MomDifCC0piNpIPS);
  infp3MomDifTopoStack->Draw("HISTsame");
  infp3MomDifData->Draw("sameE1");
  TopoLeg->Draw();
  canv_infp3MomDifTopo->Write();
  canv_infp3MomDifTopo->SaveAs("infp3MomDifTopo.pdf");
  canv_infp3MomDifTopo->SaveAs("infp3MomDifTopo.root");


  //pmu

  TCanvas* canv_mommuTopo = new TCanvas("canv_mommuTopo","canv_mommuTopo",1920,1080);
  THStack *mommuTopoStack = new THStack("mommuTopoStack","mommuTopoStack");
  mommuData->Draw("E1");
  mommuTopoStack->Add(mommuOther);
  mommuTopoStack->Add(mommuCCOther);
  mommuTopoStack->Add(mommuCC1pi);
  mommuTopoStack->Add(mommuCC0piNpOOPS);
  mommuTopoStack->Add(mommuCC0piNpIPS);
  mommuTopoStack->Draw("HISTsame");
  mommuData->Draw("sameE1");
  TopoLeg_md->Draw();
  canv_mommuTopo->Write();
  canv_mommuTopo->SaveAs("mommuTopo.pdf");
  canv_mommuTopo->SaveAs("mommuTopo.root");


  //cthmu

  TCanvas* canv_cthmuTopo = new TCanvas("canv_cthmuTopo","canv_cthmuTopo",1920,1080);
  cthmuData->GetYaxis()->SetRangeUser(0,110000);
  THStack *cthmuTopoStack = new THStack("cthmuTopoStack","cthmuTopoStack");
  cthmuData->Draw("E1");
  cthmuTopoStack->Add(cthmuOther);
  cthmuTopoStack->Add(cthmuCCOther);
  cthmuTopoStack->Add(cthmuCC1pi);
  cthmuTopoStack->Add(cthmuCC0piNpOOPS);
  cthmuTopoStack->Add(cthmuCC0piNpIPS);
  cthmuTopoStack->Draw("HISTsame");
  cthmuData->Draw("sameE1");
  TopoLegCth_md->Draw();
  canv_cthmuTopo->Write();
  canv_cthmuTopo->SaveAs("cthmuTopo.pdf");
  canv_cthmuTopo->SaveAs("cthmuTopo.root");


  //pp

  TCanvas* canv_momprTopo = new TCanvas("canv_momprTopo","canv_momprTopo",1920,1080);
  THStack *momprTopoStack = new THStack("momprTopoStack","momprTopoStack");
  momprData->Draw("E1");
  momprTopoStack->Add(momprOther);
  momprTopoStack->Add(momprCCOther);
  momprTopoStack->Add(momprCC1pi);
  momprTopoStack->Add(momprCC0piNpOOPS);
  momprTopoStack->Add(momprCC0piNpIPS);
  momprTopoStack->Draw("HISTsame");
  momprData->Draw("sameE1");
  TopoLeg_md->Draw();
  canv_momprTopo->Write();
  canv_momprTopo->SaveAs("momprTopo.pdf");
  canv_momprTopo->SaveAs("momprTopo.root");


  //cthp

  TCanvas* canv_cthprTopo = new TCanvas("canv_cthprTopo","canv_cthprTopo",1920,1080);
  THStack *cthprTopoStack = new THStack("cthprTopoStack","cthprTopoStack");
  cthprData->Draw("E1");
  cthprTopoStack->Add(cthprOther);
  cthprTopoStack->Add(cthprCCOther);
  cthprTopoStack->Add(cthprCC1pi);
  cthprTopoStack->Add(cthprCC0piNpOOPS);
  cthprTopoStack->Add(cthprCC0piNpIPS);
  cthprTopoStack->Draw("HISTsame");
  cthprData->Draw("sameE1");
  TopoLegCth_md->Draw();
  canv_cthprTopo->Write();
  canv_cthprTopo->SaveAs("cthprTopo.pdf");
  canv_cthprTopo->SaveAs("cthprTopo.root");

  // 1pi CR

  //pmu

  TCanvas* canv_mommuTopo_1pi = new TCanvas("canv_mommuTopo_1pi","canv_mommuTopo_1pi",1920,1080);
  THStack *mommuTopoStack_1pi = new THStack("mommuTopoStack_1pi","mommuTopoStack_1pi");
  mommuData_1pi->GetYaxis()->SetRangeUser(0.0,1);
  mommuData_1pi->Draw("E1");
  mommuTopoStack_1pi->Add(mommuOther_1pi);
  mommuTopoStack_1pi->Add(mommuCCOther_1pi);
  mommuTopoStack_1pi->Add(mommuCC1pi_1pi);
  mommuTopoStack_1pi->Add(mommuCC0piNpOOPS_1pi);
  mommuTopoStack_1pi->Add(mommuCC0piNpIPS_1pi);
  mommuTopoStack_1pi->Draw("HISTsame");
  mommuData_1pi->Draw("sameE1");
  TopoLeg_md->Draw();
  canv_mommuTopo_1pi->Write();
  canv_mommuTopo_1pi->SaveAs("mommuTopo_1pi.pdf");
  canv_mommuTopo_1pi->SaveAs("mommuTopo_1pi.root");


  //cthmu

  TCanvas* canv_cthmuTopo_1pi = new TCanvas("canv_cthmuTopo_1pi","canv_cthmuTopo_1pi",1920,1080);
  cthmuData_1pi->GetYaxis()->SetRangeUser(0,12500);
  cthmuData_1pi->GetXaxis()->SetRangeUser(0.4,1);
  cthmuData_1pi->Draw("E1");
  THStack *cthmuTopoStack_1pi = new THStack("cthmuTopoStack_1pi","cthmuTopoStack_1pi");
  cthmuTopoStack_1pi->Add(cthmuOther_1pi);
  cthmuTopoStack_1pi->Add(cthmuCCOther_1pi);
  cthmuTopoStack_1pi->Add(cthmuCC1pi_1pi);
  cthmuTopoStack_1pi->Add(cthmuCC0piNpOOPS_1pi);
  cthmuTopoStack_1pi->Add(cthmuCC0piNpIPS_1pi);
  cthmuTopoStack_1pi->Draw("HISTsame");
  cthmuData_1pi->Draw("sameE1");
  TopoLegCth_md->Draw();
  canv_cthmuTopo_1pi->Write();
  canv_cthmuTopo_1pi->SaveAs("cthmuTopo_1pi.pdf");
  canv_cthmuTopo_1pi->SaveAs("cthmuTopo_1pi.root");


  //pp

  TCanvas* canv_momprTopo_1pi = new TCanvas("canv_momprTopo_1pi","canv_momprTopo_1pi",1920,1080);
  momprData_1pi->GetYaxis()->SetRangeUser(0,1.4);
  momprData_1pi->Draw("E1");
  THStack *momprTopoStack_1pi = new THStack("momprTopoStack_1pi","momprTopoStack_1pi");
  momprTopoStack_1pi->Add(momprOther_1pi);
  momprTopoStack_1pi->Add(momprCCOther_1pi);
  momprTopoStack_1pi->Add(momprCC1pi_1pi);
  momprTopoStack_1pi->Add(momprCC0piNpOOPS_1pi);
  momprTopoStack_1pi->Add(momprCC0piNpIPS_1pi);
  momprTopoStack_1pi->Draw("HISTsame");
  momprData_1pi->Draw("sameE1");
  TopoLeg_md->Draw();
  canv_momprTopo_1pi->Write();
  canv_momprTopo_1pi->SaveAs("momprTopo_1pi.pdf");
  canv_momprTopo_1pi->SaveAs("momprTopo_1pi.root");


  //cthp

  TCanvas* canv_cthprTopo_1pi = new TCanvas("canv_cthprTopo_1pi","canv_cthprTopo_1pi",1920,1080);
  cthprData_1pi->GetYaxis()->SetRangeUser(0,5500);
  cthprData_1pi->GetXaxis()->SetRangeUser(0.4,1);
  cthprData_1pi->Draw("E1");
  THStack *cthprTopoStack_1pi = new THStack("cthprTopoStack_1pi","cthprTopoStack_1pi");
  cthprTopoStack_1pi->Add(cthprOther_1pi);
  cthprTopoStack_1pi->Add(cthprCCOther_1pi);
  cthprTopoStack_1pi->Add(cthprCC1pi_1pi);
  cthprTopoStack_1pi->Add(cthprCC0piNpOOPS_1pi);
  cthprTopoStack_1pi->Add(cthprCC0piNpIPS_1pi);
  cthprTopoStack_1pi->Draw("HISTsame");
  cthprData_1pi->Draw("sameE1");
  TopoLegCth_md->Draw();
  canv_cthprTopo_1pi->Write();
  canv_cthprTopo_1pi->SaveAs("cthprTopo_1pi.pdf");
  canv_cthprTopo_1pi->SaveAs("cthprTopo_1pi.root");

  // DIS CR

  //pmu

  TCanvas* canv_mommuTopo_dis = new TCanvas("canv_mommuTopo_dis","canv_mommuTopo_dis",1920,1080);
  mommuData_dis->GetYaxis()->SetRangeUser(0,1.3);
  mommuData_dis->Draw("E1");
  THStack *mommuTopoStack_dis = new THStack("mommuTopoStack_dis","mommuTopoStack_dis");
  mommuTopoStack_dis->Add(mommuOther_dis);
  mommuTopoStack_dis->Add(mommuCCOther_dis);
  mommuTopoStack_dis->Add(mommuCC1pi_dis);
  mommuTopoStack_dis->Add(mommuCC0piNpOOPS_dis);
  mommuTopoStack_dis->Add(mommuCC0piNpIPS_dis);
  mommuTopoStack_dis->Draw("HISTsame");
  mommuData_dis->Draw("sameE1");
  TopoLeg_md->Draw();
  canv_mommuTopo_dis->Write();
  canv_mommuTopo_dis->SaveAs("mommuTopo_dis.pdf");
  canv_mommuTopo_dis->SaveAs("mommuTopo_dis.root");


  //cthmu

  TCanvas* canv_cthmuTopo_dis = new TCanvas("canv_cthmuTopo_dis","canv_cthmuTopo_dis",1920,1080);
  cthmuData_dis->GetYaxis()->SetRangeUser(0,21000);
  cthmuData_dis->GetXaxis()->SetRangeUser(0.4,1);
  cthmuData_dis->Draw("E1");
  THStack *cthmuTopoStack_dis = new THStack("cthmuTopoStack_dis","cthmuTopoStack_dis");
  cthmuTopoStack_dis->Add(cthmuOther_dis);
  cthmuTopoStack_dis->Add(cthmuCCOther_dis);
  cthmuTopoStack_dis->Add(cthmuCC1pi_dis);
  cthmuTopoStack_dis->Add(cthmuCC0piNpOOPS_dis);
  cthmuTopoStack_dis->Add(cthmuCC0piNpIPS_dis);
  cthmuTopoStack_dis->Draw("HISTsame");
  cthmuData_dis->Draw("sameE1");
  TopoLegCth_md->Draw();
  canv_cthmuTopo_dis->Write();
  canv_cthmuTopo_dis->SaveAs("cthmuTopo_dis.pdf");
  canv_cthmuTopo_dis->SaveAs("cthmuTopo_dis.root");


  //pp

  TCanvas* canv_momprTopo_dis = new TCanvas("canv_momprTopo_dis","canv_momprTopo_dis",1920,1080);
  momprData_dis->GetYaxis()->SetRangeUser(0.0,1.3);
  momprData_dis->Draw("E1");
  THStack *momprTopoStack_dis = new THStack("momprTopoStack_dis","momprTopoStack_dis");
  momprTopoStack_dis->Add(momprOther_dis);
  momprTopoStack_dis->Add(momprCCOther_dis);
  momprTopoStack_dis->Add(momprCC1pi_dis);
  momprTopoStack_dis->Add(momprCC0piNpOOPS_dis);
  momprTopoStack_dis->Add(momprCC0piNpIPS_dis);
  momprTopoStack_dis->Draw("HISTsame");
  momprData_dis->Draw("sameE1");
  TopoLegCth_md->Draw();
  canv_momprTopo_dis->Write();
  canv_momprTopo_dis->SaveAs("momprTopo_dis.pdf");
  canv_momprTopo_dis->SaveAs("momprTopo_dis.root");


  //cthp

  TCanvas* canv_cthprTopo_dis = new TCanvas("canv_cthprTopo_dis","canv_cthprTopo_dis",1920,1080);
  cthprData_dis->GetYaxis()->SetRangeUser(0.0,17000);
  cthprData_dis->GetXaxis()->SetRangeUser(0.4,1);
  cthprData_dis->Draw("E1");
  THStack *cthprTopoStack_dis = new THStack("cthprTopoStack_dis","cthprTopoStack_dis");
  cthprTopoStack_dis->Add(cthprOther_dis);
  cthprTopoStack_dis->Add(cthprCCOther_dis);
  cthprTopoStack_dis->Add(cthprCC1pi_dis);
  cthprTopoStack_dis->Add(cthprCC0piNpOOPS_dis);
  cthprTopoStack_dis->Add(cthprCC0piNpIPS_dis);
  cthprTopoStack_dis->Draw("HISTsame");
  cthprData_dis->Draw("sameE1");
  TopoLegCth_md->Draw();
  canv_cthprTopo_dis->Write();
  canv_cthprTopo_dis->SaveAs("cthprTopo_dis.pdf");
  canv_cthprTopo_dis->SaveAs("cthprTopo_dis.root");

  mommuTotal_1pi->Write();
  cthmuTotal_1pi->Write();
  momprTotal_1pi->Write();
  cthprTotal_1pi->Write();
  mommuTotal_dis->Write();
  cthmuTotal_dis->Write();
  momprTotal_dis->Write();
  cthprTotal_dis->Write();

  if(makeFinalWM && !useFineBins){
    cout << "WARNING: Assuming following file exists to make WM plots: " << endl;
    cout << "  /data/t2k/dolan/paperStuff/recoPlots/nov17/coarse_WM/dptPlotRecoOut_fine.root" << endl;
    TFile* wmPlotsFile = new TFile("/data/t2k/dolan/paperStuff/recoPlots/nov17/coarse_WM/dptPlotRecoOut_fine.root");
    TH1D* mommuWM_1pi = (TH1D*)wmPlotsFile->Get("mommuTotal_1pi");
    TH1D* cthmuWM_1pi = (TH1D*)wmPlotsFile->Get("cthmuTotal_1pi");
    TH1D* momprWM_1pi = (TH1D*)wmPlotsFile->Get("momprTotal_1pi");
    TH1D* cthprWM_1pi = (TH1D*)wmPlotsFile->Get("cthprTotal_1pi");
    TH1D* mommuWM_dis = (TH1D*)wmPlotsFile->Get("mommuTotal_dis");
    TH1D* cthmuWM_dis = (TH1D*)wmPlotsFile->Get("cthmuTotal_dis");
    TH1D* momprWM_dis = (TH1D*)wmPlotsFile->Get("momprTotal_dis");
    TH1D* cthprWM_dis = (TH1D*)wmPlotsFile->Get("cthprTotal_dis");

    outfile->cd();

    TLegend* TopoLeg_WM = new TLegend(0.65,0.5,0.85,0.85);
    TopoLeg_WM->AddEntry(stvData,"Data","ep");
    TopoLeg_WM->AddEntry(stvCC0piNpIPS,"CC0#piNp p_{p}<500 MeV","lf");
    TopoLeg_WM->AddEntry(stvCC0piNpOOPS,"CC0#piNp p_{p}>500 MeV","lf");
    TopoLeg_WM->AddEntry(stvCC1pi,"CC1#pi^{+}","lf");
    TopoLeg_WM->AddEntry(stvCCOther,"CCOther","lf");
    TopoLeg_WM->AddEntry(stvOther,"Other","lf");
    TopoLeg_WM->AddEntry(mommuWM_1pi,"Postfit","l");
    TopoLeg_WM->SetFillColor(kWhite);
    TopoLeg_WM->SetLineColor(kWhite);

    TLegend* TopoLegCth_WM = new TLegend(0.25,0.5,0.45,0.85);
    TopoLegCth_WM->AddEntry(stvData,"Data","ep");
    TopoLegCth_WM->AddEntry(stvCC0piNpIPS,"CC0#piNp p_{p}<500 MeV","lf");
    TopoLegCth_WM->AddEntry(stvCC0piNpOOPS,"CC0#piNp p_{p}>500 MeV","lf");
    TopoLegCth_WM->AddEntry(stvCC1pi,"CC1#pi^{+}","lf");
    TopoLegCth_WM->AddEntry(stvCCOther,"CCOther","lf");
    TopoLegCth_WM->AddEntry(stvOther,"Other","lf");
    TopoLegCth_WM->AddEntry(mommuWM_1pi,"Postfit","l");
    TopoLegCth_WM->SetFillColor(kWhite);
    TopoLegCth_WM->SetLineColor(kWhite);

    // 1pi CR

    //pmu

    TCanvas* canv_mommuTopo_1piWM = new TCanvas("canv_mommuTopo_1piWM","canv_mommuTopo_1piWM",1920,1080);
    mommuData_1pi->GetYaxis()->SetRangeUser(0.0,1);
    mommuData_1pi->Draw("E1");
    mommuTopoStack_1pi->Draw("HISTsame");
    mommuData_1pi->Draw("sameE1");
    mommuWM_1pi->Draw("sameHIST");
    TopoLeg_WM->Draw();
    canv_mommuTopo_1piWM->Write();
    canv_mommuTopo_1piWM->SaveAs("mommuTopo_1piWM.pdf");
    canv_mommuTopo_1piWM->SaveAs("mommuTopo_1piWM.root");


    //cthmu

    TCanvas* canv_cthmuTopo_1piWM = new TCanvas("canv_cthmuTopo_1piWM","canv_cthmuTopo_1piWM",1920,1080);
    cthmuData_1pi->GetYaxis()->SetRangeUser(0,12500);
    cthmuData_1pi->GetXaxis()->SetRangeUser(0.4,1);
    cthmuData_1pi->Draw("E1");
    cthmuTopoStack_1pi->Draw("HISTsame");
    cthmuData_1pi->Draw("sameE1");
    cthmuWM_1pi->Draw("sameHIST");
    TopoLegCth_WM->Draw();
    canv_cthmuTopo_1piWM->Write();
    canv_cthmuTopo_1piWM->SaveAs("cthmuTopo_1piWM.pdf");
    canv_cthmuTopo_1piWM->SaveAs("cthmuTopo_1piWM.root");


    //pp

    TCanvas* canv_momprTopo_1piWM = new TCanvas("canv_momprTopo_1piWM","canv_momprTopo_1piWM",1920,1080);
    momprData_1pi->GetYaxis()->SetRangeUser(0,1.4);
    momprData_1pi->Draw("E1");
    momprTopoStack_1pi->Draw("HISTsame");
    momprData_1pi->Draw("sameE1");
    momprWM_1pi->Draw("sameHIST");
    TopoLeg_WM->Draw();
    canv_momprTopo_1piWM->Write();
    canv_momprTopo_1piWM->SaveAs("momprTopo_1piWM.pdf");
    canv_momprTopo_1piWM->SaveAs("momprTopo_1piWM.root");


    //cthp

    TCanvas* canv_cthprTopo_1piWM = new TCanvas("canv_cthprTopo_1piWM","canv_cthprTopo_1piWM",1920,1080);
    cthprData_1pi->GetYaxis()->SetRangeUser(0,5500);
    cthprData_1pi->GetXaxis()->SetRangeUser(0.4,1);
    cthprData_1pi->Draw("E1");
    cthprTopoStack_1pi->Draw("HISTsame");
    cthprData_1pi->Draw("sameE1");
    cthprWM_1pi->Draw("sameHIST");
    TopoLegCth_WM->Draw();
    canv_cthprTopo_1piWM->Write();
    canv_cthprTopo_1piWM->SaveAs("cthprTopo_1piWM.pdf");
    canv_cthprTopo_1piWM->SaveAs("cthprTopo_1piWM.root");

    // DIS CR

    //pmu

    TCanvas* canv_mommuTopo_disWM = new TCanvas("canv_mommuTopo_disWM","canv_mommuTopo_disWM",1920,1080);
    mommuData_dis->GetYaxis()->SetRangeUser(0,1.3);
    mommuData_dis->Draw("E1");
    mommuTopoStack_dis->Draw("HISTsame");
    mommuData_dis->Draw("sameE1");
    mommuWM_dis->Draw("sameHIST");
    TopoLeg_WM->Draw();
    canv_mommuTopo_disWM->Write();
    canv_mommuTopo_disWM->SaveAs("mommuTopo_disWM.pdf");
    canv_mommuTopo_disWM->SaveAs("mommuTopo_disWM.root");


    //cthmu

    TCanvas* canv_cthmuTopo_disWM = new TCanvas("canv_cthmuTopo_disWM","canv_cthmuTopo_disWM",1920,1080);
    cthmuData_dis->GetYaxis()->SetRangeUser(0,21000);
    cthmuData_dis->GetXaxis()->SetRangeUser(0.4,1);
    cthmuData_dis->Draw("E1");
    cthmuTopoStack_dis->Draw("HISTsame");
    cthmuData_dis->Draw("sameE1");
    cthmuWM_dis->Draw("sameHIST");
    TopoLegCth_WM->Draw();
    canv_cthmuTopo_disWM->Write();
    canv_cthmuTopo_disWM->SaveAs("cthmuTopo_disWM.pdf");
    canv_cthmuTopo_disWM->SaveAs("cthmuTopo_disWM.root");


    //pp

    TCanvas* canv_momprTopo_disWM = new TCanvas("canv_momprTopo_disWM","canv_momprTopo_disWM",1920,1080);
    momprData_dis->GetYaxis()->SetRangeUser(0.0,1.3);
    momprData_dis->Draw("E1");
    momprTopoStack_dis->Draw("HISTsame");
    momprData_dis->Draw("sameE1");
    momprWM_dis->Draw("sameHIST");
    TopoLegCth_WM->Draw();
    canv_momprTopo_disWM->Write();
    canv_momprTopo_disWM->SaveAs("momprTopo_disWM.pdf");
    canv_momprTopo_disWM->SaveAs("momprTopo_disWM.root");


    //cthp

    TCanvas* canv_cthprTopo_disWM = new TCanvas("canv_cthprTopo_disWM","canv_cthprTopo_disWM",1920,1080);
    cthprData_dis->GetYaxis()->SetRangeUser(0.0,20000);
    cthprData_dis->GetXaxis()->SetRangeUser(0.4,1);
    cthprData_dis->Draw("E1");
    cthprTopoStack_dis->Draw("HISTsame");
    cthprData_dis->Draw("sameE1");
    cthprWM_dis->Draw("sameHIST");
    TopoLegCth_WM->Draw();
    canv_cthprTopo_disWM->Write();
    canv_cthprTopo_disWM->SaveAs("cthprTopo_disWM.pdf");
    canv_cthprTopo_disWM->SaveAs("cthprTopo_disWM.root");
  }

  return 1;
}
