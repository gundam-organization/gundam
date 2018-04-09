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

Float_t getGlobBinCentre(Int_t , Float_t, Float_t, TH1D*, Int_t);

//To run on all branches:
//for(int i=1;i<8;i++){ plotReco("$XSLLHFITTER/inputs/NeutAir5_2DV2.root", "$XSLLHFITTER/inputs/p6K_rdp_allRuns_FHC_v1.root", Form("branch%d_prefit.root",i) ,i, 0.2015, "dpt", true);}  
int plotReco(TString inFilenameMC, TString inFilenameData, TString outFileName, Int_t branchSelect=0, Float_t potRatio=1.0 /*0.2015*/, TString varName="dpt", bool scaleByBinWidth=true)
{
  TFile *infileMC = new TFile(inFilenameMC);
  TTree *intreeMC = (TTree*)infileMC->Get("selectedEvents");

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
  intreeData->SetBranchAddress("weight", &weight_data);


  Float_t* bins;
  Float_t* bins_coarse;
  Int_t nbinsn = 0;
  Int_t nbinsn_coarse = 0;
  Int_t nGlobBins = 60;
  Int_t nSample = 6;
  if(varName=="dpt"){
    cout << "Using dpT"<< endl;
    nbinsn = 8;
    Float_t binsn[9] = { 0.00001, 0.08, 0.12, 0.155, 0.2, 0.26, 0.36, 0.51, 1.1};
    bins = binsn;

    nbinsn_coarse = 4;
    Float_t binsn_coarse[5] = {0.00001, 0.12, 0.2, 0.36, 1.1};
    bins_coarse = binsn_coarse;

    nGlobBins = 60;
    nSample = 6;
  }
  else if(varName=="dphiT" || varName=="dphit"){
    cout << "Using dphiT"<< endl;
    nbinsn = 8;
    Float_t binsn[9] = { 0.00001, 0.067, 0.14, 0.225, 0.34, 0.52, 0.85, 1.5, 3.14159};
    bins = binsn;
  }
  else if(varName=="dalphaT" || varName=="dalphat" || varName=="dat"){
    cout << "Using dalphaT"<< endl;
    nbinsn = 8;
    Float_t binsn[9] = { 0.00001, 0.47, 1.02, 1.54, 1.98, 2.34, 2.64, 2.89, 3.14159};
    bins = binsn;
  }
  else{
    cout << "ERROR: Unrecognised variable: " << varName << endl;
    return 0;
  }

  Float_t binsPerSam = nGlobBins/nSample;
  
  TObjArray *listOhist = new TObjArray(500);

  TH1D* allSelEvtsMC_Rec     = new TH1D("allSelEvtsMC_Rec","allSelEvtsMC_Rec",nbinsn,bins); listOhist->Add(allSelEvtsMC_Rec);
  TH1D* allSelEvtsMC_True    = new TH1D("allSelEvtsMC_True","allSelEvtsMC_True",nbinsn,bins); listOhist->Add(allSelEvtsMC_True);
  TH1D* psLimEvtsMC_Rec      = new TH1D("psLimEvtsMC_Rec","psLimEvtsMC_Rec",nbinsn,bins); listOhist->Add(psLimEvtsMC_Rec);
  TH1D* psLimEvtsMC_True     = new TH1D("psLimEvtsMC_True","psLimEvtsMC_True",nbinsn,bins); listOhist->Add(psLimEvtsMC_True);
  TH1D* psLimTrueEvtsMC_True = new TH1D("psLimTrueEvtsMC_True","psLimTrueEvtsMC_True",nbinsn,bins); listOhist->Add(psLimTrueEvtsMC_True);

  TH1D* allSelEvtsMC_globBins = new TH1D("allSelEvtsMC_globBins","allSelEvtsMC_globBins",nGlobBins,0, nGlobBins); listOhist->Add(allSelEvtsMC_globBins);
  TH1D* allSelEvtsMC_globBins_Data = new TH1D("allSelEvtsMC_globBins_Data","allSelEvtsMC_globBins_Data",nGlobBins,0, nGlobBins); listOhist->Add(allSelEvtsMC_globBins_Data);
  TH1D* allSelEvtsMC_globBins_Res = new TH1D("allSelEvtsMC_globBins_Res","allSelEvtsMC_globBins_Res",nGlobBins,0, nGlobBins); listOhist->Add(allSelEvtsMC_globBins_Data);
  TH1D* allSelEvtsMC_globBins_Chi2 = new TH1D("allSelEvtsMC_globBins_Res","allSelEvtsMC_globBins_Res",nGlobBins,0, nGlobBins); listOhist->Add(allSelEvtsMC_globBins_Data);

  TH1D* allSelEvtsData = new TH1D("allSelEvtsData","allSelEvtsData",nbinsn,bins); listOhist->Add(allSelEvtsData);
  TH1D* psLimEvtsData = new TH1D("psLimEvtsData","psLimEvtsData",nbinsn,bins); listOhist->Add(psLimEvtsData);

  //**********************************************
  // REACTION
  //**********************************************

  //All sel by reaction

  TH1D* allSelCCQEMC       = new TH1D("allSelCCQEMC","allSelCCQEMC",nbinsn,bins); listOhist->Add(allSelCCQEMC);
  TH1D* allSel2p2hMC       = new TH1D("allSel2p2hMC","allSel2p2hMC",nbinsn,bins); listOhist->Add(allSel2p2hMC);
  TH1D* allSelRESMC        = new TH1D("allSelRESMC","allSelRESMC",nbinsn,bins); listOhist->Add(allSelRESMC);
  TH1D* allSelDISMC        = new TH1D("allSelDISMC","allSelDISMC",nbinsn,bins); listOhist->Add(allSelDISMC);
  TH1D* allSelOtherReacMC  = new TH1D("allSelOtherReacMC","allSelOtherReacMC",nbinsn,bins); listOhist->Add(allSelOtherReacMC);

  //PS restricted by reaction

  TH1D* psLimCCQEMC       = new TH1D("psLimCCQEMC","psLimCCQEMC",nbinsn,bins); listOhist->Add(psLimCCQEMC);
  TH1D* psLim2p2hMC       = new TH1D("psLim2p2hMC","psLim2p2hMC",nbinsn,bins); listOhist->Add(psLim2p2hMC);
  TH1D* psLimRESMC        = new TH1D("psLimRESMC","psLimRESMC",nbinsn,bins); listOhist->Add(psLimRESMC);
  TH1D* psLimDISMC        = new TH1D("psLimDISMC","psLimDISMC",nbinsn,bins); listOhist->Add(psLimDISMC);
  TH1D* psLimOtherReacMC  = new TH1D("psLimOtherReacMC","psLimOtherReacMC",nbinsn,bins); listOhist->Add(psLimOtherReacMC);

  //True PS restricted by reaction

  TH1D* psLimTrueCCQEMC_true       = new TH1D("psLimCCQEMC_true","psLimCCQEMC_true",nbinsn,bins); listOhist->Add(psLimTrueCCQEMC_true);
  TH1D* psLimTrue2p2hMC_true       = new TH1D("psLim2p2hMC_true","psLim2p2hMC_true",nbinsn,bins); listOhist->Add(psLimTrue2p2hMC_true);
  TH1D* psLimTrueRESMC_true        = new TH1D("psLimRESMC_true","psLimRESMC_true",nbinsn,bins); listOhist->Add(psLimTrueRESMC_true);
  TH1D* psLimTrueDISMC_true        = new TH1D("psLimDISMC_true","psLimDISMC_true",nbinsn,bins); listOhist->Add(psLimTrueDISMC_true);
  TH1D* psLimTrueOtherReacMC_true  = new TH1D("psLimOtherReacMC_true","psLimOtherReacMC_true",nbinsn,bins); listOhist->Add(psLimTrueOtherReacMC_true);

  //**********************************************
  // Topology
  //**********************************************

  //All sel by topology

  TH1D* allSelCC0Pi1PMC  = new TH1D("allSelCC0Pi1PMC","allSelCC0Pi1PMC",nbinsn,bins);  listOhist->Add(allSelCC0Pi1PMC);
  TH1D* allSelCC0PiNPMC  = new TH1D("allSelCC0PiNPMC","allSelCC0PiNPMC",nbinsn,bins);  listOhist->Add(allSelCC0PiNPMC);
  TH1D* allSelCC1PiMC    = new TH1D("allSelCC1PiMC","allSelCC1PiMC",nbinsn,bins);  listOhist->Add(allSelCC1PiMC);
  TH1D* allSelCCOtherMC  = new TH1D("allSelCCOtherMC","allSelCCOtherMC",nbinsn,bins);  listOhist->Add(allSelCCOtherMC);
  TH1D* allSelOthMC      = new TH1D("allSelOthMC","allSelOthMC",nbinsn,bins);  listOhist->Add(allSelOthMC);

  //PS restricted by topology

  TH1D* psLimCC0Pi1PMC  = new TH1D("psLimCC0Pi1PMC","psLimCC0Pi1PMC",nbinsn,bins);  listOhist->Add(psLimCC0Pi1PMC);
  TH1D* psLimCC0PiNPMC  = new TH1D("psLimCC0PiNPMC","psLimCC0PiNPMC",nbinsn,bins);  listOhist->Add(psLimCC0PiNPMC);
  TH1D* psLimCC1PiMC    = new TH1D("psLimCC1PiMC","psLimCC1PiMC",nbinsn,bins);  listOhist->Add(psLimCC1PiMC);
  TH1D* psLimCCOtherMC  = new TH1D("psLimCCOtherMC","psLimCCOtherMC",nbinsn,bins);  listOhist->Add(psLimCCOtherMC);
  TH1D* psLimOthMC      = new TH1D("psLimOthMC","psLimOthMC",nbinsn,bins);  listOhist->Add(psLimOthMC);

  //True PS restricted by topology

  TH1D* psLimTrueCC0Pi1PMC_true  = new TH1D("psLimTrueCC0Pi1PMC_true","psLimTrueCC0Pi1PMC_true",nbinsn,bins);  listOhist->Add(psLimTrueCC0Pi1PMC_true);
  TH1D* psLimTrueCC0PiNPMC_true  = new TH1D("psLimTrueCC0PiNPMC_true","psLimTrueCC0PiNPMC_true",nbinsn,bins);  listOhist->Add(psLimTrueCC0PiNPMC_true);
  TH1D* psLimTrueCC1PiMC_true    = new TH1D("psLimTrueCC1PiMC_true",  "psLimTrueCC1PiMC_true",nbinsn,bins);  listOhist->Add(psLimTrueCC1PiMC_true);
  TH1D* psLimTrueCCOtherMC_true  = new TH1D("psLimTrueCCOtherMC_true","psLimTrueCCOtherMC_true",nbinsn,bins);  listOhist->Add(psLimTrueCCOtherMC_true);
  TH1D* psLimTrueOthMC_true      = new TH1D("psLimTrueOthMC_true",    "psLimTrueOthMC_true",nbinsn,bins);  listOhist->Add(psLimTrueOthMC_true);

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

  // MC Loop:
  Int_t sam, stvBin, cutBranchConv;
  Float_t globBinCentre;
  bool isSelSigBranch=false;
  Long64_t nentries = intreeMC->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    nb = intreeMC->GetEntry(jentry); nbytes += nb;
    weight = weight*potRatio;
    if(branchSelect==0) isSelSigBranch = (cutBranch==1 || cutBranch==2 || cutBranch==3 || cutBranch==7);
    else isSelSigBranch = (cutBranch==branchSelect);
    //Global binning:
    globBinCentre=getGlobBinCentre(cutBranch, D1Rec, D2Rec, allSelEvtsMC_Rec, binsPerSam);
    if(globBinCentre>=0) allSelEvtsMC_globBins->Fill(globBinCentre, weight);  

    // Events comp by branch
    if (cutBranch==piSbNum) cutBranchConv=99;
    else if (cutBranch==disSbNum) cutBranchConv=99;
    else if (cutBranch>disSbNum) cutBranchConv=cutBranch-2;
    else cutBranchConv=cutBranch;

    //Topology:
    if     (mectopology==1) allSelBranchCompCC0Pi1PMC->Fill(cutBranchConv+0.5, weight); 
    else if(mectopology==2) allSelBranchCompCC0PiNPMC->Fill(cutBranchConv+0.5, weight); 
    else if(mectopology==3) allSelBranchCompCC1PiMC->Fill(cutBranchConv+0.5, weight); 
    else if(mectopology==4) allSelBranchCompCCOtherMC->Fill(cutBranchConv+0.5, weight); 
    else                    allSelBranchCompOthMC->Fill(cutBranchConv+0.5, weight); 

    // Events selectd in branch
    if(isSelSigBranch){
      allSelEvtsMC_Rec->Fill(D1Rec, weight);
      allSelEvtsMC_True->Fill(D1True, weight);
      //Reaction:
      if     (reaction==0) allSelCCQEMC->Fill(D1Rec, weight); 
      else if(reaction==9) allSel2p2hMC->Fill(D1Rec, weight); 
      else if(reaction==1) allSelRESMC->Fill(D1Rec, weight); 
      else if(reaction==2) allSelDISMC->Fill(D1Rec, weight); 
      else                 allSelOtherReacMC->Fill(D1Rec, weight); 
      //Topology:
      if     (mectopology==1) allSelCC0Pi1PMC->Fill(D1Rec, weight); 
      else if(mectopology==2) allSelCC0PiNPMC->Fill(D1Rec, weight); 
      else if(mectopology==3) allSelCC1PiMC->Fill(D1Rec, weight); 
      else if(mectopology==4) allSelCCOtherMC->Fill(D1Rec, weight); 
      else                    allSelOthMC->Fill(D1Rec, weight); 
    }
    // Events selectd in branch that pass the reco phase space constraints
    if(isSelSigBranch && D2Rec==0){
      psLimEvtsMC_Rec->Fill(D1Rec, weight);
      psLimEvtsMC_True->Fill(D1True, weight);
      //Reaction:
      if     (reaction==0) psLimCCQEMC->Fill(D1Rec, weight); 
      else if(reaction==9) psLim2p2hMC->Fill(D1Rec, weight); 
      else if(reaction==1) psLimRESMC->Fill(D1Rec, weight); 
      else if(reaction==2) psLimDISMC->Fill(D1Rec, weight); 
      else                 psLimOtherReacMC->Fill(D1Rec, weight); 
      //Topology:
      if     (mectopology==1) psLimCC0Pi1PMC->Fill(D1Rec, weight); 
      else if(mectopology==2) psLimCC0PiNPMC->Fill(D1Rec, weight); 
      else if(mectopology==3) psLimCC1PiMC->Fill(D1Rec, weight); 
      else if(mectopology==4) psLimCCOtherMC->Fill(D1Rec, weight); 
      else                    psLimOthMC->Fill(D1Rec, weight); 
    }
    // Events selectd in branch that pass the true phase space constraints
    if(isSelSigBranch && D2True==0){
      psLimTrueEvtsMC_True->Fill(D1True, weight);
      //Reaction:
      if     (reaction==0) psLimTrueCCQEMC_true->Fill(D1True, weight); 
      else if(reaction==9) psLimTrue2p2hMC_true->Fill(D1True, weight); 
      else if(reaction==1) psLimTrueRESMC_true->Fill(D1True, weight); 
      else if(reaction==2) psLimTrueDISMC_true->Fill(D1True, weight); 
      else                 psLimTrueOtherReacMC_true->Fill(D1True, weight); 
      //Topology:
      if     (mectopology==1) psLimTrueCC0Pi1PMC_true->Fill(D1True, weight); 
      else if(mectopology==2) psLimTrueCC0PiNPMC_true->Fill(D1True, weight); 
      else if(mectopology==3) psLimTrueCC1PiMC_true->Fill(D1True, weight); 
      else if(mectopology==4) psLimTrueCCOtherMC_true->Fill(D1True, weight); 
      else                    psLimTrueOthMC_true->Fill(D1True, weight); 
    }
  }


  // Data Loop:
  bool isSelSigBranch_data;
  nentries = intreeData->GetEntriesFast();
  nbytes = 0;
  nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    nb = intreeData->GetEntry(jentry); nbytes += nb;
    if(branchSelect==0) isSelSigBranch_data = (cutBranch_data==1 || cutBranch_data==2 || cutBranch_data==3 || cutBranch_data==7);
    else isSelSigBranch_data = (cutBranch_data==branchSelect);
    //Global binning:
    globBinCentre=getGlobBinCentre(cutBranch_data, D1Rec_data, D2Rec_data, allSelEvtsData, binsPerSam);
    if(globBinCentre>=0) allSelEvtsMC_globBins_Data->Fill(globBinCentre, weight_data);  
    // Events comp by branch
    if (cutBranch_data==piSbNum) cutBranchConv=99;
    else if (cutBranch_data==disSbNum) cutBranchConv=99;
    else if (cutBranch_data>disSbNum) cutBranchConv=cutBranch_data-2;
    else cutBranchConv=cutBranch_data;
    allSelBranchCompData->Fill(cutBranchConv+0.5, weight_data);
    // Events selectd in branch
    if(isSelSigBranch_data){
      allSelEvtsData->Fill(D1Rec_data, weight_data);
    }
    // Events selectd in branch that pass the reco phase space constraints
    if(isSelSigBranch_data && D2Rec_data==0){
      psLimEvtsData->Fill(D1Rec_data, weight_data);
    }
  }

  //Data-MC comp:
  double lnl=0;
  for(int b=0; b<(allSelEvtsMC_globBins->GetNbinsX()); b++){
    double mcBin = allSelEvtsMC_globBins->GetBinContent(b);
    double dataBin = allSelEvtsMC_globBins_Data->GetBinContent(b);
    if(dataBin!=0){
      lnl = 2*(mcBin - dataBin);
      lnl += 2*dataBin*TMath::Log(dataBin/mcBin);
      allSelEvtsMC_globBins_Res->SetBinContent(b, (dataBin-mcBin)/sqrt(dataBin));
      allSelEvtsMC_globBins_Chi2->SetBinContent(b, lnl);
    }
  }


  if(scaleByBinWidth){
    for(int i=0; i<listOhist->GetEntries(); i++){
      ((TH1D*)(listOhist->At(i)))->Sumw2();
      ((TH1D*)(listOhist->At(i)))->Scale(1, "width");
    }
  }

  //Histos

  allSelEvtsMC_globBins->SetLineColor(kBlue);
  allSelEvtsMC_globBins->SetMarkerStyle(0);
  allSelEvtsMC_globBins_Data->SetMarkerStyle(20);
  allSelEvtsMC_globBins_Data->SetMarkerSize(0.5);
  allSelEvtsMC_globBins->SetYTitle("Events");
  allSelEvtsMC_globBins_Data->SetYTitle("Events");
  if(varName=="dpt"){
    allSelEvtsMC_globBins->SetXTitle("#deltap_{T} Analysis Bin");
    allSelEvtsMC_globBins_Data->SetXTitle("#deltap_{T} Analysis Bin");
  }
  else if(varName=="dphiT" || varName=="dphit"){
    allSelEvtsMC_globBins->SetXTitle("#delta#phi_{T} Analysis Bin");
    allSelEvtsMC_globBins_Data->SetXTitle("#delta#phi_{T} Analysis Bin");
  }
  else if(varName=="dalphaT" || varName=="dalphat" || varName=="dat"){
    allSelEvtsMC_globBins->SetXTitle("#delta#alpha_{T} Analysis Bin");
    allSelEvtsMC_globBins_Data->SetXTitle("#delta#alpha_{T}} Analysis Bin");
  }

  allSelEvtsMC_globBins->Write("allSelEvtsMC_globBins");
  allSelEvtsMC_globBins->Print("all");

  allSelEvtsMC_globBins_Data->Write("allSelEvtsMC_globBins_Data");

  allSelEvtsMC_globBins_Res->Write("allSelEvtsMC_globBins_Res");
  allSelEvtsMC_globBins_Chi2->Write("allSelEvtsMC_globBins_Chi2");

  allSelEvtsMC_Rec->Write();
  allSelEvtsMC_True->Write();
  psLimEvtsMC_Rec->Write();
  psLimEvtsMC_True->Write();
  psLimTrueEvtsMC_True->Write();

  allSelEvtsData->SetMarkerStyle(20);
  allSelEvtsData->SetMarkerSize(0.5);
  allSelEvtsData->Write();

  psLimEvtsData->SetMarkerStyle(20);
  psLimEvtsData->SetMarkerSize(0.5);
  psLimEvtsData->Write();

  allSelBranchCompData->SetMarkerStyle(20);
  allSelBranchCompData->SetMarkerSize(0.5);
  allSelBranchCompData->Write();

  //Reaction

  allSelCCQEMC->SetLineColor(kRed);
  allSel2p2hMC->SetLineColor(kViolet+10);
  allSelRESMC->SetLineColor(kGreen);
  allSelDISMC->SetLineColor(kBlue);
  allSelOtherReacMC->SetLineColor(kCyan);

  allSelCCQEMC->SetFillColor(kRed-7);
  allSel2p2hMC->SetFillColor(kViolet-7);
  allSelRESMC->SetFillColor(kGreen-7);
  allSelDISMC->SetFillColor(kBlue-7);
  allSelOtherReacMC->SetFillColor(kCyan-7);

  psLimCCQEMC->SetLineColor(kRed);
  psLim2p2hMC->SetLineColor(kViolet+10);
  psLimRESMC->SetLineColor(kGreen);
  psLimDISMC->SetLineColor(kBlue);
  psLimOtherReacMC->SetLineColor(kCyan);
  
  psLimCCQEMC->SetFillColor(kRed-7);
  psLim2p2hMC->SetFillColor(kViolet-7);
  psLimRESMC->SetFillColor(kGreen-7);
  psLimDISMC->SetFillColor(kBlue-7);
  psLimOtherReacMC->SetFillColor(kCyan-7);

  psLimTrueCCQEMC_true->SetLineColor(kRed);
  psLimTrue2p2hMC_true->SetLineColor(kViolet+10);
  psLimTrueRESMC_true->SetLineColor(kGreen);
  psLimTrueDISMC_true->SetLineColor(kBlue);
  psLimTrueOtherReacMC_true->SetLineColor(kCyan);
  
  psLimTrueCCQEMC_true->SetFillColor(kRed-7);
  psLimTrue2p2hMC_true->SetFillColor(kViolet-7);
  psLimTrueRESMC_true->SetFillColor(kGreen-7);
  psLimTrueDISMC_true->SetFillColor(kBlue-7);
  psLimTrueOtherReacMC_true->SetFillColor(kCyan-7);

  //Topology

  allSelBranchCompCC0Pi1PMC->SetLineColor(kRed);
  allSelBranchCompCC0PiNPMC->SetLineColor(kViolet+10);
  allSelBranchCompCC1PiMC->SetLineColor(kGreen);
  allSelBranchCompCCOtherMC->SetLineColor(kBlue);
  allSelBranchCompOthMC->SetLineColor(kCyan);

  allSelBranchCompCC0Pi1PMC->SetFillColor(kRed-7);
  allSelBranchCompCC0PiNPMC->SetFillColor(kViolet-7);
  allSelBranchCompCC1PiMC->SetFillColor(kGreen-7);
  allSelBranchCompCCOtherMC->SetFillColor(kBlue-7);
  allSelBranchCompOthMC->SetFillColor(kCyan-7);

  allSelCC0Pi1PMC->SetLineColor(kRed);
  allSelCC0PiNPMC->SetLineColor(kViolet+10);
  allSelCC1PiMC->SetLineColor(kGreen);
  allSelCCOtherMC->SetLineColor(kBlue);
  allSelOthMC->SetLineColor(kCyan);

  allSelCC0Pi1PMC->SetFillColor(kRed-7);
  allSelCC0PiNPMC->SetFillColor(kViolet-7);
  allSelCC1PiMC->SetFillColor(kGreen-7);
  allSelCCOtherMC->SetFillColor(kBlue-7);
  allSelOthMC->SetFillColor(kCyan-7);

  psLimCC0Pi1PMC->SetLineColor(kRed);
  psLimCC0PiNPMC->SetLineColor(kViolet+10);
  psLimCC1PiMC->SetLineColor(kGreen);
  psLimCCOtherMC->SetLineColor(kBlue);
  psLimOthMC->SetLineColor(kCyan);
  
  psLimCC0Pi1PMC->SetFillColor(kRed-7);
  psLimCC0PiNPMC->SetFillColor(kViolet-7);
  psLimCC1PiMC->SetFillColor(kGreen-7);
  psLimCCOtherMC->SetFillColor(kBlue-7);
  psLimOthMC->SetFillColor(kCyan-7);

  psLimTrueCC0Pi1PMC_true->SetLineColor(kRed);
  psLimTrueCC0PiNPMC_true->SetLineColor(kViolet+10);
  psLimTrueCC1PiMC_true->SetLineColor(kGreen);
  psLimTrueCCOtherMC_true->SetLineColor(kBlue);
  psLimTrueOthMC_true->SetLineColor(kCyan);
  
  psLimTrueCC0Pi1PMC_true->SetFillColor(kRed-7);
  psLimTrueCC0PiNPMC_true->SetFillColor(kViolet-7);
  psLimTrueCC1PiMC_true->SetFillColor(kGreen-7);
  psLimTrueCCOtherMC_true->SetFillColor(kBlue-7);
  psLimTrueOthMC_true->SetFillColor(kCyan-7);

  //Legends

  TLegend* ReacLeg = new TLegend(0.65,0.5,0.85,0.85);
  ReacLeg->AddEntry(allSelEvtsData,"Data","lp");
  ReacLeg->AddEntry(allSelCCQEMC,"CCQE","lf");
  ReacLeg->AddEntry(allSel2p2hMC,"2p2h","lf");
  ReacLeg->AddEntry(allSelRESMC,"RES","lf");
  ReacLeg->AddEntry(allSelDISMC,"DIS","lf");
  ReacLeg->AddEntry(allSelOtherReacMC,"Other","lf");
  ReacLeg->SetFillColor(kWhite);
  ReacLeg->SetLineColor(kWhite);
  ReacLeg->Draw();

  TLegend* TopoLeg = new TLegend(0.65,0.5,0.85,0.85);
  TopoLeg->AddEntry(allSelEvtsData,"Data","lp");
  TopoLeg->AddEntry(allSelCC0Pi1PMC,"CC0#pi1p","lf");
  TopoLeg->AddEntry(allSelCC0PiNPMC,"CC0#piNp","lf");
  TopoLeg->AddEntry(allSelCC1PiMC,"CC1#pi^{+}","lf");
  TopoLeg->AddEntry(allSelCCOtherMC,"CCOther","lf");
  TopoLeg->AddEntry(allSelOthMC,"Other","lf");
  TopoLeg->SetFillColor(kWhite);
  TopoLeg->SetLineColor(kWhite);

  //Canvas

  TCanvas* canv_allSelEvtsComp = new TCanvas("canv_allSelEvtsComp","canv_allSelEvtsComp");
  allSelEvtsMC_Rec->Draw("HIST");
  allSelEvtsData->Draw("sameE1");
  canv_allSelEvtsComp->Write();

  //Gloab binning Canvas

  TCanvas* canv_globalBins = new TCanvas("canv_globalBins","canv_globalBins");
  allSelEvtsMC_globBins->SetLineColor(kRed);
  (allSelEvtsMC_globBins->GetYaxis())->SetRangeUser(0,900);
  allSelEvtsMC_globBins->Draw("HIST");
  allSelEvtsMC_globBins_Data->Draw("sameE1");
  canv_globalBins->Write();

  // Reaction Canvas

  TCanvas* canv_allSelReacComp = new TCanvas("canv_allSelReacComp","canv_allSelReacComp");
  THStack *allSelReacStack = new THStack("allSelReacStack","allSelReacStack");
  allSelReacStack->Add(allSelOtherReacMC);
  allSelReacStack->Add(allSelDISMC);
  allSelReacStack->Add(allSelRESMC);
  allSelReacStack->Add(allSel2p2hMC);
  allSelReacStack->Add(allSelCCQEMC);
  allSelReacStack->Draw("HIST");
  allSelEvtsData->Draw("sameE1");
  ReacLeg->Draw();
  canv_allSelReacComp->Write();

  TCanvas* canv_psLimReacComp = new TCanvas("canv_psLimReacComp","canv_psLimReacComp");
  THStack *psLimReacStack = new THStack("psLimReacStack","psLimReacStack");
  psLimReacStack->Add(psLimOtherReacMC);
  psLimReacStack->Add(psLimDISMC);
  psLimReacStack->Add(psLimRESMC);
  psLimReacStack->Add(psLim2p2hMC);
  psLimReacStack->Add(psLimCCQEMC);
  psLimReacStack->Draw("HIST");
  psLimEvtsData->Draw("sameE1");
  ReacLeg->Draw();
  canv_psLimReacComp->Write();

  TCanvas* canv_psLimTrueReacComp_true = new TCanvas("canv_psLimTrueReacComp_true","canv_psLimTrueReacComp_true");
  THStack *psLimTrueReacStack_true = new THStack("psLimTrueReacStack_true","psLimTrueReacStack_true");
  psLimTrueReacStack_true->Add(psLimTrueOtherReacMC_true);
  psLimTrueReacStack_true->Add(psLimTrueDISMC_true);
  psLimTrueReacStack_true->Add(psLimTrueRESMC_true);
  psLimTrueReacStack_true->Add(psLimTrue2p2hMC_true);
  psLimTrueReacStack_true->Add(psLimTrueCCQEMC_true);
  psLimTrueReacStack_true->Draw("HIST");
  ReacLeg->Draw();
  canv_psLimTrueReacComp_true->Write();

  // Topology Canvas:

  TCanvas* canv_allSelBranchComp = new TCanvas("canv_allSelBranchComp","canv_allSelBranchComp");
  THStack *allSelBranchStack = new THStack("allSelBranchStack","allSelBranchStack");
  allSelBranchStack->Add(allSelBranchCompOthMC);
  allSelBranchStack->Add(allSelBranchCompCCOtherMC);
  allSelBranchStack->Add(allSelBranchCompCC1PiMC);
  allSelBranchStack->Add(allSelBranchCompCC0PiNPMC);
  allSelBranchStack->Add(allSelBranchCompCC0Pi1PMC);
  allSelBranchStack->Draw("HIST");
  allSelBranchStack->GetXaxis()->SetBinLabel(1, "#mu TPC (1-track)");
  allSelBranchStack->GetXaxis()->SetBinLabel(2, "#mu TPC + p TPC");
  allSelBranchStack->GetXaxis()->SetBinLabel(3, "#mu TPC + p FGD");
  allSelBranchStack->GetXaxis()->SetBinLabel(4, "#mu FGD + p TPC");
  allSelBranchStack->GetXaxis()->SetBinLabel(5, "#mu FGD (1-track)");
  allSelBranchStack->GetXaxis()->SetBinLabel(6, "#mu TPC + multi p");
  allSelBranchStack->Draw("HIST");
  allSelBranchCompData->Draw("sameE1");
  TopoLeg->Draw();
  canv_allSelBranchComp->Write();

  TCanvas* canv_allSelTopoComp = new TCanvas("canv_allSelTopoComp","canv_allSelTopoComp");
  THStack *allSelTopoStack = new THStack("allSelTopoStack","allSelTopoStack");
  allSelTopoStack->Add(allSelOthMC);
  allSelTopoStack->Add(allSelCCOtherMC);
  allSelTopoStack->Add(allSelCC1PiMC);
  allSelTopoStack->Add(allSelCC0PiNPMC);
  allSelTopoStack->Add(allSelCC0Pi1PMC);
  allSelTopoStack->Draw("HIST");
  allSelEvtsData->Draw("sameE1");
  TopoLeg->Draw();
  canv_allSelTopoComp->Write();

  TCanvas* canv_psLimTopoComp = new TCanvas("canv_psLimTopoComp","canv_psLimTopoComp");
  THStack *psLimTopoStack = new THStack("psLimTopoStack","psLimTopoStack");
  psLimTopoStack->Add(psLimOthMC);
  psLimTopoStack->Add(psLimCCOtherMC);
  psLimTopoStack->Add(psLimCC1PiMC);
  psLimTopoStack->Add(psLimCC0PiNPMC);
  psLimTopoStack->Add(psLimCC0Pi1PMC);
  psLimTopoStack->Draw("HIST");
  psLimEvtsData->Draw("sameE1");
  TopoLeg->Draw();
  canv_psLimTopoComp->Write();

  TCanvas* canv_psLimTrueTopoComp_true = new TCanvas("canv_psLimTrueTopoComp_true","canv_psLimTrueTopoComp_true");
  THStack *psLimTrueTopoStack_true = new THStack("psLimTrueTopoStack_true","psLimTrueTopoStack_true");
  psLimTrueTopoStack_true->Add(psLimTrueOthMC_true);
  psLimTrueTopoStack_true->Add(psLimTrueCCOtherMC_true);
  psLimTrueTopoStack_true->Add(psLimTrueCC1PiMC_true);
  psLimTrueTopoStack_true->Add(psLimTrueCC0PiNPMC_true);
  psLimTrueTopoStack_true->Add(psLimTrueCC0Pi1PMC_true);
  psLimTrueTopoStack_true->Draw("HIST");
  TopoLeg->Draw();
  canv_psLimTrueTopoComp_true->Write();


  return 1;
}


Float_t getGlobBinCentre(Int_t cutBranch, Float_t D1Rec, Float_t D2Rec, TH1D* refHisto, Int_t binsPerSam){
  Int_t sam = -999;
  Float_t globBinCentre;
  Int_t stvBin = refHisto->FindBin(D1Rec);

  if(D1Rec<0.0000001) return -1;

  if      (cutBranch==1) sam=0;
  else if (cutBranch==2) sam=1;
  else if (cutBranch==3) sam=2;
  else if (cutBranch==5) sam=3;
  else if (cutBranch==6) sam=4;
  else if (cutBranch==7) sam=5;
  else return -1;

  if(D2Rec==0) globBinCentre = (stvBin-1)+sam*(binsPerSam)+0.5; //-1 to discount the underflow, +0.5 to get bin centre   
  else globBinCentre = 9+sam*(binsPerSam)+0.5;

  // cout << "cutBranch " << cutBranch << endl;
  // cout << "D1Rec " << D1Rec << endl;
  // cout << "stvBin " << stvBin << endl;
  // cout << "D2Rec " << D2Rec << endl;
  // cout << "globBinCentre " << globBinCentre << endl << endl;

  return globBinCentre;
}
