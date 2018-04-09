/******************************************************

Code to convert a HL2 tree into the format required 
for the fitting code. C

Can't simply read HL2 tree directly since we don't
know what variables will be the tree

Author: Stephen Dolan
Date Created: November 2015

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
#include <TGraph.h>
#include <TTree.h>
#include <TString.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TMath.h>


using namespace std;

bool extraCuts=false;
bool psdim=true;

// specialRWmode:
//                1 - No 2p2h
//                2 - 2X 2p2h
//                3 - 3X 2p2h
//                4 - Martini Nieves Ratio RW (in E_nu)

// Example: 
// treeConvert("/data/t2k/dolan/MECProcessing/CC0Pi/mar15HL2/job_NeutWaterNoSystV2_out/allMerged.root", "default", "truth", "./NeutWater2_2D_2X2p2h_dataStats.root", 1, 0, 0, 0.1664, 2, "recDpT", "trueDpT", "trueDpT", "recDalphaT", "trueDalphaT", "trueDalphaT")

int treeConvert(TString inFileName, TString inTreeName,  TString inTreeName_T, TString outFileName,
                Float_t sigWeight, Int_t EvtStart, Int_t EvtEnd, Double_t EvtFrac, Int_t specialRWmode, 
                TString D1NameRec, TString D1NameTrue, TString D1NameTrue_T,
                TString D2NameRec, TString D2NameTrue, TString D2NameTrue_T, bool isRealData=false, bool highAngle=false)
{
  // You need to provide the number of branches in your HL2 tree
  // And the accum_level you want to cut each one at to get your selected events
  // i.e choosing n in accum_level[0][branch_i]>n
  const int nbranches = 10;
  //const int accumToCut[nbranches] =   {7,8,9,8,7,5,4,7,8,7};
  const int accumToCut[nbranches] =   {7,8,9,8,7,5,4,7,8,7};

  TFile *infile = new TFile(inFileName);
  TTree *intree = (TTree*)infile->Get(inTreeName);
  TTree *intree_T = (TTree*)infile->Get(inTreeName_T);

  TFile *outfile = new TFile(outFileName,"recreate");
  TTree *outtree = new TTree("selectedEvents", "selectedEvents");
  TTree *outtree_T = new TTree("trueEvents", "trueEvents");

  char const * xslf_env = getenv("XSLLHFITTER");
  if(!xslf_env){
    std::cerr << "[ERROR]: environment variable \"XSLLHFITTER\" not set. "
    "Cannot determine source tree location." << std::endl;
    return 1;
  }
  string inputDir = std::string(xslf_env) + "/inputs";
  string martiniNievesRatioFileName = inputDir + "/martini_nieves_ratio.root";
  TFile *martiniNievesRatioFile = new TFile(martiniNievesRatioFileName.c_str()); 
  TGraph *martiniNievesRatioGr;
  if(specialRWmode==4){
    if(!martiniNievesRatioFile){
      cout << "WARNING: cannot find martini nieves ratio file" << endl;
      return 0;
    }
    martiniNievesRatioGr = (TGraph*)martiniNievesRatioFile->Get("mec_ratio");
    martiniNievesRatioGr->Print("all");
    if(!martiniNievesRatioGr){
      cout << "WARNING: cannot find martini nieves ratio graph" << endl;
      return 0;
    }
  }


  // Declaration of leaf types
  Int_t          accum_level[1500][50];
  Int_t          reaction;
  Int_t          cutBranch=-999;
  Int_t          mectopology;
  Float_t        D1true;
  Float_t        D2true;
  Float_t        D1Reco;
  Float_t        D2Reco;
  Float_t        pMomRec;
  Float_t        pMomRecRange;
  Float_t        pThetaRec;
  Float_t        pMomTrue;
  Float_t        pThetaTrue;
  Float_t        muMomRec;
  Float_t        muMomRecRange;
  Float_t        muThetaRec;
  Float_t        muMomTrue;
  Float_t        muThetaTrue;
  Float_t        muCosThetaRec;
  Float_t        muCosThetaTrue;
  Float_t        pCosThetaRec;
  Float_t        pCosThetaTrue;
  Float_t        RecoNuEnergy=0;
  Float_t        TrueNuEnergy=0;
  Float_t        weight;

  Int_t          reaction_T;
  Int_t          mectopology_T;
  Float_t        D1true_T;
  Float_t        D2true_T; 
  Float_t        muMomTrue_T;
  Float_t        pMomTrue_T;
  Float_t        muCosThetaTrue_T;
  Float_t        pCosThetaTrue_T;
  Float_t        TrueNuEnergy_T;
  Float_t        weight_T=1.0;


  intree->SetBranchAddress("accum_level", &accum_level);
  intree->SetBranchAddress("reaction", &reaction);
  intree->SetBranchAddress("mectopology", &mectopology);
  intree->SetBranchAddress(D1NameTrue, &D1true);
  intree->SetBranchAddress(D2NameTrue, &D2true);
  intree->SetBranchAddress(D1NameRec, &D1Reco);
  intree->SetBranchAddress(D2NameRec, &D2Reco);
  intree->SetBranchAddress("selp_mom", &pMomRec);
  intree->SetBranchAddress("selp_mom_range_oarecon", &pMomRecRange);
  intree->SetBranchAddress("selp_theta" ,&pThetaRec);
  intree->SetBranchAddress("truep_truemom" ,&pMomTrue);
  intree->SetBranchAddress("truep_truecostheta" ,&pCosThetaTrue);
  intree->SetBranchAddress("selmu_mom", &muMomRec);
  intree->SetBranchAddress("selmu_mom_range_oarecon", &muMomRecRange);
  intree->SetBranchAddress("selmu_theta", &muThetaRec);
  intree->SetBranchAddress("truemu_mom", &muMomTrue);
  intree->SetBranchAddress("truemu_costheta", &muCosThetaTrue);
  //intree->SetBranchAddress("nu_trueE", &RecoNuEnergy);
  intree->SetBranchAddress("nu_trueE", &TrueNuEnergy);
  intree->SetBranchAddress("weight", &weight);

  outtree->Branch("reaction", &reaction, "reaction/I");
  outtree->Branch("cutBranch", &cutBranch, "cutBranch/I");
  outtree->Branch("mectopology", &mectopology, "mectopology/I");
  outtree->Branch("D1True", &D1true, ("D1True/F"));
  outtree->Branch("D1Rec", &D1Reco, ("D1Rec/F"));
  outtree->Branch("D2True", &D2true, ("D2True/F"));
  outtree->Branch("D2Rec", &D2Reco, ("D2Rec/F"));
  outtree->Branch("muMomRec", &muMomRec, ("muMomRec/F"));
  outtree->Branch("muMomTrue", &muMomTrue, ("muMomTrue/F"));
  outtree->Branch("muCosThetaRec", &muCosThetaRec, ("muCosThetaRec/F"));
  outtree->Branch("muCosThetaTrue", &muCosThetaTrue, ("muCosThetaTrue/F"));
  outtree->Branch("pMomRec", &pMomRec, ("pMomRec/F"));
  outtree->Branch("pMomTrue", &pMomTrue, ("pMomTrue/F"));
  outtree->Branch("pCosThetaRec", &pCosThetaRec, ("pCosThetaRec/F"));
  outtree->Branch("pCosThetaTrue", &pCosThetaTrue, ("pCosThetaTrue/F"));
  outtree->Branch("Enureco", &RecoNuEnergy, "Enureco/F");
  outtree->Branch("Enutrue", &TrueNuEnergy, "Enutrue/F");
  outtree->Branch("weight", &weight, "weight/F");

  intree_T->SetBranchAddress("reaction", &reaction_T);
  intree_T->SetBranchAddress("mectopology", &mectopology_T);
  intree_T->SetBranchAddress(D1NameTrue_T, &D1true_T);
  intree_T->SetBranchAddress(D2NameTrue_T, &D2true_T);
  intree_T->SetBranchAddress("truep_truemom" ,&pMomTrue_T);
  intree_T->SetBranchAddress("truep_truecostheta" ,&pCosThetaTrue_T);
  intree_T->SetBranchAddress("truemu_mom", &muMomTrue_T);
  intree_T->SetBranchAddress("truemu_costheta", &muCosThetaTrue_T);
  intree_T->SetBranchAddress("nu_trueE", &TrueNuEnergy_T);
  //intree_T->SetBranchAddress("weight", &weight_T);

  outtree_T->Branch("reaction", &reaction_T, "reaction/I");
  outtree_T->Branch("mectopology", &mectopology_T, "mectopology/I");
  outtree_T->Branch("D1True", &D1true_T, ("D1True/F"));
  outtree_T->Branch("D2True", &D2true_T, ("D2True/F"));
  outtree_T->Branch("muMomTrue", &muMomTrue_T, ("muMomTrue/F"));
  outtree_T->Branch("muCosThetaTrue", &muCosThetaTrue_T, ("muCosThetaTrue/F"));
  outtree_T->Branch("pMomTrue", &pMomTrue_T, ("pMomTrue/F"));
  outtree_T->Branch("pCosThetaTrue", &pCosThetaTrue_T, ("pCosThetaTrue/F"));
  outtree_T->Branch("Enutrue", &TrueNuEnergy_T, "Enutrue/F");
  outtree_T->Branch("weight", &weight_T, "weight/F");


  Long64_t nentries = intree->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  if(EvtEnd!=0) nentries=EvtEnd;
  if(EvtFrac>0.0001) nentries=nentries*EvtFrac;
  int passCount=0;
  for (Long64_t jentry=EvtStart; jentry<nentries;jentry++) {
    nb = intree->GetEntry(jentry); nbytes += nb;
    passCount=0;
    RecoNuEnergy=TrueNuEnergy;
    pCosThetaRec   = TMath::Cos(pThetaRec);
    muCosThetaRec  = TMath::Cos(muThetaRec);
    //muCosThetaTrue = TMath::Cos(muThetaTrue);
    int branches_passed[10]={0};
    for(int i=0; i<nbranches; i++){
      if(accum_level[0][i]>accumToCut[i]){
        cutBranch=i; passCount++; 
        if(cutBranch==2) pMomRec=pMomRecRange;
        if(cutBranch==3) muMomRec=muMomRecRange;
        branches_passed[i]++;
        if((( (mectopology==1)||(mectopology==2) ) && ( (pMomTrue>450)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.4) ))){
          weight = weight*sigWeight;
        }
        if(specialRWmode==1 && reaction==9) weight=0;
        if(specialRWmode==2 && reaction==9) weight=2;
        if(specialRWmode==3 && reaction==9) weight=3;
        if(specialRWmode==4 && reaction==9 && (TrueNuEnergy>200) && (TrueNuEnergy<1500)) weight=weight*martiniNievesRatioGr->Eval(TrueNuEnergy/1000);
        if(psdim){
          //Phase Space Bins:
          //printf("INFO: Applying phase space binning as specified\n");
          D2true=-1;
          D2Reco=-1;
          const int npmombins=3;
          const double pmombins[npmombins+1] = {0.0, 450.0, 1000.0, 100000.0};
          const int npthetabins=2;
          double pthetabins[npthetabins+1] = {-1.0, 0.8, 1.0};
          if(highAngle){
            pthetabins[1]=0.4;
            pthetabins[2]=0.8;
          }
          const int nmumombins=2;
          const double mumombins[nmumombins+1] = {0.0, 250.0, 100000.0};
          const int nmuthetabins=2;
          const double muthetabins[nmuthetabins+1] = {-1.0, -0.6, 1.0};
          int globalCount=1;
          for(int ii=0; ii<npmombins; ii++){
            for(int j=0; j<npthetabins; j++){
              for(int k=0; k<nmumombins; k++){
                for(int l=0; l<nmuthetabins; l++){
                  if(pMomTrue>(pmombins[ii]) && pMomTrue<(pmombins[ii+1]) && pCosThetaTrue>(pthetabins[j]) &&  pCosThetaTrue<(pthetabins[j+1]) &&
                     muMomTrue>(mumombins[k]) && muMomTrue<(mumombins[k+1]) && muCosThetaTrue>(muthetabins[l]) && muCosThetaTrue<(muthetabins[l+1])){
                    D2true=globalCount;
                  }
                  if(pMomRec>(pmombins[ii]) && pMomRec<(pmombins[ii+1]) && pCosThetaRec>(pthetabins[j]) &&  pCosThetaRec<(pthetabins[j+1]) &&
                     muMomRec>(mumombins[k]) && muMomRec<(mumombins[k+1]) && muCosThetaRec>(muthetabins[l]) && muCosThetaRec<(muthetabins[l+1])){
                    D2Reco=globalCount;
                    //cout << "ps binning applied, bin is " << globalCount << endl;
                  }
                  globalCount++;
                }
              }
            }
          }
          //Relevevent for high angle cosTheta>0.8 (to deal with the fact there's no explicit bin for it):
          if(D2true==-1) D2true=25;
          if(D2Reco==-1) D2Reco=25;

          //printf("INFO: Total number of bins is %d \n", globalCount);
          if(i==5 || i==6){ // Don't bother with ps binning for the SB
            D2true=16;
            D2Reco=16;
            //cout << "ps binning reset since using SB" << endl;
          }
          if(!isRealData && (pMomTrue<0.0001 || muMomTrue<0.0001 || pMomRec < 0.0001 || muMomRec < 0.0001)){ // Don't bother with ps binning for bad events
            D2true=16;
            D2Reco=16;
            //cout << "ps binning reset since invalid event" << endl;
          }
          if(isRealData && (pMomRec < 0.0001 || muMomRec < 0.0001)){ // Don't bother with ps binning for bad events
            D2true=16;
            D2Reco=16;
            //cout << "ps binning reset since invalid event" << endl;
          }
          //To maintain nice bin ordering put signal at the start
          if(D2true==16) D2true=0;
          if(D2Reco==16) D2Reco=0;


        }
        if(extraCuts==false) outtree->Fill();
        else if ((pMomRec>450)&&(muMomRec>250)&&(muCosThetaRec>-0.6)&&(pCosThetaRec>0.4)&&(pMomTrue<1000)){
          printf("INFO: Applying addional cuts as specified\n");
          outtree->Fill();
        }
      }
    }
    if(passCount>1){
      printf("***Warning: More than one cut branch passed***\n");
      for(int j=0;j<10;j++){
        if(branches_passed[j]==1) printf("branch %d passed ...",j);
      }
      printf("\n");
    }
  }

  Long64_t nentries_T = intree_T->GetEntriesFast();
  Long64_t nbytes_T = 0, nb_T = 0;
  if(EvtFrac>0.0001) nentries_T=nentries_T*EvtFrac;
  for (Long64_t jentry=0; jentry<nentries_T;jentry++) {
    weight_T=1;
    nb_T = intree_T->GetEntry(jentry); nbytes_T += nb_T;
    if(specialRWmode==1 && reaction_T==9) weight_T=0;
    if(specialRWmode==2 && reaction_T==9) weight_T=2;
    if(specialRWmode==3 && reaction_T==9) weight_T=3;
    if(specialRWmode==4 && reaction_T==9 && (TrueNuEnergy_T>200) && (TrueNuEnergy_T<1500)) weight_T=weight_T*martiniNievesRatioGr->Eval(TrueNuEnergy_T/1000);
    outtree_T->Fill();
  }
  
  printf("***Output Rec Tree: ***\n");
  outtree->Print();
  printf("***Output True Tree: ***\n");
  outtree_T->Print();
  outfile->Write();

  delete infile;
  delete outfile;
  return 0;
}



/*
//This is probably all low MC stats (muon super backward going)
if(pMomTrue<450 && pCosThetaTrue<0.4 && muMomTrue<250 && muCosThetaTrue<-0.6) D1true=1.0;
if(pMomRec <450 && pCosThetaRec <0.4 && muMomTRec<250 && muCosThetaTRec<-0.6) D1Reco=1.0;
if(pMomTrue>450 && pCosThetaTrue<0.4 && pMomTrue<1000 && muMomTrue<250 && muCosThetaTrue<-0.6) D1true=2.0;
if(pMomRec >450 && pCosThetaRec <0.4 && pMomTRec<1000 && muMomTRec<250 && muCosThetaTRec<-0.6) D1Reco=2.0;
if(pCosThetaTrue<0.4 && pMomTrue>1000 && muMomTrue<250 && muCosThetaTrue<-0.6) D1true=3.0;//This has MC stats issues
if(pCosThetaRec <0.4 && pMomTRec>1000 && muMomTRec<250 && muCosThetaTRec<-0.6) D1Reco=3.0;//This has MC stats issues
if(pMomTrue<450 && pCosThetaTrue>0.4 && muMomTrue<250 && muCosThetaTrue<-0.6) D1true=4.0;
if(pMomRec <450 && pCosThetaRec >0.4 && muMomTRec<250 && muCosThetaTRec<-0.6) D1Reco=4.0;
if(pMomTrue>450 && pCosThetaTrue>0.4 && pMomTrue<1000 && muMomTrue<250 && muCosThetaTrue<-0.6) D1true=5.0;
if(pMomRec >450 && pCosThetaRec >0.4 && pMomTRec<1000 && muMomTRec<250 && muCosThetaTRec<-0.6) D1Reco=5.0;
if(pCosThetaTrue>0.4 && pMomTrue>1000 && muMomTrue<250 && muCosThetaTrue<-0.6) D1true=6.0;
if(pCosThetaRec >0.4 && pMomTRec>1000 && muMomTRec<250 && muCosThetaTRec<-0.6) D1Reco=6.0;

//This is probably all low MC stats (muon super backward going)
if(pMomTrue<450 && pCosThetaTrue<0.4 && muMomTrue<250 && muCosThetaTrue<-0.6) D1true=1.0;
if(pMomRec <450 && pCosThetaRec <0.4 && muMomTRec<250 && muCosThetaTRec<-0.6) D1Reco=1.0;
if(pMomTrue>450 && pCosThetaTrue<0.4 && pMomTrue<1000 && muMomTrue<250 && muCosThetaTrue<-0.6) D1true=2.0;
if(pMomRec >450 && pCosThetaRec <0.4 && pMomTRec<1000 && muMomTRec<250 && muCosThetaTRec<-0.6) D1Reco=2.0;
if(pCosThetaTrue<0.4 && pMomTrue>1000 && muMomTrue<250 && muCosThetaTrue<-0.6) D1true=3.0;//This has MC stats issues
if(pCosThetaRec <0.4 && pMomTRec>1000 && muMomTRec<250 && muCosThetaTRec<-0.6) D1Reco=3.0;//This has MC stats issues
if(pMomTrue<450 && pCosThetaTrue>0.4 && muMomTrue<250 && muCosThetaTrue<-0.6) D1true=4.0;
if(pMomRec <450 && pCosThetaRec >0.4 && muMomTRec<250 && muCosThetaTRec<-0.6) D1Reco=4.0;
if(pMomTrue>450 && pCosThetaTrue>0.4 && pMomTrue<1000 && muMomTrue<250 && muCosThetaTrue<-0.6) D1true=5.0;
if(pMomRec >450 && pCosThetaRec >0.4 && pMomTRec<1000 && muMomTRec<250 && muCosThetaTRec<-0.6) D1Reco=5.0;
if(pCosThetaTrue>0.4 && pMomTrue>1000 && muMomTrue<250 && muCosThetaTrue<-0.6) D1true=6.0;
if(pCosThetaRec >0.4 && pMomTRec>1000 && muMomTRec<250 && muCosThetaTRec<-0.6) D1Reco=6.0;
*/
