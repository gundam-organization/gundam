//
// generate +-3 sigma varation weights for the p0dAntiNumuCC analysis samples
//
//
//
// Author Thomas Campbell <thomascampbell1@gmail.com>
//
//
#include <stdlib.h>
#include <cstdlib>

#include <iostream>

#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"
#include "TMatrix.h"

#include "T2KReWeight.h"
#include "T2KSyst.h"

#include "T2KGenieReWeight.h" 
#include "T2KGenieUtils.h"

#include "T2KNeutReWeight.h"
#include "T2KNeutUtils.h"

#include "T2KJNuBeamReWeight.h"

#ifdef __T2KRW_OAANALYSIS_ENABLED__
#include "ND__NRooTrackerVtx.h"
#endif

// For weight storer class
#include "T2KWeightsStorer.h"

#include "SK__h1.h"

//added
#include "T2KNIWGReWeight.h"
#include "T2KNIWGUtils.h"



using std::cout;
using std::cerr;
using std::endl;

using namespace t2krew;

int fNEvts = -1;
int fNEvtsSel = -1;
int sNEvts = -1; // summary tree
char * fNDFileName;
char * fSKFileName;
char * OutputFileStr;
int IsAntiNu=-1;
int N_TOYS=-1;


//added
char* KnobsNames[] = { // order by niwg-analysis-inputs
  "t2krew::kNXSec_MaCCQE", // [0]
  "t2krew::kNIWG2014a_pF_C12",
  "t2krew::kNIWG2014a_pF_O16",
  "t2krew::kNIWGMEC_Norm_C12",
  "t2krew::kNIWGMEC_Norm_O16",
  "t2krew::kNIWG2014a_Eb_C12",
  "t2krew::kNIWG2014a_Eb_O16",
  "t2krew::kNXSec_CA5RES", 
  "t2krew::kNXSec_MaNFFRES", 
  "t2krew::kNXSec_BgSclRES", 
  "t2krew::kNIWG2012a_ccnueE0",
  "t2krew::kNIWG2012a_dismpishp",
  "t2krew::kNIWG2012a_cccohE0",
  "t2krew::kNIWG2012a_nccohE0",
  "t2krew::kNIWG2012a_ncotherE0", //[14]
  "t2krew::kNIWG2014a_Eb_Al27",
  "t2krew::kNIWG2014a_Eb_Fe56",
  "t2krew::kNIWG2014a_Eb_Cu63",
  "t2krew::kNIWG2014a_Eb_Zn64",
  "t2krew::kNIWG2014a_Eb_Pb208",
  "t2krew::kNIWG2014a_pF_Al27",
  "t2krew::kNIWG2014a_pF_Fe56",
  "t2krew::kNIWG2014a_pF_Cu63",
  "t2krew::kNIWG2014a_pF_Zn64",
  "t2krew::kNIWG2014a_pF_Pb208" //[23]
};

//t2k rw variable values
t2krew::T2KSyst_t KnobsNamesValues[] = { // order by niwg-analysis-inputs
  t2krew::kNXSec_MaCCQE, // [0]
  t2krew::kNIWG2014a_pF_C12,
  t2krew::kNIWG2014a_pF_O16,
  t2krew::kNIWGMEC_Norm_C12,
  t2krew::kNIWGMEC_Norm_O16,
  t2krew::kNIWG2014a_Eb_C12,
  t2krew::kNIWG2014a_Eb_O16,
  t2krew::kNXSec_CA5RES, 
  t2krew::kNXSec_MaNFFRES, 
  t2krew::kNXSec_BgSclRES, 
  t2krew::kNIWG2012a_ccnueE0,
  t2krew::kNIWG2012a_dismpishp,
  t2krew::kNIWG2012a_cccohE0,
  t2krew::kNIWG2012a_nccohE0,
  t2krew::kNIWG2012a_ncotherE0, //[14]
  //-1,//t2krew::kNIWG2014a_Eb_Al27,
  //-1,//t2krew::kNIWG2014a_Eb_Fe56,
  t2krew::kNIWG2014a_Eb_Cu63,
  t2krew::kNIWG2014a_Eb_Zn64,
  t2krew::kNIWG2014a_Eb_Pb208,
  //-1,//t2krew::kNIWG2014a_pF_Al27,
  //-1,//t2krew::kNIWG2014a_pF_Fe56,
  t2krew::kNIWG2014a_pF_Cu63,
  t2krew::kNIWG2014a_pF_Zn64,
  t2krew::kNIWG2014a_pF_Pb208 //[23]
};



//added
//TString KnobsNamesStr[] = { // order by niwg-analysis-inputs
string KnobsNamesStr[] = { // order by niwg-analysis-inputs
  "t2krew::kNXSec_MaCCQE", // [0]
  "t2krew::kNIWG2014a_pF_C12",
  "t2krew::kNIWG2014a_pF_O16",
  "t2krew::kNIWGMEC_Norm_C12",
  "t2krew::kNIWGMEC_Norm_O16",
  "t2krew::kNIWG2014a_Eb_C12",
  "t2krew::kNIWG2014a_Eb_O16",
  "t2krew::kNXSec_CA5RES", 
  "t2krew::kNXSec_MaNFFRES", 
  "t2krew::kNXSec_BgSclRES", 
  "t2krew::kNIWG2012a_ccnueE0",
  "t2krew::kNIWG2012a_dismpishp",
  "t2krew::kNIWG2012a_cccohE0",
  "t2krew::kNIWG2012a_nccohE0",
  "t2krew::kNIWG2012a_ncotherE0", //[14]
  "t2krew::kNIWG2014a_Eb_Al27",
  "t2krew::kNIWG2014a_Eb_Fe56",
  "t2krew::kNIWG2014a_Eb_Cu63",
  "t2krew::kNIWG2014a_Eb_Zn64",
  "t2krew::kNIWG2014a_Eb_Pb208",
  "t2krew::kNIWG2014a_pF_Al27",
  "t2krew::kNIWG2014a_pF_Fe56",
  "t2krew::kNIWG2014a_pF_Cu63",
  "t2krew::kNIWG2014a_pF_Zn64",
  "t2krew::kNIWG2014a_pF_Pb208" //[23]
};


//means {t2kRW mean, NIWG recomend mean, error, ???}
Double_t MeansAndSigmas [][4] = { //<--hard coded
  {1.21, 1.15, 0.41, 0}, // GeV 20160919 0.07->0.41
  //{ 217,  223, 12.3, 0}, // MeV/c 31 -> 12.3
  { 217,  223, 31., 0}, // MeV/c 31 -> 12.3
  //{ 225,  225, 12.3, 0}, // MeV/c 31 -> 12.3
  { 225,  225, 31., 0}, // MeV/c 31 -> 12.3
  {   1,    1,    1, 0}, // (banff 2015)20161006 0.27->1 0.29->1 
  {   1,    1,    1, 0}, // (banff 2015)20161006 0.27->1 1.04->1
  {  25,   25,    9, 0}, // MeV  
  {  27,   27,    9, 0}, // MeV  
  {1.01, 1.01, 0.12, 0}, // GeV 	      
  {0.95, 0.95, 0.15, 0}, 	      
  {1.30, 1.30, 0.20, 0}, 	      
  //{  1,     1, 0.02, 0},       // 0.03 -> 0.02 (banff cov 2015)
  {  1,     1, 0.03, 0},       // 0.03 -> 0.02 (banff cov 2015)
  {  0,     0, 0.40, 0}, //[11] 
  {  1,     1, 	  1, 0},        
  //{  1,     1, 	  0.3, 0},//tuning applied?  1->0.3        
  {  1,     1,	0.3, 0},        
  {  1,     1,	0.3, 0},
  {  1,     1, 0.36, 0},
  {  1,     1, 0.36, 0},
  {  1,     1, 0.36, 0},
  {  1,     1, 0.36, 0},
  {  1,     1, 0.36, 0},
  {  1,     1,0.143, 0},
  {  1,     1,0.143, 0},
  {  1,     1,0.143, 0},
  {  1,     1,0.143, 0},
  {  1,     1,0.143, 0}
  //  {  1,     1,	  1, 1},         // how to do on/off switch
};
//indeces 3,4 Mec are correleated with 0.3*0.3 off diagonal term. 

Int_t SEED = 15798143;


void Usage();
void ParseArgs(int argc, char **argv);

int main(int argc, char *argv[])
{

  // This example only works when compiled against oaAnalysis
#ifdef __T2KRW_OAANALYSIS_ENABLED__
  ParseArgs(argc, argv);

  Int_t accum_levelMin = 3;
  if(IsAntiNu){
    accum_levelMin = 5;
    SEED = 65297143;
  }


  cout << "Starting to reweight NEUT events from ND file: " << fNDFileName << endl;

  // Load the oaAnalysis TNRooTrackerVtx tree containing
  // TClonesArray of objects inheriting from TRooTrackerBase. 
  TFile * ND_infile = new TFile(fNDFileName, "OPEN");
  if(!ND_infile){
    cerr << "Cannot open ND file!" << endl;
    exit(1);
  }
  //TTree * ND_tree = (TTree*) ND_infile->Get("TruthDir/NRooTrackerVtx");
  TTree * ND_tree = (TTree*) ND_infile->Get("NRooTrackerVtx");
  TTree * ND_treeSel = (TTree*) ND_infile->Get("default");
  TTree * ND_treeTrue = (TTree*) ND_infile->Get("truth");
  if(!ND_tree){
    cerr << "Cannot find ND_tree!" << endl;
  }

  int NVtx; 
  TClonesArray * nRooVtxs = new TClonesArray("ND::NRooTrackerVtx", 100);
  ND_tree->SetBranchAddress("Vtx", &nRooVtxs);
  ND_tree->SetBranchAddress("NVtx", &NVtx);

  int accum_level=-1;
  int RooVtxEntry=-1;
  Int_t TruthVertexID=-1;
  Int_t topology=-1;
  Float_t selmu_amom=-1.;
  Float_t selmu_costheta=-11;
  Float_t truelepton_mom=-1;
  Float_t truelepton_costheta=-1;
  Float_t* TrkChargeRatio = new Float_t[4];
  Int_t TagMichel2=-1;
  Int_t nu_pdg=-1;
  Float_t nu_trueE=-999;
  Int_t reaction=-1;
  Int_t reactionHL=-1;
  Int_t PIDnShowers=-1;
  Int_t mectopologyFill=-1;
  Int_t TempMecTopology[10];
  Float_t weight=-1;

  ND_treeSel->SetBranchAddress("mectopology",&TempMecTopology[0]);
  ND_treeSel->SetBranchAddress("RooVtxEntry",&RooVtxEntry);
  ND_treeSel->SetBranchAddress("accum_level",&accum_level);
  ND_treeSel->SetBranchAddress("TruthVertexID",&TruthVertexID);
  ND_treeSel->SetBranchAddress("topology",&topology);
  //ND_treeSel->SetBranchAddress("mectopology",&mectopologyFill);
  ND_treeSel->SetBranchAddress("selmu_amom",&selmu_amom);
  ND_treeSel->SetBranchAddress("selmu_costheta",&selmu_costheta);
  ND_treeSel->SetBranchAddress("truelepton_mom",&truelepton_mom);
  ND_treeSel->SetBranchAddress("truelepton_costheta",&truelepton_costheta);
  ND_treeSel->SetBranchAddress("TrkChargeRatio",TrkChargeRatio);
  ND_treeSel->SetBranchAddress("TagMichel2",&TagMichel2);
  ND_treeSel->SetBranchAddress("nu_pdg",&nu_pdg);
  ND_treeSel->SetBranchAddress("nu_trueE",&nu_trueE);
  ND_treeSel->SetBranchAddress("reaction",&reactionHL);
  ND_treeSel->SetBranchAddress("nu_truereac",&reaction);
  ND_treeSel->SetBranchAddress("PIDnShowers",&PIDnShowers);
  ND_treeSel->SetBranchAddress("weight",&weight);

  int RooVtxEntryTrue=-1;
  Int_t TruthVertexIDTrue=-1;
  Int_t topologyTrue=-1;
  Int_t nu_pdgTrue=-1;
  Float_t nu_trueETrue=-999;
  Float_t truelepton_momTrue=-1;
  Float_t truelepton_costhetaTrue=-11;
  Int_t reactionTrue=-1;
  Int_t reactionHLTrue=-1;
  //to add

  ND_treeTrue->SetBranchAddress("RooVtxEntry",&RooVtxEntryTrue);
  ND_treeTrue->SetBranchAddress("TruthVertexID",&TruthVertexIDTrue);
  ND_treeTrue->SetBranchAddress("topology",&topologyTrue);
  ND_treeTrue->SetBranchAddress("nu_pdg",&nu_pdgTrue);
  ND_treeTrue->SetBranchAddress("nu_trueE",&nu_trueETrue);
  ND_treeTrue->SetBranchAddress("truelepton_mom",&truelepton_momTrue);
  ND_treeTrue->SetBranchAddress("truelepton_costheta",&truelepton_costhetaTrue);
  ND_treeTrue->SetBranchAddress("reaction",&reactionHLTrue);
  ND_treeTrue->SetBranchAddress("nu_truereac",&reactionTrue);

  //if(fNEvts < 0) fNEvts = ND_tree->GetEntries();
  fNEvts = ND_treeTrue->GetEntries();
  fNEvtsSel = ND_treeSel->GetEntries();

  cout << "Will reweight ND nevents: " << fNEvts << endl;

  //TFile* outF = new TFile("TestOutF.root", "recreate");
  TFile* outF = new TFile(OutputFileStr, "recreate");
  TTree* outTdefault = new TTree("outTdefault", "output tree");
  TTree* outTtruth = new TTree("outTtruth", "output tree");
  TTree* outTConfig = new TTree("outTConfig", "ouput tree");


  //Int_t variations =3;
  //Int_t variations =N_TOYS;
  Int_t variations =7;
  Int_t nKnobs=15;


  TMatrixD MeansAndSigmasOut(nKnobs,4);
  for(int i=0; i<nKnobs; i++){
    for(int j=0; j<4; j++){
      MeansAndSigmasOut(i,j)=MeansAndSigmas[i][j];
    }
  }


  vector<string> KnobsNamesFill;
  for(int i=0; i<nKnobs; i++){
    KnobsNamesFill.push_back(KnobsNamesStr[i]);
    cout << KnobsNamesStr[i] << " " << KnobsNamesFill.at(i) << endl;
  }
  outTConfig->Branch("KnobsNames", &KnobsNamesFill);
  outTConfig->Branch("MeansAndSigmas", "TMatrixD", &MeansAndSigmasOut);
  outTConfig->Branch("nKnobs",&nKnobs,"nKnobs/I");
  outTConfig->Branch("nToys",&variations,"nToys/I");
  outTConfig->Branch("seed", &SEED, "seed/I");


  outTConfig->Fill();
  //outF->Write();
  //outF->Close();
  //return 0;


  TMatrixD WeightsMatrix(nKnobs,variations);

  outTdefault->Branch("variations",&variations,"variations/I");
  outTdefault->Branch("nKnobs",&nKnobs,"nKnobs/I");
  outTdefault->Branch("accum_level",&accum_level,"accum_level/I");
  outTdefault->Branch("topology", &topology, "topology/I");
  outTdefault->Branch("mectopology", &mectopologyFill, "mectopology/I");
  outTdefault->Branch("selmu_amom", &selmu_amom, "selmu_amom/F");
  outTdefault->Branch("selmu_costheta", &selmu_costheta, "selmu_costheta/F");
  outTdefault->Branch("truelepton_mom", &truelepton_mom, "truelepton_mom/F");
  outTdefault->Branch("truelepton_costheta", &truelepton_costheta, "truelepton_costheta/F");
  outTdefault->Branch("TrkChargeRatio", TrkChargeRatio, "TrkChargeRatio[4]/F");
  outTdefault->Branch("TagMichel2", &TagMichel2, "TagMichel2/I");
  outTdefault->Branch("WeightsMatrix","TMatrixD",&WeightsMatrix);
  outTdefault->Branch("nu_pdg", &nu_pdg, "nu_pdg/I");
  outTdefault->Branch("nu_trueE", &nu_trueE, "nu_trueE/F");
  outTdefault->Branch("seed", &SEED, "seed/I");
  outTdefault->Branch("PIDnShowers", &PIDnShowers, "PIDnShowers/I");
  outTdefault->Branch("reaction", &reactionHL, "reaction/I");
  outTdefault->Branch("weight", &weight, "weight/F");

  outTtruth->Branch("variations",&variations,"variations/I");
  outTtruth->Branch("nKnobs",&nKnobs,"nKnobs/I");
  outTtruth->Branch("topology", &topologyTrue, "topology/I");
  outTtruth->Branch("WeightsMatrix","TMatrixD",&WeightsMatrix);
  outTtruth->Branch("nu_pdg", &nu_pdgTrue, "nu_pdg/I");
  outTtruth->Branch("nu_trueE", &nu_trueETrue, "nu_trueE/F");
  outTtruth->Branch("truelepton_mom", &truelepton_momTrue, "truelepton_mom/F");
  outTtruth->Branch("truelepton_costheta", &truelepton_costhetaTrue, "truelepton_costheta/F");
  outTtruth->Branch("reaction", &reactionHLTrue, "reaction/I");
  outTtruth->Branch("seed", &SEED, "seed/I");


  // Add NEUT reweighting engine
  t2krew::T2KReWeight rw; 
  rw.AdoptWghtEngine("neut_rw", new t2krew::T2KNeutReWeight());

  rw.AdoptWghtEngine("niwg_rw", new t2krew::T2KNIWGReWeight());

  for(int i=0; i<nKnobs; i++){
    rw.Systematics().Include(KnobsNamesValues[i]);
    rw.Systematics().SetAbsTwk(KnobsNamesValues[i]);
  }


  // flags 
  // SF  mode  SF_RFG = 0; VecFFCCQE = 402
  // RFG mode  SF_RFG = 1; VecFFCCQE = 2
  // RFG needs norm = 1 and shape = 0 relative or -1 non-relavite
  rw.Systematics().Include(t2krew::kNIWG2014a_SF_RFG);
  rw.Systematics().Include(t2krew::kNXSec_VecFFCCQE); 
  rw.Systematics().Include(t2krew::kNIWG_rpaCCQE_norm);
  rw.Systematics().Include(t2krew::kNIWG_rpaCCQE_shape);

  // default SF mode
  // these flags defaults are 0, 402 (SF mode)
  // for rpa mode set them to 1, 2
  // 2016Apr06 T2K default RFG relavite i.e. 
  // SF_RFG 1, VecFF 2, RFG norm 1, RFG shape 0
  int useRFGModel=0;
  cout<<" useRFGModel is set to "<<useRFGModel<<endl;
  if (useRFGModel == 1) {
    rw.Systematics().SetTwkDial(t2krew::kNIWG2014a_SF_RFG, 1);
    rw.Systematics().SetTwkDial(t2krew::kNXSec_VecFFCCQE, 2);
    rw.Systematics().SetTwkDial(t2krew::kNIWG_rpaCCQE_norm, 1);
    rw.Systematics().SetTwkDial(t2krew::kNIWG_rpaCCQE_shape, 0);
  }
  else { // SF model
    rw.Systematics().SetTwkDial(t2krew::kNIWG2014a_SF_RFG, 0);
    rw.Systematics().SetTwkDial(t2krew::kNXSec_VecFFCCQE, 402);
  }



#ifdef __T2KRW_NEUT_ENABLED__

  //Cov Matix:
  TMatrixD Cov(nKnobs,nKnobs);
  for(int i=0; i<nKnobs; i++){
    for(int j=0; j<nKnobs; j++){
      if(i==j){
	if(MeansAndSigmas[i][1]>0){
	  Cov(i,j)=MeansAndSigmas[i][2]*MeansAndSigmas[i][2]/(MeansAndSigmas[i][1]*MeansAndSigmas[i][1]);
	}
	else Cov(i,j)=MeansAndSigmas[i][2]*MeansAndSigmas[i][2];
      }
      else Cov(i,j)=0.0;
    }
  }
  //correlations
  //3,4 (MEC) correleated with 0.3^2 off diagonal term
  Cov(3,4)=0.3*0.3;
  Cov(4,3)=0.3*0.3;
  //Have Cov Matrix
  //Decomp 
  TDecompChol Chol(Cov);
  Chol.Decompose();
  TMatrixD TDecompCov=Chol.GetU();
  //Check: outtemp should be very very close to inital cov matrix
  TMatrixD outtemp = TMatrixD(TDecompCov,TMatrixD::kTransposeMult,TDecompCov);
  TDecompCov.T();
  //TDecompCov should now be ready for multiplication by random vector to get variations


  //disable output cout
  std::cout.setstate(std::ios_base::failbit);



  for(int iEvent = 0; iEvent < fNEvts; iEvent++){

    if(!(iEvent%(fNEvts/100))){
      //reenable cout
      std::cout.clear();
      cout << "On Event: " << iEvent+1 << "/" << fNEvts << " " << iEvent*100./fNEvts << "%" << endl;
      //disable output cout
      std::cout.setstate(std::ios_base::failbit);
    }


    //truth tree

    ND_treeTrue->GetEntry(iEvent);
    Double_t weight = 1.0;
    ND_tree->GetEntry(RooVtxEntryTrue);

    ND::NRooTrackerVtx * foundVtx;
    int foundVtxFlag=-1;

    if(topologyTrue==0){

      //flip signs here so that if statemnts work, flip back before fill
      if(IsAntiNu){
	reactionTrue=(-1)*reactionTrue;
	nu_pdgTrue=(-1)*nu_pdgTrue;
      }

      for(int jVtx = 0; jVtx<NVtx; jVtx++){
	ND::NRooTrackerVtx * vtx = (ND::NRooTrackerVtx*) nRooVtxs->At(jVtx);
	if(!vtx){
	  cerr << "Cannot find NRooTrackerVtx object - skipping weight for this vertex!";
	  continue;
	}

	if((vtx->TruthVertexID)!=TruthVertexIDTrue) continue;
	else{
	  foundVtx = vtx;
	  foundVtxFlag=1;
	}


	Double_t TwkDialValue=-1.;
	Double_t* TwkDialValueVec = new Double_t[nKnobs];
	for(int i=0; i<nKnobs; i++){
	  TwkDialValueVec[i]=0.0;
	}
	TRandom3 r;
	r.SetSeed(SEED);
	Float_t tempRand=-1;
	TVectorD RandVec(nKnobs);
	TVectorD VarVec(nKnobs);



	for(int varIndex=0; varIndex<variations; varIndex++){

	  tempRand=r.Gaus();
	  for(int i=0; i<nKnobs; i++){
	    RandVec(i)=r.Gaus();
	  }
	  VarVec=TDecompCov*RandVec;

	  for(int KnobIndex=0; KnobIndex<nKnobs; KnobIndex++){

	    Int_t skipFlag=0;

	    if(nu_pdgTrue==14 && (reactionTrue==1)){
	      if(!(KnobIndex==0 || KnobIndex==1 || KnobIndex==2 || KnobIndex==5 || KnobIndex==6)){
		WeightsMatrix(KnobIndex,varIndex)=1.0;
		skipFlag=1;
	      }
	    }
	    else if(nu_pdgTrue==14 && (reactionTrue==2)){
	      if(!(KnobIndex==5 || KnobIndex==6)){
		WeightsMatrix(KnobIndex,varIndex)=1.0;
		skipFlag=1;
	      }
	    }
	    else if(nu_pdgTrue==14 && (reactionTrue==11 || reactionTrue==12 || reactionTrue==13 || (reactionTrue<=34 && reactionTrue>=31))){
	      if(!(KnobIndex<=9 && KnobIndex>=7)){
		WeightsMatrix(KnobIndex,varIndex)=1.0;
		skipFlag=1;
	      }
	    }
	    else if(nu_pdgTrue==14 && (reactionTrue==17 || reactionTrue==21 || reactionTrue==22 || reactionTrue==23 || reactionTrue==26)){
	      if(KnobIndex!=11){
		WeightsMatrix(KnobIndex,varIndex)=1.0;
		skipFlag=1;
	      }
	    }
	    else if(nu_pdgTrue==14 && (reactionTrue==16)){
	      if(KnobIndex!=12){
		WeightsMatrix(KnobIndex,varIndex)=1.0;
		skipFlag=1;
	      }
	    }
	    else if(nu_pdgTrue==14 && (reactionTrue!=36)){
	      if(KnobIndex!=13){
		WeightsMatrix(KnobIndex,varIndex)=1.0;
		skipFlag=1;
	      }
	    }
	    else if(nu_pdgTrue==14 && (reactionTrue==38 || reactionTrue==39 || (reactionTrue>=41 && reactionTrue<=46) || reactionTrue==51 || reactionTrue==52)){
	      if(KnobIndex!=14){
		WeightsMatrix(KnobIndex,varIndex)=1.0;
		skipFlag=1;
	      }
	    }
	    else if((nu_pdgTrue==12 || nu_pdgTrue==-12) && (reactionTrue>=1 && reactionTrue<=26)){
	      if(KnobIndex!=10){
		WeightsMatrix(KnobIndex,varIndex)=1.0;
		skipFlag=1;
	      }
	    }

	    if(skipFlag==1) continue;


	    //TwkDialValue=VarVec(KnobIndex);
	    if(MeansAndSigmas[KnobIndex][1]>0){
	      TwkDialValue=(varIndex-3)*MeansAndSigmas[KnobIndex][2]/MeansAndSigmas[KnobIndex][1];
	    }
	    else{
	      TwkDialValue=(varIndex-3)*MeansAndSigmas[KnobIndex][2];
	    }



	    rw.Systematics().SetTwkDial(KnobsNamesValues[KnobIndex], TwkDialValue);

	    rw.Reconfigure();
	     
	    weight = rw.CalcWeight(foundVtx);
	    WeightsMatrix(KnobIndex,varIndex)=weight;

	    // reset to zero the above twk
	    TwkDialValue = 0;
	    rw.Systematics().SetTwkDial(KnobsNamesValues[KnobIndex], TwkDialValue);
	  }//knobs
	}//variations
	break;//this is to get out of for-vtx-in-event loop after the proper vtx has been found and used
      } //vtx in event

      //flip back signs
      if(IsAntiNu){
	reactionTrue=(-1)*reactionTrue;
	nu_pdgTrue=(-1)*nu_pdgTrue;
      }
      outTtruth->Fill();
    }//if generated topology


    //default tree
    if(iEvent>=fNEvtsSel) continue;


    ND_treeSel->GetEntry(iEvent);
    mectopologyFill=TempMecTopology[0];
    //wtf!?!?

    //std::cout.clear();
    //cout << mectopologyFill << endl;
    //cout << TempMecTopology[0] << endl;
    //std::cout.setstate(std::ios_base::failbit);

    weight = 1.0;
    if(accum_level<accum_levelMin) continue;
    ND_tree->GetEntry(RooVtxEntry);
    //flip signs for anti nu mode for if statements
    if(IsAntiNu){
      reaction=(-1)*reaction;
      nu_pdg=(-1)*nu_pdg;
    }

    foundVtx = NULL;
    foundVtxFlag=-1;
    for(int jVtx = 0; jVtx<NVtx; jVtx++){
      ND::NRooTrackerVtx * vtx = (ND::NRooTrackerVtx*) nRooVtxs->At(jVtx);
      if(!vtx){
	cerr << "Cannot find NRooTrackerVtx object - skipping weight for this vertex!";
	continue;
      }

      if((vtx->TruthVertexID)!=TruthVertexID) continue;
      else{
	foundVtx = vtx;
	foundVtxFlag=1;
      }


      Double_t TwkDialValue=-1.;
      TRandom3 r;
      r.SetSeed(SEED);
      Float_t tempRand=-1;
      TVectorD RandVec(nKnobs);
      TVectorD VarVec(nKnobs);

      for(int varIndex=0; varIndex<variations; varIndex++){

	tempRand=r.Gaus();
	for(int i=0; i<nKnobs; i++){
	  RandVec(i)=r.Gaus();
	}
	VarVec=TDecompCov*RandVec;


	for(int KnobIndex=0; KnobIndex<nKnobs; KnobIndex++){

	  if(nu_pdg==14 && (reaction==1)){
	    if(!(KnobIndex==0 || KnobIndex==1 || KnobIndex==2 || KnobIndex==5 || KnobIndex==6)){
	      WeightsMatrix(KnobIndex,varIndex)=1.0;
	      continue;
	    }
	  }
	  else if(nu_pdg==14 && (reaction==2)){
	    if(!(KnobIndex==5 || KnobIndex==6)){
	      WeightsMatrix(KnobIndex,varIndex)=1.0;
	      continue;
	    }
	  }
	  else if(nu_pdg==14 && (reaction==11 || reaction==12 || reaction==13 || (reaction<=34 && reaction>=31))){
	    if(!(KnobIndex<=9 && KnobIndex>=7)){
	      WeightsMatrix(KnobIndex,varIndex)=1.0;
	      continue;
	    }
	  }
	  else if(nu_pdg==14 && (reaction==17 || reaction==21 || reaction==22 || reaction==23 || reaction==26)){
	    if(KnobIndex!=11){
	      WeightsMatrix(KnobIndex,varIndex)=1.0;
	      continue;
	    }
	  }
	  else if(nu_pdg==14 && (reaction==16)){
	    if(KnobIndex!=12){
	      WeightsMatrix(KnobIndex,varIndex)=1.0;
	      continue;
	    }
	  }
	  else if(nu_pdg==14 && (reaction!=36)){
	    if(KnobIndex!=13){
	      WeightsMatrix(KnobIndex,varIndex)=1.0;
	      continue;
	    }
	  }
	  else if(nu_pdg==14 && (reaction==38 || reaction==39 || (reaction>=41 && reaction<=46) || reaction==51 || reaction==52)){
	    if(KnobIndex!=14){
	      WeightsMatrix(KnobIndex,varIndex)=1.0;
	      continue;
	    }
	  }
	  else if((nu_pdg==12 || nu_pdg==-12) && (reaction>=1 && reaction<=26)){
	    if(KnobIndex!=10){
	      WeightsMatrix(KnobIndex,varIndex)=1.0;
	      continue;
	    }
	  }

	  //TwkDialValue=VarVec(KnobIndex);
	  if(MeansAndSigmas[KnobIndex][1]>0){
	    TwkDialValue=(varIndex-3)*MeansAndSigmas[KnobIndex][2]/MeansAndSigmas[KnobIndex][1];
	  }
	  else{
	    TwkDialValue=(varIndex-3)*MeansAndSigmas[KnobIndex][2];
	  }

	  rw.Systematics().SetTwkDial(KnobsNamesValues[KnobIndex], TwkDialValue);

	  rw.Reconfigure();

	  weight = rw.CalcWeight(foundVtx);
	  WeightsMatrix(KnobIndex,varIndex)=weight;

	  // reset to zero the above twk
	  TwkDialValue = 0;

	  rw.Systematics().SetTwkDial(KnobsNamesValues[KnobIndex], TwkDialValue);

	}//knobs
      }//variations
      break;//this is to get out of for-vtx-in-event loop after the proper vtx has been found and used
    } //vtx in event
    //flip signs back before fill
    if(IsAntiNu){
      reaction=(-1)*reaction;
      nu_pdg=(-1)*nu_pdg;
    }
    outTdefault->Fill();
  }//event loop

  outF->Write();
  outF->Close();


#endif



#endif // __T2KRW_OAANALYSIS_ENABLED__
  return 0;
}

// Print the cmd line syntax
void Usage(){
  cout << "Cmd line syntax should be:" << endl;
  //cout << "genWeightsFromJReWeight.exe -n nd_inputfile -s sk_inputfile [-e nevents]" << endl;
  cout << "genToy****.exe -n nd_inputfile -a IsAntiNu -o OutputFile.root -t nToys [-e nevents]" << endl;
}

// Messy way to process cmd line arguments.
void ParseArgs(int argc, char **argv){
  int nargs = 1; 
  if(argc<(nargs*2+1)){ Usage(); exit(1); }
  for(int i = 1; i < argc; i+=2){
    if(string(argv[i]) == "-n") fNDFileName = argv[i+1];
    else if(string(argv[i]) == "-s") fSKFileName = argv[i+1];
    else if(string(argv[i]) == "-e") fNEvts = std::atoi(argv[i+1]);
    else if(string(argv[i]) == "-a") IsAntiNu = std::atoi(argv[i+1]);
    else if(string(argv[i]) == "-o") OutputFileStr = argv[i+1];
    else if(string(argv[i]) == "-t") N_TOYS = std::atoi(argv[i+1]);
    else {  
      cout << "Invalid argument:" << argv[i] << " "<< argv[i+1] << endl;
      Usage();
      exit(1);
    }
  } 
}

