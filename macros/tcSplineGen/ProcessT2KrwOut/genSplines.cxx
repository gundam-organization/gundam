#include "processT2KrwOutIncludes.h"
#include "TCanvas.h"

using std::cout;
using std::cerr;
using std::endl;

using namespace t2krew;


#include "GetDiffBin.C"

/*
   int fNEvts = -1;
   int fNEvtsSel = -1;
   int sNEvts = -1; // summary tree
   char * fNDFileName;
   char * fSKFileName;
   char * OutputFileStr;
   int IsAntiNu=-1;
   int N_TOYS=-1;
   int SingleKnob=-1;
   char* inFStr;
 */


//for splines:
int IsAntiNu=-1;
char* inFStr;
string dialName;
int dialIndex=-1;
int topo=-1;
int reac=-1;



void Usage();
void ParseArgs(int argc, char **argv);

int main(int argc, char *argv[])
{
  ParseArgs(argc, argv);



  //TFile* inF = new TFile("genToyOut1000.root", "OPEN");
  TFile* inF = new TFile(inFStr, "OPEN");
  TFile* inFAir = new TFile("/home/other/tcampbell/p0dCCAnalysis/Run6Air/T2KrwOutFiles/MergedFiles/FullRun6Air.root","OPEN");
  if(!inF){
    cout << "Damn it all to hell!" << endl;
    return 1;
  }
  if(!inFAir){
    cout << "Damn it all to hell!" << endl;
    return 1;
  }
  TTree* inTdefault = (TTree*)inF->Get("outTdefault");
  TTree* inTdefaultAir = (TTree*)inFAir->Get("outTdefault");
  TTree* inTConfig = (TTree*)inF->Get("outTConfig");

  //Int_t nToys = -1;
  Int_t nToys = -1;
  Int_t nKnobs = -1;
  //Int_t nKnobs = -1;
  //Int_t nBins = 60;
/*  
  Int_t nPBins = 9;
  Int_t nCosBins = 2;
  Int_t nInitTrashBins=1;//subtract off inital bins (only one cos bin analysis bin for initial trash pbin
  Int_t nAnaBins=16;

  Float_t PBins[10]={0,400,530,670,800,1000,1380,2010,3410,50000};
  Float_t CosBins[9][3]={
    {-1,1,1},//only one bin here
    {-1,0.84,1},
    {-1,0.83,1},
    {-1,0.8,1},
    {-1,0.8,1},
    {-1,0.85,1},
    {-1,0.87,1},
    {-1,0.95,1},
    {-1,1,1},//only one bin here
  };
*/
/* 
  Int_t nPBins = 9;
  Int_t nCosBins = 4;
  Int_t nAnaBins=26;

  Float_t PBins[10]={0,400,530,670,800,1000,1380,2010,3410,50000};
  Float_t CosBins[9][5]={
    {-1,1,1,1,1},//only one bin here
    {-1,0.84,0.93,1,1},//three bins
    {-1,0.83,0.886,0.95,1},//four bins
    {-1,0.8,0.87,0.95,1},//four bins
    {-1,0.8,0.87,0.95,1},//four bins
    {-1,0.85,0.9,0.95,1},//four bins
    {-1,0.87,0.95,1,1},//three bins
    {-1,0.95,1,1,1},//two bins
    {-1,1,1,1,1},//only one bin here
  };
*/
  Int_t nPBins = 9;
  Int_t nCosBins = 3;
  Int_t nAnaBins=21;

  Float_t PBins[10]={0,400,530,670,800,1000,1380,2010,3410,50000};
  Float_t CosBins[9][4]={
    {-1,1,1,1},//only one bin here
    {-1,0.84,1,1},//only two bins here
    {-1,0.83,0.95,1},
    {-1,0.8,0.94,1},
    {-1,0.8,0.94,1},
    {-1,0.85,0.94,1},
    {-1,0.87,0.95,1},
    {-1,0.95,1,1},//only two bins here
    {-1,1,1},//only one bin here
  };
 
/*
  Float_t PBins[10]={0,400,530,670,800,1000,1380,2010,3410,50000};
  Float_t CosBins[9][4]={
    {-1,1,1,1},//only one bin here
    {-1,0.84,0.95,1},
    {-1,0.83,0.95,1},
    {-1,0.8,0.95,1},
    {-1,0.8,0.95,1},
    {-1,0.85,0.95,1},
    {-1,0.87,0.95,1},
    {-1,0.95,1,1},//only two bins here
    {-1,1,1},//only one bin here
  };
*/
/*
  //Float_t PBins[10]={0,490,580,670,790,1000,1380,2010,3410,50000};
  Float_t PBins[10]={0,400,530,670,800,1000,1380,2010,3410,50000};
  Float_t CosBins[9][3]={
    {-1,1,1},//only one bin here
    {-1,0.8,1},
    {-1,0.9,1},
    {-1,0.9,1},
    {-1,0.9,1},
    {-1,0.9,1},
    {-1,0.95,1},
    {-1,0.97,1},
    {-1,1,1},//only one bin here
  };
*/
  //total bins=44 (7*6+2)
  //bins to be populated: 0, 6-48
  //if(bin==0) anaBin=bin;
  //else anaBin=bin-5;
/*
  //course double diff
  Float_t PBins[10]={0,400,530,670,800,1000,1380,2010,3410,10000};
  Float_t CosBins[9][5]={
    {-1,1.0,0.924,1,1},//only one bin here
    {-1,0.843,0.932,1,1},//three bins
    {-1,0.83,0.886,0.95,1},//four bins
    {-1,0.8,0.905,0.96,1},//four bins
    {-1,0.8,0.92,0.967,1},//for bins
    {-1,0.85,0.934,0.973,1},//forbins
    {-1,0.88,0.957,0.982,1},//Col. Forbin's
    {-1,0.94,0.976,1,1},//three bins
    {-1,1.0,0.996,1,1}//only one bin here
  };

  Int_t nPBins=9;
  Int_t nCosBins=4;
  Int_t nAnaBins=28;

  //nAnaBins=9*4-3-3-1-1=28
  //nPBins=9
  //nCosBins=4
  //Analysis bins (starting from 0 index): 2,3 | 5,6,7 | 9,10,11 | 13,14,15 | 17,18,19 | 21,22,23 | 25,26 
  //anaDiffBin==(2,3) or {!=(4,8,12,16,20,24): Bin%4==0}, or ==(25,26)
  //if((AnaDiffBin%4==0) || AnaDiffBin==1 || AnaDiffBin==27) continue; 

  //going from DiffBin to AnaDiffBin:
  //if(Diff==0) Ana=0
  //else if(Diff<7) Ana=Diff-3
  //else if(Diff>7 && Diff<31) Ana=Diff-4
  //else if(Diff==32) Ana=27
*/


  
  float* PBinsPass = new float[nPBins+1];
  for(int i=0; i<(nPBins+1); i++){
    PBinsPass[i]=PBins[i];
  }

  float** CosBinsPass = new float*[nPBins];
  for(int i=0; i<nPBins; i++){
    CosBinsPass[i] = new float[nCosBins+1];
  }
  for(int i=0; i<nPBins; i++){
    for(int j=0; j<(nCosBins+1); j++){
      CosBinsPass[i][j] = CosBins[i][j];
    }
  }


  //inTConfig->SetBranchAddress("KnobsNames",&KnobsNames);
  inTConfig->SetBranchAddress("nToys",&nToys);
  inTConfig->SetBranchAddress("nKnobs",&nKnobs);

  inTConfig->GetEntry(0);

  //TMatrixD MeansAndSigmas(nKnobs,nToys);
  TMatrixD MeansAndSigmas(nKnobs,4);
  TMatrixD* MeansAndSigmasP = &MeansAndSigmas;

  vector<string> KnobsNames;
  vector<string>* KnobsNamesP = &KnobsNames;
  inTConfig->SetBranchAddress("KnobsNames",&KnobsNamesP);
  inTConfig->SetBranchAddress("MeansAndSigmas",&MeansAndSigmasP);

  inTConfig->GetEntry(0);

  //=====
  //Can use these variables to isolate a single or sequential subset of knobs to run on

  Int_t nKnobsStart = 0; //needs to be declared, 0 will run over all knobs
  nKnobsStart=dialIndex;
  nKnobs = nKnobsStart+1;

  //======


  TMatrixD WeightsMatrix(nKnobs,nToys);
  TMatrixD* WeightsMatrixP = &WeightsMatrix;

  Int_t topology=-1;
  Int_t accum_level=-1;
  Float_t truelepton_mom=-999;
  Float_t truelepton_costheta=-11;
  Float_t TrkChargeRatio[4];
  Int_t TagMichel2=-1;
  Int_t nu_pdg=-1;
  Float_t nu_trueE=-1;
  Float_t selmu_amom=-1;
  Float_t selmu_costheta=-11;
  Int_t PIDnShowers=-1;
  Int_t reaction=-1;

  //inTdefault->SetBranchAddress("topology",&topology);
  inTdefault->SetBranchAddress("mectopology",&topology);
  inTdefault->SetBranchAddress("accum_level",&accum_level);
  inTdefault->SetBranchAddress("truelepton_mom",&truelepton_mom);
  inTdefault->SetBranchAddress("truelepton_costheta",&truelepton_costheta);
  inTdefault->SetBranchAddress("WeightsMatrix",&WeightsMatrixP);
  inTdefault->SetBranchAddress("TrkChargeRatio",TrkChargeRatio);
  inTdefault->SetBranchAddress("TagMichel2",&TagMichel2);
  inTdefault->SetBranchAddress("nu_pdg",&nu_pdg);
  inTdefault->SetBranchAddress("nu_trueE",&nu_trueE);
  inTdefault->SetBranchAddress("selmu_amom",&selmu_amom);
  inTdefault->SetBranchAddress("selmu_costheta",&selmu_costheta);
  inTdefault->SetBranchAddress("PIDnShowers",&PIDnShowers);
  inTdefault->SetBranchAddress("reaction",&reaction);

  Int_t nEntriesDefault = inTdefault->GetEntries();

  //air stuff
  //
  //inTdefault->SetBranchAddress("topology",&topology);
  inTdefaultAir->SetBranchAddress("mectopology",&topology);
  inTdefaultAir->SetBranchAddress("accum_level",&accum_level);
  inTdefaultAir->SetBranchAddress("truelepton_mom",&truelepton_mom);
  inTdefaultAir->SetBranchAddress("truelepton_costheta",&truelepton_costheta);
  inTdefaultAir->SetBranchAddress("WeightsMatrix",&WeightsMatrixP);
  inTdefaultAir->SetBranchAddress("TrkChargeRatio",TrkChargeRatio);
  inTdefaultAir->SetBranchAddress("TagMichel2",&TagMichel2);
  inTdefaultAir->SetBranchAddress("nu_pdg",&nu_pdg);
  inTdefaultAir->SetBranchAddress("nu_trueE",&nu_trueE);
  inTdefaultAir->SetBranchAddress("selmu_amom",&selmu_amom);
  inTdefaultAir->SetBranchAddress("selmu_costheta",&selmu_costheta);
  inTdefaultAir->SetBranchAddress("PIDnShowers",&PIDnShowers);
  inTdefaultAir->SetBranchAddress("reaction",&reaction);

  Int_t nEntriesDefaultAir = inTdefaultAir->GetEntries();


  //bools for cuts
  //Selection and SBs
  bool CutsSig; 
  bool CutsSB1pi;
  bool CutsSBOther;

  //ready to loop


  // N[toy][bin]
  /*
  //Selections
  Float_t** nSig = new Float_t*[nToys];
  //Sidebands
  Float_t** nSB1pi = new Float_t*[nToys];
  Float_t** nSBOther = new Float_t*[nToys];
   */
  //with "reaction" which is really topology
  //Selections
  Float_t*** nSig = new Float_t**[nToys];
  Float_t*** nSigAir = new Float_t**[nToys];
  //Sidebands
  Float_t*** nSB1pi = new Float_t**[nToys];
  Float_t*** nSB1piAir = new Float_t**[nToys];
  Float_t*** nSBOther = new Float_t**[nToys];



  for(int i=0; i<nToys; i++){
    nSig[i] = new Float_t*[nAnaBins];
    nSigAir[i] = new Float_t*[nAnaBins];
    nSB1pi[i] = new Float_t*[nAnaBins];
    nSB1piAir[i] = new Float_t*[nAnaBins];
    nSBOther[i] = new Float_t*[nAnaBins];
    for(int j=0; j<nAnaBins; j++){
      nSig[i][j]= new Float_t[8];
      nSigAir[i][j]= new Float_t[8];
      nSB1pi[i][j]= new Float_t[8];
      nSB1piAir[i][j]= new Float_t[8];
      nSBOther[i][j]= new Float_t[8];
    }
  }

  for(int i=0; i<nToys; i++){
    for(int j=0; j<nAnaBins; j++){
      for(int k=0; k<8; k++){
	nSig[i][j][k]=0.0;
	nSigAir[i][j][k]=0.0;
	nSB1pi[i][j][k]=0.0;
	nSB1piAir[i][j][k]=0.0;
	nSBOther[i][j][k]=0.0;
      }
    }
  }


  int DiffBin=-1;
  int anaDiffBin=-1;
  Float_t weight=1.0;

  for(int iEntry=0; iEntry<nEntriesDefault; iEntry++){

    if(!(iEntry%(nEntriesDefault/100))) cout << "On Event: " << iEntry << "/" << nEntriesDefault << " " << iEntry*100./nEntriesDefault << "%" << endl;

    inTdefault->GetEntry(iEntry);
    DiffBin=-1;
    //DiffBin=GetDiffBin(selmu_amom, selmu_costheta, nPBins, nCosBins, PBinsPass, CosBinsPass);
    DiffBin=GetDiffBin(truelepton_mom, truelepton_costheta, nPBins, nCosBins, PBinsPass, CosBinsPass);
    if(DiffBin==0) anaDiffBin=0;
    else if(DiffBin>2 && DiffBin<5) anaDiffBin=DiffBin-2;
    else if(DiffBin>5 && DiffBin<24) anaDiffBin=DiffBin-3;
    else if(DiffBin==24) anaDiffBin=20;
    else continue;


    //toy loop
    for(int iToy=0; iToy<nToys; iToy++){
      weight=1.0;
      for(int iKnob=nKnobsStart; iKnob<nKnobs; iKnob++){
	//if(WeightsMatrixTrue(iKnob,iToy)<0 || WeightsMatrixTrue(iKnob,iToy)>1.8) continue;
	if(WeightsMatrix(iKnob,iToy)<0 || WeightsMatrix(iKnob,iToy)>10.) continue;
	weight*=WeightsMatrix(iKnob,iToy);
      }

      //Cuts:
      //Selection and SBs
      if(!(IsAntiNu)){//Nu
	CutsSig = accum_level>2 && PIDnShowers<1 && TrkChargeRatio[1]>1.5 && TagMichel2<1;
	CutsSB1pi = accum_level>2 && PIDnShowers<1 && TrkChargeRatio[2]>2.0 && TagMichel2==1 && !(TrkChargeRatio[1]>1.5 && TagMichel2<1);
	CutsSBOther = accum_level>2 && PIDnShowers<1 && !(TrkChargeRatio[2]>2.0 && TagMichel2==1 && !(TrkChargeRatio[1]>1.5 && TagMichel2<1)) && !(TrkChargeRatio[1]>1.5 && TagMichel2<1);
      }
      else{//AntiNu
	CutsSig = accum_level>4 && PIDnShowers<1 && TrkChargeRatio[1]>1.5;
	CutsSB1pi = accum_level>4 && PIDnShowers<1 && TrkChargeRatio[2]>1.5 && !(TrkChargeRatio[1]>1.5);
	CutsSBOther = accum_level>4 && PIDnShowers<1 && !(TrkChargeRatio[2]>1.5 && !(TrkChargeRatio[1]>1.5)) && !(TrkChargeRatio[1]>1.5);
      }



      //increment event vectors
      //cout << iToy << " " << DiffBin << " " << anaDiffBin << " " << topology << endl;
      if(topology>=0 && topology<8){
	//cout << "topology is: " << topology << endl;
	if(CutsSig) nSig[iToy][anaDiffBin][topology]+=weight;
	if(CutsSB1pi) nSB1pi[iToy][anaDiffBin][topology]+=weight;
	if(CutsSBOther) nSBOther[iToy][anaDiffBin][topology]+=weight;
      }
    }//toy loop
  }//event loop

  //Air
  for(int iEntry=0; iEntry<nEntriesDefaultAir; iEntry++){

    if(!(iEntry%(nEntriesDefaultAir/100))) cout << "On Event: " << iEntry << "/" << nEntriesDefaultAir << " " << iEntry*100./nEntriesDefaultAir << "%" << endl;

    inTdefaultAir->GetEntry(iEntry);
    DiffBin=-1;
    //DiffBin=GetDiffBin(selmu_amom, selmu_costheta, nPBins, nCosBins, PBinsPass, CosBinsPass);
    DiffBin=GetDiffBin(truelepton_mom, truelepton_costheta, nPBins, nCosBins, PBinsPass, CosBinsPass);
    if(DiffBin==0) anaDiffBin=0;
    else if(DiffBin>2 && DiffBin<5) anaDiffBin=DiffBin-2;
    else if(DiffBin>5 && DiffBin<24) anaDiffBin=DiffBin-3;
    else if(DiffBin==24) anaDiffBin=20;
    else continue;


   //toy loop
    for(int iToy=0; iToy<nToys; iToy++){
      weight=1.0;
      for(int iKnob=nKnobsStart; iKnob<nKnobs; iKnob++){
	//if(WeightsMatrixTrue(iKnob,iToy)<0 || WeightsMatrixTrue(iKnob,iToy)>1.8) continue;
	if(WeightsMatrix(iKnob,iToy)<0 || WeightsMatrix(iKnob,iToy)>10.) continue;
	weight*=WeightsMatrix(iKnob,iToy);
      }

      //Cuts:
      //Selection and SBs
      if(!(IsAntiNu)){//Nu
	CutsSig = accum_level>2 && PIDnShowers<1 && TrkChargeRatio[1]>1.5 && TagMichel2<1;
	CutsSB1pi = accum_level>2 && PIDnShowers<1 && TrkChargeRatio[2]>2.0 && TagMichel2==1 && !(TrkChargeRatio[1]>1.5 && TagMichel2<1);
	CutsSBOther = accum_level>2 && PIDnShowers<1 && !(TrkChargeRatio[2]>2.0 && TagMichel2==1 && !(TrkChargeRatio[1]>1.5 && TagMichel2<1)) && !(TrkChargeRatio[1]>1.5 && TagMichel2<1);
      }
      else{//AntiNu
	CutsSig = accum_level>4 && PIDnShowers<1 && TrkChargeRatio[1]>1.5;
	CutsSB1pi = accum_level>4 && PIDnShowers<1 && TrkChargeRatio[2]>1.5 && !(TrkChargeRatio[1]>1.5);
	CutsSBOther = accum_level>4 && PIDnShowers<1 && !(TrkChargeRatio[2]>1.5 && !(TrkChargeRatio[1]>1.5)) && !(TrkChargeRatio[1]>1.5);
      }



      //increment event vectors
      if(topology>=0 && topology<8){
	//cout << "topology is: " << topology << endl;
	if(CutsSig) nSigAir[iToy][anaDiffBin][topology]+=weight;
	if(CutsSB1pi) nSB1piAir[iToy][anaDiffBin][topology]+=weight;
	//if(CutsSBOther) nSBOther[iToy][anaDiffBin][topology]+=weight;
      }
    }//toy loop
  }//event loop




  //cut on topo, tell to used either Sig, SB1pi, or SBOther
  //0: sig, 1: SB1pi, 2: SBOther

  //added from genResponse


  //need to loop over topology and reaction?  look at output files stephen uses, email him if not clear

  int nTopologies=4;
  int nReacs=8;

  //string outname = Form("topo%d_reac%d.root",topo,reac);
  string outputFileName = "./output/resFunc"+dialName+"allVar.root";
  cout << "Output file name is: " << outputFileName << endl; 

  const int nWeights = 7;

  //cout << "Total number of bins is: " << globalCount << endl;
  cout << "Total number of bins is: " << nAnaBins << endl;
  //Write TGraph in the output file
  TFile *output = new TFile(outputFileName.c_str(),"RECREATE");   

  //TGraph *ReWeight[nTotalBins][nTotalBins][nTopologies];
  //TGraph *ReWeight[nAnaBins][nTopologies];
  TGraph *ReWeightTemp;

  double MA[7];
  for(int w = 0; w < 7; w++){
    //MA[w]=bestfit-error*(3-w);
    if(MeansAndSigmas[dialIndex][1]>0){
      MA[w]=(MeansAndSigmas[dialIndex][1]-MeansAndSigmas[dialIndex][2]*(3-w))/MeansAndSigmas[dialIndex][1];
    }
    else{
      MA[w]=1.+MeansAndSigmas[dialIndex][1]-MeansAndSigmas[dialIndex][2]*(3-w);
    }
  }  


  for(int t=1; t<=nTopologies; t++){  
    output->mkdir(Form("topology_%d",t));
  }
  char dir[200];


  for(int iBranch=1; iBranch<=nTopologies; iBranch++){
    for(int iReac=0; iReac<nReacs; iReac++){
      for(int bt = 0; bt < nAnaBins; bt++){//true kinematics bin
	reac=iReac;
	topo=iBranch;
	//cout << "On true bin and reco bin: " << br << " and " << bt << endl;
	cout << "On true bin " << bt << endl;
	sprintf(dir,"topology_%d",topo);
	output->cd(dir);
	char nameHisto[256];
	//sprintf(nameHisto,"RecBin_%d_trueBin_%d_topology_%d_reac_%d",br,bt,t,r);
	sprintf(nameHisto,"trueBin_%d_topology_%d_reac_%d",bt,topo,reac);

	ReWeightTemp = new TGraph(7);
	ReWeightTemp->SetName(nameHisto);
	ReWeightTemp->SetTitle(nameHisto);
	ReWeightTemp->SetMarkerStyle(20);
	ReWeightTemp->SetMarkerColor(2);

	for(int w=0;w<nWeights;w++){
	  if(topo==1){
	    if(nSig[3][bt][reac]>0) ReWeightTemp->SetPoint(w,MA[w],nSig[w][bt][reac]/nSig[3][bt][reac]);
	    else ReWeightTemp->SetPoint(w,MA[w],1);
	  }
	  else if(topo==2){
	    if(nSB1pi[3][bt][reac]>0) ReWeightTemp->SetPoint(w,MA[w],nSB1pi[w][bt][reac]/nSB1pi[3][bt][reac]);
	    else ReWeightTemp->SetPoint(w,MA[w],1);
	  }
	  else if(topo==3){
	    if(nSigAir[3][bt][reac]>0) ReWeightTemp->SetPoint(w,MA[w],nSigAir[w][bt][reac]/nSigAir[3][bt][reac]);
	    else ReWeightTemp->SetPoint(w,MA[w],1);
	  }
	  else if(topo==4){
	    if(nSB1piAir[3][bt][reac]>0) ReWeightTemp->SetPoint(w,MA[w],nSB1piAir[w][bt][reac]/nSB1piAir[3][bt][reac]);
	    else ReWeightTemp->SetPoint(w,MA[w],1);
	  }
	  //if(nSig[3][bt][reac]>0 ){
	    //if(topo==1) ReWeightTemp->SetPoint(w,MA[w],nSig[w][bt][reac]/nSig[3][bt][reac]);
	    //if(topo==2) ReWeightTemp->SetPoint(w,MA[w],nSB1pi[w][bt][reac]/nSB1pi[3][bt][reac]);
	    //if(topo==3) ReWeightTemp->SetPoint(w,MA[w],nSigAir[w][bt][reac]/nSigAir[3][bt][reac]);
	    //if(topo==4) ReWeightTemp->SetPoint(w,MA[w],nSB1piAir[w][bt][reac]/nSB1piAir[3][bt][reac]);
	  //}
	  //else{
	    //cout << "No events in true bin " << endl;
	    //ReWeightTemp->SetPoint(w,MA[w],1);
	  //}
	  ReWeightTemp->GetYaxis()->SetTitle("weight");
	}
	ReWeightTemp->Write();
      }  
    }//reaction (topology) loop
  }//branches loop
  output->Close();
  return 0;
}


// Print the cmd line syntax
void Usage(){
  cout << "Cmd line syntax should be:" << endl;
  //cout << "genWeightsFromJReWeight.exe -n nd_inputfile -s sk_inputfile [-e nevents]" << endl;
  cout << "process****.exe -n inputfileFromT2Krw -a IsAntiNu -t Branch -r HLreac -i dialIndex -d dialName" << endl;
}

// Messy way to process cmd line arguments.
void ParseArgs(int argc, char **argv){
  int nargs = 1; 
  if(argc<(nargs*2+1)){ Usage(); exit(1); }
  for(int i = 1; i < argc; i+=2){
    if(string(argv[i]) == "-n") inFStr = argv[i+1];
    else if(string(argv[i]) == "-a") IsAntiNu = std::atoi(argv[i+1]);
    else if(string(argv[i]) == "-t") topo = std::atoi(argv[i+1]);
    else if(string(argv[i]) == "-r") reac = std::atoi(argv[i+1]);
    else if(string(argv[i]) == "-i") dialIndex = std::atoi(argv[i+1]);
    else if(string(argv[i]) == "-d") dialName = argv[i+1];
    else {  
      cout << "Invalid argument:" << argv[i] << " "<< argv[i+1] << endl;
      Usage();
      exit(1);
    }
  } 
}
