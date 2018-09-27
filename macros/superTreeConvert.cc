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
#include <TRandom3.h>

using namespace std;

bool extraCuts=false;
bool psdim=false;

// specialRWmode:
//                1 - No 2p2h
//                2 - 2X 2p2h
//                3 - 3X 2p2h
//                4 - Martini Nieves Ratio RW (in E_nu)

// Example:
// treeConvert("/data/t2k/dolan/MECProcessing/CC0Pi/mar15HL2/job_NeutWaterNoSystV2_out/allMerged.root", "default", "truth", "./NeutWater2_2D_2X2p2h_dataStats.root", 1, 0, 0, 0.1664, 2, "recDpT", "trueDpT", "trueDpT", "recDalphaT", "trueDalphaT", "trueDalphaT")

int superTreeConvert(
        TString nd5fileName, TString nd5treeName, TString nd5treeName_T,
        TString nd2fileName, TString nd2treeName, TString nd2treeName_T,
        TString outFileName, Float_t sigWeight, Int_t EvtStart, Int_t EvtEnd, Double_t EvtFrac, Int_t specialRWmode,
        TString D1NameRec, TString D1NameTrue, TString D1NameTrue_T,
        TString D2NameRec, TString D2NameTrue, TString D2NameTrue_T, bool isRealData=false, bool useAltPp=false)
{
    // You need to provide the number of branches in your HL2 tree
    // And the accum_level you want to cut each one at to get your selected events
    // i.e choosing n in accum_level[0][branch_i]>n
    const int nbranches = 11;
    const int accumToCut[nbranches] =   {7,8,9,8,7,5,4,7,8,7,1};

    TFile *nd5file = new TFile(nd5fileName);
    TTree *nd5tree = (TTree*)nd5file->Get(nd5treeName);
    TTree *nd5tree_T = (TTree*)nd5file->Get(nd5treeName_T);

    TFile *nd2file = new TFile(nd2fileName);
    TTree *nd2tree = (TTree*)nd2file->Get(nd2treeName);
    TTree *nd2tree_T = (TTree*)nd2file->Get(nd2treeName_T);

    TFile *outfile = new TFile(outFileName,"recreate");
    TTree *outtree = new TTree("selectedEvents", "selectedEvents");
    TTree *outtree_T = new TTree("trueEvents", "trueEvents");

    char const * xslf_env = getenv("XSLLHFITTER");
    if(!xslf_env){
        std::cerr << "[ERROR]: environment variable \"XSLLHFITTER\" not set. "
            "Cannot determine source tree location." << std::endl;
        return 1;
    }

    // Declaration of leaf types
    Int_t          accum_level[1500][50];
    Int_t          nutype;
    Int_t          reaction;
    Int_t          cutBranch=-999;
    Int_t          mectopology;
    Float_t        D1True;
    Float_t        D2True;
    Float_t        D1Reco;
    Float_t        D2Reco;
    Float_t        pMomRec;
    Float_t        pMomRecRange;
    Float_t        pThetaRec;
    Float_t        pMomTrue;
    Float_t        pThetaTrue;
    Float_t        pMomTrueAlt;
    Float_t        pThetaTrueAlt;
    Float_t        pCosThetaTrueAlt;
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
    Float_t        weightIn[2];

    Int_t          nutype_T;
    Int_t          reaction_T;
    Int_t          cutBranch_T=-999;
    Int_t          mectopology_T;
    Float_t        D1True_T;
    Float_t        D2True_T;
    Float_t        muMomTrue_T;
    Float_t        pMomTrue_T;
    Float_t        muCosThetaTrue_T;
    Float_t        pCosThetaTrue_T;
    Float_t        TrueNuEnergy_T;
    Float_t        weight_T;
    Float_t        weightIn_T[2];

    nd5tree->SetBranchAddress("accum_level", &accum_level);
    nd5tree->SetBranchAddress("nutype", &nutype);
    nd5tree->SetBranchAddress("reaction", &reaction);
    nd5tree->SetBranchAddress("mectopology", &mectopology);
    //nd5tree->SetBranchAddress(D1NameTrue, &D1True);
    //nd5tree->SetBranchAddress(D2NameTrue, &D2True);
    //nd5tree->SetBranchAddress(D1NameRec, &D1Reco);
    //nd5tree->SetBranchAddress(D2NameRec, &D2Reco);
    nd5tree->SetBranchAddress("selp_mom", &pMomRec);
    nd5tree->SetBranchAddress("selp_mom_range_oarecon", &pMomRecRange);
    nd5tree->SetBranchAddress("selp_costheta" ,&pCosThetaRec);
    nd5tree->SetBranchAddress("truep_truemom" ,&pMomTrue);
    nd5tree->SetBranchAddress("truep_truecostheta" ,&pCosThetaTrue);
    nd5tree->SetBranchAddress("selp_truemom" ,&pMomTrueAlt);
    nd5tree->SetBranchAddress("selp_trueztheta" ,&pThetaTrueAlt);
    nd5tree->SetBranchAddress("selmu_mom", &muMomRec);
    nd5tree->SetBranchAddress("selmu_mom_range_oarecon", &muMomRecRange);
    nd5tree->SetBranchAddress("selmu_costheta", &muCosThetaRec);
    nd5tree->SetBranchAddress("truelepton_mom", &muMomTrue);
    nd5tree->SetBranchAddress("truelepton_costheta", &muCosThetaTrue);
    nd5tree->SetBranchAddress("nu_trueE", &TrueNuEnergy);
    nd5tree->SetBranchAddress("weight_syst_total", &weight);

    outtree->Branch("nutype", &nutype, "nutype/I");
    outtree->Branch("reaction", &reaction, "reaction/I");
    outtree->Branch("cutBranch", &cutBranch, "cutBranch/I");
    outtree->Branch("mectopology", &mectopology, "mectopology/I");
    outtree->Branch("D1True", &D1True, ("D1True/F"));
    outtree->Branch("D1Rec", &D1Reco, ("D1Rec/F"));
    outtree->Branch("D2True", &D2True, ("D2True/F"));
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

    nd5tree_T->SetBranchAddress("nutype", &nutype_T);
    nd5tree_T->SetBranchAddress("reaction", &reaction_T);
    nd5tree_T->SetBranchAddress("mectopology", &mectopology_T);
    //nd5tree_T->SetBranchAddress(D1NameTrue_T, &D1True_T);
    //nd5tree_T->SetBranchAddress(D2NameTrue_T, &D2True_T);
    nd5tree_T->SetBranchAddress("truehm_proton_truemom" ,&pMomTrue_T);
    nd5tree_T->SetBranchAddress("truehm_proton_truecth" ,&pCosThetaTrue_T);
    nd5tree_T->SetBranchAddress("truelepton_mom", &muMomTrue_T);
    nd5tree_T->SetBranchAddress("truelepton_costheta", &muCosThetaTrue_T);
    nd5tree_T->SetBranchAddress("nu_trueE", &TrueNuEnergy_T);
    nd5tree_T->SetBranchAddress("weight", &weight_T);
    //nd5tree_T->SetBranchAddress("weight", &weightIn_T);

    outtree_T->Branch("nutype", &nutype_T, "nutype/I");
    outtree_T->Branch("reaction", &reaction_T, "reaction/I");
    outtree_T->Branch("cutBranch", &cutBranch_T, "cutBranch/I");
    outtree_T->Branch("mectopology", &mectopology_T, "mectopology/I");
    outtree_T->Branch("D1True", &D1True_T, ("D1True/F"));
    outtree_T->Branch("D2True", &D2True_T, ("D2True/F"));
    outtree_T->Branch("muMomTrue", &muMomTrue_T, ("muMomTrue/F"));
    outtree_T->Branch("muCosThetaTrue", &muCosThetaTrue_T, ("muCosThetaTrue/F"));
    outtree_T->Branch("pMomTrue", &pMomTrue_T, ("pMomTrue/F"));
    outtree_T->Branch("pCosThetaTrue", &pCosThetaTrue_T, ("pCosThetaTrue/F"));
    outtree_T->Branch("Enutrue", &TrueNuEnergy_T, "Enutrue/F");
    outtree_T->Branch("weight", &weight_T, "weight/F");

    Float_t muang;
    Float_t pang;
    Float_t opening;
    Float_t nuE;
    Float_t range;
    Int_t fileIndex;
    Int_t inttype;
    Int_t mupdg;
    Int_t ppdg;
    Int_t npioncount;
    Int_t nprotoncount;

    TRandom3* rng = new TRandom3();

    nd2tree->SetBranchAddress("muang", &muang);
    nd2tree->SetBranchAddress("pang", &pang);
    nd2tree->SetBranchAddress("opening", &opening);
    nd2tree->SetBranchAddress("nuE", &nuE);
    nd2tree->SetBranchAddress("fileIndex", &fileIndex);
    nd2tree->SetBranchAddress("inttype", &inttype);
    nd2tree->SetBranchAddress("mupdg", &mupdg);
    nd2tree->SetBranchAddress("ppdg", &ppdg);
    nd2tree->SetBranchAddress("range", &range);
    nd2tree->SetBranchAddress("npioncount", &npioncount);
    nd2tree->SetBranchAddress("nprotoncount", &nprotoncount);

    Long64_t nentries = nd5tree -> GetEntriesFast();
    Long64_t nbytes = 0, nb = 0;

    if(EvtEnd!=0) nentries=EvtEnd;
    if(EvtFrac>0.0001) nentries=nentries*EvtFrac;

    int passCount=0;
    for (Long64_t jentry=EvtStart; jentry < nentries; jentry++) {
        nb = nd5tree->GetEntry(jentry);
        nbytes += nb;
        passCount = 0;
        RecoNuEnergy = TrueNuEnergy;
        weight = 1.0;
        //pCosThetaRec  = TMath::Cos(pThetaRec);
        //muCosThetaRec = TMath::Cos(muThetaRec);

        //D1True = TMath::Cos(TMath::ACos(muCosThetaTrue) - TMath::ACos(pCosThetaTrue));
        //D1Reco = TMath::Cos(TMath::ACos(muCosThetaRec) - TMath::ACos(pCosThetaRec));
        D1True = muMomTrue;
        D1Reco = muMomRec;
        D2True = muCosThetaTrue;
        D2Reco = muCosThetaRec;

        if(mectopology == 1 || mectopology == 2)
            weight = 1.0;

        if(useAltPp){
            pMomTrue=pMomTrueAlt;
            pCosThetaTrue=TMath::Cos(pThetaTrueAlt);
        }

        int branches_passed[10]={0};
        for(int i=0; i < nbranches; i++){
            if(accum_level[0][i] > accumToCut[i]){
                cutBranch=i; passCount++;
                if(cutBranch==2) pMomRec=pMomRecRange;
                if(cutBranch==3) muMomRec=muMomRecRange;
                branches_passed[i]++;
                if((( (mectopology==1)||(mectopology==2) ) && ( (pMomTrue>450)&&(muMomTrue>250)&&(muCosThetaTrue>-0.6)&&(pCosThetaTrue>0.4) ))){
                    weight = 1.0; // weight*sigWeight;
                }
                if(psdim){
                    //Phase Space Bins:
                    //printf("INFO: Applying phase space binning as specified\n");
                    const int npmombins=3;
                    const double pmombins[npmombins+1] = {0.0, 450.0, 1000.0, 100000.0};
                    const int npthetabins=2;
                    const double pthetabins[npthetabins+1] = {-1.0, 0.4, 1.0};
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
                                        D2True=globalCount;
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
                    //printf("INFO: Total number of bins is %d \n", globalCount);
                    if(i==5 || i==6){ // Don't bother with ps binning for the SB
                        D2True=16;
                        D2Reco=16;
                        //cout << "ps binning reset since using SB" << endl;
                    }
                    if(!isRealData && (pMomTrue<0.0001 || muMomTrue<0.0001 || pMomRec < 0.0001 || muMomRec < 0.0001)){ // Don't bother with ps binning for bad events
                        D2True=16;
                        D2Reco=16;
                        //cout << "ps binning reset since invalid event" << endl;
                    }
                    if(isRealData && (pMomRec < 0.0001 || muMomRec < 0.0001)){ // Don't bother with ps binning for bad events
                        D2True=16;
                        D2Reco=16;
                        //cout << "ps binning reset since invalid event" << endl;
                    }
                    //To maintain nice bin ordering put signal at the start
                    if(D2True==16) D2True=0;
                    else D2True=2;
                    if(D2Reco==16) D2Reco=0;
                    else D2Reco=2;

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

    Long64_t nentries_T = nd5tree_T->GetEntriesFast();
    Long64_t nbytes_T = 0, nb_T = 0;
    if(EvtFrac>0.0001) nentries_T=nentries_T*EvtFrac;

    for (Long64_t jentry=0; jentry<nentries_T;jentry++) {
        nb_T = nd5tree_T->GetEntry(jentry);
        nbytes_T += nb_T;
        weight_T = 1.0; // 1.0

        //D1True_T = TMath::Cos(TMath::ACos(muCosThetaTrue_T) - TMath::ACos(pCosThetaTrue_T));
        D1True_T = muMomTrue_T;
        D2True_T = muCosThetaTrue_T;
        cutBranch_T = -1;
        outtree_T->Fill();
    }

    nentries = nd2tree -> GetEntriesFast();
    nbytes = 0;
    nb = 0;

    if(EvtEnd!=0) nentries=EvtEnd;
    if(EvtFrac>0.0001) nentries=nentries*EvtFrac;

    for(Long64_t jentry=2000000; jentry < 2030000; jentry++) {
        nb = nd2tree->GetEntry(jentry);
        nbytes += nb;
        if(fileIndex != 1)
            continue;

        RecoNuEnergy = nuE * 1000;
        TrueNuEnergy = nuE * 1000;

        muMomRec = 0;
        muMomTrue = 0;
        pMomRec = 0;
        pMomTrue = 0;
        weight = 1.0;
        nutype = 14;
        cutBranch = 10;

        if(inttype == 1)
            reaction = 0;
        else if(inttype == 2)
            reaction = 9;
        else if(inttype == 11 || inttype == 12 || inttype == 13)
            reaction = 1;
        else if(inttype == 21 || inttype == 26)
            reaction = 2;
        else if(inttype == 16)
            reaction = 3;
        else if(inttype > 30 && inttype < 47)
            reaction = 4;
        else if(inttype < 0)
            reaction = 5;
        else
            reaction = -1;

        if(fileIndex == 1 && npioncount == 0 && mupdg == 13 && ppdg == 2212 && nprotoncount == 1)
        {
            mectopology = 1;
            weight = 1.0;
        }
        else if(fileIndex == 1 && npioncount == 0 && mupdg == 13 && ppdg == 2212 && nprotoncount > 1)
        {
            mectopology = 2;
            weight = 1.0;
        }
        else if(fileIndex == 1 && npioncount == 1 && mupdg == 13 && ppdg == 211 && nprotoncount == 0)
            mectopology = 3;
        //else if(fileIndex == 1 && npioncount > 0 && mupdg == 13)
        //    mectopology = 4;
        else
            continue;

        //std::cout << "Muon Angle: " << muang << std::endl;
        //std::cout << "fileIndex: " << fileIndex << std::endl;
        //std::cout << "npion: " << npioncount << std::endl;
        //std::cout << "mupdg: " << mupdg << std::endl;
        //std::cout << "ppdg: " << ppdg << std::endl;
        //std::cout << "nuE: " << nuE << std::endl;

        double muang_true = TMath::DegToRad() * muang;
        double pang_true = TMath::DegToRad() * pang;
        double muang_reco = muang_true * rng -> Gaus(1, 0.1);
        double pang_reco = pang_true * rng -> Gaus(1, 0.1);

        muCosThetaRec = TMath::Cos(muang_reco);
        muCosThetaTrue = TMath::Cos(muang_true);

        pCosThetaRec = TMath::Cos(pang_reco);
        pCosThetaTrue = TMath::Cos(pang_true);

        double mu_mom_true = range * 0.0114127 + 0.230608;
        double mu_mom_reco = (range * rng -> Gaus(1, 0.1)) * 0.0114127 + 0.230608;

        muMomTrue = mu_mom_true * 1000.0;
        muMomRec = mu_mom_reco * 1000.0;

        //D1True = TMath::Cos(muang_true - pang_true);
        //D1Reco = TMath::Cos(muang_reco - pang_reco);
        D1True = muMomTrue;
        D1Reco = muMomRec;
        D2True = TMath::Cos(muang_true);
        D2Reco = TMath::Cos(muang_reco);

        outtree -> Fill();

        D1True_T = D1True;
        D2True_T = D2True;
        TrueNuEnergy_T = TrueNuEnergy;
        cutBranch_T = -2;
        nutype_T = 14;
        mectopology_T = mectopology;
        weight_T = weight;
        outtree_T -> Fill();
    }

    printf("***Output Rec Tree: ***\n");
    outtree->Print();
    printf("***Output True Tree: ***\n");
    outtree_T->Print();
    outfile->Write();

    delete nd5file;
    delete outfile;
    return 0;
}
