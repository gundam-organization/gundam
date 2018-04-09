#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <list>
#include <functional>
#include <numeric>
#include <TClonesArray.h>
#include <TRefArray.h>
#include <TMath.h>

#include <TSystem.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TF1.h>
#include <TH1.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TFile.h>
#include <TChain.h>
#include <TClonesArray.h>
#include <TTree.h>
#include "TDirectory.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TString.h"
#include <TLegend.h>
#include "TLine.h"
#include "TMatrixDSym.h"

#define DEBUG


void calcCovMat_fine(TString inFileName, TString outFileName, TString varName, Int_t ntoys)
{
    // You need to provide the number of branches in your HL2 tree
    // And the accum_level you want to cut each one at to get your selected events
    // i.e choosing n in accum_level[0][branch_i]>n
    const int nbranches = 8;
    const int nusedbranches = 6;//IGNORING 1 AND 4

    //const int accumToCut[nbranches] =   {8,8,9,8,8,5,4,7,8,7}; // should ignore branches 0 and 4 (muon only)
    const int accumToCut[nbranches] =   {8,8,9,8,8,5,4,7}; // should ignore branches 0 and 4 (muon only)

    TFile* inFile = new TFile(inFileName);
    TFile* outFile = new TFile(outFileName, "RECREATE");

    TTree* systTree = (TTree*) inFile->Get("all_syst");
    TTree* nomTree = (TTree*) inFile->Get("default");

    Float_t* bins;
    Int_t nbinsn;
    if(varName=="recDpT"){
      nbinsn = 9;
      Float_t bins[9+1] = { 0.0, 0.08, 0.12, 0.155, 0.2, 0.26, 0.36, 0.51, 1.1, 100.0};
    }
    else if(varName=="recDphiT"){
      nbinsn = 8;
      Float_t bins[8+1] = { 0.0, 0.067, 0.14, 0.225, 0.34, 0.52, 0.85, 1.5, 3.14159};
    }
    else if(varName=="recDalphaT"){
      nbinsn = 8;
      Float_t bins[8+1] = { 0.0, 0.47, 1.02, 1.54, 1.98, 2.34, 2.64, 2.89, 3.14159};
    }
    else{
      cout << "ERROR: Unrecognised variable: " << varName << endl;
      return;
    }
    const Int_t nbins = nbinsn;
    cout << "Using "<< nbins << " bins" << endl;


    TMatrixDSym covar(nbins*(nusedbranches)); //IGNORING 1 AND 4
    TMatrixDSym covar_norm(nbins*(nusedbranches)); //IGNORING 1 AND 4


    vector<TH1D*> nomhists;
    vector<TH1D*> toyhists;

    TString nomhname, toyhname;
    TString cutString, drawString;

    Int_t brCnt=0;

    for(Int_t h=0; h<nbranches; h++){
        if((h==4)||(h==0)) continue;
        cout << "on branch " << h << "used branch" << brCnt << endl;
        nomhname.Form("nomhist%d",brCnt);
        drawString.Form(">>nomhist%d",brCnt);
        cutString.Form("accum_level[0][%d]>%d",h,accumToCut[h]);
        TH1D* nomhist = new TH1D(nomhname, nomhname, nbins, bins);
        cout << " cutting on: " << cutString << endl; 
        nomTree->Draw(varName+drawString,cutString);
        cout << "drew histo" << endl;
        #ifdef DEBUG 
                std::cout << "Nom hist first bin is: " << nomhist->GetBinContent(1) << std::endl; 
                nomhist->Print();
        #endif
        nomhists.push_back(nomhist);
        cout << "pushed back histo" << endl;
        #ifdef DEBUG 
                std::cout << "Nom hist first bin is: " << nomhists[0]->GetBinContent(1) << std::endl; 
                nomhists[0]->Print();
        #endif
        brCnt++;

    }

    brCnt=0;


    for(Int_t t=0; t<ntoys; t++){
        if((t+1)%10 == 0) std::cout << "Processing toy: " << (t+1) << " out of " << ntoys << std::endl;
        #ifdef DEBUG 
                std::cout << "Processing toy: " << (t+1) << " out of " << ntoys << std::endl; 
        #endif
        for(Int_t h=0; h<nbranches; h++){
            if((h==4)||(h==0)) continue;
            cout << "*** on branch " << h << " used branch " << brCnt << endl;
            toyhname.Form("toyhist%d",brCnt);
            cutString.Form("accum_level[%d][%d]>%d",t,h,accumToCut[h]);
            drawString.Form("[%d]>>toyhist%d",t,brCnt);
            TH1D* toyhist = new TH1D(toyhname, toyhname, nbins, bins);
            cout << " Drawing " << varName+drawString << ", " << cutString<< endl; 
            systTree->Draw(varName+drawString,cutString);
            cout << "drew histo" << endl;
            toyhists.push_back(toyhist);       
            cout << "pushed back histo" << endl; 
            #ifdef DEBUG 
                    std::cout << "toy hist first bin is: " << toyhist->GetBinContent(1) << std::endl; 
                    toyhist->Print();
            #endif
            brCnt++;

        }
            for(Int_t ibr=0; ibr<nusedbranches; ibr++){
                for(Int_t jbr=0; jbr<nusedbranches; jbr++){
                    for(Int_t i=0; i<nbins; i++){
                        for(Int_t j=0; j<nbins; j++){
                            Int_t iindex = (nbins*ibr)+i; 
                            Int_t jindex = (nbins*jbr)+j; 
                            covar[iindex][jindex]+=(toyhists[ibr]->GetBinContent(i+1)-nomhists[ibr]->GetBinContent(i+1))*(toyhists[jbr]->GetBinContent(j+1)-nomhists[jbr]->GetBinContent(j+1)); 
                            covar_norm[iindex][jindex]+=covar[iindex][jindex]/(nomhists[ibr]->GetBinContent(i+1)*nomhists[jbr]->GetBinContent(j+1)); 
                            #ifdef DEBUG 
                                std::cout << "covar element" << iindex << "," << jindex <<" now is " << covar[iindex][jindex] << std::endl; 
                                std::cout << "Divisor is: " << (nomhists[ibr]->GetBinContent(i+1)*nomhists[jbr]->GetBinContent(j+1)) << std::endl;                                 
                                std::cout << "covar_norm element" << iindex << "," << jindex <<" now is " << covar_norm[iindex][jindex] << std::endl; 
                            #endif
                        }
                    }
                }
            }
            toyhists.clear();
            brCnt=0;
        }

    covar*=1.0/(Float_t)ntoys;
    covar_norm*=1.0/(Float_t)ntoys;
    covar.Print();
    covar_norm.Print();
    inFile->Close();
    outFile->cd();
    covar.Write("covMat");
    covar_norm.Write("covMat_norm");
    outFile->Close();

  
}