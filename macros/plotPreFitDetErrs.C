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


void plotPreFitDetErrors(TString inFileName, TString outFileName)
{
    // You need to provide the number of branches in your HL2 tree
    // And the accum_level you want to cut each one at to get your selected events
    // i.e choosing n in accum_level[0][branch_i]>n
    const int nbranches = 8;


    TFile* inFile = new TFile(inFileName);
    TFile* outFile = new TFile(outFileName, "RECREATE");

    TMatrixDSym* covmat = (TMatrixDSym*)inFile->Get("covMat_norm");
    if(!covmat) {cout << "Failed to read matrix"; return;}


    //const Int_t nbins = 8;
    const Int_t nbins = 4;
    if(nbins!=((covmat->GetNcols())/nbranches)) { cout << "Binning mismatch"; return;}
    //Float_t bins[nbins+1] = { 0.0, 0.08, 0.12, 0.155, 0.2, 0.26, 0.36, 0.51, 1.1};
    Float_t bins[nbins+1] = {0.0, 0.12, 0.2, 0.36, 1.1};

    TH1D* mutpcptpc = new TH1D("mutpcptpc","mutpcptpc",nbins,bins);
    TH1D* mutpcpfgd = new TH1D("mutpcpfgd","mutpcpfgd",nbins,bins);
    TH1D* mufgdptpc = new TH1D("mufgdptpc","mufgdptpc",nbins,bins);
    TH1D* onepi = new TH1D("onepi","onepi",nbins,bins);
    TH1D* pip = new TH1D("pip","pip",nbins,bins);
    TH1D* mutpcnp = new TH1D("mutpcnp","mutpcnp",nbins,bins);
    TH1D* mufgdptpcnp = new TH1D("mufgdptpcnp","mufgdptpcnp",nbins,bins);
    TH1D* mufgdnp = new TH1D("mufgdnp","mufgdnp",nbins,bins);

    for(int i=0;i<nbins;i++){
        mutpcptpc->SetBinContent(i+1, sqrt((*covmat)[i][i]));
        mutpcpfgd->SetBinContent(i+1, sqrt((*covmat)[(nbins)+i][(nbins)+i]));
        mufgdptpc->SetBinContent(i+1, sqrt((*covmat)[(2*nbins)+i][(2*nbins)+i]));
        onepi->SetBinContent(i+1, sqrt((*covmat)[(3*nbins)+i][(3*nbins)+i]));
        pip->SetBinContent(i+1, sqrt((*covmat)[(4*nbins)+i][(4*nbins)+i]));
        mutpcnp->SetBinContent(i+1, sqrt((*covmat)[(5*nbins)+i][(5*nbins)+i]));
        mufgdptpcnp->SetBinContent(i+1, sqrt((*covmat)[(6*nbins)+i][(6*nbins)+i]));
        mufgdnp->SetBinContent(i+1, sqrt((*covmat)[(7*nbins)+i][(7*nbins)+i]));
    }

    mutpcptpc->GetYaxis()->SetRangeUser(0,0.5);
    mutpcptpc->SetLineColor(kBlack);
    mutpcptpc->SetLineWidth(2);
    mutpcpfgd->SetLineColor(kRed);
    mutpcpfgd->SetLineWidth(2);
    mufgdptpc->SetLineColor(kGreen);
    mufgdptpc->SetLineWidth(2);
    onepi->SetLineColor(kOrange);
    onepi->SetLineStyle(2);
    onepi->SetLineWidth(2);
    pip->SetLineColor(kMagenta);
    pip->SetLineStyle(2);
    pip->SetLineWidth(2);
    mutpcnp->SetLineColor(kBlue);
    mutpcnp->SetLineWidth(2);

    inFile->Close();
    outFile->cd();
    TCanvas* canv = new TCanvas("detsyst","detsyst");
    mutpcptpc->Draw();
    mutpcpfgd->Draw("same");
    mufgdptpc->Draw("same");
    onepi->Draw("same");
    pip->Draw("same");
    mutpcnp->Draw("same");
    canv->Write();
    mutpcptpc->Write("mutpcptpc");
    mutpcpfgd->Write("mutpcpfgd");
    mufgdptpc->Write("mufgdptpc");
    onepi->Write("onepi");
    pip->Write("pip");
    mutpcnp->Write("mutpcnp");
    mufgdptpcnp->Write("mufgdptpcnp");
    mufgdnp->Write("mufgdnp");
    outFile->Close();

  
}