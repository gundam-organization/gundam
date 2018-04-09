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

//#define DEBUG

using namespace std;


void calcCovMat_eff(TString inFileName, TString outFileName, TString varName, const Int_t ntoys, Int_t toyToProcess=-1)
{
    // You need to provide the number of branches in your HL2 tree
    // And the accum_level you want to cut each one at to get your selected events
    // i.e choosing n in accum_level[0][branch_i]>n
    const int nbranches = 8;
    const int nusedbranches = 6;//IGNORING 1 AND 4 (and now 8 and 9)
    //const int nusedbranches = 1;//IGNORING 1 AND 4 (and now 8 and 9)

    //const int accumToCut[nbranches] =   {8,8,9,8,8,5,4,7,8,7}; // should ignore branches 0 and 4 (muon only)
    const int accumToCut[nbranches] =   {8,8,9,8,8,5,4,7}; // should ignore branches 0 and 4 (muon only)

    TFile* inFile = new TFile(inFileName);

    TTree* systTree = (TTree*) inFile->Get("all_syst");
    TTree* nomTree = (TTree*) inFile->Get("default");

    Float_t* bins;
    Int_t nbinsn;
    if(varName=="trueDpT"){
      cout << "Using dpT"<< endl;
      nbinsn = 8;
      Float_t binsn[8+1] = { 0.0, 0.08, 0.12, 0.155, 0.2, 0.26, 0.36, 0.51, 100.0};
      bins = binsn;
    }
    else if(varName=="trueDphiT"){
      cout << "Using dphiT"<< endl;
      nbinsn = 8;
      Float_t binsn[8+1] = { 0.0, 0.067, 0.14, 0.225, 0.34, 0.52, 0.85, 1.5, 3.14159};
      bins = binsn;
    }
    else if(varName=="trueDalphaT"){
      cout << "Using dalphaT"<< endl;
      nbinsn = 8;
      Float_t binsn[8+1] = { 0.0, 0.47, 1.02, 1.54, 1.98, 2.34, 2.64, 2.89, 3.14159};
      bins = binsn;
    }
    else{
      cout << "ERROR: Unrecognised variable: " << varName << endl;
      return;
    }

    cout << "Chosen IPS binning: " << endl;
    for(int i=0;i<nbinsn;i++)cout << "Bin " << i+1 << " is from " << bins[i] << " to " << bins[i+1] << endl;

    Int_t noopsbins = 0;

    const Int_t nbins = nbinsn+noopsbins;
    cout << "Using "<< nbins << " bins" << endl;

    TString psCut = "((selp_truemom>450)&&(selp_truemom<1000)&&(truelepton_mom>250)&&((truelepton_costheta)>-0.6)&&(cos(selp_trueztheta)>0.4)&&((mectopology==1)||(mectopology==2)))";
    TString oopsCut  = "((selp_truemom<450)||(selp_truemom>1000)||(truelepton_mom<250)||((truelepton_costheta)<-0.6)||(cos(selp_trueztheta)<0.4))";

    TMatrixDSym covar(nbins); //IGNORING 1 AND 4
    TMatrixDSym covar_mean(nbins); //IGNORING 1 AND 4
    TMatrixDSym covar_norm(nbins); //IGNORING 1 AND 4
    TMatrixDSym covar_mean_norm(nbins); //IGNORING 1 AND 4

    TH1D* binSpreadHist[10];
    for(int b=0;b<(nbins);b++){
      binSpreadHist[b] = new TH1D(Form("binSpreadHist%d_distribution", b), Form("binSpreadHist%d_distribution", b), 11000, -1000, 10000); 
    }


    vector<TH1D*> nomhists;
    vector<TH1D*> toyhists[ntoys+1];

    TH1D* nomhistSum = new TH1D("nomHistSum", "nomHistSum", nbins, 0, nbins);
    TH1D* toyhistSum[ntoys+1];

    for(int t=0;t<(ntoys+1);t++){
      toyhistSum[t] = new TH1D(Form("toyhistSum%d", t), Form("toyhistSum%d", t), nbins, 0, nbins);
    }

    TString nomhname, toyhname;
    TString ipsnomhname, ipstoyhname;
    TString oopsnomhname, oopstoyhname;
    TString cutString, drawString, weightstring, systweightstring;
    TString ipsdrawString, oopsdrawString;

    bool IsSB;

    Int_t brCnt=0;

    for(Int_t h=0; h<nbranches; h++){
        if((h==4)||(h==0)) continue;

        IsSB = false;

        cout << "on branch " << h << " used branch " << brCnt << endl;
        cutString.Form("accum_level[][%d]>%d",h,accumToCut[h]);

        ipsnomhname.Form("ipsnomhist%d",brCnt);
        ipsdrawString.Form(">>ipsnomhist%d",brCnt);
        TH1D* ipsnomhist = new TH1D(ipsnomhname, ipsnomhname, nbinsn, bins);
        cout << " cutting on: " << cutString << endl; 
        if(!IsSB) systTree->Draw(varName+ipsdrawString,"(weight_corr_total[]/NTOYS)*(" + cutString + " && " + psCut + ")");
        else if(IsSB) systTree->Draw(varName+ipsdrawString,"(weight_corr_total[]/NTOYS)*(" + cutString + ")");
        cout << "drew ips histo" << endl;
        ipsnomhist->Print("all");

        nomhname.Form("nomhist%d",brCnt);
        TH1D* nomhist = new TH1D(nomhname, nomhname, nbins, 0, nbins);
        for(Int_t b=1; b<(nbins+1); b++){
            nomhist->SetBinContent(b,ipsnomhist->GetBinContent(b));
        }
        cout << "Formed nom histo: " << endl;
        nomhist->Print("all");



        #ifdef DEBUG 
                std::cout << "Nom hist first bin is: " << nomhist->GetBinContent(1) << std::endl; 
                nomhist->Print();
        #endif                

        nomhists.push_back(nomhist);
        nomhistSum->Add(nomhist);

        cout << "pushed back and summed histo" << endl;
        cout << "Sum is now: " << endl;
        nomhistSum->Print("all");

        #ifdef DEBUG 
                std::cout << "Nom hist first bin is: " << nomhists[0]->GetBinContent(1) << std::endl; 
                nomhists[0]->Print();
        #endif

        brCnt++;

    }

    brCnt=0;


    for(Int_t t=0; t<ntoys; t++){
        if(toyToProcess!=-1 && t!=toyToProcess) continue;
        else if (t==toyToProcess) cout << "Running in para mode: processing toy " << t << endl;

        if((t+1)%10 == 0) std::cout << "Processing toy: " << (t+1) << " out of " << ntoys << std::endl;
        #ifdef DEBUG 
                std::cout << "Processing toy: " << (t+1) << " out of " << ntoys << std::endl; 
        #endif
        for(Int_t h=0; h<nbranches; h++){
            if((h==4)||(h==0)) continue;

            IsSB = false;

            #ifdef DEBUG 
              cout << "*** on branch " << h << " used branch " << brCnt << endl;
            #endif
            //Apply all weight syts then remove the flux (the flux is fit seperately)
            systweightstring.Form("(weight_syst_total[%d])*",t);       
            weightstring.Form("(weight_corr_total[%d])*(",t);       
            cutString.Form("accum_level[%d][%d]>%d",t,h,accumToCut[h]);

            ipstoyhname.Form("ipstoyhist%d",brCnt);
            ipsdrawString.Form("[%d]>>ipstoyhist%d",t,brCnt);
            TH1D* ipstoyhist = new TH1D(ipstoyhname, ipstoyhname, nbinsn, bins);
            //cout << " Drawing " << varName+ipsdrawString << ", " << cutString<< endl; 
            if(!IsSB) systTree->Draw(varName+ipsdrawString, systweightstring + weightstring + cutString + " && " + psCut + ")");
            else if(IsSB) systTree->Draw(varName+ipsdrawString, systweightstring + weightstring + cutString + ")");
            //cout << "drew histo" << endl;

            toyhname.Form("toyhist%d",brCnt);
            TH1D* toyhist = new TH1D(toyhname, toyhname, nbins, 0, nbins);
            for(Int_t b=1; b<(nbins+1); b++){
                toyhist->SetBinContent(b,ipstoyhist->GetBinContent(b));
            }


            toyhists[t].push_back(toyhist);       
            toyhistSum[t]->Add(toyhist);       

            #ifdef DEBUG 
                    cout << "pushed back histo" << endl;
                    toyhist->Print("all"); 
                    std::cout << "toy hist first bin is: " << toyhist->GetBinContent(1) << std::endl; 
                    toyhist->Print();
                    cout << "weightstring is: " << weightstring << endl;
                    cout << "systweightstring is: " << systweightstring << endl;
            #endif
            brCnt++;

        }

        for(Int_t i=0; i<nbins; i++){
            for(Int_t j=0; j<nbins; j++){
                if(i==j) binSpreadHist[i]->Fill(toyhistSum[t]->GetBinContent(i+1)); 
                covar[i][j]+=(toyhistSum[t]->GetBinContent(i+1)-nomhistSum->GetBinContent(i+1))*(toyhistSum[t]->GetBinContent(j+1)-nomhistSum->GetBinContent(j+1)); 
                #ifdef DEBUG 
                    std::cout << "covar element" << i << "," << j <<" now is " << covar[i][j] << std::endl; 
                    std::cout << "Divisor is: " << (nomhists[i]->GetBinContent(i+1)*nomhists[j]->GetBinContent(j+1)) << std::endl;                                 
                #endif
            }
        }

        //toyhists.clear();
        brCnt=0;
    }

    //Running in paralell mode
    if(toyToProcess!=-1){
        TFile* outFilePara = new TFile(Form("paraOut_%d.root",toyToProcess), "RECREATE");
        for(int b=0;b<(nbins);b++){    
          binSpreadHist[b]->Write();
        }
        for(Int_t ibr=0; ibr<nusedbranches; ibr++){
            toyhistSum[toyToProcess]->Write();
            nomhistSum->Write();
        }
        return;
    }
    for(Int_t t=0; t<ntoys; t++){
        for(Int_t i=0; i<nbins; i++){
            for(Int_t j=0; j<nbins; j++){
                covar_mean[i][j]+=(toyhistSum[t]->GetBinContent(i+1)-binSpreadHist[i]->GetMean())*(toyhistSum[t]->GetBinContent(j+1)-binSpreadHist[j]->GetMean()); 
            }
        }
    }


    covar*=1.0/(Float_t)ntoys;
    covar_mean*=1.0/(Float_t)ntoys;
    //covar_norm*=1.0/(Float_t)ntoys;

    for(Int_t i=0; i<nbins; i++){
        for(Int_t j=0; j<nbins; j++){
            covar_norm[i][j]+=covar[i][j]/(nomhistSum->GetBinContent(i+1)*nomhistSum->GetBinContent(j+1)); 
            covar_mean_norm[i][j]+=covar_mean[i][j]/(binSpreadHist[i]->GetMean()*binSpreadHist[j]->GetMean()); 
        }
    }


    TFile* outFile = new TFile(outFileName, "RECREATE");

    covar.Print();
    covar_norm.Print();

    outFile->cd();
    covar.Write("covMat");
    covar_norm.Write("covMat_norm");

    covar_mean.Write("covMat_mean");
    covar_mean_norm.Write("covMat_mean_norm");

    cout << "Writting nomhistSum:" << endl;

    nomhistSum->Print("all");
    nomhistSum->Write("nomhistSum");

    cout << "Writting nomhist from each branch:" << endl;
    for(Int_t ibr=0; ibr<nusedbranches; ibr++){ nomhists[ibr]->Write(); }

    cout << "Writting bin spread from each bin:" << endl;
    for(int b=0;b<(nbins);b++){
      cout << "writting bin " << b << endl; 
      binSpreadHist[b]->Print();
      binSpreadHist[b]->Write(); 
    }
    outFile->Close();
    inFile->Close();

}