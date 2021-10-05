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


void calcCovMat_fine_v2(TString inFileName, TString outFileName, TString varName, const Int_t ntoys, Int_t toyToProcess=-1)
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
    if(varName=="recDpT"){
      cout << "Using dpT"<< endl;
      nbinsn = 8;
      Float_t binsn[8+1] = { 0.0, 0.08, 0.12, 0.155, 0.2, 0.26, 0.36, 0.51, 100.0};
      bins = binsn;
    }
    else if(varName=="recDphiT"){
      cout << "Using dphiT"<< endl;
      nbinsn = 8;
      Float_t binsn[8+1] = { 0.0, 0.067, 0.14, 0.225, 0.34, 0.52, 0.85, 1.5, 3.14159};
      bins = binsn;
    }
    else if(varName=="recDalphaT"){
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

    Int_t noopsbins = 1;

    const Int_t nbins = nbinsn+noopsbins;
    cout << "Using "<< nbins << " bins" << endl;

    TString psCut, oopsCut;

    TString psCut_pmuTPC = "((selp_mom>450)&&(selp_mom<1000)&&(selmu_mom>250)&&((selmu_dir[2])>-0.6)&&(cos(selp_theta)>0.4))";
    TString oopsCut_pmuTPC = "((selp_mom<450)||(selp_mom>1000)||(selmu_mom<250)||((selmu_dir[2])<-0.6)||(cos(selp_theta)<0.4))";

    TString psCut_pFGD = "((selp_mom_range_oarecon>450)&&(selp_mom_range_oarecon<1000)&&(selmu_mom>250)&&((selmu_dir[2])>-0.6)&&(cos(selp_theta)>0.4))";
    TString oopsCut_pFGD = "((selp_mom_range_oarecon<450)||(selp_mom_range_oarecon>1000)||(selmu_mom<250)||((selmu_dir[2])<-0.6)||(cos(selp_theta)<0.4))";

    TString psCut_muFGD = "((selp_mom>450)&&(selp_mom<1000)&&(selmu_mom_range_oarecon>250)&&((selmu_dir[2])>-0.6)&&(cos(selp_theta)>0.4))";
    TString oopsCut_muFGD = "((selp_mom<450)||(selp_mom>1000)||(selmu_mom_range_oarecon<250)||((selmu_dir[2])<-0.6)||(cos(selp_theta)<0.4))";

    TMatrixDSym covar(nbins*(nusedbranches)); //IGNORING 1 AND 4
    TMatrixDSym covar_mean(nbins*(nusedbranches)); //IGNORING 1 AND 4
    TMatrixDSym covar_norm(nbins*(nusedbranches)); //IGNORING 1 AND 4
    TMatrixDSym covar_mean_norm(nbins*(nusedbranches)); //IGNORING 1 AND 4

    TH1D* binSpreadHist[200];
    for(int b=0;b<(nbins*(nusedbranches));b++){
      binSpreadHist[b] = new TH1D(Form("binSpreadHist%d_distribution", b), Form("binSpreadHist%d_distribution", b), 11000, -1000, 10000); 
    }


    vector<TH1D*> nomhists;
    vector<TH1D*> toyhists[ntoys+1];

    TString nomhname, toyhname;
    TString ipsnomhname, ipstoyhname;
    TString oopsnomhname, oopstoyhname;
    TString cutString, drawString, weightstring, systweightstring;
    TString ipsdrawString, oopsdrawString;

    bool IsSB;

    Int_t brCnt=0;

    for(Int_t h=0; h<nbranches; h++){
        if((h==4)||(h==0)) continue;

        // Don't bother with OOPS binning for SB
        if((h==5)||(h==6)) IsSB = true;
        else IsSB = false;

        switch(h){
            case 2:
                psCut = psCut_pFGD;
                oopsCut = oopsCut_pFGD;
                break;
            case 3:
                psCut = psCut_muFGD;
                oopsCut = oopsCut_muFGD;
                break;
            default:
                psCut = psCut_pmuTPC;
                oopsCut = oopsCut_pmuTPC;
        }

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

        oopsnomhname.Form("oopsnomhist%d",brCnt);
        oopsdrawString.Form(">>oopsnomhist%d",brCnt);
        cout << "Diagnostics: " << nbinsn << endl;
        cout << " Making OOPS histo from var value " << bins[0] << " to " << bins[nbinsn] << endl; 
        TH1D* oopsnomhist = new TH1D(oopsnomhname, oopsnomhname, 1, bins[0], bins[nbinsn]);
        cout << " cutting on: " << cutString << endl; 
        if(!IsSB) systTree->Draw(varName+oopsdrawString,"(weight_corr_total[]/NTOYS)*(" + cutString + " && " + oopsCut + ")");
        else if(IsSB) for(int i=0;i<noopsbins;i++) oopsnomhist->SetBinContent(i+1,1.0); // Add dummy event to avoid covar chaos
        cout << "drew oops histo" << endl;
        oopsnomhist->Print("all");

        nomhname.Form("nomhist%d",brCnt);
        TH1D* nomhist = new TH1D(nomhname, nomhname, nbins, 0, nbins);
        for(Int_t b=1; b<(nbins+1); b++){
            if(b<(nbinsn+1)) nomhist->SetBinContent(b,ipsnomhist->GetBinContent(b));
            else nomhist->SetBinContent(b,oopsnomhist->GetBinContent(b-nbinsn));
        }
        cout << "Formed nom histo: " << endl;
        nomhist->Print("all");



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
        if(toyToProcess!=-1 && t!=toyToProcess) continue;
        else if (t==toyToProcess) cout << "Running in para mode: processing toy " << t << endl;

        if((t+1)%10 == 0) std::cout << "Processing toy: " << (t+1) << " out of " << ntoys << std::endl;
        #ifdef DEBUG 
                std::cout << "Processing toy: " << (t+1) << " out of " << ntoys << std::endl; 
        #endif
        psCut_pmuTPC = Form("((selp_mom[%d]>450)&&(selp_mom[%d]<1000)&&(selmu_mom[%d]>250)&&((selmu_dir[0])>-0.6)&&(cos(selp_theta[%d])>0.4))",t,t,t,t);
        oopsCut_pmuTPC = Form("((selp_mom[%d]<450)||(selp_mom[%d]>1000)||(selmu_mom[%d]<250)||((selmu_dir[0])<-0.6)||(cos(selp_theta[%d])<0.4))",t,t,t,t);

        psCut_pFGD = Form("((selp_mom_range_oarecon[%d]>450)&&(selp_mom_range_oarecon[%d]<1000)&&(selmu_mom[%d]>250)&&((selmu_dir[0])>-0.6)&&(cos(selp_theta[%d])>0.4))",t,t,t,t);
        oopsCut_pFGD = Form("((selp_mom_range_oarecon[%d]<450)||(selp_mom_range_oarecon[%d]>1000)||(selmu_mom[%d]<250)||((selmu_dir[0])<-0.6)||(cos(selp_theta[%d])<0.4))",t,t,t,t);

        psCut_muFGD = Form("((selp_mom[%d]>450)&&(selp_mom[%d]<1000)&&(selmu_mom_range_oarecon[%d]>250)&&((selmu_dir[0])>-0.6)&&(cos(selp_theta[%d])>0.4))",t,t,t,t);
        oopsCut_muFGD = Form("((selp_mom[%d]<450)||(selp_mom[%d]>1000)||(selmu_mom_range_oarecon[%d]<250)||((selmu_dir[0])<-0.6)||(cos(selp_theta[%d])<0.4))",t,t,t,t);
        for(Int_t h=0; h<nbranches; h++){
            if((h==4)||(h==0)) continue;

            // Don't bother with OOPS binning for SB
            if((h==5)||(h==6)) IsSB = true;
            else IsSB = false;

            switch(h){
                case 2:
                    psCut = psCut_pFGD;
                    oopsCut = oopsCut_pFGD;
                    break;
                case 3:
                    psCut = psCut_muFGD;
                    oopsCut = oopsCut_muFGD;
                    break;
                default:
                    psCut = psCut_pmuTPC;
                    oopsCut = oopsCut_pmuTPC;
            }

            #ifdef DEBUG 
              cout << "*** on branch " << h << " used branch " << brCnt << endl;
            #endif
            //Apply all weight syts then remove the flux (the flux is fit seperately)
            systweightstring.Form("(weight_syst_total[%d])*",t,t);       
            weightstring.Form("(weight_corr_total[%d])*(",t);       
            cutString.Form("accum_level[%d][%d]>%d",t,h,accumToCut[h]);

            ipstoyhname.Form("ipstoyhist%d",brCnt);
            ipsdrawString.Form("[%d]>>ipstoyhist%d",t,brCnt);
            TH1D* ipstoyhist = new TH1D(ipstoyhname, ipstoyhname, nbinsn, bins);
            //cout << " Drawing " << varName+ipsdrawString << ", " << cutString<< endl; 
            if(!IsSB) systTree->Draw(varName+ipsdrawString, systweightstring + weightstring + cutString + " && " + psCut + ")");
            else if(IsSB) systTree->Draw(varName+ipsdrawString, systweightstring + weightstring + cutString + ")");
            //cout << "drew histo" << endl;

            oopstoyhname.Form("oopstoyhist%d",brCnt);
            oopsdrawString.Form("[%d]>>oopstoyhist%d",t,brCnt);
            TH1D* oopstoyhist = new TH1D(oopstoyhname, oopstoyhname, 1, bins[0], bins[nbinsn]);
            if(!IsSB) systTree->Draw(varName+oopsdrawString,  systweightstring + weightstring + cutString + " && " + oopsCut + ")");
            else if(IsSB) for(int i=0;i<noopsbins;i++) oopstoyhist->SetBinContent(i+1,1.0); // Add dummy event to avoid covar chaos

            toyhname.Form("toyhist%d",brCnt);
            TH1D* toyhist = new TH1D(toyhname, toyhname, nbins, 0, nbins);
            for(Int_t b=1; b<(nbins+1); b++){
                if(b<(nbinsn+1)) toyhist->SetBinContent(b,ipstoyhist->GetBinContent(b));
                else toyhist->SetBinContent(b,oopstoyhist->GetBinContent(b-nbinsn));
            }


            toyhists[t].push_back(toyhist);       

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
        for(Int_t ibr=0; ibr<nusedbranches; ibr++){
            for(Int_t jbr=0; jbr<nusedbranches; jbr++){
                for(Int_t i=0; i<nbins; i++){
                    for(Int_t j=0; j<nbins; j++){
                        Int_t iindex = (nbins*ibr)+i; 
                        Int_t jindex = (nbins*jbr)+j; 
                        if(iindex==jindex) binSpreadHist[iindex]->Fill(toyhists[t][ibr]->GetBinContent(i+1)); 
                        covar[iindex][jindex]+=(toyhists[t][ibr]->GetBinContent(i+1)-nomhists[ibr]->GetBinContent(i+1))*(toyhists[t][jbr]->GetBinContent(j+1)-nomhists[jbr]->GetBinContent(j+1)); 
                        #ifdef DEBUG 
                            std::cout << "covar element" << iindex << "," << jindex <<" now is " << covar[iindex][jindex] << std::endl; 
                            std::cout << "Divisor is: " << (nomhists[ibr]->GetBinContent(i+1)*nomhists[jbr]->GetBinContent(j+1)) << std::endl;                                 
                        #endif
                    }
                }
            }
        }
        //toyhists.clear();
        brCnt=0;
    }

    //Running in paralell mode
    if(toyToProcess!=-1){
        TFile* outFilePara = new TFile(Form("paraOut_%d.root",toyToProcess), "RECREATE");
        for(int b=0;b<(nbins*(nusedbranches));b++){    
          binSpreadHist[b]->Write();
        }
        for(Int_t ibr=0; ibr<nusedbranches; ibr++){
            toyhists[toyToProcess][ibr]->Write();
            nomhists[ibr]->Write();
        }
        return;
    }

    for(Int_t t=0; t<ntoys; t++){
        for(Int_t ibr=0; ibr<nusedbranches; ibr++){
            for(Int_t jbr=0; jbr<nusedbranches; jbr++){
                for(Int_t i=0; i<nbins; i++){
                    for(Int_t j=0; j<nbins; j++){
                        Int_t iindex = (nbins*ibr)+i; 
                        Int_t jindex = (nbins*jbr)+j; 
                        covar_mean[iindex][jindex]+=(toyhists[t][ibr]->GetBinContent(i+1)-binSpreadHist[iindex]->GetMean())*(toyhists[t][jbr]->GetBinContent(j+1)-binSpreadHist[jindex]->GetMean()); 
                        #ifdef DEBUG 
                            std::cout << "covar element" << iindex << "," << jindex <<" now is " << covar[iindex][jindex] << std::endl; 
                            std::cout << "Divisor is: " << (nomhists[ibr]->GetBinContent(i+1)*nomhists[jbr]->GetBinContent(j+1)) << std::endl;                                 
                        #endif
                    }
                }
            }
        }
    }


    covar*=1.0/(Float_t)ntoys;
    covar_mean*=1.0/(Float_t)ntoys;
    //covar_norm*=1.0/(Float_t)ntoys;

    for(Int_t ibr=0; ibr<nusedbranches; ibr++){
        for(Int_t jbr=0; jbr<nusedbranches; jbr++){
            for(Int_t i=0; i<nbins; i++){
                for(Int_t j=0; j<nbins; j++){
                    Int_t iindex = (nbins*ibr)+i; 
                    Int_t jindex = (nbins*jbr)+j; 
                    covar_norm[iindex][jindex]+=covar[iindex][jindex]/(nomhists[ibr]->GetBinContent(i+1)*nomhists[jbr]->GetBinContent(j+1)); 
                    covar_mean_norm[iindex][jindex]+=covar_mean[iindex][jindex]/(binSpreadHist[iindex]->GetMean()*binSpreadHist[jindex]->GetMean()); 
                }
            }
        }
    }

    TFile* outFile = new TFile(outFileName, "RECREATE");

    covar.Print();
    covar_norm.Print();
    inFile->Close();
    outFile->cd();
    covar.Write("covMat");
    covar_norm.Write("covMat_norm");

    covar_mean.Write("covMat_mean");
    covar_mean_norm.Write("covMat_mean_norm");


    for(Int_t ibr=0; ibr<nusedbranches; ibr++){ nomhists[ibr]->Write(); }

    for(int b=0;b<(nbins*(nusedbranches));b++){
      binSpreadHist[b]->Write(); 
    }
    outFile->Close();

  
}