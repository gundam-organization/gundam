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
#include <TGraph.h>
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
#include "TSystemDirectory.h"
#include "TList.h"

using namespace std;

void buildLcurve(const char * outFileName="lcurveOut.root",  const char *dirname="./", const char *ext=".root")
{
  TSystemDirectory dir(dirname, dirname);
  TList *files = dir.GetListOfFiles();
  int fileCount=0;
  int finalIter=0;
  Double_t reg[100] = {0};
  Double_t reg2[100] = {0};
  Double_t chi2reg[100] = {0};
  Double_t chi2reg2[100] = {0};
  Double_t chi2[100] = {0};
  Double_t chi2rnorm[100] = {0};
  Double_t chi2rnorm2[100] = {0};
  Double_t chi2rnormtot[100] = {0};
  Double_t lgrad[100] = {0};
  Double_t ldgrad[100] = {0};
  if(files){
    TSystemFile *file;
    TString fname;
    TIter next(files);
    while ((file=(TSystemFile*)next())) {
      fname = file->GetName();
      if(!file->IsDirectory() && fname.EndsWith(ext) && fname!=outFileName && fname!="noreg.root") {
        cout << fname.Data() << endl;
        TFile* inFile = new TFile(fname);
        if(!inFile) continue;
        TH1D* chi2_stat_periter = (TH1D*)inFile->Get("chi2_stat_periter");
        TH1D* chi2_sys_periter = (TH1D*)inFile->Get("chi2_sys_periter");
        TH1D* chi2_reg_periter = (TH1D*)inFile->Get("chi2_reg_periter");
        TH1D* chi2_reg2_periter = (TH1D*)inFile->Get("chi2_reg2_periter");
        TH1D* chi2_tot_periter = (TH1D*)inFile->Get("chi2_tot_periter");
        TH1D* reg_param = (TH1D*)inFile->Get("reg_param");
        TH1D* reg_param2 = (TH1D*)inFile->Get("reg_param2");
        if(chi2_stat_periter && chi2_sys_periter && chi2_reg_periter && chi2_tot_periter && reg_param) cout << "Chi2 Info Found" << endl;
        else{
          cout << "Cannot find chi2 info in current file, skipping" << endl;
          continue;
        }
        finalIter = (chi2_stat_periter->GetNbinsX())-1;
        reg[fileCount]=reg_param->GetMean();
        reg2[fileCount]=reg_param2->GetMean();
        chi2reg[fileCount]=chi2_reg_periter->GetBinContent(finalIter);
        chi2reg2[fileCount]=chi2_reg2_periter->GetBinContent(finalIter);
        chi2[fileCount]=chi2_tot_periter->GetBinContent(finalIter);
        chi2rnorm[fileCount]=chi2reg[fileCount]/reg[fileCount];
        chi2rnorm2[fileCount]=chi2reg2[fileCount]/reg2[fileCount];
        chi2rnormtot[fileCount]=chi2rnorm[fileCount]+chi2rnorm2[fileCount];
        if(fileCount!=0) lgrad[fileCount]  = (chi2rnorm[fileCount]-chi2rnorm[fileCount-1])/(chi2[fileCount]-chi2[fileCount-1]);
        if(fileCount>1)  ldgrad[fileCount] = (lgrad[fileCount]-lgrad[fileCount-1])/(chi2[fileCount]-chi2[fileCount-1]);
        cout << "Info for current file: " << endl;
        cout << "Reg param1 is: " << reg[fileCount] << ", Reg param2 is: " << reg2[fileCount] << endl;
        cout << "Final iter = " << finalIter << endl;
        cout << "chi2reg = " << chi2reg[fileCount] << ", chi2reg2 = " << chi2reg2[fileCount] << endl;
        cout << "chi2 tot = " << chi2[fileCount] << ", chi2rnorm = " << chi2rnorm[fileCount] << endl;

        inFile->Close();
        fileCount++;
        cout << fileCount << endl;
        if(fileCount>99){
          cout << "Warning, too many files, will need to increase array size." << endl;
          return;
        }
      }
    }
  }
  TFile* outFile = new TFile(outFileName, "RECREATE");
  outFile->cd();

  TCanvas* c1 = new TCanvas();
  c1->cd();


  TGraph* lc = new TGraph(fileCount, chi2, chi2rnorm);
  lc->SetMarkerColor(kBlack);
  lc->SetMarkerStyle(3);
  TGraph* comp1 = new TGraph(fileCount, reg, chi2);
  comp1->SetMarkerColor(kRed);
  comp1->SetMarkerStyle(4);
  TGraph* comp2 = new TGraph(fileCount, reg, chi2rnorm);
  comp2->SetMarkerColor(kBlue);
  comp2->SetMarkerStyle(25);
  TGraph* comp3 = new TGraph(fileCount, reg, chi2reg);
  comp1->SetMarkerColor(kRed);
  comp1->SetMarkerStyle(4);

  TGraph* lc2 = new TGraph(fileCount, chi2, chi2rnorm2);
  lc2->SetMarkerColor(kBlack);
  lc2->SetMarkerStyle(3);
  TGraph* comp12 = new TGraph(fileCount, reg2, chi2);
  comp12->SetMarkerColor(kRed);
  comp12->SetMarkerStyle(4);
  TGraph* comp22 = new TGraph(fileCount, reg2, chi2rnorm2);
  comp22->SetMarkerColor(kBlue);
  comp22->SetMarkerStyle(25);
  TGraph* comp32 = new TGraph(fileCount, reg2, chi2reg2);
  comp22->SetMarkerColor(kBlue);
  comp22->SetMarkerStyle(25);

  TGraph* lccomb = new TGraph(fileCount, chi2, chi2rnormtot);
  lccomb->SetMarkerColor(kBlack);
  lccomb->SetMarkerStyle(3);
  TGraph* compcomb1 = new TGraph(fileCount, reg, chi2rnormtot);
  compcomb1->SetMarkerColor(kBlue);
  compcomb1->SetMarkerStyle(25);
  TGraph* compcomb2 = new TGraph(fileCount, reg2, chi2rnormtot);
  compcomb2->SetMarkerColor(kGreen);
  compcomb2->SetMarkerStyle(4);


  // TGraph* lcgrad = new TGraph(fileCount, chi2, lgrad);
  // lcgrad->SetMarkerColor(kBlack);
  // lcgrad->SetMarkerStyle(3);
  // TGraph* lcdgrad = new TGraph(fileCount, chi2, ldgrad);
  // lcdgrad->SetMarkerColor(kBlack);
  // lcdgrad->SetMarkerStyle(3);
  // TGraph* lcgrad_preg = new TGraph(fileCount, reg, lgrad);
  // lcgrad->SetMarkerColor(kBlack);
  // lcgrad->SetMarkerStyle(3);
  // TGraph* lcdgrad_preg = new TGraph(fileCount, reg, ldgrad);
  // lcdgrad->SetMarkerColor(kBlack);
  // lcdgrad->SetMarkerStyle(3);


  comp1->Write("chi2vsreg1");
  comp2->Write("chi2regnormvsreg1");
  comp3->Write("chi2reg1vsreg1");
  comp12->Write("chi2vsreg2");
  comp22->Write("chi2regnormvsreg2");
  comp32->Write("chi2reg2vsreg2");
  compcomb1->Write("chi2regnormTotvsreg1");
  compcomb2->Write("chi2regnormTotvsreg2");

  TCanvas *canvasComp = new TCanvas("canvasComp1", "canvasComp1");
  comp1->Draw("AP");
  comp1->GetYaxis()->SetTitle("#\chi^{2} contribution");
  comp1->GetYaxis()->SetRangeUser(1,1100);
  comp1->GetXaxis()->SetTitle("IPS Regularisation Parameter");
  comp1->SetTitle("");
  comp2->Draw("Psame");
  canvasComp->SetTickx(); canvasComp->SetTicky();
  canvasComp->SetLogx(); canvasComp->SetLogy();
  canvasComp->SetTitle("");
  canvasComp->Write();  

  TCanvas *canvasComp2 = new TCanvas("canvasComp2", "canvasComp2");
  comp12->Draw("AP");
  comp12->GetYaxis()->SetTitle("#\chi^{2} contribution");
  comp12->GetYaxis()->SetRangeUser(1,1100);
  comp12->GetXaxis()->SetTitle("OOFV Regularisation Parameter");
  comp12->SetTitle("");
  comp22->Draw("Psame");
  canvasComp2->SetTickx(); canvasComp2->SetTicky();
  canvasComp2->SetLogx(); canvasComp2->SetLogy();
  canvasComp2->SetTitle("");
  canvasComp2->Write();


  TCanvas *canvasL = new TCanvas("canvasL","canvasL");
  lc->Draw("AP");
  lc->GetYaxis()->SetTitle("IPS Normalised Penalty");
  lc->GetXaxis()->SetTitle("#\chi^{2} of Fit");
  lc->SetTitle("");
  lc->Write("lcurve1");
  canvasL->SetTickx(); canvasL->SetTicky();
  canvasL->SetTitle("");
  canvasL->Write();

  TCanvas *canvasL2 = new TCanvas("canvasL2","canvasL2");
  lc2->Draw("AP");
  lc2->GetYaxis()->SetTitle("OOPS Normalised Penalty");
  lc2->GetXaxis()->SetTitle("#\chi^{2} of Fit");
  lc2->SetTitle("");
  lc2->Write("lcurve2");
  canvasL2->SetTickx(); canvasL2->SetTicky();
  canvasL2->SetTitle("");
  canvasL2->Write();

  TCanvas *canvasLcomb = new TCanvas("canvasL3","canvasL3");
  lccomb->Draw("AC*");
  //lccomb->GetYaxis()->SetTitle("Combined Normalised Penalty");
  lccomb->GetYaxis()->SetTitle("Normalised Penalty");
  lccomb->GetXaxis()->SetTitle("-2log(L) of Fit");
  lccomb->SetTitle("");
  lccomb->Write("lcurvecomb");
  canvasLcomb->SetTickx(); canvasLcomb->SetTicky();
  canvasLcomb->SetTitle("");
  canvasLcomb->Write();
  canvasLcomb->SaveAs("lCurve.png");

  TCanvas *canvasLcomb_noC = new TCanvas("canvasL3_noC","canvasL3_noC");
  lccomb->Draw("A*");
  //lccomb->GetYaxis()->SetTitle("Combined Normalised Penalty");
  lccomb->GetYaxis()->SetTitle("Normalised Penalty");
  lccomb->GetXaxis()->SetTitle("-2log(L) of Fit");
  lccomb->SetTitle("");
  canvasLcomb_noC->SetTickx(); canvasLcomb_noC->SetTicky();
  canvasLcomb_noC->SetTitle("");
  canvasLcomb_noC->Write();
  canvasLcomb_noC->SaveAs("lCurve_noC.png");




  // lcgrad->Write("lcgrad");
  // lcdgrad->Write("lcdgrad");
  // lcgrad_preg->Write("lcgrad_preg");
  // lcdgrad_preg->Write("lcdgrad_preg");



  //outFile->Close();
}

