//
// Created by Adrien BLANCHET on 12/05/2022.
//

#include <TLegend.h>
#include "GundamGreetings.h"

#include "Logger.h"
#include "CmdLineParser.h"
#include "GenericToolbox.Root.h"

#include "nlohmann/json.hpp"
#include "TKey.h"
#include "TFile.h"

#include "string"
#include "vector"

LoggerInit([]{
  Logger::setUserHeaderStr("[gundamFitCompare.cxx]");
});

CmdLineParser clp;
TFile* outFile{nullptr};

void makeSampleComparePlots(bool usePrefit_);
void makeScanComparePlots(bool usePrefit_);
void makeErrorComparePlots(bool usePrefit_, bool useNomVal_);

int main( int argc, char** argv ){
  GundamGreetings g;
  g.setAppName("FitCompare");
  g.hello();

  clp.addCmdLineArgs(argc, argv);

  // files
  clp.addOption("file-1", {"-f1"}, "Path to first output fit file.", 1);
  clp.addOption("file-2", {"-f2"}, "Path to second output fit file.", 1);

  // display name
  clp.addOption("name-1", {"-n1"}, "Set display name of the first fit file.", 1);
  clp.addOption("name-2", {"-n2"}, "Set display name of the  fit file.", 1);

  // display name
  clp.addOption("algo-1", {"-a1"}, "Specify algo folder to compare for the first fit file.", 1);
  clp.addOption("algo-2", {"-a2"}, "Specify algo folder to compare for the second fit file.", 1);

  clp.addOption("output", {"-o"}, "Output file.", 1);

  LogInfo << "Options list:" << std::endl;
  Logger::setIndentStr("  ");
  LogInfo << clp.getConfigSummary() << std::endl;
  Logger::setIndentStr("");

  clp.parseCmdLine();

  if( clp.isNoOptionTriggered()
      or not clp.isOptionTriggered("file-1")
      or not clp.isOptionTriggered("file-2")
      or not clp.isOptionTriggered("output")
      ){
    LogError << "Missing options." << std::endl;
    exit(EXIT_FAILURE);
  }

  LogInfo << "Reading config..." << std::endl;

  auto outPath = clp.getOptionVal<std::string>("output");


  std::string strBuffer;

  outFile = TFile::Open(outPath.c_str(), "RECREATE");

  LogInfo << "Comparing preFit scans..." << std::endl;
  makeScanComparePlots(true);
  LogInfo << "Comparing postFit scans..." << std::endl;
  makeScanComparePlots(false);

  LogInfo << "Comparing preFit samples..." << std::endl;
  makeSampleComparePlots(true);
  LogInfo << "Comparing postFit samples..." << std::endl;
  makeSampleComparePlots(false);

  LogInfo << "Comparing preFit errors..." << std::endl;
  makeErrorComparePlots(true, false);
  LogInfo << "Comparing preFit (normalized) errors..." << std::endl;
  makeErrorComparePlots(true, true);
  LogInfo << "Comparing postFit errors..." << std::endl;
  makeErrorComparePlots(false, false);
  LogInfo << "Comparing postFit (normalized) errors..." << std::endl;
  makeErrorComparePlots(false, true);

  LogWarning << "Closing: " << outPath << std::endl;
  outFile->Close();

  g.goodbye();
  return EXIT_SUCCESS;
}


void makeSampleComparePlots(bool usePrefit_){
  auto filePath1 = clp.getOptionVal<std::string>("file-1");
  auto filePath2 = clp.getOptionVal<std::string>("file-2");

  auto name1 = clp.getOptionVal("name-1", filePath1);
  auto name2 = clp.getOptionVal("name-2", filePath2);

  auto* file1 = GenericToolbox::openExistingTFile(filePath1);
  auto* file2 = GenericToolbox::openExistingTFile(filePath2);

  std::string strBuffer;

  strBuffer = Form("FitterEngine/%s/samples", (usePrefit_? "preFit": "postFit"));
  auto* dir1 = file1->Get<TDirectory>(strBuffer.c_str());
  LogThrowIf(dir1== nullptr, "Could not find \"" << strBuffer << "\" within " << filePath1);

  auto* dir2 = file2->Get<TDirectory>(strBuffer.c_str());
  LogThrowIf(dir2== nullptr, "Could not find \"" << strBuffer << "\" within " << filePath2);

  std::vector<std::string> pathBuffer;
  pathBuffer.emplace_back(Form("%s/samples", (usePrefit_? "preFit": "postFit")));
  std::function<void(TDirectory* dir1_, TDirectory* dir2_)> recurseSampleCompareGraph;
  recurseSampleCompareGraph = [&](TDirectory* dir1_, TDirectory* dir2_){

    for( int iKey = 0 ; iKey < dir1_->GetListOfKeys()->GetEntries() ; iKey++ ){
      if( dir2_->Get(dir1_->GetListOfKeys()->At(iKey)->GetName()) == nullptr ) continue;
      TKey* keyObj = (TKey*) dir1_->GetListOfKeys()->At(iKey);


      if( (gROOT->GetClass( keyObj->GetClassName() ))->InheritsFrom("TDirectory") ){
        // recursive part
        pathBuffer.emplace_back( dir1_->GetListOfKeys()->At(iKey)->GetName() );

        LogTrace << GenericToolbox::joinVectorString(pathBuffer, "/") << std::endl;

        recurseSampleCompareGraph(
            dir1_->GetDirectory(dir1_->GetListOfKeys()->At(iKey)->GetName()),
            dir2_->GetDirectory(dir1_->GetListOfKeys()->At(iKey)->GetName())
        );

        pathBuffer.pop_back();
      }
      else if( (gROOT->GetClass( keyObj->GetClassName() ))->InheritsFrom("TH1") ){
        auto* h1 = dir1_->Get<TH1D>( dir1_->GetListOfKeys()->At(iKey)->GetName() );
        auto* h2 = dir2_->Get<TH1D>( dir1_->GetListOfKeys()->At(iKey)->GetName() );

        LogContinueIf(h1->GetNbinsX() != h2->GetNbinsX(), "");

        auto* hCompValues = (TH1D*) h1->Clone();
        hCompValues->Add(h2, -1);
        GenericToolbox::transformBinContent(hCompValues, [](TH1D* h_, int bin_){ h_->SetBinError(bin_, 0); });

        hCompValues->SetTitle(Form("Comparing \"%s\"", dir1_->GetListOfKeys()->At(iKey)->GetName()));
        hCompValues->GetYaxis()->SetTitle("Bin content difference");

        GenericToolbox::mkdirTFile(outFile, GenericToolbox::joinVectorString(pathBuffer, "/"))->cd();
        hCompValues->Write(dir1_->GetListOfKeys()->At(iKey)->GetName());
      }
    }

  };


  recurseSampleCompareGraph(dir1, dir2);

}
void makeScanComparePlots(bool usePrefit_){

  auto filePath1 = clp.getOptionVal<std::string>("file-1");
  auto filePath2 = clp.getOptionVal<std::string>("file-2");

  auto name1 = clp.getOptionVal("name-1", filePath1);
  auto name2 = clp.getOptionVal("name-2", filePath2);

  auto* file1 = GenericToolbox::openExistingTFile(filePath1);
  auto* file2 = GenericToolbox::openExistingTFile(filePath2);

  std::vector<std::string> pathBuffer;
  pathBuffer.emplace_back(Form("%s/scan", (usePrefit_? "preFit": "postFit")));
  std::function<void(TDirectory* dir1_, TDirectory* dir2_)> recurseScanCompareGraph;
  recurseScanCompareGraph = [&](TDirectory* dir1_, TDirectory* dir2_){

    for( int iKey = 0 ; iKey < dir1_->GetListOfKeys()->GetEntries() ; iKey++ ){
      if( dir2_->Get(dir1_->GetListOfKeys()->At(iKey)->GetName()) == nullptr ) continue;
      TKey* keyObj = (TKey*) dir1_->GetListOfKeys()->At(iKey);


      if( (gROOT->GetClass( keyObj->GetClassName() ))->InheritsFrom("TDirectory") ){
        // recursive part
        pathBuffer.emplace_back( dir1_->GetListOfKeys()->At(iKey)->GetName() );

        recurseScanCompareGraph(
            dir1_->GetDirectory(dir1_->GetListOfKeys()->At(iKey)->GetName()),
            dir2_->GetDirectory(dir1_->GetListOfKeys()->At(iKey)->GetName())
            );

        pathBuffer.pop_back();
      }
      else if( (gROOT->GetClass( keyObj->GetClassName() ))->InheritsFrom("TGraph") ){
        auto* gr1 = dir1_->Get<TGraph>( dir1_->GetListOfKeys()->At(iKey)->GetName() );
        auto* gr2 = dir2_->Get<TGraph>( dir1_->GetListOfKeys()->At(iKey)->GetName() );

        auto* overlayCanvas = new TCanvas( dir1_->GetListOfKeys()->At(iKey)->GetName() , Form("Comparing %s scan: \"%s\"", (usePrefit_? "preFit": "postFit"), dir1_->GetListOfKeys()->At(iKey)->GetName()), 800, 600);

        overlayCanvas->cd();

        gr1->SetMarkerStyle(kFullSquare);
        gr1->SetLineColor(kBlue-7);
        gr1->SetMarkerColor(kBlue-7);

        gr2->SetLineColor(kBlack);
        gr2->SetMarkerColor(kBlack);

        gr1->SetTitle( dir1_->GetListOfKeys()->At(iKey)->GetName() );

        std::pair<double, double> xBounds{
            std::min( gr1->GetXaxis()->GetXmin(), gr2->GetXaxis()->GetXmin() ),
            std::max( gr1->GetXaxis()->GetXmax(), gr2->GetXaxis()->GetXmax() )
        };
        std::pair<double, double> yBounds{
          std::min( gr1->GetMinimum(), gr2->GetMinimum() ),
          std::max( gr1->GetMaximum(), gr2->GetMaximum() )
        };

        gr1->GetXaxis()->SetLimits( xBounds.first - ( xBounds.second - xBounds.first )*0.1, xBounds.second + ( xBounds.second - xBounds.first )*0.1 );
        gr1->SetMinimum( yBounds.first - ( yBounds.second - yBounds.first )*0.1 ); // not working
        gr1->SetMaximum( yBounds.second + ( yBounds.second - yBounds.first )*0.2 ); // not working

        gr1->Draw();
        gr2->Draw("LPSAME");

        TLegend l(0.6, 0.79, 0.89, 0.89);
        l.AddEntry(gr1, Form("%s", name1.c_str()));
        l.AddEntry(gr2, Form("%s", name2.c_str()));
        l.Draw();

        gPad->SetGridx();
        gPad->SetGridy();

        GenericToolbox::mkdirTFile(outFile, GenericToolbox::joinVectorString(pathBuffer, "/"))->cd();
        overlayCanvas->Write();
        delete overlayCanvas;
      }
    }

  };

  std::string strBuffer;
  strBuffer = Form("FitterEngine/%s/scan", (usePrefit_? "preFit": "postFit"));
  auto* dir1 = file1->Get<TDirectory>(strBuffer.c_str());
  LogReturnIf(dir1== nullptr, "Could not find \"" << strBuffer << "\" within " << filePath1);
  auto* dir2 = file2->Get<TDirectory>(strBuffer.c_str());
  LogReturnIf(dir2== nullptr, "Could not find \"" << strBuffer << "\" within " << filePath2);

  recurseScanCompareGraph(dir1, dir2);
}
void makeErrorComparePlots(bool usePrefit_, bool useNomVal_) {

  auto filePath1 = clp.getOptionVal<std::string>("file-1");
  auto filePath2 = clp.getOptionVal<std::string>("file-2");

  auto algo1 = clp.getOptionVal("algo-1", "Hesse");
  auto algo2 = clp.getOptionVal("algo-2", "Hesse");

  auto name1 = clp.getOptionVal("name-1", filePath1);
  auto name2 = clp.getOptionVal("name-2", filePath2);

  auto* file1 = GenericToolbox::openExistingTFile(filePath1);
  auto* file2 = GenericToolbox::openExistingTFile(filePath2);


  std::string strBuffer;

  strBuffer = Form("FitterEngine/postFit/%s/errors", algo1.c_str());
  auto* dir1 = file1->Get<TDirectory>(strBuffer.c_str());
  LogReturnIf(dir1== nullptr, "Could not find \"" << strBuffer << "\" within " << filePath1);

  strBuffer = Form("FitterEngine/postFit/%s/errors", algo2.c_str());
  auto* dir2 = file2->Get<TDirectory>(strBuffer.c_str());
  LogReturnIf(dir2== nullptr, "Could not find \"" << strBuffer << "\" within " << filePath2);

  // loop over parSets
  auto* outDir = GenericToolbox::mkdirTFile(outFile, Form("%s/errors%s", (usePrefit_? "preFit": "postFit"), (useNomVal_? "Norm": "")));
  for( int iKey = 0 ; iKey < dir1->GetListOfKeys()->GetEntries() ; iKey++ ){
    Logger::setIndentStr("  ");
    std::string parSet = dir1->GetListOfKeys()->At(iKey)->GetName();

    strBuffer = Form("%s/values%s/%sErrors_TH1D", parSet.c_str(), (useNomVal_? "Norm": ""), (usePrefit_? "preFit": "postFit"));
    auto* hist1 = dir1->Get<TH1D>(strBuffer.c_str());
    LogContinueIf(hist1 == nullptr, "Could no find parSet \"" << strBuffer << "\" in " << file1->GetPath());

    strBuffer = Form("%s/values%s/%sErrors_TH1D", parSet.c_str(), (useNomVal_? "Norm": ""), (usePrefit_? "preFit": "postFit"));
    auto* hist2 = dir2->Get<TH1D>(strBuffer.c_str());
    LogContinueIf(hist2 == nullptr, "Could no find parSet \"" << strBuffer << "\" in " << file2->GetPath());

    LogInfo << "Processing parameter set: \"" << parSet << "\"" << std::endl;

    auto yBounds = GenericToolbox::getYBounds({hist1, hist2});

    auto* overlayCanvas = new TCanvas( "overlay_TCanvas" , Form("Comparing %s parameters: \"%s\"", (usePrefit_? "preFit": "postFit"), parSet.c_str()), 800, 600);
    hist1->SetFillColor(kRed-9);
    hist1->SetLineColor(kRed-3);
    hist1->SetMarkerStyle(kFullDotLarge);
    hist1->SetMarkerColor(kRed-3);
    hist1->SetMarkerSize(0);
    hist1->SetLabelSize(0.02);
    hist1->SetTitle(Form("%s (%s)", name1.c_str(), algo1.c_str()));
    hist1->GetXaxis()->SetLabelSize(0.03);
    hist1->GetXaxis()->LabelsOption("v");
    hist1->GetYaxis()->SetRangeUser(yBounds.first, yBounds.second);
    useNomVal_ ? hist1->GetYaxis()->SetTitle("Parameter values (normalized to the prior)"): hist1->GetYaxis()->SetTitle("Parameter values (a.u.)");
    hist1->Draw("E2");

    TH1D hist1Line = TH1D("hist1Line", "hist1Line",
                          hist1->GetNbinsX(),
                          hist1->GetXaxis()->GetXmin(),
                          hist1->GetXaxis()->GetXmax()
    );
    GenericToolbox::transformBinContent(&hist1Line, [&](TH1D* h_, int b_){
      h_->SetBinContent(b_, hist1->GetBinContent(b_));
    });

    hist1Line.SetLineColor(kRed-3);
    hist1Line.Draw("SAME");

    hist2->SetLineColor(9);
    hist2->SetLineWidth(2);
    hist2->SetFillColor(kWhite);
    hist2->SetMarkerColor(9);
    hist2->SetMarkerStyle(kFullDotLarge);
    hist2->SetTitle(Form("%s (%s)", name2.c_str(), algo2.c_str()));
    hist2->Draw("E1 X0 SAME");

    gPad->SetGridx();
    gPad->SetGridy();

    size_t longestTitleSize{0};
    for( int iBin = 1 ; iBin <= hist1->GetNbinsX() ; iBin++ ){
      longestTitleSize = std::max(longestTitleSize, std::string(hist1->GetXaxis()->GetBinLabel(iBin)).size());
    }
    gPad->SetBottomMargin(float(0.1*(1. + double(longestTitleSize)/15.)));

    TLegend l(0.6, 0.79, 0.89, 0.89);
    l.AddEntry(hist1, hist1->GetTitle());
    l.AddEntry(hist2, hist2->GetTitle());
    l.Draw();

    hist1->SetTitle(Form("Comparing %s parameters: \"%s\"", (usePrefit_? "preFit": "postFit"), parSet.c_str()));
    GenericToolbox::mkdirTFile(outDir, parSet)->cd();
    overlayCanvas->Write();
    delete overlayCanvas;
    Logger::setIndentStr("");
  }
}