//
// Created by Adrien BLANCHET on 12/05/2022.
//

#include "GundamGreetings.h"
#include "GundamUtils.h"

#include "Logger.h"
#include "CmdLineParser.h"
#include "GenericToolbox.Root.h"

#include "nlohmann/json.hpp"
#include "TKey.h"
#include "TFile.h"
#include <TLegend.h>

#include <string>
#include <vector>
#include <memory>
#include <utility>


LoggerInit([]{
  Logger::getUserHeader() << "[" << FILENAME << "]";
});


CmdLineParser clp;
TFile* outFile{nullptr};
bool verbose{false};


void makeSampleComparePlots(bool usePrefit_);
void makeScanComparePlots(bool usePrefit_);
void makeErrorComparePlots(bool usePrefit_, bool useNomVal_);


int main( int argc, char** argv ){
  GundamGreetings g;
  g.setAppName("fit compare tool");
  g.hello();

  // files
  clp.addDummyOption("Main options");
  clp.addOption("file-1", {"-f1"}, "Path to first output fit file.", 1);
  clp.addOption("file-2", {"-f2"}, "Path to second output fit file.", 1);
  clp.addOption("name-1", {"-n1"}, "Set display name of the first fit file.", 1);
  clp.addOption("name-2", {"-n2"}, "Set display name of the  fit file.", 1);
  clp.addOption("algo-1", {"-a1"}, "Specify algo folder to compare for the first fit file.", 1);
  clp.addOption("algo-2", {"-a2"}, "Specify algo folder to compare for the second fit file.", 1);
  clp.addOption("output", {"-o"}, "Output file.", 1);

  // compare post-fit with pre-fit data (useful when f1 and f2 are the same file)
  clp.addDummyOption("Trigger options");
  clp.addTriggerOption("use-prefit-1", {"--prefit-1"}, "Use prefit data only for file 1.");
  clp.addTriggerOption("use-prefit-2", {"--prefit-2"}, "Use prefit data only for file 2.");
  clp.addTriggerOption("verbose", {"-v"}, "Recursive verbosity printout.");

  clp.addDummyOption("");

  LogInfo << "Options list:" << std::endl;
  LogInfo << clp.getConfigSummary() << std::endl << std::endl;

  // read cli
  clp.parseCmdLine(argc, argv);

  // printout what's been fired
  LogInfo << "Fired options are: " << std::endl << clp.getValueSummary() << std::endl;

  if( clp.isNoOptionTriggered()
      or not clp.isOptionTriggered("file-1")
      or not clp.isOptionTriggered("file-2")
      ){
    LogError << "Missing options." << std::endl;
    exit(EXIT_FAILURE);
  }

  LogThrowIf(clp.isOptionTriggered("use-prefit-1") and clp.isOptionTriggered("use-prefit-2"),
             "Remove the two options to see prefit comparison of both files");

  LogInfo << "Reading config..." << std::endl;
  verbose = clp.isOptionTriggered("verbose");

  std::string outPath{};
  if( clp.isOptionTriggered("output") ){ outPath = clp.getOptionVal<std::string>("output"); }
  else{
    // auto generate
    std::vector<std::pair<std::string, std::string>> appendixDict;

    // file1
    if( clp.isOptionTriggered("name-1") ){ appendixDict.emplace_back("name-1", "%s"); }
    else                                 { appendixDict.emplace_back("file-1", "%s"); }
    if( clp.isOptionTriggered("use-prefit-1") ){ appendixDict.emplace_back("use-prefit-1", "PreFit"); }

    // file2
    if( clp.isOptionTriggered("name-2") ){ appendixDict.emplace_back("name-2", "vs_%s"); }
    else                                 { appendixDict.emplace_back("file-2", "vs_%s"); }
    if( clp.isOptionTriggered("use-prefit-2") ){ appendixDict.emplace_back("use-prefit-2", "PreFit"); }

    outPath = "fitCompare_" + GundamUtils::generateFileName(clp, appendixDict) + ".root";
    LogWarning << "Output file: " << outPath << std::endl;
  }
  outFile = TFile::Open(outPath.c_str(), "RECREATE");


  LogAlert << std::endl << "Starting comparison..." << std::endl;

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

  LogAlert << std::endl << "Closing: " << outPath << std::endl;
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

  strBuffer = Form("FitterEngine/%s/samples", ((usePrefit_ or clp.isOptionTriggered("use-prefit-1"))? "preFit": "postFit"));
  auto* dir1 = file1->Get<TDirectory>(strBuffer.c_str());
  LogReturnIf(dir1== nullptr, "Could not find \"" << strBuffer << "\" within " << filePath1);

  strBuffer = Form("FitterEngine/%s/samples", ((usePrefit_ or clp.isOptionTriggered("use-prefit-2"))? "preFit": "postFit"));
  auto* dir2 = file2->Get<TDirectory>(strBuffer.c_str());
  LogReturnIf(dir2== nullptr, "Could not find \"" << strBuffer << "\" within " << filePath2);

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
        GenericToolbox::transformBinContent(hCompValues, [](TH1D* h_, int bin_){
          h_->SetBinError(bin_, 0);
          if( std::isnan(h_->GetBinContent(bin_)) ){ h_->SetBinContent(bin_, 0); }
        });

        hCompValues->SetTitle(Form("Comparing \"%s\"", dir1_->GetListOfKeys()->At(iKey)->GetName()));
        hCompValues->GetYaxis()->SetTitle("Bin content difference");

        GenericToolbox::writeInTFile(
            GenericToolbox::mkdirTFile(outFile, GenericToolbox::joinVectorString(pathBuffer, "/")),
            hCompValues,
            dir1_->GetListOfKeys()->At(iKey)->GetName()
            );

        auto* hCompValuesRatio = (TH1D*) hCompValues->Clone();
        GenericToolbox::transformBinContent(hCompValuesRatio, [h1](TH1D* h_, int bin_){
          h_->SetBinContent(bin_, 100. * h_->GetBinContent(bin_)/h1->GetBinContent(bin_));
          h_->SetBinError(bin_, 0);
          if( std::isnan(h_->GetBinContent(bin_)) ){ h_->SetBinContent(bin_, 0); }
        });

        hCompValuesRatio->SetTitle(Form("Comparing \"%s\"", dir1_->GetListOfKeys()->At(iKey)->GetName()));
        hCompValuesRatio->GetYaxis()->SetTitle("Bin content relative difference (%)");

        GenericToolbox::writeInTFile(
            GenericToolbox::mkdirTFile(outFile, GenericToolbox::joinVectorString(pathBuffer, "/")),
            hCompValuesRatio,
            dir1_->GetListOfKeys()->At(iKey)->GetName() + std::string("_Ratio")
        );

      }
    }

  };

  recurseSampleCompareGraph(dir1, dir2);
  GenericToolbox::triggerTFileWrite( outFile );
}
void makeScanComparePlots(bool usePrefit_){

  auto filePath1 = clp.getOptionVal<std::string>("file-1");
  auto filePath2 = clp.getOptionVal<std::string>("file-2");

  auto name1 = clp.getOptionVal("name-1", filePath1);
  auto name2 = clp.getOptionVal("name-2", filePath2);

  auto* file1 = GenericToolbox::openExistingTFile(filePath1);
  auto* file2 = GenericToolbox::openExistingTFile(filePath2);

  // path buffer should not care about "use-prefit" option
  // it is used for the output file.
  std::vector<std::string> pathBuffer;
  pathBuffer.emplace_back( Form( "%s/scan", ( usePrefit_ ? "preFit": "postFit")) );
  std::function<void(TDirectory* dir1_, TDirectory* dir2_)> recurseScanCompareGraph;
  recurseScanCompareGraph = [&](TDirectory* dir1_, TDirectory* dir2_){

    for( int iKey = 0 ; iKey < dir1_->GetListOfKeys()->GetEntries() ; iKey++ ){
      if( dir2_->Get(dir1_->GetListOfKeys()->At(iKey)->GetName()) == nullptr ) continue;
      TKey* keyObj = (TKey*) dir1_->GetListOfKeys()->At(iKey);


      if     ( (gROOT->GetClass( keyObj->GetClassName() ))->InheritsFrom("TDirectory") ){
        // recursive part
        pathBuffer.emplace_back( dir1_->GetListOfKeys()->At(iKey)->GetName() );

        LogDebugIf(verbose) << "Exploring: " << GenericToolbox::joinPath(pathBuffer) << std::endl;
        recurseScanCompareGraph(
            dir1_->GetDirectory(dir1_->GetListOfKeys()->At(iKey)->GetName()),
            dir2_->GetDirectory(dir1_->GetListOfKeys()->At(iKey)->GetName())
            );

        pathBuffer.pop_back();
      }
      else if( (gROOT->GetClass( keyObj->GetClassName() ))->InheritsFrom("TGraph") ){
        LogDebugIf(verbose) << "Found: " << dir1_->GetListOfKeys()->At(iKey)->GetName() << std::endl;

        auto* gr1 = dir1_->Get<TGraph>( dir1_->GetListOfKeys()->At(iKey)->GetName() );
        auto* gr2 = dir2_->Get<TGraph>( dir1_->GetListOfKeys()->At(iKey)->GetName() ); // should be the same keyname.

        // look for current value:
        std::string parValObjName = dir1_->GetListOfKeys()->At(iKey)->GetName();
        // removing "_TGraph"
        parValObjName = GenericToolbox::joinVectorString(GenericToolbox::splitString(parValObjName, "_"), "_", 0, -1);
        parValObjName += "_CurrentPar_TVectorT_double";
        auto* val1 = dir1_->GetDirectory("../")->Get<TVectorD>(parValObjName.c_str());
        auto* val2 = dir2_->GetDirectory("../")->Get<TVectorD>(parValObjName.c_str());

        auto* overlayCanvas = new TCanvas(
            dir1_->GetListOfKeys()->At(iKey)->GetName() ,
            Form("Comparing %s scan: \"%s\"", (usePrefit_? "preFit": "postFit"),
                 dir1_->GetListOfKeys()->At(iKey)->GetName()
            ),
            800, 600
        );
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

        if( val1 != nullptr ){
          // vertical lines
          overlayCanvas->Update(); // update Y1 and Y2
          auto* line1 = new TLine((*val1)[0], gPad->GetFrame()->GetY1(), (*val1)[0], gPad->GetFrame()->GetY2());
          line1->SetLineColor(kBlue-7);
          line1->SetLineStyle(2);
          line1->SetLineWidth(3);
          line1->Draw();
        }
        if( val2 != nullptr ){
          // vertical lines
          overlayCanvas->Update();
          auto* line2 = new TLine((*val2)[0], gPad->GetFrame()->GetY1(), (*val2)[0], gPad->GetFrame()->GetY2());
          line2->SetLineColor(kBlack);
          line2->SetLineStyle(3);
          line2->SetLineWidth(3);
          line2->Draw();
        }

        gPad->SetGridx();
        gPad->SetGridy();

        GenericToolbox::writeInTFile(
            GenericToolbox::mkdirTFile(outFile, GenericToolbox::joinVectorString(pathBuffer, "/")),
            overlayCanvas
            );
        delete overlayCanvas;
        delete val1;
        delete val2;
      }
    }

  };

  std::string strBuffer;

  strBuffer = Form("FitterEngine/%s/scan", ((usePrefit_ or clp.isOptionTriggered("use-prefit-1"))? "preFit": "postFit"));
  auto* dir1 = file1->Get<TDirectory>(strBuffer.c_str());
  LogReturnIf(dir1 == nullptr, "Could not find \"" << strBuffer << "\" within " << filePath1);

  strBuffer = Form("FitterEngine/%s/scan", ((usePrefit_ or clp.isOptionTriggered("use-prefit-2"))? "preFit": "postFit"));
  auto* dir2 = file2->Get<TDirectory>(strBuffer.c_str());
  LogReturnIf(dir2 == nullptr, "Could not find \"" << strBuffer << "\" within " << filePath2);

  recurseScanCompareGraph(dir1, dir2);
  GenericToolbox::triggerTFileWrite( outFile );
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

  strBuffer = Form("FitterEngine/%s/%s/errors", ((usePrefit_ or clp.isOptionTriggered("use-prefit-1"))? "preFit": "postFit"), algo1.c_str());
  auto* dir1 = file1->Get<TDirectory>(strBuffer.c_str());
  LogReturnIf(dir1== nullptr, "Could not find \"" << strBuffer << "\" within " << filePath1);

  strBuffer = Form("FitterEngine/%s/%s/errors", ((usePrefit_ or clp.isOptionTriggered("use-prefit-2"))? "preFit": "postFit"), algo2.c_str());
  auto* dir2 = file2->Get<TDirectory>(strBuffer.c_str());
  LogReturnIf(dir2== nullptr, "Could not find \"" << strBuffer << "\" within " << filePath2);

  // loop over parSets
  auto* outDir = GenericToolbox::mkdirTFile(outFile, Form("%s/errors%s", (usePrefit_? "preFit": "postFit"), (useNomVal_? "Norm": "")));
  for( int iKey = 0 ; iKey < dir1->GetListOfKeys()->GetEntries() ; iKey++ ){
    LogScopeIndent;
    std::string parSet = dir1->GetListOfKeys()->At(iKey)->GetName();

    strBuffer = Form("%s/values%s/%sErrors_TH1D", parSet.c_str(), (useNomVal_? "Norm": ""), (usePrefit_? "preFit": "postFit"));
    auto* hist1 = dir1->Get<TH1D>(strBuffer.c_str());
    if( hist1 == nullptr ){
      // legacy
      strBuffer = Form("%s/values%s/%sErrors", parSet.c_str(), (useNomVal_? "Norm": ""), (usePrefit_? "preFit": "postFit"));
      hist1 = dir1->Get<TH1D>(strBuffer.c_str());
    }
    LogContinueIf(hist1 == nullptr, "Could no find parSet \"" << strBuffer << "\" in " << file1->GetPath());

    strBuffer = Form("%s/values%s/%sErrors_TH1D", parSet.c_str(), (useNomVal_? "Norm": ""), (usePrefit_? "preFit": "postFit"));
    auto* hist2 = dir2->Get<TH1D>(strBuffer.c_str());
    if( hist2 == nullptr ){
      // legacy
      strBuffer = Form("%s/values%s/%sErrors", parSet.c_str(), (useNomVal_? "Norm": ""), (usePrefit_? "preFit": "postFit"));
      hist2 = dir2->Get<TH1D>(strBuffer.c_str());
    }
    LogContinueIf(hist2 == nullptr, "Could no find parSet \"" << strBuffer << "\" in " << file2->GetPath());

    LogInfo << "Processing parameter set: \"" << parSet << "\"" << std::endl;

    auto yBounds = GenericToolbox::getYBounds({hist1, hist2});

    auto* overlayCanvas = new TCanvas( "overlay" , Form("Comparing %s parameters: \"%s\"", (usePrefit_? "preFit": "postFit"), parSet.c_str()), 800, 600);
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
    GenericToolbox::writeInTFile( GenericToolbox::mkdirTFile(outDir, parSet), overlayCanvas );

    std::map<std::string, TH1D*> compHist{
        {"ScaledComp", nullptr},
        {"ValueDiff", nullptr},
        {"ValueRatio", nullptr},
        {"ErrorDiff", nullptr},
        {"ErrorRatio", nullptr},
        {"ValueDiffAbs", nullptr},
        {"ValueRatioAbs", nullptr},
        {"ErrorDiffAbs", nullptr},
        {"ErrorRatioAbs", nullptr}
    };

    for( auto& histEntry : compHist ){
      histEntry.second = (TH1D*) hist2->Clone();
      GenericToolbox::resetHistogram(histEntry.second);
      histEntry.second->SetName(histEntry.first.c_str());
      histEntry.second->GetXaxis()->LabelsOption("v");
      histEntry.second->SetTitleSize(0.03);
      histEntry.second->SetTitle(
          Form(R"(Comparing "%s" %s parameters: "%s"/%s [1] and "%s"/%s [2])",
               parSet.c_str(), (usePrefit_? "preFit": "postFit"),
               name1.c_str(), algo1.c_str(), name2.c_str(), algo2.c_str()));
    }

    for( int iBin = 1 ; iBin <= hist1->GetNbinsX() ; iBin++ ){
      double hist1Val = hist1->GetBinContent(iBin);
      double hist2Val = hist2->GetBinContent(iBin);
      double hist1Err = hist1->GetBinError(iBin);
      double hist2Err = hist2->GetBinError(iBin);
      double diffVal = hist2Val - hist1Val;
      double diffErr = hist2Err - hist1Err;

      compHist["ValueDiff"]->SetBinContent( iBin, diffVal );
      compHist["ValueRatio"]->SetBinContent( iBin, ( hist1Val > 0 ) ? 100* (diffVal/hist1Val) : 0 );
      compHist["ErrorDiff"]->SetBinContent( iBin, diffErr );
      compHist["ErrorRatio"]->SetBinContent( iBin, ( hist1Err > 0 ) ? 100* (diffErr/hist1Err) : 0 );
      compHist["ValueDiffAbs"]->SetBinContent( iBin, TMath::Abs(compHist["ValueDiff"]->GetBinContent(iBin)) );
      compHist["ValueRatioAbs"]->SetBinContent( iBin, TMath::Abs(compHist["ValueRatio"]->GetBinContent(iBin)) );
      compHist["ErrorDiffAbs"]->SetBinContent( iBin, TMath::Abs(compHist["ErrorDiff"]->GetBinContent(iBin)) );
      compHist["ErrorRatioAbs"]->SetBinContent( iBin, TMath::Abs(compHist["ErrorRatio"]->GetBinContent(iBin)) );
      compHist["ScaledComp"]->SetBinContent( iBin, diffVal ); compHist["ScaledComp"]->SetBinError( iBin, ( hist1Err > 0 ) ? hist2Err/hist1Err : 0 );
    }

    compHist["ValueDiff"]->GetYaxis()->SetTitle("#mu_{2} - #mu_{1}");
    compHist["ValueRatio"]->GetYaxis()->SetTitle("#mu_{2} / #mu_{1} - 1 (%)");
    compHist["ErrorDiff"]->GetYaxis()->SetTitle("#sigma_{2} - #sigma_{1}");
    compHist["ErrorRatio"]->GetYaxis()->SetTitle("#sigma_{2} / #sigma_{1} - 1 (%)");
    compHist["ValueDiffAbs"]->GetYaxis()->SetTitle("#left|#mu_{2} - #mu_{1}#right|");
    compHist["ValueRatioAbs"]->GetYaxis()->SetTitle("#left|#mu_{2} / #mu_{1} - 1#right|  (%)");
    compHist["ErrorDiffAbs"]->GetYaxis()->SetTitle("#left|#sigma_{2} - #sigma_{1}#right|");
    compHist["ErrorRatioAbs"]->GetYaxis()->SetTitle("#left|#sigma_{2} / #sigma_{1} - 1#right| (%)");
    compHist["ScaledComp"]->GetYaxis()->SetTitle("(#mu_{2} - #mu_{1}) #pm #sigma_{2}/#sigma_{1}");

    overlayCanvas->cd();

    for( auto& histEntry : compHist ){
      overlayCanvas->cd();

      histEntry.second->Draw("E1");
      overlayCanvas->Update();
      TLine* line1{nullptr}, *line2{nullptr};
      if( histEntry.first == "ScaledComp" ){
        histEntry.second->GetYaxis()->SetRangeUser(-2, 2);
        line1 = new TLine(gPad->GetFrame()->GetX1(), 1, gPad->GetFrame()->GetX2(), 1); line1->SetLineColor(kRed); line1->SetLineStyle(2); line1->Draw();
        line2 = new TLine(gPad->GetFrame()->GetX1(), -1, gPad->GetFrame()->GetX2(), -1); line2->SetLineColor(kRed); line2->SetLineStyle(2); line2->Draw();
      }

      GenericToolbox::writeInTFile(
          GenericToolbox::mkdirTFile(outDir, parSet),
          overlayCanvas,
          histEntry.first
          );

      delete histEntry.second; delete line1; delete line2;
    }

    delete overlayCanvas;
  }

  GenericToolbox::triggerTFileWrite( outFile );
}