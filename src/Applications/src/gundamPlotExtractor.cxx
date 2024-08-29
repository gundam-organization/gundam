//
// Created by Adrien BLANCHET on 28/01/2022.
//

#include "GundamGreetings.h"

#include "CmdLineParser.h"
#include "Logger.h"
#include "GenericToolbox.Root.h"

#include "TStyle.h"
#include "TDirectory.h"
#include "TClass.h"
#include "TKey.h"
#include "TSystem.h"

#include <string>
#include <vector>
#include <cstdlib>




std::vector<std::string> outExtensions;
int nPlots{0};

void walkAndUnfoldTDirectory(TDirectory* dir_, const std::string &saveFolderPath_);
void init();

int main( int argc, char** argv ){

  GundamGreetings g;
  g.setAppName("plot extractor tool");
  g.hello();

  CmdLineParser clp(argc, argv);
  clp.addOption("root-file", {"-f"}, "Provide ROOT file path.", 1);
  clp.addOption("output-folder-path", {"-o"}, "Set output folder name.", 1);
  clp.addOption("output-plot-extensions", {"-x"}, "Set output plot extensions.", -1);

  LogInfo << "Available options: " << std::endl;
  LogInfo << clp.getConfigSummary() << std::endl;

  clp.parseCmdLine();

  LogWarning << "Command line options:" << std::endl;
  LogWarning << clp.getValueSummary() << std::endl;

  LogInfo << "Reading config..." << std::endl;
  auto rootFilePath = clp.getOptionVal<std::string>("root-file");
  auto outFolderPath = clp.getOptionVal<std::string>("output-folder-path");
  outExtensions = clp.getOptionValList<std::string>("output-plot-extensions");

  if( outExtensions.empty() ) outExtensions.emplace_back("pdf");
  LogInfo << "Output plot extensions: " << GenericToolbox::toString(outExtensions) << std::endl;

  auto* rootFile = GenericToolbox::openExistingTFile(rootFilePath);

  init();
  walkAndUnfoldTDirectory(rootFile, outFolderPath);

  LogInfo << nPlots << " plots written." << std::endl;

  rootFile->Close();
  delete rootFile;

  g.goodbye();
  return EXIT_SUCCESS;
}


void walkAndUnfoldTDirectory(TDirectory* dir_, const std::string &saveFolderPath_){

  TCanvas* canObj;
  TClass* classObj;
  TKey* keyObj;
  for( int iObj = 0 ; iObj < dir_->GetListOfKeys()->GetEntries() ; iObj++ ){
    keyObj = (TKey*) dir_->GetListOfKeys()->At(iObj);

    classObj = gROOT->GetClass( keyObj->GetClassName() );
    if( classObj->InheritsFrom("TDirectory") ){
      // recursive walk
      walkAndUnfoldTDirectory(
          dir_->GetDirectory(dir_->GetListOfKeys()->At(iObj)->GetName()),
          saveFolderPath_ + "/" + dir_->GetListOfKeys()->At(iObj)->GetName()
          );
    }
    else if( classObj->InheritsFrom("TCanvas") ){
      canObj = dir_->Get<TCanvas>(dir_->GetListOfKeys()->At(iObj)->GetName());

      LogContinueIf( canObj == nullptr, "Could not cast \"" << dir_->GetListOfKeys()->At(iObj)->GetName() << "\" as TCanvas" );

//      canObj->Draw();
//      canObj->Update();
////      canObj->GetWindowHeight()
//
//      bool WhileBool = true;
//      while(WhileBool){
//        gPad->Modified(); gPad->Update();
//        gSystem->ProcessEvents();
//        std::cout<<"Enter 0 to exit the macro. Enter 0 to stay in while loop: "<< std::endl;
//        std::cin >> WhileBool;
//      }

      GenericToolbox::mkdir(saveFolderPath_);
      nPlots++;
      for( auto& ext : outExtensions ){
        std::string outPath =
            Form("%s/%s.%s",
              saveFolderPath_.c_str(),
              dir_->GetListOfKeys()->At(iObj)->GetName(),
              ext.c_str()
            );

        LogWarning << outPath << std::endl;
        if( GenericToolbox::isFile(outPath) ){
          std::remove(outPath.c_str());
        }

        GenericToolbox::muteRoot();
        canObj->SaveAs( outPath.c_str() );
        GenericToolbox::unmuteRoot();
      }

    }
  }

}

void init(){
  gROOT->SetStyle("Plain");
  gStyle->SetTitleBorderSize(0);
  gStyle->SetStatFont(42);
  gStyle->SetOptFit(1111);
  gStyle->SetOptStat(0);

  gStyle->SetLabelFont(42,"xyz");
  gStyle->SetLabelSize(0.05,"xyz");
  gStyle->SetLabelOffset(0.015,"x");
  gStyle->SetLabelOffset(0.015,"y");

  gStyle->SetTitleFont(42);
  gStyle->SetTitleFontSize(0.06);
  gStyle->SetLegendFont(42);

  gStyle->SetTitleFont(42,"xyz");
  gStyle->SetTitleSize(0.06,"xyz");
  gStyle->SetTitleOffset(1.05,"x");
  gStyle->SetTitleOffset(1.20,"y");

  gStyle->SetStripDecimals(kFALSE);

  gStyle->SetPadLeftMargin(0.16);
  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadRightMargin(0.07);
  gStyle->SetPadTopMargin(0.09);

  // gStyle->SetStatW(0.35);
  // gStyle->SetStatH(0.25);

  gStyle->SetPadTickX(kTRUE);
  gStyle->SetPadTickY(kTRUE);

  gStyle->SetGridStyle(3);
  gStyle->SetGridWidth(1);
  gStyle->SetPadGridX(true);
  gStyle->SetPadGridY(true);

  //  gStyle->SetPalette(1);

  gStyle->SetLineWidth(2);
  gStyle->SetHistLineWidth(3);
  gStyle->SetFuncWidth(3);
  gStyle->SetFrameLineWidth(2);

  gStyle->SetMarkerSize(1.2);
}
