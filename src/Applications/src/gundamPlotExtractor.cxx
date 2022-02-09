//
// Created by Adrien BLANCHET on 28/01/2022.
//

#include "GundamGreetings.h"

#include "CmdLineParser.h"
#include "Logger.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.h"

#include "TDirectory.h"
#include "TClass.h"
#include "TKey.h"

#include <cstdlib>


LoggerInit([]{
  Logger::setUserHeaderStr("[gundamPlotExtractor.cxx]");
})

std::vector<std::string> outExtensions;
int nPlots{0};

void walkAndUnfoldTDirectory(TDirectory* dir_, const std::string &saveFolderPath_);

int main( int argc, char** argv ){

  GundamGreetings g;
  g.setAppName("PlotExtractor");
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
  LogInfo << "Output plot extensions: " << GenericToolbox::parseVectorAsString(outExtensions) << std::endl;

  auto* rootFile = GenericToolbox::openExistingTFile(rootFilePath);

  walkAndUnfoldTDirectory(rootFile, outFolderPath);

  LogInfo << nPlots << " plots writen." << std::endl;

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
      canObj = (TCanvas*) dir_->Get(dir_->GetListOfKeys()->At(iObj)->GetName());
      GenericToolbox::mkdirPath(saveFolderPath_);
      nPlots++;
      for( auto& ext : outExtensions ){
        std::string outPath =
            Form("%s/%s.%s",
              saveFolderPath_.c_str(),
              dir_->GetListOfKeys()->At(iObj)->GetName(),
              ext.c_str()
            );

        LogWarning << outPath << std::endl;

        GenericToolbox::muteRoot();
        canObj->SaveAs( outPath.c_str() );
        GenericToolbox::unmuteRoot();
      }

    }
  }

}
