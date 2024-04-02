//
// Created by Nadrino on 02/04/2024.
//

#include "GundamGreetings.h"
#include "GundamApp.h"
#include "ConfigUtils.h"
#include "GundamUtils.h"

#include "Logger.h"
#include "CmdLineParser.h"
#include "GenericToolbox.Json.h"
#include "GenericToolbox.Root.h"

#include <vector>
#include <utility>
#include <algorithm>


LoggerInit([]{
  Logger::getUserHeader() << "[" << FILENAME << "]";
});


int main( int argc, char** argv ){
  GundamApp app{"fit compare tool"};

  CmdLineParser clp;

  clp.addDummyOption("Main options");
  clp.addOption("configFile", {"-c"}, "Specify config file.", 1);
  clp.addOption("overrideFiles", {"-of", "--override-files"}, "Provide config files that will override keys", -1);
  clp.addOption("outputFilePath", {"-o", "--out-file"}, "Specify the output file", 1);
  clp.addOption("appendix", {"--appendix"}, "Add appendix to the output file name", 1);

  clp.addDummyOption("");

  LogInfo << "Options list:" << std::endl;
  LogInfo << clp.getConfigSummary() << std::endl << std::endl;

  // read cli
  clp.parseCmdLine(argc, argv);

  // printout what's been fired
  LogInfo << "Fired options are: " << std::endl << clp.getValueSummary() << std::endl;

  if( not clp.isOptionTriggered("configFile") ){
    LogError << "No config file provided." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Reading configuration
  auto configFilePath = clp.getOptionVal("configFile", "");
  LogThrowIf(configFilePath.empty(), "Config file not provided.");

  ConfigUtils::ConfigHandler configHandler(configFilePath);
  configHandler.override( clp.getOptionValList<std::string>("overrideFiles") );

  // Output file path
  std::string outFileName;
  if( clp.isOptionTriggered("outputFilePath") ){ outFileName = clp.getOptionVal("outputFilePath", outFileName + ".root"); }
  else{

    std::string outFolder{"./output"};

    // appendixDict["optionName"] = "Appendix"
    // this list insure all appendices will appear in the same order
    std::vector<std::pair<std::string, std::string>> appendixDict{
        {"configFile", "%s"},
        {"overrideFiles", "With_%s"},
        {"appendix", "%s"}
    };

    outFileName = GenericToolbox::joinPath(
        outFolder,
        GundamUtils::generateFileName(clp, appendixDict)
    ) + ".root";
  }


  // to write cmdLine info
  app.setCmdLinePtr( &clp );

  // unfolded config
  app.setConfigString( GenericToolbox::Json::toReadableString(configHandler.getConfig()) );

  // Ok, we should run. Create the out file.
  app.openOutputFile(outFileName);
  app.writeAppInfo();

  auto fitPlotConfig = GenericToolbox::Json::fetchValue<JsonType>(configHandler.getConfig(), "fitPlotConfig");
  auto fitEntryList = GenericToolbox::Json::fetchValue<JsonType>(fitPlotConfig, "fitEntryList");

  // initialise plots
  struct GraphHolder{
    std::string name{}; // what systematics to plot
    std::string path{};

    struct PointList{
      std::vector<double> xPoints{};
      std::vector<double> yPoints{};

      void addPoint(double x_, double y_){
        xPoints.emplace_back(x_);
        yPoints.emplace_back(y_);
      }

      void sort(){
        auto p = GenericToolbox::getSortPermutation(xPoints, [](double a_, double b_){ return a_ < b_; });

        GenericToolbox::applyPermutation(xPoints, p);
        GenericToolbox::applyPermutation(yPoints, p);
      }
    };

    PointList pointList{};

    TGraph* generateGraph(){
      if( pointList.xPoints.empty() ) return nullptr;

      auto* out = new TGraph(pointList.xPoints.size(), pointList.xPoints.data(), pointList.yPoints.data());
      out->SetName( name.c_str() );
      out->SetTitle( name.c_str() );

      out->SetDrawOption("LP");
      out->SetMarkerStyle(kFullDotLarge);

      double yMax{pointList.yPoints[0]};
      for( auto& y : pointList.yPoints ){ if( y>yMax ){ yMax = y; }}

      out->GetYaxis()->SetRangeUser(0, yMax*1.2);

      return out;
    }
  };
  std::vector<GraphHolder> graphHolder;

  auto xTitle = GenericToolbox::Json::fetchValue<std::string>(fitPlotConfig, "xTitle");

  // add points
  for( auto& fitEntry : fitEntryList ){
    auto filePath = GenericToolbox::Json::fetchValue<std::string>(fitEntry, "filePath");
    auto xValue = GenericToolbox::Json::fetchValue<double>(fitEntry, "xValue");
    LogInfo << "Opening file: " << filePath << std::endl; LogScopeIndent;
    auto* file = GenericToolbox::openExistingTFile( filePath );

    auto* errorsDir = file->Get<TDirectory>("FitterEngine/postFit/Hesse/errors");
    LogThrowIf(errorsDir == nullptr);
    for( int iParSet = 0 ; iParSet < errorsDir->GetListOfKeys()->GetEntries() ; iParSet++ ){
      auto* parSetDir{errorsDir->Get<TDirectory>(errorsDir->GetListOfKeys()->At(iParSet)->GetName())};
      auto* hist = parSetDir->Get<TH1D>("valuesNorm/postFitErrors_TH1D");

      for( int iBin = 0 ; iBin < hist->GetNbinsX() ; iBin++ ){
        std::string binName = hist->GetXaxis()->GetBinLabel(iBin+1);
        std::string graphName{GenericToolbox::joinPath(parSetDir->GetName(), binName)};

        GraphHolder* selectedGraph{nullptr};
        for(auto& graph: graphHolder){ if( graph.name == graphName ){ selectedGraph = &graph; break; } }

        if( selectedGraph == nullptr ){
          selectedGraph = &graphHolder.emplace_back();
          selectedGraph->name = graphName;
          selectedGraph->path = parSetDir->GetName();
        }

        selectedGraph->pointList.addPoint( xValue, hist->GetBinError(iBin+1) );
      }
    }
  }

  for( auto& graphEntry : graphHolder ){
    auto* outGraph = graphEntry.generateGraph();

    outGraph->GetYaxis()->SetTitle( "post-fit constraint (%)" );
    outGraph->GetXaxis()->SetTitle( xTitle.c_str() );

    auto path = GenericToolbox::mkdirTFile(
        app.getOutfilePtr(),
        GenericToolbox::joinPath("systematics",graphEntry.path)
    );
    GenericToolbox::writeInTFile(path, outGraph);
  }

  return EXIT_SUCCESS;
}
