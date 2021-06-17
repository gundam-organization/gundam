//
// Created by Adrien BLANCHET on 16/06/2021.
//

#include "string"
#include "vector"
#include "sstream"

#include "TCanvas.h"
#include "TH1D.h"

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolboxRootExt.h"

#include "JsonUtils.h"
#include "SamplePlotGenerator.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[SamplePlotGenerator]");
})


SamplePlotGenerator::SamplePlotGenerator() { this->reset(); }
SamplePlotGenerator::~SamplePlotGenerator() { this->reset(); }

void SamplePlotGenerator::reset() {
  _saveTDirectory_ = nullptr;


  defaultColorWheel = {
    kGreen-3, kTeal+3, kAzure+7,
    kCyan-2, kBlue-7, kBlue+2,
    kOrange+1, kOrange+9, kRed+2, kPink+9
  };

}

void SamplePlotGenerator::setSaveTDirectory(TDirectory *saveTDirectory_) {
  _saveTDirectory_ = saveTDirectory_;
}
void SamplePlotGenerator::setConfig(const nlohmann::json &config) {
  _config_ = config;
  while( _config_.is_string() ){
    _config_ = JsonUtils::readConfigFile(_config_.get<std::string>());
  }
}

void SamplePlotGenerator::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  if( _config_.empty() ){
    LogError << "_config_ not set." << std::endl;
    throw std::logic_error("_config_ not set.");
  }

  _varDictionary_ = JsonUtils::fetchValue(_config_, "varDictionnaries", nlohmann::json());
  _canvasParameters_ = JsonUtils::fetchValue(_config_, "canvasParameters", nlohmann::json());
  _histogramsDefinition_ = JsonUtils::fetchValue(_config_, "histogramsDefinition", nlohmann::json());

}

void SamplePlotGenerator::saveSamplePlots(TDirectory *saveTDirectory_, const std::vector<AnaSample> &sampleList_) {

  LogWarning << "Generating sample plots." << std::endl;

  if( _histogramsDefinition_.empty() ){
    LogError << "No histogram has been defined." << std::endl;
    return;
  }

  auto* lastDir = gDirectory;
  saveTDirectory_->cd();
  LogInfo << "Samples plots will be writen in: " << saveTDirectory_->GetPath() << std::endl;
  int sampleCounter = -1;

  // Generating histograms
  std::map<std::string, TH1D*> histogramList;

  for( const auto& sample : sampleList_ ){
    sampleCounter++;

    short unsetSplitValueColor = kGray; // will increment if needed
    for( const auto& histDef : _histogramsDefinition_ ){

      std::string varToPlot = JsonUtils::fetchValue<std::string>(histDef, "varToPlot");
      std::vector<std::string> splitVars = JsonUtils::fetchValue(histDef, "splitVars", std::vector<std::string>{""});

      for( const auto& splitVar : splitVars ){

        std::vector<int> splitVarValues;

        std::stringstream ssHistPath;
        ssHistPath << sample.GetName() << "/";
        ssHistPath << varToPlot << "/";
        if( not splitVar.empty() ){
          ssHistPath << splitVar << "/";
          for( const auto& event : sample.GetConstMcEvents() ){
            int splitValue = event.GetEventVarInt(splitVar);
            if( not GenericToolbox::doesElementIsInVector(splitValue, splitVarValues) ){
              LogDebug << GET_VAR_NAME_VALUE(splitValue) << std::endl;
              splitVarValues.emplace_back(splitValue);
            }
          }
        }
        else{
          splitVarValues.emplace_back(0); // just for the loop
        }

        int splitValueIndex = -1;
        for( const auto& splitValue : splitVarValues ){
          splitValueIndex++;

          std::string histDirPath = ssHistPath.str();
          if( not splitVar.empty() ){
            histDirPath += std::to_string(splitValue) + "/";
          }

          for( const std::string& histType: {"mc", "data"}){

            if( // don't build data histogram in those cases
                histType == "data" and
                ( not splitVar.empty()  // don't use split var for data
                  or not JsonUtils::fetchValue(histDef, "noData", false) // explicitly no data
                )
              ){
              continue;
            }

            std::string histPath = histDirPath + histType;
            histogramList[histPath] = nullptr;

            if( varToPlot == "Raw" ){
              if( histType == "mc" ) histogramList[histPath] = (TH1D*) sample.GetPredHisto().Clone();
              else if( histType == "data" ) histogramList[histPath] = (TH1D*) sample.GetDataHisto().Clone();
            }
            else {

              std::vector<double> xLowBinEdges;
              if( JsonUtils::fetchValue(histDef, "useSampleBinning", false) ){

                int dimIndex = GenericToolbox::findElementIndex(varToPlot, sample.GetFitPhaseSpace());
                if( dimIndex == -1 ){
                  LogDebug << "Can't useSampleBinning since \"" << varToPlot
                           << "\" is not a dimension of the sample \"" << sample.GetName()
                           << "\": " << GenericToolbox::parseVectorAsString(sample.GetFitPhaseSpace()) << std::endl;
                  continue; // skip this sample
                }

                for( const auto& bin : sample.GetBinEdges() ){
                  if( xLowBinEdges.empty()) xLowBinEdges.emplace_back(bin.getBinLowEdge(dimIndex));
                  if(not GenericToolbox::doesElementIsInVector(bin.getBinHighEdge(dimIndex), xLowBinEdges)){
                    xLowBinEdges.emplace_back(bin.getBinHighEdge(dimIndex));
                  }
                }
                if( xLowBinEdges.empty() ) continue; // skip
                std::sort(xLowBinEdges.begin(), xLowBinEdges.end());

                histogramList[histPath] = new TH1D(
                  histPath.c_str(), histType.c_str(),int(xLowBinEdges.size())-1, &xLowBinEdges[0]
                );
                histogramList[histPath]->GetXaxis()->SetTitle( varToPlot.c_str() );

              }
              else{
                LogError << "Unsupported yet." << std::endl;
                throw std::logic_error("unsupported yet");
              }

              // Fill the histogram
              for( const auto& event : sample.GetConstMcEvents() ){

                if( // Condition to fill:
                    splitVar.empty() // no split var: every event in the hist
                    or event.GetEventVarInt(splitVar) == splitValue // it's the corresponding splitValue
                    ){
                  histogramList[histPath]->Fill( event.GetEventVarAsDouble(varToPlot), event.GetEventWeight() );
                }

              }

              // Normalize by bin width
              for( int iBin = 0 ; iBin <= histogramList[histPath]->GetNbinsX()+1 ; iBin++ ){
                histogramList[histPath]->SetBinContent(
                  iBin,
                  (
                    histogramList[histPath]->GetBinContent(iBin) / histogramList[histPath]->GetBinWidth(iBin)
                  )
                );
              }

              if( histType == "mc" ){
                histogramList[histPath]->Scale( sample.GetNorm() );
              }
              else if( histType == "data" and sample.GetDataType() == DataType::kAsimov ){
                // Asimov is just MC, so it should be normalized the same way as MC
                histogramList[histPath]->Scale( sample.GetNorm() );
              }


            } // varToPlot is not Raw?

            if( histType == "data" ){
              histogramList[histPath]->SetLineColor(kBlack);
              histogramList[histPath]->SetMarkerColor(kBlack);
              histogramList[histPath]->SetMarkerStyle(kFullDotLarge);
              histogramList[histPath]->SetOption("EP");
            }
            else{


              if( splitVar.empty() ){
                histogramList[histPath]->SetLineColor(defaultColorWheel[sampleCounter % defaultColorWheel.size() ]);
                histogramList[histPath]->SetFillColor(defaultColorWheel[sampleCounter % defaultColorWheel.size() ]);
                histogramList[histPath]->SetMarkerColor(defaultColorWheel[sampleCounter % defaultColorWheel.size() ]);
              }
              else{

                // default
                std::string valueTitle = splitVar + " = " + std::to_string(splitValue);
                short valueColor = defaultColorWheel[splitValueIndex % defaultColorWheel.size() ];

                auto varDict = JsonUtils::fetchEntry(_varDictionary_, "name", splitVar); // does the cosmetic pars are configured?
                if( not varDict.empty() ){

                  if( varDict["dictionary"].is_null() ){
                    LogError << R"(Could not find "dictionary" key in JSON config for var: ")" << splitVar << "\"" << std::endl;
                    throw std::runtime_error("dictionary not found, by variable name found in JSON.");
                  }

                  // Look for the value we want
                  auto valDict = JsonUtils::fetchEntry(varDict["dictionary"], "value", std::to_string(splitValue));

                  valueTitle = JsonUtils::fetchValue(valDict, "title", valueTitle);
                  valueColor = std::stoi(JsonUtils::fetchValue(valDict, "color", std::to_string(unsetSplitValueColor)));
                  if( valueColor == unsetSplitValueColor ) unsetSplitValueColor++; // increment for the next ones

                }

                histogramList[histPath]->SetTitle( valueTitle.c_str() );
                histogramList[histPath]->SetLineColor( valueColor );
                histogramList[histPath]->SetFillColor( valueColor );
                histogramList[histPath]->SetMarkerColor( valueColor );

              } // splitVar not empty

            } // is MC


            LogDebug << "Writing histo: " << histPath << std::endl;
            GenericToolbox::mkdirTFile(saveTDirectory_, histDirPath)->cd();
            histogramList[histPath]->SetName(histType.c_str());
            histogramList[histPath]->Write(histType.c_str());
            saveTDirectory_->cd();

          } // mc or data
        }

      } // splitVar

    } // histDef

  } // sample

  lastDir->cd();

//
////    double maxD1Scale = 2000;
//  double maxD1Scale = -1;
//
//  // TODO: do the same thing for histograms
//
//  // Select which canvas to plot
//  std::vector<std::string> canvasSubFolderList;
//  canvasSubFolderList.emplace_back("Raw"); // special case
//  canvasSubFolderList.emplace_back("Raw/reactions");
//  canvasSubFolderList.emplace_back("D1");             // varToPlot
//  canvasSubFolderList.emplace_back("D1/reactions");   // varToPlot/splitHist
//  canvasSubFolderList.emplace_back("D2");             // varToPlot
//  canvasSubFolderList.emplace_back("D2/reactions");   // varToPlot/splitHist
//
//
//
//  LogInfo << "Generating and writing sample histograms..." << std::endl;
//  std::map<std::string, TH1D*> TH1D_handler;
//  std::map<std::string, int> splitVarColor;
//  int sampleCounter = 0;
//  for(const auto& anaSample : sampleList_){
//
//    std::map<std::string, TH1D*> tempHistMap;
//
//    LogDebug << "Processing histograms for: " << anaSample->GetName() << std::endl;
//
//    std::string histNameBuffer;
//
//    // Sample's raw histograms (what's actually fitted)
//    histNameBuffer              = anaSample->GetName() + "/Raw/MC";
//    tempHistMap[histNameBuffer] = (TH1D*) anaSample->GetPredHisto().Clone();
//    tempHistMap[histNameBuffer]->SetName(histNameBuffer.c_str());
//    tempHistMap[histNameBuffer]->SetTitle("MC");
//    histNameBuffer              = anaSample->GetName() + "/Raw/Data";
//    tempHistMap[histNameBuffer] = (TH1D*) anaSample->GetDataHisto().Clone();
//    tempHistMap[histNameBuffer]->SetName(histNameBuffer.c_str());
//    tempHistMap[histNameBuffer]->SetTitle("Data");
//
//    // Build the histograms binning (D1)
//    // TODO: check if a simple binning can be applied
//    bool isSimpleBinning = true;
//    std::vector<double> D1binning;
//    auto bins = anaSample->GetBinEdges();
//    for(const auto& bin : bins){
//      if(D1binning.empty()) D1binning.emplace_back(bin.getBinLowEdge(1));
//      if(not GenericToolbox::doesElementIsInVector(bin.getBinHighEdge(1), D1binning)){
//        D1binning.emplace_back(bin.getBinHighEdge(1));
//      }
//    }
//    std::sort(D1binning.begin(), D1binning.end());
//
//    histNameBuffer              = anaSample->GetName() + "/D1/MC";
//    tempHistMap[histNameBuffer] = new TH1D(histNameBuffer.c_str(), "MC",
//                                           D1binning.size() - 1, &D1binning[0]);
//    histNameBuffer              = anaSample->GetName() + "/D1/Data";
//    tempHistMap[histNameBuffer] = new TH1D(histNameBuffer.c_str(), "Data",
//                                           D1binning.size() - 1, &D1binning[0]);
//
//    // Build the histograms binning (D2)
//    // TODO: check if a simple binning can be applied
//    isSimpleBinning = true;
//    std::vector<double> D2binning;
//    for(const auto& bin : bins){
//      if(D2binning.empty()) D2binning.emplace_back(bin.getBinLowEdge(0));
//      if(not GenericToolbox::doesElementIsInVector(bin.getBinHighEdge(0), D2binning)){
//        D2binning.emplace_back(bin.getBinHighEdge(0));
//      }
//    }
//    std::sort(D2binning.begin(), D2binning.end());
//
//    histNameBuffer              = anaSample->GetName() + "/D2/MC";
//    tempHistMap[histNameBuffer] = new TH1D(histNameBuffer.c_str(), "MC",
//                                           D2binning.size() - 1, &D2binning[0]);
//    histNameBuffer              = anaSample->GetName() + "/D2/Data";
//    tempHistMap[histNameBuffer] = new TH1D(histNameBuffer.c_str(), "Data",
//                                           D2binning.size() - 1, &D2binning[0]);
//
//    // Get the list of valid sub-divisions...
//    {   // Reaction
//      splitVarColor["reactions"] = 0;
//      std::vector<int> reactionCodesList;
//      for( size_t iEvent = 0 ; iEvent < anaSample->GetMcEvents().size() ; iEvent++ ){
//        auto* anaEvent = anaSample->GetEvent(iEvent);
//        if(not GenericToolbox::doesElementIsInVector(anaEvent->GetReaction(),
//                                                     reactionCodesList)){
//          reactionCodesList.emplace_back(anaEvent->GetReaction());
//        }
//      }
//      for(const auto& thisReactionCode : reactionCodesList){
//        histNameBuffer = anaSample->GetName() + "/D1/reactions/" + std::to_string(thisReactionCode);
//        tempHistMap[histNameBuffer] = (TH1D*) tempHistMap[anaSample->GetName() + "/D1/MC"]->Clone();
//        tempHistMap[histNameBuffer]->Reset("ICESM");
//        tempHistMap[histNameBuffer]->SetName(histNameBuffer.c_str());
//        tempHistMap[histNameBuffer]->SetTitle(reactionNamesAndColors[thisReactionCode].first.c_str());
//
//        histNameBuffer = anaSample->GetName() + "/D2/reactions/" + std::to_string(thisReactionCode);
//        tempHistMap[histNameBuffer] = (TH1D*) tempHistMap[anaSample->GetName() + "/D2/MC"]->Clone();
//        tempHistMap[histNameBuffer]->Reset("ICESM");
//        tempHistMap[histNameBuffer]->SetName(histNameBuffer.c_str());
//        tempHistMap[histNameBuffer]->SetTitle(reactionNamesAndColors[thisReactionCode].first.c_str());
//
//        histNameBuffer = anaSample->GetName() + "/Raw/reactions/" + std::to_string(thisReactionCode);
//        tempHistMap[histNameBuffer] = (TH1D*) tempHistMap[anaSample->GetName() + "/Raw/MC"]->Clone();
//        tempHistMap[histNameBuffer]->Reset("ICESM");
//        tempHistMap[histNameBuffer]->SetName(histNameBuffer.c_str());
//        tempHistMap[histNameBuffer]->SetTitle(reactionNamesAndColors[thisReactionCode].first.c_str());
//      }
//    }
//
//
//
//    // Fill the histograms (MC)
//    for( size_t iEvent = 0 ; iEvent < anaSample->GetMcEvents().size() ; iEvent++ ){
//      auto* anaEvent = &anaSample->GetMcEvents()[iEvent];
//      tempHistMap[anaSample->GetName() + "/D1/MC"]->Fill(
//        anaEvent->GetRecoD1(), anaEvent->GetEvWght()
//      );
//      tempHistMap[anaSample->GetName() + "/D2/MC"]->Fill(
//        anaEvent->GetRecoD2(), anaEvent->GetEvWght()
//      );
//      tempHistMap[anaSample->GetName() + "/D1/reactions/" + std::to_string(anaEvent->GetReaction())]->Fill(
//        anaEvent->GetRecoD1(), anaEvent->GetEvWght()
//      );
//      tempHistMap[anaSample->GetName() + "/D2/reactions/" + std::to_string(anaEvent->GetReaction())]->Fill(
//        anaEvent->GetRecoD2(), anaEvent->GetEvWght()
//      );
//      tempHistMap[anaSample->GetName() + "/Raw/reactions/" + std::to_string(anaEvent->GetReaction())]->Fill(
//        anaEvent->GetRecoBinIndex() + 0.5, anaEvent->GetEvWght()
//      );
//    }
//
//    // Fill the histograms (Data)
//    for( size_t iEvent = 0 ; iEvent < anaSample->GetDataEvents().size() ; iEvent++ ){
//      auto* anaEvent = &anaSample->GetDataEvents()[iEvent];
//      tempHistMap[anaSample->GetName() + "/D1/Data"]->Fill(
//        anaEvent->GetRecoD1(), anaEvent->GetEvWght()
//      );
//      tempHistMap[anaSample->GetName() + "/D2/Data"]->Fill(
//        anaEvent->GetRecoD2(), anaEvent->GetEvWght()
//      );
//    }
//
//    // Cosmetics, Normalization and Write
//    for(auto& histPair : tempHistMap){
//
//      auto pathElements = GenericToolbox::splitString(histPair.first, "/");
//      std::string xVarName = pathElements[1]; // SampleName/Var/SplitVar/...
//      std::string splitVarName;
//      std::string splitVarCode;
//      if(pathElements.size() >= 4){
//        splitVarName = pathElements[2]; // "reactions" for example
//        splitVarCode = pathElements[3]; // "0" for example
//      }
//
//      histPair.second->GetXaxis()->SetTitle(xVarName.c_str());
//      if(xVarName == "D1"){
//        // Get Number of counts per 100 MeV
//        for(int iBin = 0 ; iBin <= histPair.second->GetNbinsX()+1 ; iBin++){
//          histPair.second->SetBinContent( iBin, histPair.second->GetBinContent(iBin)/histPair.second->GetBinWidth(iBin)*100.);
//          histPair.second->SetBinError( iBin, TMath::Sqrt(histPair.second->GetBinContent(iBin))/histPair.second->GetBinWidth(iBin)*100.);
//        }
//        histPair.second->GetYaxis()->SetTitle("Counts/(100 MeV)");
//        histPair.second->GetXaxis()->SetRangeUser(histPair.second->GetXaxis()->GetXmin(),maxD1Scale);
//      }
//      else {
//        histPair.second->GetYaxis()->SetTitle("Counts");
//      }
//
//      if(pathElements.back() == "Data"){
//        // IS DATA
//        if( anaSample->GetDataType() == DataType::kAsimov
//            and xVarName != "Raw" // RAW IS NOT RENORMALIZED
//          ){
//          histPair.second->Scale(anaSample->GetNorm());
//        }
//        for( int iBin = 0 ; iBin <= histPair.second->GetNbinsX()+1 ; iBin++ ){
//          histPair.second->SetBinError(iBin, TMath::Sqrt(histPair.second->GetBinContent(iBin)));
//        }
//        histPair.second->SetLineColor(kBlack);
//        histPair.second->SetMarkerColor(kBlack);
//        histPair.second->SetMarkerStyle(kFullDotLarge);
//        histPair.second->SetOption("EP");
//      }
//      else{
//        // IS MC (if it's broken down by reaction, is MC too)
//        if(xVarName != "Raw" or not splitVarName.empty()){ // RAW IS NOT RENORMALIZED, unless we rebuild it (for split var)
//          histPair.second->Scale(anaSample->GetNorm());
//        }
//
//        if(splitVarName == "reactions"){
//          histPair.second->SetLineColor(reactionNamesAndColors[stoi(splitVarCode)].second);
////                    histPair.second->SetFillColorAlpha(reactionNamesAndColors[stoi(splitVarCode)].second, 0.8);
//          histPair.second->SetFillColor(reactionNamesAndColors[stoi(splitVarCode)].second);
//          histPair.second->SetMarkerColor(reactionNamesAndColors[stoi(splitVarCode)].second);
//        }
//        else if( not splitVarName.empty() ){
//          histPair.second->SetLineColor(
//            defaultColorWheel[splitVarColor[splitVarName]% defaultColorWheel.size()]);
//          histPair.second->SetFillColor(
//            defaultColorWheel[splitVarColor[splitVarName]% defaultColorWheel.size()]);
//          histPair.second->SetMarkerColor(
//            defaultColorWheel[splitVarColor[splitVarName]% defaultColorWheel.size()]);
//          splitVarColor[splitVarName]++;
//        }
//        else{
//          histPair.second->SetLineColor(defaultColorWheel[sampleCounter% defaultColorWheel.size()]);
//          histPair.second->SetFillColor(defaultColorWheel[sampleCounter% defaultColorWheel.size()]);
//          histPair.second->SetMarkerColor(defaultColorWheel[sampleCounter% defaultColorWheel.size()]);
//        }
//
//        histPair.second->SetOption("HIST");
//      }
//
//      histPair.second->SetLineWidth(2);
//      histPair.second->GetYaxis()->SetRangeUser(
//        histPair.second->GetMinimum(0), // 0 or lower as min will prevent to set log scale
//        histPair.second->GetMaximum()*1.2
//      );
//
//      // Writing the histogram
//      std::string subFolderPath = GenericToolbox::joinVectorString(pathElements, "/", 0, -1);
//      GenericToolbox::mkdirTFile(samplesDir, subFolderPath)->cd();
//      TH1D* tempHistPtr = (TH1D*) histPair.second->Clone();
//      tempHistPtr->Write((pathElements.back() + "_TH1D").c_str());
//
//    }
//
//    GenericToolbox::appendToMap(TH1D_handler, tempHistMap);
//
//    // Next Loop
//    sampleCounter++;
//
//  }
//
//
//  // Building canvas
//  LogInfo << "Generating and writing sample canvas..." << std::endl;
//  std::map<std::string, std::vector<TCanvas*>> canvasHandler;
//  int nbXPlots       = JsonUtils::fetchValue(_canvasParameters_, "nbXplots", 3);
//  int nbYPlots       = JsonUtils::fetchValue(_canvasParameters_, "nbXplots", 2);
//  int nbSamples      = sampleList_.size();
//
//
//  int sampleCounter = 0;
//  while(sampleCounter != nbSamples){
//    std::stringstream canvasName;
//    canvasName << "samples_" << sampleCounter+1;
//    sampleCounter += nbXPlots*nbYPlots;
//    if(sampleCounter > nbSamples){
//      sampleCounter = nbSamples;
//    }
//    canvasName << "_to_" << sampleCounter;
//
//    std::string pathBuffer;
//    for(auto& canvasSubFolder : canvasSubFolderList){
//      pathBuffer = canvasSubFolder + "/" + canvasName.str();
//      canvasHandler[canvasSubFolder].emplace_back(
//        new TCanvas(pathBuffer.c_str(), canvasName.str().c_str(),
//                    JsonUtils::fetchValue(_canvasParameters_, "width", 1200),
//                    JsonUtils::fetchValue(_canvasParameters_, "height", 700)
//                    ));
//      canvasHandler[canvasSubFolder].back()->Divide(nbXPlots,nbYPlots);
//    }
//  }
//
//
//  for(auto& canvasFolderPair : canvasHandler){
//
//    int canvasIndex = 0;
//    int iSlot       = 1;
//    for(const auto& anaSample : sampleList_){
//
//      if(iSlot > nbXPlots*nbYPlots){
//        canvasIndex++;
//        iSlot = 1;
//      }
//
//      canvasFolderPair.second[canvasIndex]->cd(iSlot);
//
//      auto subFolderList = GenericToolbox::splitString(canvasFolderPair.first, "/", true);
//      std::string xVarName = subFolderList[0];
//
//      TH1D* dataSampleHist = TH1D_handler[anaSample->GetName() + "/" + xVarName + "/Data"];
//      std::vector<TH1D*> mcSampleHistList;
//
//      if(subFolderList.size() == 1){ // no splitting of MC hist
//        mcSampleHistList.emplace_back(TH1D_handler[anaSample->GetName() + "/" + xVarName + "/MC"]);
//      }
//      else{
//        std::string splitVarName = subFolderList[1];
//        for(auto& histNamePair : TH1D_handler){
//          if(GenericToolbox::doesStringStartsWithSubstring(
//            histNamePair.first,
//            anaSample->GetName() + "/" + xVarName + "/" + splitVarName + "/"
//          )){
//            mcSampleHistList.emplace_back(histNamePair.second);
//          }
//        }
//      }
//
//      if(mcSampleHistList.empty()) continue;
//
//      // Sorting histograms by norm (lowest stat first)
//      std::function<bool(TH1D*, TH1D*)> aGoesFirst = [maxD1Scale](TH1D* histA_, TH1D* histB_){
//        bool aGoesFirst = true; // A is smaller = A goes first
//        double Xmax = histA_->GetXaxis()->GetXmax();
//        if(maxD1Scale > 0) Xmax = maxD1Scale;
//        if(  histA_->Integral(histA_->FindBin(0), histA_->FindBin(Xmax)-1)
//             > histB_->Integral(histB_->FindBin(0), histB_->FindBin(Xmax)-1) ) aGoesFirst = false;
//        return aGoesFirst;
//      };
//      auto p = GenericToolbox::getSortPermutation( mcSampleHistList, aGoesFirst );
//      mcSampleHistList = GenericToolbox::applyPermutation(mcSampleHistList, p);
//
//      // Stacking histograms
//      TH1D* histPileBuffer = nullptr;
//      double minYValue = 1;
//      for( size_t iHist = 0 ; iHist < mcSampleHistList.size() ; iHist++ ){
//        if(minYValue > mcSampleHistList.back()->GetMinimum(0)){
//          minYValue = mcSampleHistList.back()->GetMinimum(0);
//        }
//        if(histPileBuffer != nullptr) mcSampleHistList[iHist]->Add(histPileBuffer);
//        histPileBuffer = mcSampleHistList[iHist];
//      }
//
//      // Draw (the one on top of the pile should be drawn first, otherwise it will hide the others...)
//      int lastIndex = mcSampleHistList.size()-1;
//      for( int iHist = lastIndex ; iHist >= 0 ; iHist-- ){
//        if( iHist == lastIndex ){
//          mcSampleHistList[iHist]->GetYaxis()->SetRangeUser(
//            minYValue,
//            mcSampleHistList[iHist]->GetMaximum()*1.2
//          );
//          mcSampleHistList[iHist]->Draw("HIST");
//        }
//        else {
//          mcSampleHistList[iHist]->Draw("HISTSAME");
//        }
//      }
//
//      dataSampleHist->SetTitle("Data");
//      dataSampleHist->Draw("EPSAME");
//
//      // Legend
//      double Xmax = 0.9;
//      double Ymax = 0.9;
//      double Xmin = 0.5;
//      double Ymin = Ymax - 0.04*(mcSampleHistList.size()+1);
//      gPad->BuildLegend(Xmin,Ymin,Xmax,Ymax);
//
//      mcSampleHistList[lastIndex]->SetTitle(anaSample->GetName().c_str()); // the actual title
//      gPad->SetGridx();
//      gPad->SetGridy();
//      iSlot++;
//
//    } // sample
//
//    samplesDir->cd();
//    GenericToolbox::mkdirTFile(samplesDir, "canvas/" + canvasFolderPair.first)->cd();
//    for(auto& canvas : canvasFolderPair.second){
//      canvas->Write((canvas->GetTitle() + std::string("_TCanvas")).c_str());
//    }
//
//  } // canvas

}


