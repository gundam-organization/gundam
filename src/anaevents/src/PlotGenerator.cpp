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
#include "PlotGenerator.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[PlotGenerator]");
})


PlotGenerator::PlotGenerator() { this->reset(); }
PlotGenerator::~PlotGenerator() { this->reset(); }

void PlotGenerator::reset() {

  defaultColorWheel = {
    kGreen-3, kTeal+3, kAzure+7,
    kCyan-2, kBlue-7, kBlue+2,
    kOrange+1, kOrange+9, kRed+2, kPink+9
  };

  _config_.clear();
  _sampleListPtr_ = nullptr;

}

void PlotGenerator::setConfig(const nlohmann::json &config) {
  _config_ = config;
  while( _config_.is_string() ){
    _config_ = JsonUtils::readConfigFile(_config_.get<std::string>());
  }
}
void PlotGenerator::setSampleListPtr(const std::vector<AnaSample> *sampleListPtr_) {
  PlotGenerator::_sampleListPtr_ = sampleListPtr_;
}

void PlotGenerator::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  if(_sampleListPtr_ == nullptr ){
    LogError << "_sampleListPtr_ not set." << std::endl;
    throw std::logic_error("_sampleListPtr_ not set.");
  }

  if( _config_.empty() ){
    LogError << "_config_ not set." << std::endl;
    throw std::logic_error("_config_ not set.");
  }

  _varDictionary_ = JsonUtils::fetchValue(_config_, "varDictionnaries", nlohmann::json());
  _canvasParameters_ = JsonUtils::fetchValue(_config_, "canvasParameters", nlohmann::json());
  _histogramsDefinition_ = JsonUtils::fetchValue(_config_, "histogramsDefinition", nlohmann::json());

  this->readHistogramsConfig();

}

const std::vector<HistHolder> &PlotGenerator::getHistHolderList() const {
  return _histHolderList_;
}
std::map<std::string, TCanvas *> PlotGenerator::getBufferCanvasList() const {
  return _bufferCanvasList_;
}

void PlotGenerator::generateSamplePlots(TDirectory *saveDir_) {

  LogWarning << "Generating sample plots..." << std::endl;

  this->generateSampleHistograms(GenericToolbox::mkdirTFile(saveDir_, "histograms"));
  this->generateCanvas(_histHolderList_, GenericToolbox::mkdirTFile(saveDir_, "canvas"));

}
void PlotGenerator::generateSampleHistograms(TDirectory *saveDir_) {

  if( _histogramsDefinition_.empty() ){
    LogError << "No histogram has been defined." << std::endl;
    return;
  }

  auto* lastDir = gDirectory;
  if(saveDir_ != nullptr ){
    saveDir_->cd();
    LogInfo << "Samples plots will be writen in: " << saveDir_->GetPath() << std::endl;
  }

  // Create histograms
  for( auto& histDef : _histHolderList_ ){

    TH1D* hist;
    if( histDef.varToPlot == "Raw" ){
      if( histDef.isData ) hist = (TH1D*) histDef.samplePtr->GetDataHisto().Clone();
      else hist = (TH1D*) histDef.samplePtr->GetPredHisto().Clone();
    }
    else{
      hist = new TH1D(
        histDef.histName.c_str(), histDef.histTitle.c_str(),
        int(histDef.xEdges.size()) - 1, &histDef.xEdges[0]
      );
    }

    auto* dir = GenericToolbox::mkdirTFile(saveDir_, histDef.folderPath);
    hist->SetDirectory(dir); // memory handled by ROOT

    histDef.histPtr = hist;
  }

  // Fill histograms
  for( const auto& sample : *_sampleListPtr_ ){

    // Data sets:
    for( bool isData : { false, true } ){

      const std::vector<AnaEvent>* eventListPtr;
      std::vector<HistHolder*> histPtrToFillList;

      if( isData ){
        eventListPtr = &sample.GetConstDataEvents();

        // which hist should be filled?
        for( auto& histDef : _histHolderList_ ){
          if( histDef.isData and histDef.samplePtr == &sample and histDef.fillFunction ){
            histPtrToFillList.emplace_back( &histDef );
          }
        }
      }
      else{
        eventListPtr = &sample.GetConstMcEvents();

        // which hist should be filled?
        for( auto& histDef : _histHolderList_ ){
          if( not histDef.isData and histDef.samplePtr == &sample and histDef.fillFunction ){
            histPtrToFillList.emplace_back( &histDef );
          }
        }
      }

      // Filling the selected histograms
      LogTrace << "Filling the selected histograms for: " << sample.GetName() << " / " << (isData ? "data": "mc")  << std::endl;
      for( const auto& event : *eventListPtr ){
        for( auto& hist : histPtrToFillList ){
          hist->fillFunction(hist->histPtr ,&event);
        }
      }

    } // isData loop

  } // sample

  // Post-processing (norm, color)
  for( auto& histDefPair : _histHolderList_ ){

    if( not histDefPair.isData or histDefPair.samplePtr->GetDataType() == kAsimov ){
      histDefPair.histPtr->Scale( histDefPair.samplePtr->GetNorm() );
    }

    for(int iBin = 0 ; iBin <= histDefPair.histPtr->GetNbinsX() + 1 ; iBin++ ){
      double binContent = histDefPair.histPtr->GetBinContent(iBin); // this is the real number of counts
      double errorFraction = TMath::Sqrt(binContent)/binContent; // error fraction will not change with renorm of bin width

      if( histDefPair.rescaleAsBinWidth ) binContent /= histDefPair.histPtr->GetBinWidth(iBin);
      if( histDefPair.rescaleBinFactor != 1. ) binContent *= histDefPair.rescaleBinFactor;
      histDefPair.histPtr->SetBinContent( iBin, binContent );
      histDefPair.histPtr->SetBinError( iBin, binContent * errorFraction );

      histDefPair.histPtr->SetDrawOption("EP");
    }

    histDefPair.histPtr->GetXaxis()->SetTitle(histDefPair.xTitle.c_str());
    histDefPair.histPtr->GetYaxis()->SetTitle(histDefPair.yTitle.c_str());

    histDefPair.histPtr->SetLineWidth(2);
    histDefPair.histPtr->SetLineColor(histDefPair.histColor);

    histDefPair.histPtr->SetMarkerStyle( kFullDotLarge );
    histDefPair.histPtr->SetMarkerColor(histDefPair.histColor);

    histDefPair.histPtr->SetFillColor(histDefPair.histColor);
  }

  // Saving
  if( saveDir_ != nullptr ){
    for( auto& histDefPair : _histHolderList_ ){
//      LogDebug << "Writing histogram: " << histDefPair.histPtr->GetName() << " in " << histDefPair.folderPath << std::endl;
      GenericToolbox::mkdirTFile( saveDir_, histDefPair.folderPath )->cd();
      histDefPair.histPtr->Write( histDefPair.histName.c_str() );
      saveDir_->cd();
    }
  }

  lastDir->cd();

}
void PlotGenerator::generateCanvas(const std::vector<HistHolder> &histHolderList_, TDirectory *saveDir_, bool stackHist_){
  auto *lastDir = gDirectory;
  if (saveDir_ != nullptr) {
    saveDir_->cd();
    LogInfo << "Samples plots will be writen in: " << saveDir_->GetPath() << std::endl;
  }

  auto buildCanvasPath = [](const HistHolder* hist_){
    std::stringstream ss;
    ss << hist_->varToPlot << hist_->prefix << "/";
    if( not hist_->splitVarName.empty() ) ss << hist_->splitVarName << "/";
    return ss.str();
  };

  std::map<std::string, std::map<const AnaSample*, std::vector<const HistHolder*>>> histsToStackMap; // histsToStackMap[path][samplePtr] = listOfTh1d
  for( auto& histHolder : histHolderList_ ){
    if( histHolder.isData ) continue; // data associated to each later
    histsToStackMap[buildCanvasPath(&histHolder)][histHolder.samplePtr].emplace_back( &histHolder );
  }

  // Now search for the associated data hist
  for( auto& histsSamples : histsToStackMap ){
    const std::string& canvasPath = histsSamples.first;
    for( auto& histsToStack : histsSamples.second ){
      for( auto& histHolder : histHolderList_ ){
        const AnaSample* samplePtr = histsToStack.first;

        // we are searching for data
        if( not histHolder.isData ) continue;

        // canvasPath may have the trailing splitVar -> need to check the beginning of the path with the Data hist
        if( not GenericToolbox::doesStringStartsWithSubstring(canvasPath, buildCanvasPath(&histHolder)) ) continue;

        // same sample?
        if( samplePtr != histHolder.samplePtr ) continue;

        histsToStack.second.emplace_back( &histHolder );
        break; // no need to check for more
      }

    }
  }


  int canvasHeight = JsonUtils::fetchValue(_canvasParameters_, "height", 700);
  int canvasWidth = JsonUtils::fetchValue(_canvasParameters_, "width", 1200);
  int canvasNbXplots = JsonUtils::fetchValue(_canvasParameters_, "nbXplots", 3);
  int canvasNbYplots = JsonUtils::fetchValue(_canvasParameters_, "nbYplots", 2);

  // Canvas builder
  for ( const auto &histsToStackPair : histsToStackMap ) {
    const std::string canvasFolderPath = histsToStackPair.first;

    int canvasIndex = 0;
    int iSampleSlot = 0;
    for (const auto &histList : histsToStackPair.second) {
      const AnaSample* samplePtr = histList.first;
      iSampleSlot++;

      if (iSampleSlot > canvasNbXplots * canvasNbYplots) {
        canvasIndex++;
        iSampleSlot = 1;
      }

      std::string canvasName = "samples_n" + std::to_string(canvasIndex);
      std::string canvasPath = canvasFolderPath + canvasName;
      if (not GenericToolbox::doesKeyIsInMap(canvasPath, _bufferCanvasList_)) {
        _bufferCanvasList_[canvasPath] = new TCanvas(canvasPath.c_str(), canvasPath.c_str(), canvasWidth, canvasHeight);
        _bufferCanvasList_[canvasPath]->Divide(canvasNbXplots, canvasNbYplots);
      }
      _bufferCanvasList_[canvasPath]->cd(iSampleSlot);

      // separating histograms
      TH1D *dataSampleHist{nullptr};
      std::vector<TH1D *> mcSampleHistList;
      double minYValue = 1;
      for( const auto* histHolder : histList.second ) {
        TH1D* hist = histHolder->histPtr;
        if ( histHolder->isData ) {
          dataSampleHist = hist;
        }
        else {
          mcSampleHistList.emplace_back(hist);
          minYValue = std::min(minYValue, hist->GetMinimum(0));
        }
      }

      TH1D* firstHistToPlot{nullptr};

      // process mc part
      std::vector<TH1D *> mcSampleHistAccumulatorList;
      if (not mcSampleHistList.empty()) {

        if (stackHist_) {
          // Sorting histograms by norm (lowest stat first)
          std::function<bool(TH1D *, TH1D *)> aGoesFirst = [](TH1D *histA_, TH1D *histB_) {
            return (histA_->Integral(histA_->FindBin(0), histA_->FindBin(histA_->GetXaxis()->GetXmax()))
                    < histB_->Integral(histB_->FindBin(0), histB_->FindBin(histB_->GetXaxis()->GetXmax())));
          };
          auto p = GenericToolbox::getSortPermutation(mcSampleHistList, aGoesFirst);
          mcSampleHistList = GenericToolbox::applyPermutation(mcSampleHistList, p);

          // Stacking histograms
          TH1D *histPileBuffer = nullptr;
          for (auto &hist : mcSampleHistList) {
            mcSampleHistAccumulatorList.emplace_back((TH1D *) hist->Clone()); // inheriting cosmetics as well
            if (histPileBuffer != nullptr) {
              mcSampleHistAccumulatorList.back()->Add(histPileBuffer);
            }
            histPileBuffer = mcSampleHistAccumulatorList.back();
          }

          // Draw the stack
          int lastIndex = int(mcSampleHistAccumulatorList.size()) - 1;
          for (int iHist = lastIndex; iHist >= 0; iHist--) {
            if ( firstHistToPlot == nullptr ) {
              firstHistToPlot = mcSampleHistAccumulatorList[iHist];
              mcSampleHistAccumulatorList[iHist]->GetYaxis()->SetRangeUser(minYValue,
                                                                           mcSampleHistAccumulatorList[iHist]->GetMaximum() *
                                                                           1.2);
              mcSampleHistAccumulatorList[iHist]->Draw("HIST");
            } else {
              mcSampleHistAccumulatorList[iHist]->Draw("HISTSAME");
            }
          }
        } else {
          // Just draw each hist on the same plot
          for (auto &mcHist : mcSampleHistList) {
            if ( firstHistToPlot == nullptr ) {
              firstHistToPlot = mcHist;
              if (mcSampleHistList.size() == 1) {
                // only one: draw error bars
                mcHist->Draw("EP");
              }
              else {
                // don't draw multiple error bars
                mcHist->Draw("HIST P");
              }
            }
            else {
              if (mcSampleHistList.size() == 1) { mcHist->Draw("EPSAME"); }
              else { mcHist->Draw("HIST P SAME"); }
            }
          } // mcHist
        } // stack?


      } // mcHistList empty?

      // Draw the data hist on top
      if (dataSampleHist != nullptr) {
        std::string originalTitle = dataSampleHist->GetTitle(); // title can be used for figuring out the type of the histogram
        dataSampleHist->SetTitle("Data");
        if ( firstHistToPlot != nullptr ) {
          dataSampleHist->Draw("EPSAME");
        } else {
          firstHistToPlot = dataSampleHist;
          dataSampleHist->Draw("EP");
        }
        dataSampleHist->SetTitle(originalTitle.c_str()); // restore
      }

      if( firstHistToPlot == nullptr ){
        // Nothing to plot here
        continue;
      }

      // Legend
      double Xmax = 0.9;
      double Ymax = 0.9;
      double Xmin = 0.5;
      double Ymin = Ymax - 0.04 * double(mcSampleHistList.size() + 1);
      gPad->BuildLegend(Xmin, Ymin, Xmax, Ymax);

      firstHistToPlot->SetTitle( samplePtr->GetName().c_str() ); // the actual displayed title
      gPad->SetGridx();
      gPad->SetGridy();

//      LogWarning << histsToStackPair.first << ": " << histList.first << " -> "
//                 << GenericToolbox::parseVectorAsString(histList.second) << std::endl;
    } // sample

  }

  // Write
  if (saveDir_ != nullptr) {
    for (auto &canvas : _bufferCanvasList_) {
      auto pathSplit = GenericToolbox::splitString(canvas.first, "/");
      std::string folderPath = GenericToolbox::joinVectorString(pathSplit, "/", 0, -1);
      std::string canvasName = pathSplit.back();
      GenericToolbox::mkdirTFile(saveDir_, folderPath)->cd();
      canvas.second->Write(canvasName.c_str());
      canvas.second->SetName((saveDir_->GetPath() + canvas.first).c_str()); // full path to avoid ROOT deleting
      saveDir_->cd();
    }
  }

  lastDir->cd();
}

void PlotGenerator::generateComparisonPlots(
  const std::vector<HistHolder> &histsToStackOther_,
  const std::vector<HistHolder> &histsToStackReference_,
  TDirectory *saveDir_){

  LogWarning << "Generating comparison plots..." << std::endl;

  this->generateComparisonHistograms(histsToStackOther_, histsToStackReference_, GenericToolbox::mkdirTFile(saveDir_, "histograms"));
  this->generateCanvas(_comparisonHistHolderList_, GenericToolbox::mkdirTFile(saveDir_, "canvas"), false);

}
void PlotGenerator::generateComparisonHistograms(const std::vector<HistHolder> &histList_, const std::vector<HistHolder> &refHistsList_, TDirectory *saveDir_) {

  auto* curDir = gDirectory;

  if(saveDir_ != nullptr){
    saveDir_->cd();
    LogInfo << "Comparison histograms will be writen in: " << saveDir_->GetPath() << std::endl;
  }
  _comparisonHistHolderList_.clear();

  for( const auto& histHolder : histList_ ){

    if( histHolder.isData ){
      // don't compare data
      continue;
    }

    // search for the corresponding reference
    const HistHolder* refHistHolderPtr{nullptr};
    for( const auto& refHistHolderCandidate : refHistsList_ ){
      if(histHolder.folderPath == refHistHolderCandidate.folderPath
      and histHolder.histName == refHistHolderCandidate.histName
      and histHolder.xEdges.size() == refHistHolderCandidate.xEdges.size()
      ){
        // FOUND!
        refHistHolderPtr = &refHistHolderCandidate;
        break;
      }
    }
    if( refHistHolderPtr == nullptr ) continue; // no available ref

    _comparisonHistHolderList_.emplace_back( histHolder ); // copy
    _comparisonHistHolderList_.back().histPtr = (TH1D*) histHolder.histPtr->Clone();
    TH1D* compHistPtr = _comparisonHistHolderList_.back().histPtr;

    for( int iBin = 0 ; iBin <= compHistPtr->GetNbinsX()+1 ; iBin++ ){

      if( refHistHolderPtr->histPtr->GetBinContent(iBin) == 0 ){
        // no division by 0
        compHistPtr->SetBinContent( iBin, 0 );
      }

      double binContent = compHistPtr->GetBinContent( iBin );
      binContent /= refHistHolderPtr->histPtr->GetBinContent(iBin);
      binContent -= 1;
      binContent *= 100.;
      compHistPtr->SetBinContent( iBin, binContent );
      compHistPtr->SetBinError( iBin, histHolder.histPtr->GetBinError(iBin) / histHolder.histPtr->GetBinContent(iBin) * 100 );

    } // iBin
    compHistPtr->GetYaxis()->SetTitle("Relative Deviation (%)");

    // Y axis
    double Ymin = 0;
    double Ymax = 0;
    for(int iBin = 0 ; iBin <= compHistPtr->GetNbinsX() + 1 ; iBin++){
      double val = compHistPtr->GetBinContent(iBin);
      double error = compHistPtr->GetBinError(iBin);
      if(val + error > Ymax){
        Ymax = val + error;
      }
      if(val - error < Ymin){
        Ymin = val - error;
      }
    }

    // Add 20% margin
    Ymin += Ymin*0.2;
    Ymax += Ymax*0.2;

    // Force showing Y=0
    if(Ymin > -0.2) Ymin = -0.2;
    if(Ymax < 0.2) Ymax = 0.2;

    compHistPtr->GetYaxis()->SetRangeUser(Ymin, Ymax);

    if( saveDir_ != nullptr ){
      GenericToolbox::mkdirTFile( saveDir_, _comparisonHistHolderList_.back().folderPath )->cd();
      compHistPtr->Write( _comparisonHistHolderList_.back().histName.c_str() );
      saveDir_->cd();
    }
  }

  curDir->cd();

}

void PlotGenerator::readHistogramsConfig() {
  _histHolderList_.clear();

  HistHolder histDefBase;
  int sampleCounter = -1;
  for( const auto& sample : *_sampleListPtr_ ){
    sampleCounter++;
    histDefBase.samplePtr = &sample;
    short unsetSplitValueColor = kGray; // will increment if needed

    std::map<std::string, std::vector<int>> splitValuesList;

    // Definition of histograms
    for( const auto& histConfig : _histogramsDefinition_ ){

      histDefBase.varToPlot = JsonUtils::fetchValue<std::string>(histConfig, "varToPlot");
      histDefBase.prefix = JsonUtils::fetchValue(histConfig, "prefix", "");
      histDefBase.rescaleAsBinWidth = JsonUtils::fetchValue(histConfig, "rescaleAsBinWidth", true);
      histDefBase.rescaleBinFactor = JsonUtils::fetchValue(histConfig, "rescaleBinFactor", 1.);

      std::vector<std::string> splitVars = JsonUtils::fetchValue(histConfig, "splitVars", std::vector<std::string>{""});

      for( const auto& splitVar : splitVars ){

        histDefBase.splitVarName = splitVar;

        // Fetch the appearing split vars in the sample
        if( not GenericToolbox::doesKeyIsInMap(histDefBase.splitVarName, splitValuesList) ){
          splitValuesList[histDefBase.splitVarName] = std::vector<int>();
          if( histDefBase.splitVarName.empty() ){
            splitValuesList[histDefBase.splitVarName].emplace_back(0); // just for the loop
          }
          else{
            for( const auto& event : sample.GetConstMcEvents() ){
              int splitValue = event.GetEventVarInt(histDefBase.splitVarName);
              if( not GenericToolbox::doesElementIsInVector(splitValue, splitValuesList[histDefBase.splitVarName]) ){
                splitValuesList[histDefBase.splitVarName].emplace_back(splitValue);
              }
            }
          }
        }

        // Loop over split vars
        int splitValueIndex = -1;
        for( const auto& splitValue : splitValuesList[histDefBase.splitVarName] ){
          splitValueIndex++;
          histDefBase.splitVarValue = splitValue;

          for( bool isData: {false, true} ){
            histDefBase.isData = isData;

            if( histDefBase.isData and
                ( not histDefBase.splitVarName.empty()
                  or JsonUtils::fetchValue(histConfig, "noData", false)
                ) ){
              continue;
            }

            if( histDefBase.varToPlot != "Raw" ){
              // Then filling the histo is needed

              // Binning
              histDefBase.xEdges.clear();

              histDefBase.xMin = JsonUtils::fetchValue(histConfig, "xMin", std::nan("nan"));;
              histDefBase.xMax = JsonUtils::fetchValue(histConfig, "xMax", std::nan("nan"));

              if( JsonUtils::fetchValue(histConfig, "useSampleBinning", false) ){

                int dimIndex = GenericToolbox::findElementIndex(histDefBase.varToPlot, sample.GetFitPhaseSpace());
                if( dimIndex == -1 ){
                  LogWarning << "Can't useSampleBinning since \"" << histDefBase.varToPlot
                             << "\" is not a dimension of the sample \"" << sample.GetName()
                             << "\": " << GenericToolbox::parseVectorAsString(sample.GetFitPhaseSpace()) << std::endl;
                  continue; // skip this sample
                }

                // Creating binning
                for( const auto& bin : sample.GetBinEdges() ){
                  auto edges = { bin.getBinLowEdge(dimIndex), bin.getBinHighEdge(dimIndex) };
                  for( const auto& edge : edges ){
                    if( ( histDefBase.xMin != histDefBase.xMin or histDefBase.xMin <= edge )
                        and ( histDefBase.xMax != histDefBase.xMax or histDefBase.xMax >= edge ) ){
                      // either NaN or in bounds
                      if( not GenericToolbox::doesElementIsInVector(edge, histDefBase.xEdges) ){
                        histDefBase.xEdges.emplace_back(edge);
                      }
                    }
                  }
                }
                if( histDefBase.xEdges.empty() ) continue; // skip
                std::sort( histDefBase.xEdges.begin(), histDefBase.xEdges.end() ); // sort for ROOT

              } // sample binning ?
              else{
                LogError << "Unsupported yet." << std::endl;
                throw std::logic_error("unsupported yet");
              }

              // Hist fill function
              auto splitVarValue = histDefBase.splitVarValue;
              auto varToPlot = histDefBase.varToPlot;
              auto splitVarName = histDefBase.splitVarName;
              histDefBase.fillFunction =
                [ splitVarValue, varToPlot, splitVarName ](TH1D* hist_, const AnaEvent* event_){
                  if( splitVarName.empty() or event_->GetEventVarInt(splitVarName) == splitVarValue){
                    hist_->Fill(event_->GetEventVarAsDouble(varToPlot), event_->GetEventWeight());
                  }
                };

            } // not Raw?

            histDefBase.xTitle = JsonUtils::fetchValue(histConfig, "xTitle", histDefBase.varToPlot);
            histDefBase.yTitle = JsonUtils::fetchValue(histConfig, "yTitle", "");
            if( histDefBase.yTitle.empty() ){
              histDefBase.yTitle = "Counts";
              if( histDefBase.rescaleAsBinWidth ) histDefBase.yTitle += " (/bin width)";
              if( histDefBase.rescaleBinFactor != 1. ) histDefBase.yTitle += "*" + std::to_string(histDefBase.rescaleBinFactor);
            }

            // Colors / Title (legend) / Name
            if( histDefBase.isData ){
              histDefBase.histName = "Data_TH1D";
              histDefBase.histTitle = "Data";
              histDefBase.histColor = kBlack;
            }
            else{

              histDefBase.histName = "MC_TH1D";

              if( histDefBase.splitVarName.empty() ){
                histDefBase.histTitle = "Prediction";
                histDefBase.histColor = defaultColorWheel[ sampleCounter % defaultColorWheel.size() ];
              }
              else{
                histDefBase.histTitle = "Prediction (" + splitVar + " == " + std::to_string(splitValue) + ")";
                histDefBase.histColor = defaultColorWheel[ splitValueIndex % defaultColorWheel.size() ];

                // User defined color?
                auto varDict = JsonUtils::fetchMatchingEntry(_varDictionary_, "name", splitVar); // does the cosmetic pars are configured?
                if( not varDict.empty() ){

                  if( varDict["dictionary"].is_null() ){
                    LogError << R"(Could not find "dictionary" key in JSON config for var: ")" << splitVar << "\"" << std::endl;
                    throw std::runtime_error("dictionary not found, by variable name found in JSON.");
                  }

                  // Look for the value we want
                  auto valDict = JsonUtils::fetchMatchingEntry(varDict["dictionary"], "value", splitValue);

                  histDefBase.histTitle = JsonUtils::fetchValue(valDict, "title", histDefBase.histTitle);
                  histDefBase.histColor = JsonUtils::fetchValue(valDict, "color", unsetSplitValueColor);
                  if( histDefBase.histColor == unsetSplitValueColor ) unsetSplitValueColor++; // increment for the next ones

                } // var dict?

              } // splitVar ?

            } // isData?

            // Config DONE : creating save path
            histDefBase.folderPath = sample.GetName();
            histDefBase.folderPath += "/" + histDefBase.varToPlot;
            if( not histDefBase.splitVarName.empty() ){
              histDefBase.folderPath += "/" + histDefBase.splitVarName;
              histDefBase.folderPath += "/" + std::to_string(histDefBase.splitVarValue);
            }

            // Config DONE
            _histHolderList_.emplace_back(histDefBase);

          } // isData
        } // splitValue
      } // splitVar
    } // histDef
  } // sample
}
