//
// Created by Nadrino on 16/06/2021.
//

#include "JsonUtils.h"
#include "GlobalVariables.h"
#include "PlotGenerator.h"

#include <memory>

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

#include "TCanvas.h"
#include "TH1D.h"
#include "TLegend.h"

#include "string"
#include "vector"
#include "sstream"

LoggerInit([]{
  Logger::setUserHeaderStr("[PlotGenerator]");
});


PlotGenerator::PlotGenerator() { this->reset(); }
PlotGenerator::~PlotGenerator() { this->reset(); }

void PlotGenerator::reset() {

  gStyle->SetOptStat(0);

  defaultColorWheel = {
    kGreen-3, kTeal+3, kAzure+7,
    kCyan-2, kBlue-7, kBlue+2,
    kOrange+1, kOrange+9, kRed+2, kPink+9
  };

  _config_.clear();
  _fitSampleSetPtr_ = nullptr;

  _histHolderCacheList_.resize(1);
}

void PlotGenerator::setConfig(const nlohmann::json &config_) {
  _config_ = config_;
  while( _config_.is_string() ){
    _config_ = JsonUtils::readConfigFile(_config_.get<std::string>());
  }
}
void PlotGenerator::setFitSampleSetPtr(const FitSampleSet *fitSampleSetPtr) {
  _fitSampleSetPtr_ = fitSampleSetPtr;
}

void PlotGenerator::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

//  LogThrowIf(_config_.empty(), "Config is not set.");
  _varDictionary_ = JsonUtils::fetchValue(_config_, "varDictionnaries", nlohmann::json());
  _canvasParameters_ = JsonUtils::fetchValue(_config_, "canvasParameters", nlohmann::json());
  _histogramsDefinition_ = JsonUtils::fetchValue(_config_, "histogramsDefinition", nlohmann::json());
}
void PlotGenerator::defineHistogramHolders() {
  LogWarning << __METHOD_NAME__ << std::endl;
  _histHolderCacheList_[0].clear();

  HistHolder histDefBase;
  if( _fitSampleSetPtr_ != nullptr ){

    LogInfo << "Fetching appearing split vars..." << std::endl;
    std::map<std::string, std::map<const FitSample*, std::vector<int>>> splitVarsDictionary;
    for( const auto& histConfig : _histogramsDefinition_ ){
      auto splitVars = JsonUtils::fetchValue(histConfig, "splitVars", std::vector<std::string>{""});
      for( auto& splitVar : splitVars ){
        if( not GenericToolbox::doesKeyIsInMap(splitVar, splitVarsDictionary) ){
          for( const auto& sample : _fitSampleSetPtr_->getFitSampleList() ){
            splitVarsDictionary[splitVar][&sample] = std::vector<int>();
            if( splitVar.empty() ) splitVarsDictionary[splitVar][&sample].emplace_back(0); // placeholder for no split var
          }
        }
      }
    }

    std::function<void(int)> fetchSplitVar = [&](int iThread_){
      int iSample{-1};
      int iHistDef;
      for( const auto& sample : _fitSampleSetPtr_->getFitSampleList() ){
        iSample++;
        if( iSample % GlobalVariables::getNbThreads() != iThread_ ){ continue; }
        for( const auto& event : sample.getMcContainer().eventList ){
          for( auto& splitVarInstance: splitVarsDictionary ){
            if(splitVarInstance.first.empty()) continue;
            int splitValue = event.getVarValue<int>(splitVarInstance.first);
            if( not GenericToolbox::doesElementIsInVector(splitValue, splitVarInstance.second[&sample]) ){
              splitVarInstance.second[&sample].emplace_back(splitValue);
            }
          } // splitVarList
        } // Event
      } // Sample
    };
    GlobalVariables::getParallelWorker().addJob("fetchSplitVar", fetchSplitVar);
    GlobalVariables::getParallelWorker().runJob("fetchSplitVar");
    GlobalVariables::getParallelWorker().removeJob("fetchSplitVar");

    int sampleCounter = -1;
    for( const auto& sample : _fitSampleSetPtr_->getFitSampleList() ){
      LogInfo << "Defining holders for sample: \"" << sample.getName() << "\"" << std::endl;
      sampleCounter++;
      histDefBase.fitSamplePtr = &sample;
      short unsetSplitValueColor = kGray; // will increment if needed

      // Definition of histograms
      for( const auto& histConfig : _histogramsDefinition_ ){

        histDefBase.varToPlot         = JsonUtils::fetchValue<std::string>(histConfig, "varToPlot");
        histDefBase.prefix            = JsonUtils::fetchValue(histConfig, "prefix", "");
        histDefBase.rescaleAsBinWidth = JsonUtils::fetchValue(histConfig, "rescaleAsBinWidth", true);
        histDefBase.rescaleBinFactor  = JsonUtils::fetchValue(histConfig, "rescaleBinFactor", 1.);

        if( GenericToolbox::splitString(histDefBase.varToPlot, ":").size() >= 2 ){
          LogAlert << "Skipping 2D plot def: " << histDefBase.varToPlot << std::endl;
          continue;
        }
        else{
          auto splitVars = JsonUtils::fetchValue(histConfig, "splitVars", std::vector<std::string>{""});

          for( const auto& splitVar : splitVars ){

            histDefBase.splitVarName = splitVar;

            // Loop over split vars
            int splitValueIndex = -1;
            for( const auto& splitValue : splitVarsDictionary[histDefBase.splitVarName][&sample] ){
              splitValueIndex++;
              histDefBase.splitVarValue = splitValue;

              for( bool isData: {false, true} ){
                histDefBase.isData = isData;
                bool buildFillFunction = false;

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

                    std::string sampleObsBinning = JsonUtils::fetchValue(histConfig, "useSampleBinningOfObservable", histDefBase.varToPlot);

                    for( const auto& bin : sample.getBinning().getBinsList() ){
                      const auto& edges = bin.getVarEdges(sampleObsBinning);
                      for( const auto& edge : { edges.first, edges.second } ) {
                        if ((histDefBase.xMin != histDefBase.xMin or histDefBase.xMin <= edge)
                            and (histDefBase.xMax != histDefBase.xMax or histDefBase.xMax >= edge)) {
                          // either NaN or in bounds
                          if (not GenericToolbox::doesElementIsInVector(edge, histDefBase.xEdges)) {
                            histDefBase.xEdges.emplace_back(edge);
                          }
                        }
                      }
                    }
                    if( histDefBase.xEdges.empty() ) continue; // skip
                    std::sort( histDefBase.xEdges.begin(), histDefBase.xEdges.end() ); // sort for ROOT

                  } // sample binning ?
                  else if( JsonUtils::doKeyExist(histConfig, "binningFile") ){
                    DataBinSet b;
                    b.readBinningDefinition(JsonUtils::fetchValue<std::string>(histConfig, "binningFile") );
                    LogThrowIf(b.getBinVariables().size()!=1, "Binning should be defined with only one variable, here: " << GenericToolbox::parseVectorAsString(b.getBinVariables()))

                    for(const auto& bin: b.getBinsList()){
                      const auto& edges = bin.getVarEdges(b.getBinVariables()[0]);
                      for( const auto& edge : { edges.first, edges.second } ) {
                        if ((histDefBase.xMin != histDefBase.xMin or histDefBase.xMin <= edge)
                            and (histDefBase.xMax != histDefBase.xMax or histDefBase.xMax >= edge)) {
                          // either NaN or in bounds
                          if (not GenericToolbox::doesElementIsInVector(edge, histDefBase.xEdges)) {
                            histDefBase.xEdges.emplace_back(edge);
                          }
                        }
                      }
                    }
                  }
                  else{
                    LogThrow("Could not find the binning definition.")
                  }

                  // Hist fill function
                  buildFillFunction = true; // build Later

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
                  histDefBase.fillStyle = 1001;
                }
                else{
                  histDefBase.histName = "MC_TH1D";

                  if( histDefBase.splitVarName.empty() ){
                    histDefBase.histTitle = "Prediction";
                    histDefBase.histColor = defaultColorWheel[ sampleCounter % defaultColorWheel.size() ];
                    histDefBase.fillStyle = 1001;
                  }
                  else{
                    histDefBase.histTitle = "Prediction (" + splitVar + " == " + std::to_string(splitValue) + ")";
                    histDefBase.histColor = defaultColorWheel[ splitValueIndex % defaultColorWheel.size() ];

                    // User defined color?
                    auto varDict = JsonUtils::fetchMatchingEntry(_varDictionary_, "name", splitVar); // does the cosmetic pars are configured?
                    auto dictEntries = varDict["dictionary"];
                    JsonUtils::forwardConfig(dictEntries);

                    if( not varDict.empty() ){

                      if( dictEntries.is_null() ){
                        LogError << R"(Could not find "dictionary" key in JSON config for var: ")" << splitVar << "\"" << std::endl;
                        throw std::runtime_error("dictionary not found, by variable name found in JSON.");
                      }

                      // Look for the value we want
                      auto valDict = JsonUtils::fetchMatchingEntry(dictEntries, "value", splitValue);

                      histDefBase.histTitle = JsonUtils::fetchValue(valDict, "title", histDefBase.histTitle);
                      histDefBase.fillStyle = JsonUtils::fetchValue(valDict, "fillStyle", short(1001));
                      histDefBase.histColor = JsonUtils::fetchValue(valDict, "color", unsetSplitValueColor);
                      if( histDefBase.histColor == unsetSplitValueColor ) unsetSplitValueColor++; // increment for the next ones

                    } // var dict?

                  } // splitVar ?

                } // isData?

                // Config DONE : creating save path
                histDefBase.folderPath = sample.getName();
                histDefBase.folderPath += "/" + histDefBase.varToPlot;
                if( not histDefBase.splitVarName.empty() ){
                  histDefBase.folderPath += "/" + histDefBase.splitVarName;
                  histDefBase.folderPath += "/" + std::to_string(histDefBase.splitVarValue);
                }

                // Config DONE
                _histHolderCacheList_[0].emplace_back(histDefBase);
                if( buildFillFunction ){
                  auto splitVarValue = _histHolderCacheList_[0].back().splitVarValue;
                  auto varToPlot = _histHolderCacheList_[0].back().varToPlot;
                  auto splitVarName = _histHolderCacheList_[0].back().splitVarName;
                }

              } // isData
            } // splitValue
          } // splitVar
        }

      } // histDef
    }
  }
  else{
    LogThrow("No sample has been set.");
  }
}

const std::vector<HistHolder> &PlotGenerator::getHistHolderList(int cacheSlot_) const {
  return _histHolderCacheList_[cacheSlot_];
}
const std::vector<HistHolder> &PlotGenerator::getComparisonHistHolderList() const {
  return _comparisonHistHolderList_;
}
std::map<std::string, std::shared_ptr<TCanvas>> PlotGenerator::getBufferCanvasList() const {
  return _bufferCanvasList_;
}

bool PlotGenerator::isEmpty() const{
  return _histHolderCacheList_[0].empty();
}
void PlotGenerator::generateSamplePlots(TDirectory *saveDir_, int cacheSlot_) {
  LogInfo << "Generating sample plots..." << std::endl;
  this->generateSampleHistograms(GenericToolbox::mkdirTFile(saveDir_, "histograms"), cacheSlot_);
  this->generateCanvas(_histHolderCacheList_[cacheSlot_], GenericToolbox::mkdirTFile(saveDir_, "canvas"));
}
void PlotGenerator::generateSampleHistograms(TDirectory *saveDir_, int cacheSlot_) {
  LogWarning << __METHOD_NAME__ << std::endl;

  if( _histogramsDefinition_.empty() ){
    LogError << "No histogram has been defined." << std::endl;
    return;
  }
  if( cacheSlot_ >= _histHolderCacheList_.size() ){
    size_t oldSize = _histHolderCacheList_.size();
    _histHolderCacheList_.resize(cacheSlot_+1);
    for(size_t iSlot = oldSize ; iSlot < _histHolderCacheList_.size(); iSlot++){
      _histHolderCacheList_[iSlot] = _histHolderCacheList_[0]; // inherit properties from the first one
      for( auto& histDef : _histHolderCacheList_[iSlot] ){
        histDef.histPtr = nullptr; // remove the pointer: hist will be recreated
      }
    }
  }

  auto* lastDir = GenericToolbox::getCurrentTDirectory();
  if(saveDir_ != nullptr ){
    saveDir_->cd();
  }

  // Create histograms
  int iHistCount = 0;
  for( auto& histDef : _histHolderCacheList_[cacheSlot_] ){

    if( histDef.histPtr == nullptr ){
      if( histDef.varToPlot == "Raw" ){
        if( _fitSampleSetPtr_ != nullptr ){
          if( histDef.isData ) histDef.histPtr = std::shared_ptr<TH1D>( (TH1D*) histDef.fitSamplePtr->getDataContainer().histogram->Clone() );
          else histDef.histPtr = std::shared_ptr<TH1D>( (TH1D*) histDef.fitSamplePtr->getMcContainer().histogram->Clone() );
        }
        else{
          LogThrow("Samples not set.")
        }
      }
      else{
        histDef.histPtr = std::make_shared<TH1D>(
            histDef.histName.c_str(), histDef.histTitle.c_str(),
            int(histDef.xEdges.size()) - 1, &histDef.xEdges[0]
        );
      }

      histDef.histPtr->SetName(Form("%p", (void *) histDef.histPtr.get()));

//      auto* dir = GenericToolbox::mkdirTFile(saveDir_, histDef.folderPath);
//      histDef.histPtr->SetDirectory(dir); // memory handled by ROOT
      histDef.histPtr->SetDirectory(nullptr); // memory handled by US
    }
    else{
      histDef.histPtr->Reset();
    }
  }

  // Fill histograms
  if( _fitSampleSetPtr_ != nullptr ){
    for( const auto& sample : _fitSampleSetPtr_->getFitSampleList() ){
      // Data sets:
      for( bool isData : { false, true } ){

        const std::vector<PhysicsEvent>* eventListPtr;
        std::vector<HistHolder*> histPtrToFillList;

        if( isData ){
          eventListPtr = &sample.getDataContainer().eventList;

          // which hist should be filled?
          for( auto& histDef : _histHolderCacheList_[cacheSlot_] ){
            if(histDef.isData and histDef.fitSamplePtr == &sample ){
              histPtrToFillList.emplace_back( &histDef );
            }
          }
        }
        else{
          eventListPtr = &sample.getMcContainer().eventList;

          // which hist should be filled?
          for( auto& histDef : _histHolderCacheList_[cacheSlot_] ){
            if(not histDef.isData and histDef.fitSamplePtr == &sample ){
              histPtrToFillList.emplace_back( &histDef );
            }
          }
        }

        for( auto& histPtrToFill : histPtrToFillList ){
          if( not histPtrToFill->isBinCacheBuilt ){
            // If any, launch rebuild cache
            LogInfo << "Build event bin cache for sample \"" << sample.getName() << "\"" << std::endl;
            this->buildEventBinCache(histPtrToFillList, eventListPtr, isData);
            break; // all caches done at once
          }
        }


        // Filling the selected histograms
        int nThreads = GlobalVariables::getNbThreads();
        std::function<void(int)> fillJob = [&]( int iThread ){
          int iEvent = -1;
          int iHist = -1;
          std::string pBarTitle;
          if( iThread == 0 ) pBarTitle = LogDebug.getPrefixString() + "Filling histograms of sample \"" + sample.getName() + "\"";

          for( auto& hist : histPtrToFillList ){
            for( int iBin = iThread+1 ; iBin <= hist->histPtr->GetNbinsX() ; iBin += nThreads ){
              hist->histPtr->SetBinContent(iBin, 0);
              for( auto* evtPtr : hist->_binEventPtrList_[iBin-1] ){
                hist->histPtr->AddBinContent(iBin, evtPtr->getEventWeight());
              }
              hist->histPtr->SetBinError(iBin, TMath::Sqrt(hist->histPtr->GetBinContent(iBin)));
            }
          }
        };

        GlobalVariables::getParallelWorker().addJob("fillJob", fillJob);
        GlobalVariables::getParallelWorker().runJob("fillJob");
        GlobalVariables::getParallelWorker().removeJob("fillJob");

      } // isData loop
    } // sample
  }
  else{
    LogThrow("Samples not set.")
  }

  // Post-processing (norm, color)
  for( auto& histDefPair : _histHolderCacheList_[cacheSlot_] ){

    if( _fitSampleSetPtr_ != nullptr ){
      if( histDefPair.isData ){
        histDefPair.histPtr->Scale( histDefPair.fitSamplePtr->getDataContainer().histScale );
      }
      else{
        histDefPair.histPtr->Scale( histDefPair.fitSamplePtr->getMcContainer().histScale );
      }
    }
    else{
      LogThrow("Samples not set.")
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
    histDefPair.histPtr->SetFillStyle(1001);
    if( histDefPair.fillStyle != short(1001) ){
      histDefPair.histPtr->SetFillStyle(histDefPair.fillStyle);
    }
  }

  // Saving
  if( saveDir_ != nullptr ){
    for( auto& histDefPair : _histHolderCacheList_[cacheSlot_] ){
      GenericToolbox::mkdirTFile( saveDir_, histDefPair.folderPath )->cd();
      histDefPair.histPtr->Write( histDefPair.histName.c_str() );
      saveDir_->cd();
    }
  }

  lastDir->cd();

}
void PlotGenerator::generateCanvas(const std::vector<HistHolder> &histHolderList_, TDirectory *saveDir_, bool stackHist_){
  LogWarning << __METHOD_NAME__ << std::endl;

  auto *lastDir = GenericToolbox::getCurrentTDirectory();
  if (saveDir_ != nullptr) { saveDir_->cd(); }

  auto buildCanvasPath = [](const HistHolder* hist_){
    std::stringstream ss;
    ss << hist_->varToPlot << hist_->prefix << "/";
    if( not hist_->splitVarName.empty() ) ss << hist_->splitVarName << "/";
    return ss.str();
  };

  int canvasHeight = JsonUtils::fetchValue(_canvasParameters_, "height", 700);
  int canvasWidth = JsonUtils::fetchValue(_canvasParameters_, "width", 1200);
  int canvasNbXplots = JsonUtils::fetchValue(_canvasParameters_, "nbXplots", 3);
  int canvasNbYplots = JsonUtils::fetchValue(_canvasParameters_, "nbYplots", 2);

  if( _fitSampleSetPtr_ != nullptr ){
    std::map<std::string, std::map<const FitSample*, std::vector<const HistHolder*>>> histsToStackMap; // histsToStackMap[path][FitSample] = listOfTh1d
    for( auto& histHolder : histHolderList_ ){
      if( histHolder.isData ) continue; // data associated to each later
      histsToStackMap[buildCanvasPath(&histHolder)][histHolder.fitSamplePtr].emplace_back(&histHolder );
    }

    // Now search for the associated data hist
    for( auto& histsSamples : histsToStackMap ){
      const std::string& canvasPath = histsSamples.first;
      for( auto& histsToStack : histsSamples.second ){
        for( auto& histHolder : histHolderList_ ){
          const FitSample* samplePtr = histsToStack.first;

          // we are searching for data
          if( not histHolder.isData ) continue;

          // canvasPath may have the trailing splitVar -> need to check the beginning of the path with the External hist
          if( not GenericToolbox::doesStringStartsWithSubstring(canvasPath, buildCanvasPath(&histHolder)) ) continue;

          // same sample?
          if( samplePtr != histHolder.fitSamplePtr ) continue;

          histsToStack.second.emplace_back( &histHolder );
          break; // no need to check for more
        }

      }
    }

    // Canvas builder
    for ( const auto &histsToStackPair : histsToStackMap ) {
      const std::string canvasFolderPath = histsToStackPair.first;

      int canvasIndex = 0;
      int iSampleSlot = 0;
      for (const auto &histList : histsToStackPair.second) {
        const FitSample* samplePtr = histList.first;
        iSampleSlot++;

        if (iSampleSlot > canvasNbXplots * canvasNbYplots) {
          canvasIndex++;
          iSampleSlot = 1;
        }

        std::string canvasName = "samples_n" + std::to_string(canvasIndex);
        std::string canvasPath = canvasFolderPath + canvasName;
        if (not GenericToolbox::doesKeyIsInMap(canvasPath, _bufferCanvasList_)) {
          _bufferCanvasList_[canvasPath] = std::make_shared<TCanvas>( canvasPath.c_str(), canvasPath.c_str(), canvasWidth, canvasHeight );
          _bufferCanvasList_[canvasPath]->Divide(canvasNbXplots, canvasNbYplots);
        }
        _bufferCanvasList_[canvasPath]->cd(iSampleSlot);

        // separating histograms
        TH1D *dataSampleHist{nullptr};
        std::vector<TH1D *> mcSampleHistList;
        double minYValue = 1;
        for( const auto* histHolder : histList.second ) {
          TH1D* hist = histHolder->histPtr.get();
          if ( histHolder->isData ) {
            dataSampleHist = hist;
          }
          else {
            mcSampleHistList.emplace_back(hist);
            minYValue = std::min(minYValue, hist->GetMinimum(0));
          }
        }

        TH1D* firstHistToPlot{nullptr};

        // Legend
        double Xmax = 0.9;
        double Ymax = 0.9;
        double Xmin = 0.5;
        double Ymin = Ymax - 0.04 * _maxLegendLength_;
        auto* splitLegend = new TLegend(Xmin, Ymin, Xmax, Ymax); // ptr required to transfert ownership
        int nLegend{0};

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

              if( nLegend < _maxLegendLength_ - 1 ){
                nLegend++;
                splitLegend->AddEntry(mcSampleHistAccumulatorList[iHist], mcSampleHistAccumulatorList[iHist]->GetTitle(), "f");
              }
              else{
                mcSampleHistAccumulatorList[iHist]->SetFillColor(kGray);
                mcSampleHistAccumulatorList[iHist]->SetLineColor(kGray);
                if( nLegend == _maxLegendLength_ - 1 ){
                  nLegend++;
                  splitLegend->AddEntry(mcSampleHistAccumulatorList[iHist], "others", "f");
                }
              }

              if ( firstHistToPlot == nullptr ) {
                firstHistToPlot = mcSampleHistAccumulatorList[iHist];
                mcSampleHistAccumulatorList[iHist]->GetYaxis()->SetRangeUser(minYValue,
                                                                             mcSampleHistAccumulatorList[iHist]->GetMaximum() *
                                                                             1.2);
                mcSampleHistAccumulatorList[iHist]->Draw("HIST GOFF");
              }
              else {
                mcSampleHistAccumulatorList[iHist]->Draw("HIST SAME GOFF");
              }


            }
          }
          else {
            // Just draw each hist on the same plot
            for (auto &mcHist : mcSampleHistList) {
              if ( firstHistToPlot == nullptr ) {
                firstHistToPlot = mcHist;
                if (mcSampleHistList.size() == 1) {
                  // only one: draw error bars
                  mcHist->Draw("EP GOFF");
                }
                else {
                  // don't draw multiple error bars
                  mcHist->Draw("HIST P GOFF");
                }
              }
              else {
                if (mcSampleHistList.size() == 1) { mcHist->Draw("EPSAME GOFF"); }
                else { mcHist->Draw("HIST P SAME GOFF"); }
              }

              if( nLegend < _maxLegendLength_ - 1 ){
                nLegend++;
                splitLegend->AddEntry(mcHist, mcHist->GetTitle(), "lep");
              }
            } // mcHist
          } // stack?


        } // mcHistList empty?

        // Draw the data hist on top
        if (dataSampleHist != nullptr) {
          std::string originalTitle = dataSampleHist->GetTitle(); // title can be used for figuring out the type of the histogram
          dataSampleHist->SetTitle("Data");
          splitLegend->AddEntry(dataSampleHist, dataSampleHist->GetTitle(), "lep"); nLegend++;
          if ( firstHistToPlot != nullptr ) {
            dataSampleHist->Draw("EPSAME GOFF");
          }
          else {
            firstHistToPlot = dataSampleHist;
            dataSampleHist->Draw("EP GOFF");
          }
          dataSampleHist->SetTitle(originalTitle.c_str()); // restore
        }

        if( firstHistToPlot == nullptr ){
          // Nothing to plot here
          continue;
        }

        if(nLegend != _maxLegendLength_){
          Ymin = Ymax - 0.04 * double(nLegend);
          splitLegend->SetY1(Ymin);
        }
        gPad->cd();
        splitLegend->Draw();

        firstHistToPlot->SetTitle( samplePtr->getName().c_str() ); // the actual displayed title
        gPad->SetGridx();
        gPad->SetGridy();
      } // sample

    } // Hist to stack
  }
  else{
    LogThrow("Samples not set.")
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

void PlotGenerator::generateComparisonPlots(const std::vector<HistHolder> &histsToStackOther_, const std::vector<HistHolder> &histsToStackReference_, TDirectory *saveDir_){
  this->generateComparisonHistograms(histsToStackOther_, histsToStackReference_, GenericToolbox::mkdirTFile(saveDir_, "histograms"));
  this->generateCanvas(_comparisonHistHolderList_, GenericToolbox::mkdirTFile(saveDir_, "canvas"), false);
}
void PlotGenerator::generateComparisonHistograms(const std::vector<HistHolder> &histList_, const std::vector<HistHolder> &refHistsList_, TDirectory *saveDir_) {
  LogWarning << __METHOD_NAME__ << std::endl;

  auto* curDir = GenericToolbox::getCurrentTDirectory();

  if(saveDir_ != nullptr){
    saveDir_->cd();
  }

  bool newHistHolder{false};
  if( _comparisonHistHolderList_.empty() ) newHistHolder = true;
  int iHistComp = -1;


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

    iHistComp++;
    if( newHistHolder ){
      _comparisonHistHolderList_.emplace_back( histHolder ); // copy all variables
      _comparisonHistHolderList_[iHistComp].histPtr = std::shared_ptr<TH1D>((TH1D*)histHolder.histPtr->Clone());
      _comparisonHistHolderList_[iHistComp].histPtr->SetName(Form("%p", (void*) _comparisonHistHolderList_[iHistComp].histPtr.get()));
    }
    else{
      _comparisonHistHolderList_[iHistComp].histPtr->Reset();
    }
    _comparisonHistHolderList_[iHistComp].folderPath = histHolder.folderPath;
    _comparisonHistHolderList_[iHistComp].histPtr->SetDirectory(nullptr);

    TH1D* compHistPtr = _comparisonHistHolderList_[iHistComp].histPtr.get();

    for( int iBin = 0 ; iBin <= compHistPtr->GetNbinsX()+1 ; iBin++ ){

      if( refHistHolderPtr->histPtr->GetBinContent(iBin) == 0 ){
        // no division by 0
        compHistPtr->SetBinContent( iBin, 0 );
        compHistPtr->SetBinError( iBin, 0 );
      }
      else{
        double binContent = histHolder.histPtr->GetBinContent( iBin );
        binContent /= refHistHolderPtr->histPtr->GetBinContent(iBin);
        binContent -= 1;
        binContent *= 100.;
        compHistPtr->SetBinContent( iBin, binContent );
        compHistPtr->SetBinError( iBin, histHolder.histPtr->GetBinError(iBin) / histHolder.histPtr->GetBinContent(iBin) * 100 );
      }

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
      GenericToolbox::mkdirTFile( saveDir_, _comparisonHistHolderList_[iHistComp].folderPath )->cd();
      compHistPtr->Write( _comparisonHistHolderList_[iHistComp].histName.c_str() );
      saveDir_->cd();
    }

  } // histList_

  curDir->cd();


}

std::vector<std::string> PlotGenerator::fetchListOfVarToPlot(bool isData_){
//  LogThrowIf(_config_.empty(), "Config not set, can't call " << __METHOD_NAME__);

  std::vector<std::string> varNameList;
  _histogramsDefinition_ = JsonUtils::fetchValue(_config_, "histogramsDefinition", nlohmann::json());
  for( const auto& histConfig : _histogramsDefinition_ ){
    auto varToPlot = JsonUtils::fetchValue<std::string>(histConfig, "varToPlot");
    if( varToPlot != "Raw"
        and not GenericToolbox::doesElementIsInVector(varToPlot, varNameList)
        and GenericToolbox::splitString(varToPlot, ":").size() < 2
        and not ( isData_ and JsonUtils::fetchValue(histConfig, "noData", false) )
        ){
      varNameList.emplace_back(varToPlot);
    }
  }
  return varNameList;
}
std::vector<std::string> PlotGenerator::fetchListOfSplitVarNames(){
//  LogThrowIf(_config_.empty(), "Config not set, can't call " << __METHOD_NAME__);

  std::vector<std::string> varNameList;
  _histogramsDefinition_ = JsonUtils::fetchValue(_config_, "histogramsDefinition", nlohmann::json());
  for( const auto& histConfig : _histogramsDefinition_ ){
    auto splitVars = JsonUtils::fetchValue(histConfig, "splitVars", std::vector<std::string>{""});
    for( const auto& splitVar : splitVars ){
      if( not splitVar.empty() and not GenericToolbox::doesElementIsInVector(splitVar, varNameList) ){
        varNameList.emplace_back(splitVar);
      }
    }
  }
  return varNameList;
}
std::vector<std::string> PlotGenerator::fetchRequestedLeafNames(){
//  LogAssert(not _config_.empty(), "Config not set, can't call " << __METHOD_NAME__);

  std::vector<std::string> varNameList;
  auto varToPlotList = this->fetchListOfVarToPlot();
  auto splitVarNameList = this->fetchListOfSplitVarNames();

  varNameList.reserve( varToPlotList.size() + splitVarNameList.size() ); // preallocate memory
  varNameList.insert( varNameList.end(), varToPlotList.begin(), varToPlotList.end() );
  varNameList.insert( varNameList.end(), splitVarNameList.begin(), splitVarNameList.end() );
  return varNameList;
}

void PlotGenerator::buildEventBinCache(const std::vector<HistHolder *> &histPtrToFillList, const std::vector<PhysicsEvent> *eventListPtr, bool isData_) {

  std::function<void(int)> fillEventHistCache = [&](int iThread_){
    int iHistCache{-1};
    std::string pBarTitle;
    for( auto& histPtrToFill : histPtrToFillList ){
      iHistCache++;

      if( iHistCache % GlobalVariables::getNbThreads() != iThread_ ) continue;

      if( not histPtrToFill->isBinCacheBuilt ){

        histPtrToFill->_binEventPtrList_.resize(histPtrToFill->histPtr->GetNbinsX());
        int iBin{-1};
        for( const auto& event : *eventListPtr ){
          if( histPtrToFill->splitVarName.empty() or event.getVarValue<int>(histPtrToFill->splitVarName) == histPtrToFill->splitVarValue){
            if( histPtrToFill->varToPlot == "Raw" ) iBin = event.getSampleBinIndex();
            else iBin = histPtrToFill->histPtr->FindBin(event.getVarAsDouble(histPtrToFill->varToPlot));
            if( iBin > 0 and iBin <= histPtrToFill->histPtr->GetNbinsX() ){
              // so it's a valid bin!
              histPtrToFill->_binEventPtrList_[iBin-1].emplace_back(&event);
            }
          }
        }
        histPtrToFill->isBinCacheBuilt = true;
      }
    }
  };

  GlobalVariables::getParallelWorker().addJob("fillEventHistCache", fillEventHistCache);
  GlobalVariables::getParallelWorker().runJob("fillEventHistCache");
  GlobalVariables::getParallelWorker().removeJob("fillEventHistCache");

}
