//
// Created by Nadrino on 16/06/2021.
//

#include "GenericToolbox.Json.h"
#include "GundamGlobals.h"
#include "PlotGenerator.h"
#include "ConfigUtils.h"

#include "Logger.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Map.h"

#include "TCanvas.h"
#include "TH1D.h"
#include "TLegend.h"

#include <string>
#include <vector>
#include <memory>
#include <sstream>

LoggerInit([]{
  Logger::setUserHeaderStr("[PlotGenerator]");
});


void PlotGenerator::readConfigImpl(){
  LogWarning << __METHOD_NAME__ << std::endl;
  gStyle->SetOptStat(0);
  _histHolderCacheList_.resize(1);

  if( not GenericToolbox::Json::fetchValue(_config_, "isEnabled", true) ){ return; }

  _varDictionary_ = GenericToolbox::Json::fetchValue(_config_, {{"varDictionaries"}, {"varDictionnaries"}}, JsonType());
  _canvasParameters_ = GenericToolbox::Json::fetchValue(_config_, "canvasParameters", JsonType());
  _histogramsDefinition_ = GenericToolbox::Json::fetchValue(_config_, "histogramsDefinition", JsonType());

  _writeGeneratedHistograms_ = GenericToolbox::Json::fetchValue(_config_, "writeGeneratedHistograms", _writeGeneratedHistograms_);
}
void PlotGenerator::initializeImpl() {
  LogWarning << __METHOD_NAME__ << std::endl;
  LogThrowIf(_sampleSetPtr_ == nullptr);
}


// Getters
bool PlotGenerator::isEmpty() const{
  return _histHolderCacheList_[0].empty();
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


std::vector<HistHolder> &PlotGenerator::getHistHolderList(int cacheSlot_){
  return _histHolderCacheList_[cacheSlot_];
}

// Core
void PlotGenerator::generateSamplePlots(TDirectory *saveDir_, int cacheSlot_) {
  LogThrowIf( not isInitialized() );
  LogScopeIndent;
  this->generateSampleHistograms(GenericToolbox::mkdirTFile(saveDir_, "histograms"), cacheSlot_);
  this->generateCanvas(_histHolderCacheList_[cacheSlot_], GenericToolbox::mkdirTFile(saveDir_, "canvas"));
}
void PlotGenerator::generateSampleHistograms(TDirectory *saveDir_, int cacheSlot_) {
  LogThrowIf(not isInitialized());

  if( _histogramsDefinition_.empty() ){
    LogWarning << "No histogram has been defined." << std::endl;
    return;
  }
  if( cacheSlot_ >= _histHolderCacheList_.size() ){
    size_t oldSize = _histHolderCacheList_.size();
    _histHolderCacheList_.resize(cacheSlot_+1); // copy all properties from slot 0
    for(size_t iSlot = oldSize ; iSlot < _histHolderCacheList_.size(); iSlot++){
      _histHolderCacheList_[iSlot] = _histHolderCacheList_[0];
      for( auto& histDef : _histHolderCacheList_[iSlot] ){
        histDef.histPtr = nullptr; // as shared ptr the copy did copy the ptr value. We want to create new hists
      }
    }
  }

  // Create histograms
  int iHistCount = 0;
  for( auto& histDef : _histHolderCacheList_[cacheSlot_] ){

    if( histDef.histPtr == nullptr ){
      if( histDef.varToPlot == "Raw" ){
        if( _sampleSetPtr_ != nullptr ){
          if( histDef.isData ) { histDef.histPtr = histDef.samplePtr->getDataContainer().generateRootHistogram(); }
          else { histDef.histPtr = histDef.samplePtr->getMcContainer().generateRootHistogram(); }
        }
        else{
          LogThrow("Samples not set.");
        }
      }
      else{
        histDef.histPtr = std::make_shared<TH1D>(
            histDef.histName.c_str(), histDef.histTitle.c_str(),
            int(histDef.xEdges.size()) - 1, &histDef.xEdges[0]
        );
      }

      // SHOULD NOT BE SAVED with the ptr name?
      histDef.histPtr->SetName(Form("%s_slot%i_%p", histDef.histName.c_str(), cacheSlot_, (void *) histDef.histPtr.get()));
      histDef.histPtr->SetDirectory(nullptr); // memory handled by US
    }
    else{
      histDef.histPtr->Reset();
    }
  }

  // Fill histograms
  for( const auto& sample : _sampleSetPtr_->getSampleList() ){
      // Datasets:
      for( bool isData : { false, true } ){

        const std::vector<Event>* eventListPtr;
        std::vector<HistHolder*> histPtrToFillList;

        if( isData ){
          eventListPtr = &sample.getDataContainer().getEventList();

          // which hist should be filled?
          for( auto& histDef : _histHolderCacheList_[cacheSlot_] ){
            if(histDef.isData and histDef.samplePtr == &sample ){
              histPtrToFillList.emplace_back( &histDef );
            }
          }
        }
        else{
          eventListPtr = &sample.getMcContainer().getEventList();

          // which hist should be filled?
          for( auto& histDef : _histHolderCacheList_[cacheSlot_] ){
            if(not histDef.isData and histDef.samplePtr == &sample ){
              histPtrToFillList.emplace_back( &histDef );
            }
          }
        }

        for( auto& histPtrToFill : histPtrToFillList ){
          if( not histPtrToFill->isBinCacheBuilt ){
            // If any, launch rebuild cache
            LogInfo << "Build event bin cache for sample \"" << sample.getName() << "\" " << (isData? "(data)":"(mc)") << std::endl;
            PlotGenerator::buildEventBinCache(histPtrToFillList, eventListPtr, isData);
            break; // all caches done at once
          }
        }

        // Filling the selected histograms
        std::function<void(int)> fillJob = [&]( int iThread_ ){

          for( auto* hist : histPtrToFillList ){

            auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
                iThread_, GundamGlobals::getParallelWorker().getNbThreads(),
                hist->histPtr->GetNbinsX()
            );

            for( int iBin = bounds.beginIndex+1 ; iBin <= bounds.endIndex ; iBin++ ){
              hist->histPtr->SetBinContent(iBin, 0);
              for( auto* evtPtr : hist->_binEventPtrList_[iBin-1] ){
                hist->histPtr->AddBinContent(iBin, evtPtr->getEventWeight());
              }
              hist->histPtr->SetBinError(iBin, TMath::Sqrt(hist->histPtr->GetBinContent(iBin)));
            }
          }
        };

        fillJob(-1);

//        GundamGlobals::getParallelWorker().addJob("fillJob", fillJob);
//        GundamGlobals::getParallelWorker().runJob("fillJob");
//        GundamGlobals::getParallelWorker().removeJob("fillJob");

      } // isData loop
    } // sample

  // Post-processing (norm, color)
  for( auto& histHolderCached : _histHolderCacheList_[cacheSlot_] ){

    LogThrowIf( _sampleSetPtr_ == nullptr, "Samples not set.");

    for(int iBin = 0 ; iBin <= histHolderCached.histPtr->GetNbinsX() + 1 ; iBin++ ){
      double binContent = histHolderCached.histPtr->GetBinContent(iBin); // this is the real number of counts
      double errorFraction = TMath::Sqrt(binContent)/binContent; // error fraction will not change with renorm of bin width

      if( histHolderCached.rescaleAsBinWidth ) binContent /= histHolderCached.histPtr->GetBinWidth(iBin);
      if(histHolderCached.rescaleBinFactor != 1. ) binContent *= histHolderCached.rescaleBinFactor;
      histHolderCached.histPtr->SetBinContent(iBin, binContent );
      histHolderCached.histPtr->SetBinError(iBin, binContent * errorFraction );

      histHolderCached.histPtr->SetDrawOption("EP");
    }

    histHolderCached.histPtr->GetXaxis()->SetTitle(histHolderCached.xTitle.c_str());
    histHolderCached.histPtr->GetYaxis()->SetTitle(histHolderCached.yTitle.c_str());

    histHolderCached.histPtr->SetLineWidth(2);
    histHolderCached.histPtr->SetLineColor(histHolderCached.histColor);

    histHolderCached.histPtr->SetMarkerStyle(kFullDotLarge );
    histHolderCached.histPtr->SetMarkerColor(histHolderCached.histColor);

    histHolderCached.histPtr->SetFillColor(histHolderCached.histColor);
    histHolderCached.histPtr->SetFillStyle(1001);
    if(histHolderCached.fillStyle != short(1001) ){
      histHolderCached.histPtr->SetFillStyle(histHolderCached.fillStyle);
    }

    // Cleanup nan
    for(int iBin = 0 ; iBin <= histHolderCached.histPtr->GetNbinsX() + 1 ; iBin++ ){
      if( std::isnan(histHolderCached.histPtr->GetBinContent(iBin)) ){ histHolderCached.histPtr->SetBinContent(iBin, 0); }
      if( std::isnan(histHolderCached.histPtr->GetBinError(iBin)) ){ histHolderCached.histPtr->SetBinError(iBin, 0); }
    }
  }

  // Saving
  if( saveDir_ != nullptr and not _writeGeneratedHistograms_ ){
    for( auto& histHolderCached : _histHolderCacheList_[cacheSlot_] ){
      GenericToolbox::writeInTFile(
          GenericToolbox::mkdirTFile(saveDir_, histHolderCached.folderPath ),
          histHolderCached.histPtr.get(), histHolderCached.histName, false
          );
    }
  }

}
void PlotGenerator::generateCanvas(const std::vector<HistHolder> &histHolderList_, TDirectory *saveDir_, bool stackHist_){
  LogThrowIf(not isInitialized());

  auto buildCanvasPath = [](const HistHolder* hist_){
    std::stringstream ss;
    ss << hist_->varToPlot << hist_->prefix << "/";
    if( not hist_->splitVarName.empty() ) ss << hist_->splitVarName << "/";
    return ss.str();
  };

  int canvasHeight = GenericToolbox::Json::fetchValue(_canvasParameters_, "height", 700);
  int canvasWidth = GenericToolbox::Json::fetchValue(_canvasParameters_, "width", 1200);
  int canvasNbXplots = GenericToolbox::Json::fetchValue(_canvasParameters_, "nbXplots", 3);
  int canvasNbYplots = GenericToolbox::Json::fetchValue(_canvasParameters_, "nbYplots", 2);

  std::map<std::string, std::map<const Sample*, std::vector<const HistHolder*>>> histsToStackMap; // histsToStackMap[path][sample] = listOfTh1d
  for( auto& histHolder : histHolderList_ ){
    if( histHolder.isData ) continue; // data associated to each later
    histsToStackMap[buildCanvasPath(&histHolder)][histHolder.samplePtr].emplace_back(&histHolder );
  }

  // Now search for the associated data hist
  for( auto& histsSamples : histsToStackMap ){
    const std::string& canvasPath = histsSamples.first;
    for( auto& histsToStack : histsSamples.second ){
      for( auto& histHolder : histHolderList_ ){
        const Sample* samplePtr = histsToStack.first;

        // we are searching for data
        if( not histHolder.isData ) continue;

        // canvasPath may have the trailing splitVar -> need to check the beginning of the path with the External hist
        if( not GenericToolbox::startsWith(canvasPath, buildCanvasPath(&histHolder)) ) continue;

        // same sample?
        if( samplePtr != histHolder.samplePtr ) continue;

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
      const Sample* samplePtr = histList.first;
      iSampleSlot++;

      if (iSampleSlot > canvasNbXplots * canvasNbYplots) {
        canvasIndex++;
        iSampleSlot = 1;
      }

      std::string canvasName = "samples_n" + std::to_string(canvasIndex);
      std::string canvasPath = canvasFolderPath + canvasName;
      if (not GenericToolbox::isIn(canvasPath, _bufferCanvasList_)) {
        _bufferCanvasList_[canvasPath] = std::make_shared<TCanvas>( canvasPath.c_str(), canvasPath.c_str(), canvasWidth, canvasHeight );
        _bufferCanvasList_[canvasPath]->Divide(canvasNbXplots, canvasNbYplots);
      }
      _bufferCanvasList_[canvasPath]->cd(iSampleSlot);
      _bufferCanvasList_[canvasPath]->GetPad(iSampleSlot)->SetLeftMargin(0.12); // Y title prints ok
      _bufferCanvasList_[canvasPath]->GetPad(iSampleSlot)->SetTopMargin(0.105); // 10^XX print correctly
      _bufferCanvasList_[canvasPath]->GetPad(iSampleSlot)->SetRightMargin(0.); // don't lose space on the right
      _bufferCanvasList_[canvasPath]->GetPad(iSampleSlot)->SetBottomMargin(0.11); // Leave a bit of space at the bottom

      // separating histograms
      TH1D *dataSampleHist{nullptr};
      std::vector<TH1D *> mcSampleHistList;
      double minYValue{std::nan("unset")};
      double maxYValue{std::nan("unset")};
      for( const auto* histHolder : histList.second ) {
        TH1D* hist{ histHolder->histPtr.get() };
        std::pair<double, double> yBounds;
        if( histHolder->isData ){
          dataSampleHist = hist;
          yBounds = GenericToolbox::fetchYRange(hist);
        }
        else{
          mcSampleHistList.emplace_back(hist);
          yBounds = GenericToolbox::fetchYRange(hist, false); // don't consider un-ploted errors on MC
        }
        minYValue = std::min(yBounds.first, minYValue); // NAN on the right!!
        maxYValue = std::max(yBounds.second, maxYValue);
      }

      TH1D* firstHistToPlot{nullptr};

      // Legend
      double Xmax = 1;
      double Ymax = 0.9;
      double Xmin = 0.6;
      double Ymin = Ymax - 0.04 * _maxLegendLength_;
      auto* splitLegend = new TLegend(Xmin, Ymin, Xmax, Ymax); // ptr required to transfert ownership
      int nLegend{0};

      // process mc part
      std::vector<TH1D *> mcSampleHistAccumulatorList;
      if (not mcSampleHistList.empty()) {

        if (stackHist_) {
          // Sorting histograms by norm (lowest stat first)
          GenericToolbox::sortVector(mcSampleHistList,  [](TH1D *histA_, TH1D *histB_) {
            return (histA_->Integral(histA_->FindBin(0), histA_->FindBin(histA_->GetXaxis()->GetXmax()))
                    < histB_->Integral(histB_->FindBin(0), histB_->FindBin(histB_->GetXaxis()->GetXmax())));
          });

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
        maxYValue = std::max(dataSampleHist->GetMaximum(), maxYValue);
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

      firstHistToPlot->GetYaxis()->SetRangeUser(
          minYValue,
          maxYValue * 1.2
      );

      firstHistToPlot->SetTitle( samplePtr->getName().c_str() ); // the actual displayed title
      gPad->SetGridx();
      gPad->SetGridy();
    } // sample

  } // Hist to stack

  if (saveDir_ != nullptr) {
    for (auto &canvas : _bufferCanvasList_) {
      auto pathSplit = GenericToolbox::splitString(canvas.first, "/");
      std::string folderPath = GenericToolbox::joinVectorString(pathSplit, "/", 0, -1);
      std::string canvasName = pathSplit.back();
      GenericToolbox::writeInTFile(
          GenericToolbox::mkdirTFile(saveDir_, folderPath),
          canvas.second.get(), canvasName, false
          );
    }
  }
}
void PlotGenerator::generateComparisonPlots(const std::vector<HistHolder> &histsToStackOther_, const std::vector<HistHolder> &histsToStackReference_, TDirectory *saveDir_){
  this->generateComparisonHistograms(
      histsToStackOther_, histsToStackReference_,
//      GenericToolbox::mkdirTFile(saveDir_, "histograms")
      nullptr
  );
  this->generateCanvas(_comparisonHistHolderList_, GenericToolbox::mkdirTFile(saveDir_, "canvas"), false);
}
void PlotGenerator::generateComparisonHistograms(const std::vector<HistHolder> &histList_, const std::vector<HistHolder> &refHistsList_, TDirectory *saveDir_) {
  LogThrowIf(not isInitialized());

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
      _comparisonHistHolderList_[iHistComp].histPtr = std::make_shared<TH1D>( *histHolder.histPtr );
    }
    else{
      _comparisonHistHolderList_[iHistComp].histPtr->Reset();
    }
    _comparisonHistHolderList_[iHistComp].folderPath = histHolder.folderPath;
    _comparisonHistHolderList_[iHistComp].histPtr->SetDirectory(nullptr);

    TH1D* compHistPtr = _comparisonHistHolderList_[iHistComp].histPtr.get();

    for( int iBin = 0 ; iBin <= compHistPtr->GetNbinsX()+1 ; iBin++ ){

      // cleanup
      if( std::isnan(compHistPtr->GetBinContent(iBin)) ){ compHistPtr->SetBinContent(iBin, 0); }
      if( std::isnan(compHistPtr->GetBinError(iBin)) ){ compHistPtr->SetBinError(iBin, 0); }

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

    if( saveDir_ != nullptr and not _writeGeneratedHistograms_ ){
      GenericToolbox::writeInTFile(
          GenericToolbox::mkdirTFile( saveDir_, _comparisonHistHolderList_[iHistComp].folderPath ),
          compHistPtr, _comparisonHistHolderList_[iHistComp].histName, false
          );
    }
  } // histList_
}

// Misc
std::vector<std::string> PlotGenerator::fetchListOfVarToPlot(bool isData_){
  std::vector<std::string> varNameList;
  _histogramsDefinition_ = GenericToolbox::Json::fetchValue(_config_, "histogramsDefinition", JsonType());
  for( const auto& histConfig : _histogramsDefinition_ ){
    if( not GenericToolbox::Json::fetchValue(histConfig, "isEnabled", true) ){ continue; }
    auto varToPlot = GenericToolbox::Json::fetchValue<std::string>(histConfig, "varToPlot");
    if( varToPlot != "Raw"
        and not GenericToolbox::doesElementIsInVector(varToPlot, varNameList)
        and GenericToolbox::splitString(varToPlot, ":").size() < 2
        and not ( isData_ and GenericToolbox::Json::fetchValue(histConfig, "noData", false) )
        ){
      varNameList.emplace_back(varToPlot);
    }
  }
  return varNameList;
}
std::vector<std::string> PlotGenerator::fetchListOfSplitVarNames(){
//  LogThrowIf(_config_.empty(), "Config not set, can't call " << __METHOD_NAME__);

  std::vector<std::string> varNameList;
  _histogramsDefinition_ = GenericToolbox::Json::fetchValue(_config_, "histogramsDefinition", JsonType());
  for( const auto& histConfig : _histogramsDefinition_ ){
    auto splitVars = GenericToolbox::Json::fetchValue(histConfig, "splitVars", std::vector<std::string>{""});
    for( const auto& splitVar : splitVars ){
      if( not splitVar.empty() and not GenericToolbox::doesElementIsInVector(splitVar, varNameList) ){
        varNameList.emplace_back(splitVar);
      }
    }
  }
  return varNameList;
}

// Internals
void PlotGenerator::defineHistogramHolders() {
  LogWarning << __METHOD_NAME__ << std::endl;
  _histHolderCacheList_[0].clear();

  LogInfo << "Fetching appearing split vars..." << std::endl;

  struct SplitVariableDictionary{
    struct Entry{
      std::string name{};
      struct SplitSample{
        const Sample* samplePtr{nullptr};
        std::vector<int> splitValueList{};
      };
      std::vector<SplitSample> sampleList{};

      [[nodiscard]] const SplitSample& fetchSample(const Sample* samplePtr_) const{
        int idx{GenericToolbox::findElementIndex(samplePtr_, sampleList, [](const SplitSample& s_){ return s_.samplePtr; })};
        LogThrowIf(idx==-1, "Can't find SplitVariableDictionary for: " << samplePtr_->getName());
        return sampleList[idx];
      }
      SplitSample& fetchSample(const Sample* samplePtr_){
        int idx{GenericToolbox::findElementIndex(samplePtr_, sampleList, [](const SplitSample& s_){ return s_.samplePtr; })};
        LogThrowIf(idx==-1, "Can't find SplitVariableDictionary for: " << samplePtr_->getName());
        return sampleList[idx];
      }
    };
    std::vector<Entry> entryList{};

    [[nodiscard]] bool hasEntry(const std::string& name_) const{
      return GenericToolbox::isIn(name_, entryList, [](const Entry& e_){ return e_.name; });
    }
    [[nodiscard]] const Entry& fetchEntry(const std::string& name_) const{
      int idx{GenericToolbox::findElementIndex(name_, entryList, [](const Entry& e_){ return e_.name; })};
      LogThrowIf(idx==-1, "Can't find SplitVariableDictionary for: " << name_);
      return entryList[idx];
    }
  };


  SplitVariableDictionary splitVarsDictionary{};
  for( const auto& histConfig : _histogramsDefinition_ ){
    auto splitVars = GenericToolbox::Json::fetchValue(histConfig, "splitVars", std::vector<std::string>{""});
    for( auto& splitVar : splitVars ){
      if( not splitVarsDictionary.hasEntry(splitVar) ){
        splitVarsDictionary.entryList.emplace_back();
        auto& entry = splitVarsDictionary.entryList.back();
        entry.name = splitVar;
        entry.sampleList.reserve( _sampleSetPtr_->getSampleList().size() );
        for( const auto& sample : _sampleSetPtr_->getSampleList() ){
          entry.sampleList.emplace_back();
          auto& sampleSplit = entry.sampleList.back();
          sampleSplit.samplePtr = &sample;
          if( splitVar.empty() ){ sampleSplit.splitValueList.emplace_back(0); } // placeholder for no split var
        }
      }
    }
  }

  std::function<void(int)> fetchSplitVar = [&](int iThread_){
    auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
        iThread_, GundamGlobals::getParallelWorker().getNbThreads(),
        int(_sampleSetPtr_->getSampleList().size())
    );

    for( int iSample = bounds.beginIndex ; iSample < bounds.endIndex ; iSample++ ){
      const Sample* samplePtr = &_sampleSetPtr_->getSampleList()[iSample];
      for( auto& event : samplePtr->getMcContainer().getEventList() ){
        for( auto& entry : splitVarsDictionary.entryList ){
          if( entry.name.empty() ){ continue; }
          auto splitValue = int( event.getVariables().getVarList()[event.getVariables().findVarIndex( entry.name )].getVarAsDouble() );
          GenericToolbox::addIfNotInVector(splitValue, entry.fetchSample( samplePtr ).splitValueList);
        } // splitVarList
      } // Event
    }
  };

  GundamGlobals::getParallelWorker().addJob("fetchSplitVar", fetchSplitVar);
  GundamGlobals::getParallelWorker().runJob("fetchSplitVar");
  GundamGlobals::getParallelWorker().removeJob("fetchSplitVar");

  int sampleCounter = -1;
  HistHolder histDefBase;
  for( const auto& sample : _sampleSetPtr_->getSampleList() ){
    LogScopeIndent;
    LogInfo << "Defining holders for sample: \"" << sample.getName() << "\"" << std::endl;
    sampleCounter++;
    histDefBase.samplePtr = &sample;
    short unsetSplitValueColor = kGray; // will increment if needed

    // Definition of histograms
    for( const auto& histConfig : _histogramsDefinition_ ){

      histDefBase.varToPlot         = GenericToolbox::Json::fetchValue<std::string>(histConfig, "varToPlot");
      histDefBase.prefix            = GenericToolbox::Json::fetchValue(histConfig, "prefix", "");
      histDefBase.rescaleAsBinWidth = GenericToolbox::Json::fetchValue(histConfig, "rescaleAsBinWidth", true);
      histDefBase.rescaleBinFactor  = GenericToolbox::Json::fetchValue(histConfig, "rescaleBinFactor", 1.);

      if( GenericToolbox::splitString(histDefBase.varToPlot, ":").size() >= 2 ){
        LogAlert << "Skipping 2D plot def: " << histDefBase.varToPlot << std::endl;
        continue;
      }
      else{
        auto splitVars = GenericToolbox::Json::fetchValue(histConfig, "splitVars", std::vector<std::string>{""});

        for( const auto& splitVar : splitVars ){

          histDefBase.splitVarName = splitVar;

          // Loop over split vars
          int splitValueIndex = -1;
          auto& splitValueList = splitVarsDictionary.fetchEntry(histDefBase.splitVarName).fetchSample(&sample).splitValueList;
          for( auto& splitValue : splitValueList ){
            splitValueIndex++;
            histDefBase.splitVarValue = splitValue;

            for( bool isData: {false, true} ){
              histDefBase.isData = isData;
              bool buildFillFunction = false;

              if( histDefBase.isData and
                  ( not histDefBase.splitVarName.empty()
                    or GenericToolbox::Json::fetchValue(histConfig, "noData", false)
                  ) ){
                continue;
              }

              if( histDefBase.varToPlot != "Raw" ){
                // Then filling the histo is needed

                // Binning
                histDefBase.xEdges.clear();

                histDefBase.xMin = GenericToolbox::Json::fetchValue(histConfig, "xMin", std::nan("nan"));;
                histDefBase.xMax = GenericToolbox::Json::fetchValue(histConfig, "xMax", std::nan("nan"));

                if( GenericToolbox::Json::fetchValue(histConfig, "useSampleBinning", false) ){

                  bool varNotAvailable{false};
                  std::string sampleObsBinning = GenericToolbox::Json::fetchValue(histConfig, "useSampleBinningOfObservable", histDefBase.varToPlot);

                  for( const auto& bin : sample.getBinning().getBinList() ){
                    std::string variableNameForBinning{sampleObsBinning};

                    if( not GenericToolbox::doesElementIsInVector(sampleObsBinning, bin.getEdgesList(), [](const DataBin::Edges& e){ return e.varName; }) ){
                      if( GenericToolbox::Json::doKeyExist(histConfig, "sampleVariableIfNotAvailable") ){
                        for( auto& varSubstitution : GenericToolbox::Json::fetchValue<std::vector<std::string>>(histConfig, "sampleVariableIfNotAvailable") ){
                          if( GenericToolbox::doesElementIsInVector(varSubstitution, bin.getEdgesList(), [](const DataBin::Edges& e){ return e.varName; }) ){
                            variableNameForBinning = varSubstitution;
                            break;
                          }
                        }
                      } // sampleVariableIfNotAvailable
                    } // sampleObsBinning not in the sample binning

                    const DataBin::Edges* edges{bin.getVarEdgesPtr(variableNameForBinning)};
                    if( edges == nullptr ){
                      LogAlert << "Can't use sample binning for var " << variableNameForBinning << " and sample " << sample.getName() << std::endl;
                      varNotAvailable = true;
                      break;
                    }

                    for( const auto& edge : { edges->min, edges->max } ) {
                      if (    ( std::isnan( histDefBase.xMin ) or histDefBase.xMin <= edge )
                          and ( std::isnan( histDefBase.xMax ) or histDefBase.xMax >= edge)) {
                        // either NaN or in bounds
                        if (not GenericToolbox::doesElementIsInVector(edge, histDefBase.xEdges)) {
                          histDefBase.xEdges.emplace_back(edge);
                        }
                      }
                    }
                  }


                  if( varNotAvailable ) break;
                  if( histDefBase.xEdges.empty() ) continue; // skip
                  std::sort( histDefBase.xEdges.begin(), histDefBase.xEdges.end() ); // sort for ROOT

                } // sample binning ?
                else if( GenericToolbox::Json::doKeyExist(histConfig, "binningFile") ){
                  DataBinSet b;
                  b.readBinningDefinition(GenericToolbox::Json::fetchValue(histConfig, "binningFile", JsonType()) );
                  b.sortBins();

                  auto varList{b.buildVariableNameList()};
                  LogThrowIf(varList.size()!=1, "Binning should be defined with only one variable, here: " << GenericToolbox::toString(varList))

                  for(const auto& bin: b.getBinList()){
                    const auto& edges = bin.getVarEdges(varList[0]);
                    for( const auto& edge : { edges.min, edges.max } ) {
                      if (    ( std::isnan( histDefBase.xMin ) or histDefBase.xMin <= edge)
                          and ( std::isnan( histDefBase.xMax ) or histDefBase.xMax >= edge)) {
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

              histDefBase.xTitle = GenericToolbox::Json::fetchValue(histConfig, "xTitle", histDefBase.varToPlot);
              histDefBase.yTitle = GenericToolbox::Json::fetchValue(histConfig, "yTitle", "");
              if( histDefBase.yTitle.empty() ){
                histDefBase.yTitle = "Counts";
                if( histDefBase.rescaleAsBinWidth ) histDefBase.yTitle += " (/bin width)";
                if( histDefBase.rescaleBinFactor != 1. ) histDefBase.yTitle += "*" + std::to_string(histDefBase.rescaleBinFactor);
              }

              // Colors / Title (legend) / Name
              if( histDefBase.isData ){
                histDefBase.histName = "Data";
                histDefBase.histTitle = "Data";
                histDefBase.histColor = kBlack;
                histDefBase.fillStyle = 1001;
              }
              else{
                histDefBase.histName = "MC";

                if( histDefBase.splitVarName.empty() ){
                  histDefBase.histTitle = "Model";
                  histDefBase.histColor = defaultColorWheel[ sampleCounter % defaultColorWheel.size() ];
                  histDefBase.fillStyle = 1001;
                }
                else{
                  histDefBase.histTitle = "Model (" + splitVar + " == " + std::to_string(splitValue) + ")";
                  histDefBase.histColor = defaultColorWheel[ splitValueIndex % defaultColorWheel.size() ];

                  // User defined color?
                  JsonType varDict{};

                  for( auto& varDictEntry : _varDictionary_ ){
                    if( not GenericToolbox::Json::doKeyExist(varDictEntry, "name") ){ continue; }
                    if( GenericToolbox::Json::fetchValue<std::string>(varDictEntry, "name") != splitVar ){ continue; }
                    varDict = varDictEntry;
                    break;
                  }

                  if( not varDict.empty() ){

                    auto dictEntries = varDict["dictionary"];
                    if( dictEntries.is_null() ){
                      LogError << R"(Could not find "dictionary" key in JSON config for var: ")" << splitVar << "\"" << std::endl;
                      throw std::runtime_error("dictionary not found, by variable name found in JSON.");
                    }

                    // Look for the value we want
                    JsonType valDict{};
                    for( auto& dictEntry : dictEntries ){
                      if( GenericToolbox::Json::fetchValue<int>(dictEntry, "value") == splitValue ){
                        valDict = dictEntry;
                        break;
                      }
                    }

                    histDefBase.histTitle = GenericToolbox::Json::fetchValue(valDict, "title", histDefBase.histTitle);
                    histDefBase.fillStyle = GenericToolbox::Json::fetchValue(valDict, "fillStyle", short(1001));

                    histDefBase.histColor = unsetSplitValueColor;
                    histDefBase.histColor = GenericToolbox::Json::fetchValue(valDict, {{"colorRoot"}, {"color"}}, histDefBase.histColor);
                    if( GenericToolbox::Json::doKeyExist(valDict, "colorHex") ){
                      TColor::SetColorThreshold(0.1); // will fetch the closest color
                      histDefBase.histColor = short( TColor::GetColor( GenericToolbox::Json::fetchValue<std::string>(valDict, "colorHex").c_str() ) );
                    }

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
void PlotGenerator::buildEventBinCache( const std::vector<HistHolder *> &histPtrToFillList, const std::vector<Event> *eventListPtr, bool isData_) {

  std::function<void()> prepareCacheFct = [&]() {
    for (auto *holder: histPtrToFillList) {
      if (not holder->isBinCacheBuilt) {
        if (holder->histPtr == nullptr) { continue; }

        // pre-allocate
        holder->_binEventPtrList_.resize(holder->histPtr->GetNbinsX());
        for (auto &evtList: holder->_binEventPtrList_) { evtList.reserve(eventListPtr->size()); }
      }
    }
  };
  std::function<void(int)> fillEventHistCache = [&](int iThread_){

    auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
        iThread_, GundamGlobals::getParallelWorker().getNbThreads(),
        int(histPtrToFillList.size())
    );

    HistHolder* histPtr{nullptr};
    for( int iHist = bounds.beginIndex ; iHist < bounds.endIndex ; iHist++ ){
      histPtr = histPtrToFillList[iHist];

      if( not histPtr->isBinCacheBuilt ){
        int iBin{-1};
        for( const auto& event : *eventListPtr ){
          int splitValue;
          if( not histPtr->splitVarName.empty() ){
            splitValue = event.getVariables().fetchVariable(histPtr->splitVarName).get().getValue<int>();
          }

          if( histPtr->splitVarName.empty() or splitValue == histPtr->splitVarValue){

            if( histPtr->varToPlot == "Raw" ){ iBin = event.getIndices().bin + 1; }
            else                             { iBin = histPtr->histPtr->FindBin(event.getVariables().fetchVariable(histPtr->varToPlot).getVarAsDouble()); }

            if( iBin > 0 and iBin <= histPtr->histPtr->GetNbinsX() ){
              // so it's a valid bin!
              histPtr->_binEventPtrList_[iBin-1].emplace_back( &event );
            }
          }
        }
        histPtr->isBinCacheBuilt = true;
      }
    }
  };
  std::function<void()> shrinkAllocationsFct = [&]() {
    for( auto* holder : histPtrToFillList ){
      for( auto& evtList : holder->_binEventPtrList_ ){ evtList.shrink_to_fit(); }
    }
  };

  // Single thread test
//  prepareCacheFct();
//  fillEventHistCache(-1);
//  shrinkAllocationsFct();

  GundamGlobals::getParallelWorker().addJob("fillEventHistCache", fillEventHistCache);
  GundamGlobals::getParallelWorker().setPreParallelJob("fillEventHistCache", prepareCacheFct);
  GundamGlobals::getParallelWorker().setPostParallelJob("fillEventHistCache", shrinkAllocationsFct);
  GundamGlobals::getParallelWorker().runJob("fillEventHistCache");
  GundamGlobals::getParallelWorker().removeJob("fillEventHistCache");

}

