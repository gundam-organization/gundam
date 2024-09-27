//
// Created by Nadrino on 16/06/2021.
//

#ifndef GUNDAM_PLOT_GENERATOR_H
#define GUNDAM_PLOT_GENERATOR_H

#include "Sample.h"
#include "Event.h"

#include "GenericToolbox.Wrappers.h"
#include "GenericToolbox.Thread.h"

#include "TDirectory.h"
#include "TH1D.h"

#include <map>
#include <mutex>
#include <memory>
#include <vector>
#include <string>
#include <functional>


struct HistHolder{
  // Hist
  std::shared_ptr<TH1D> histPtr{nullptr};
//  GenericToolbox::NoCopyWrapper<std::unique_ptr<TH1D>> histPtr;
//  TH1D hist;

  // Path
  std::string folderPath;
  std::string histName;
  std::string histTitle;

  // Data
  bool isData{false};
  const Sample* samplePtr{nullptr};
  std::function<void(TH1D*, const Event*)> fillFunctionSample;

  // X axis
  std::string varToPlot;
  std::string prefix;
  std::string xTitle;
  double xMin;
  double xMax;
  std::vector<double> xEdges;

  // Y axis
  std::string yTitle;

  // sub-Histogram
  std::string splitVarName;
  int splitVarValue;

  // display Properties
  bool rescaleAsBinWidth{true};
  double rescaleBinFactor{1.};
  short histColor;
  short fillStyle{1001};

  // Flags
  bool isBaseSplitHist{false};

  // Caches
  bool isBinCacheBuilt{false};
  std::vector<std::vector<const Event*>> _binEventPtrList_;
};

struct CanvasHolder{
  int canvasHeight = 700;
  int canvasWidth = 1200;
  int canvasNbXplots = 3;
  int canvasNbYplots = 2;
};



class PlotGenerator : public JsonBaseClass {

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  // Setters
  void setModelSampleSetPtr(const std::vector<Sample> *modelSampleListPtr_){ _modelSampleListPtr_ = modelSampleListPtr_; }
  void setDataSampleSetPtr(const std::vector<Sample> *dataSampleListPtr_){ _dataSampleListPtr_ = dataSampleListPtr_; }

  // Getters
  [[nodiscard]] bool isEmpty() const;
  [[nodiscard]] const std::vector<HistHolder> &getHistHolderList(int cacheSlot_ = 0) const;
  [[nodiscard]] const std::vector<HistHolder> &getComparisonHistHolderList() const { return _comparisonHistHolderList_; }
  [[nodiscard]] std::map<std::string, std::shared_ptr<TCanvas>> getBufferCanvasList() const { return _bufferCanvasList_; }

  // non-const getters
  std::vector<HistHolder> &getHistHolderList(int cacheSlot_ = 0);

  // Core
  void generateSamplePlots(TDirectory *saveDir_ = nullptr, int cacheSlot_ = 0);
  void generateSampleHistograms(TDirectory *saveDir_ = nullptr, int cacheSlot_ = 0);
  void generateCanvas(const std::vector<HistHolder> &histHolderList_, TDirectory *saveDir_ = nullptr, bool stackHist_ = true);
  void generateComparisonPlots(const std::vector<HistHolder> &histsToStackOther_, const std::vector<HistHolder> &histsToStackReference_, TDirectory *saveDir_ = nullptr);
  void generateComparisonHistograms(const std::vector<HistHolder> &histList_, const std::vector<HistHolder> &refHistsList_, TDirectory *saveDir_ = nullptr);

  // Misc
  std::vector<std::string> fetchListOfVarToPlot(bool isData_ = false) const;
  std::vector<std::string> fetchListOfSplitVarNames() const;

  void defineHistogramHolders();


protected:
  // Internals
  void buildEventBinCache( const std::vector<HistHolder *> &histPtrToFillList, const std::vector<Event> *eventListPtr, bool isData_);

private:
  // Parameters
  bool _isEnabled_{true};
  bool _writeGeneratedHistograms_{false};
  int _maxLegendLength_{15};
  std::vector<Color_t> defaultColorWheel {
      kGreen-3, kTeal+3, kAzure+7,
      kCyan-2, kBlue-7, kBlue+2,
      kOrange+1, kOrange+9, kRed+2, kPink+9
  };

  struct HistogramDefinition{
    bool noData{false};
    bool rescaleAsBinWidth{true};
    bool useSampleBinning{false};
    double rescaleBinFactor{1.};
    double xMin{std::nan("unset")};
    double xMax{std::nan("unset")};
    std::string prefix{};
    std::string varToPlot{};
    std::string xTitle{};
    std::string yTitle{};
    std::string useSampleBinningOfVar{};
    std::vector<std::string> splitVarList{};
    std::vector<std::string> sampleVariableIfNotAvailable{};
    BinSet binning{};
  };
  std::vector<HistogramDefinition> _histDefList_{};

  struct CanvasParameters{
    int height{700};
    int width{1200};
    int nbXplots{3};
    int nbYplots{2};
  };
  CanvasParameters _canvasParameters_{};

  struct VarDictionary{
    std::string name{};

    struct DictEntry{
      int value{};
      std::string title{};
      short fillStyle{1001};
      short color{};
    };
    std::vector<DictEntry> dictEntryList{};

    const DictEntry* fetchDictEntryPtr(int value_) const {
      for( auto& dictEntry : dictEntryList ){
        if( dictEntry.value == value_ ){ return &dictEntry; }
      }
      return nullptr;
    }
  };
  std::vector<VarDictionary> _varDictionaryList_{};
  const VarDictionary* fetchVarDictionaryPtr(const std::string& name_){
    for( auto& varDict : _varDictionaryList_ ){
      if( varDict.name == name_ ){ return &varDict; }
    }
    return nullptr;
  }

  // Internals
  const std::vector<Sample>* _modelSampleListPtr_{nullptr};
  const std::vector<Sample>* _dataSampleListPtr_{nullptr};
  std::vector<std::vector<HistHolder>> _histHolderCacheList_{};
  std::vector<HistHolder> _comparisonHistHolderList_;
  std::map<std::string, std::shared_ptr<TCanvas>> _bufferCanvasList_;

  GenericToolbox::ParallelWorker _threadPool_{};

};


#endif //GUNDAM_PLOT_GENERATOR_H
