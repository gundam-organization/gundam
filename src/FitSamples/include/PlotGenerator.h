//
// Created by Nadrino on 16/06/2021.
//

#ifndef GUNDAM_PLOTGENERATOR_H
#define GUNDAM_PLOTGENERATOR_H

#include "FitSampleSet.h"
#include "PhysicsEvent.h"

#include "GenericToolbox.Wrappers.h"

#include "nlohmann/json.hpp"
#include "TDirectory.h"
#include "TH1D.h"

#include "memory"
#include "vector"
#include "functional"
#include "mutex"
#include "string"
#include "map"


struct HistHolder{
  // Hist
  std::shared_ptr<TH1D> histPtr{nullptr};

  // Path
  std::string folderPath;
  std::string histName;
  std::string histTitle;

  // Data
  bool isData{false};
  const FitSample* fitSamplePtr{nullptr};
  std::function<void(TH1D*, const PhysicsEvent*)> fillFunctionFitSample;

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
  std::vector<std::vector<const PhysicsEvent*>> _binEventPtrList_;
};

struct CanvasHolder{
  int canvasHeight = 700;
  int canvasWidth = 1200;
  int canvasNbXplots = 3;
  int canvasNbYplots = 2;
};



class PlotGenerator {

public:
  PlotGenerator();
  virtual ~PlotGenerator();

  // Reset
  void reset();

  // Setters
  void setConfig(const nlohmann::json &config_);
  void setFitSampleSetPtr(const FitSampleSet *fitSampleSetPtr);

  // Init
  void initialize();
  void defineHistogramHolders();

  // Getters
  const std::vector<HistHolder> &getHistHolderList(int cacheSlot_ = 0) const;
  const std::vector<HistHolder> &getComparisonHistHolderList() const;
  std::map<std::string, std::shared_ptr<TCanvas>> getBufferCanvasList() const;

  // Core
  bool isEmpty() const;

  void generateSamplePlots(TDirectory *saveDir_ = nullptr, int cacheSlot_ = 0);
  void generateSampleHistograms(TDirectory *saveDir_ = nullptr, int cacheSlot_ = 0);
  void generateCanvas(const std::vector<HistHolder> &histHolderList_, TDirectory *saveDir_ = nullptr, bool stackHist_ = true);

  void generateComparisonPlots(const std::vector<HistHolder> &histsToStackOther_, const std::vector<HistHolder> &histsToStackReference_, TDirectory *saveDir_ = nullptr);
  void generateComparisonHistograms(const std::vector<HistHolder> &histList_, const std::vector<HistHolder> &refHistsList_, TDirectory *saveDir_ = nullptr);

  // Misc
  std::vector<std::string> fetchListOfVarToPlot(bool isData_ = false);
  std::vector<std::string> fetchListOfSplitVarNames();
  std::vector<std::string> fetchRequestedLeafNames();

protected:
  void buildEventBinCache(const std::vector<HistHolder *> &histPtrToFillList, const std::vector<PhysicsEvent> *eventListPtr, bool isData_);

private:
  nlohmann::json _config_;
  const FitSampleSet* _fitSampleSetPtr_{nullptr};
  int _maxLegendLength_{15};

  // Internals
  nlohmann::json _varDictionary_;
  nlohmann::json _canvasParameters_;
  nlohmann::json _histogramsDefinition_;
  std::vector<Color_t> defaultColorWheel;

  std::vector<std::vector<HistHolder>> _histHolderCacheList_;
  std::vector<HistHolder> _comparisonHistHolderList_;
  std::map<std::string, std::shared_ptr<TCanvas>> _bufferCanvasList_;

};


#endif //GUNDAM_PLOTGENERATOR_H
