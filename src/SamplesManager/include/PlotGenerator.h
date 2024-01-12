//
// Created by Nadrino on 16/06/2021.
//

#ifndef GUNDAM_PLOTGENERATOR_H
#define GUNDAM_PLOTGENERATOR_H

#include "SampleSet.h"
#include "PhysicsEvent.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.Wrappers.h"

#include "nlohmann/json.hpp"
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
  const Sample* fitSamplePtr{nullptr};
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



class PlotGenerator : public JsonBaseClass {

public:
  // Setters
  void setFitSampleSetPtr(const SampleSet *fitSampleSetPtr);

  // Getters
  [[nodiscard]] bool isEmpty() const;
  [[nodiscard]] const std::vector<HistHolder> &getHistHolderList(int cacheSlot_ = 0) const;
  [[nodiscard]] const std::vector<HistHolder> &getComparisonHistHolderList() const;
  [[nodiscard]] std::map<std::string, std::shared_ptr<TCanvas>> getBufferCanvasList() const;

  // mutable getters
  std::vector<HistHolder> &getHistHolderList(int cacheSlot_ = 0);

  // Core
  void generateSamplePlots(TDirectory *saveDir_ = nullptr, int cacheSlot_ = 0);
  void generateSampleHistograms(TDirectory *saveDir_ = nullptr, int cacheSlot_ = 0);
  void generateCanvas(const std::vector<HistHolder> &histHolderList_, TDirectory *saveDir_ = nullptr, bool stackHist_ = true);
  void generateComparisonPlots(const std::vector<HistHolder> &histsToStackOther_, const std::vector<HistHolder> &histsToStackReference_, TDirectory *saveDir_ = nullptr);
  void generateComparisonHistograms(const std::vector<HistHolder> &histList_, const std::vector<HistHolder> &refHistsList_, TDirectory *saveDir_ = nullptr);

  // Misc
  std::vector<std::string> fetchListOfVarToPlot(bool isData_ = false);
  std::vector<std::string> fetchListOfSplitVarNames();

  void defineHistogramHolders();
protected:
  void readConfigImpl() override;
  void initializeImpl() override;

  // Internals
  static void buildEventBinCache(const std::vector<HistHolder *> &histPtrToFillList, const std::vector<PhysicsEvent> *eventListPtr, bool isData_);

private:
  // Parameters
  bool _writeGeneratedHistograms_{false};
  int _maxLegendLength_{15};
  JsonType _varDictionary_;
  JsonType _canvasParameters_;
  JsonType _histogramsDefinition_;
  std::vector<Color_t> defaultColorWheel {
      kGreen-3, kTeal+3, kAzure+7,
      kCyan-2, kBlue-7, kBlue+2,
      kOrange+1, kOrange+9, kRed+2, kPink+9
  };

  // Internals
  const SampleSet* _fitSampleSetPtr_{nullptr};
  std::vector<std::vector<HistHolder>> _histHolderCacheList_{};
  std::vector<HistHolder> _comparisonHistHolderList_;
  std::map<std::string, std::shared_ptr<TCanvas>> _bufferCanvasList_;

};


#endif //GUNDAM_PLOTGENERATOR_H
