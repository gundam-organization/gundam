//
// Created by Nadrino on 16/06/2021.
//

#ifndef XSLLHFITTER_PLOTGENERATOR_H
#define XSLLHFITTER_PLOTGENERATOR_H

#include "json.hpp"
#include "TDirectory.h"

#include "AnaSample.hh"

struct HistHolder{
  // Hist
  TH1D* histPtr{nullptr};

  // Path
  std::string folderPath;
  std::string histName;
  std::string histTitle;

  // Data
  bool isData{false};
  const AnaSample* samplePtr{nullptr};
  std::mutex* fillMutexPtr{nullptr};
  std::function<void(TH1D*, const AnaEvent*)> fillFunction;

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

  // Flags
  bool isBaseSplitHist{false};
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
  void setSampleListPtr(const std::vector<AnaSample> *sampleListPtr_);

  // Init
  void initialize();

  // Getters
  const std::vector<HistHolder> &getHistHolderList() const;
  const std::vector<HistHolder> &getComparisonHistHolderList() const;
  std::map<std::string, TCanvas *> getBufferCanvasList() const;

  // Core
  void generateSamplePlots(TDirectory *saveDir_ = nullptr);
  void generateSampleHistograms(TDirectory *saveDir_ = nullptr);
  void generateCanvas(const std::vector<HistHolder> &histHolderList_, TDirectory *saveDir_ = nullptr, bool stackHist_ = true);

  void generateComparisonPlots(const std::vector<HistHolder> &histsToStackOther_, const std::vector<HistHolder> &histsToStackReference_, TDirectory *saveDir_ = nullptr);
  void generateComparisonHistograms(const std::vector<HistHolder> &histList_, const std::vector<HistHolder> &refHistsList_, TDirectory *saveDir_ = nullptr);

  // Misc
  std::vector<std::string> fetchListOfVarToPlot();
  std::vector<std::string> fetchListOfSplitVarNames();
  std::vector<std::string> fetchRequestedLeafNames();


protected:
  void defineHistogramHolders();


private:
  nlohmann::json _config_;
  const std::vector<AnaSample>* _sampleListPtr_;

  // Internals
  nlohmann::json _varDictionary_;
  nlohmann::json _canvasParameters_;
  nlohmann::json _histogramsDefinition_;
  std::vector<Color_t> defaultColorWheel;

  std::vector<HistHolder> _histHolderList_;
  std::vector<HistHolder> _comparisonHistHolderList_;
  std::map<std::string, TCanvas*> _bufferCanvasList_;


};


#endif //XSLLHFITTER_PLOTGENERATOR_H
