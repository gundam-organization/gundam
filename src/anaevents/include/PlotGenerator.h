//
// Created by Adrien BLANCHET on 16/06/2021.
//

#ifndef XSLLHFITTER_PLOTGENERATOR_H
#define XSLLHFITTER_PLOTGENERATOR_H

#include "json.hpp"
#include "TDirectory.h"

#include "AnaSample.hh"

class PlotGenerator {

public:
  PlotGenerator();
  virtual ~PlotGenerator();

  // Reset
  void reset();

  // Setters
  void setConfig(const nlohmann::json &config);

  // Init
  void initialize();

  // Getters
  std::map<std::string, TH1D *> getBufferHistogramList() const; // copies of the maps (not the ptr)
  std::map<std::string, TCanvas *> getBufferCanvasList() const;
  std::map<std::string, std::map<std::string, std::vector<TH1D *>>> getHistsToStack() const;
  std::map<std::string, std::map<std::string, std::vector<TH1D *>>> getCompHistsToStack() const;

  // Core
  void generateSamplePlots(const std::vector<AnaSample> &sampleList_, TDirectory *saveTDirectory_ = nullptr);
  void generateSampleHistograms(const std::vector<AnaSample> &sampleList_, TDirectory *saveDir_ = nullptr);
  void generateCanvas(const std::map<std::string, std::map<std::string, std::vector<TH1D*>>>& histsToStack_, TDirectory *saveDir_ = nullptr, bool stackHist_ = true);

  void generateComparisonPlots(
    const std::map<std::string, std::map<std::string, std::vector<TH1D *>>> &histsToStackOther_,
    const std::map<std::string, std::map<std::string, std::vector<TH1D *>>> &histsToStackReference_,
    TDirectory *saveDir_ = nullptr);
  void generateComparisonHistograms(
    const std::map<std::string, std::map<std::string, std::vector<TH1D *>>> &histsToStackOther_,
    const std::map<std::string, std::map<std::string, std::vector<TH1D *>>> &histsToStackReference_,
    TDirectory *saveDir_ = nullptr);

private:
  nlohmann::json _config_;

  // Internals
  nlohmann::json _varDictionary_;
  nlohmann::json _canvasParameters_;
  nlohmann::json _histogramsDefinition_;
  std::vector<Color_t> defaultColorWheel;
  std::map<std::string, TH1D*> _bufferHistogramList_;
  std::map<std::string, std::map<std::string, std::vector<TH1D*>>> _histsToStack_;
  std::map<std::string, std::map<std::string, std::vector<TH1D*>>> _compHistsToStack_;
  std::map<std::string, TCanvas*> _bufferCanvasList_;


};


#endif //XSLLHFITTER_PLOTGENERATOR_H
