//
// Created by Nadrino on 31/05/2024.
//

#ifndef GUNDAM_FIGURE_UTILS_H
#define GUNDAM_FIGURE_UTILS_H

#include "Event.h"
#include "Sample.h"

#include "TCanvas.h"
#include "TH1D.h"

#include <string>


struct FigureInstance {

  std::string saveFolderPath{};
  std::string subFolderPath{};

  int canvasHeight = 700;
  int canvasWidth = 1200;
  int canvasNbXplots = 3;
  int canvasNbYplots = 2;

  std::shared_ptr<TCanvas> canvas{nullptr};

};


struct PadInstance{

  std::string padTitle{};
  std::string xTitle{};
  std::string yTitle{};

  std::string xVariable{};
  std::vector<double> xEdges{};

  double xMin{std::nan("unset")};
  double xMax{std::nan("unset")};

};


struct HistogramInstance{

  std::string legendTitle{};

  short histColor{};
  short fillStyle{1001};

  std::shared_ptr<TH1D> histPtr{nullptr};

  bool isBinCacheBuilt{false};
  std::vector<std::vector<const Event*>> _binEventPtrList_;

  const Sample* samplePtr{nullptr};
  std::function<void(TH1D*, const Event*)> fillFunctionSample;

};



#endif //GUNDAM_FIGURE_UTILS_H
