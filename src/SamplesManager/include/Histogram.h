//
// Created by Nadrino on 19/10/2024.
//

#ifndef GUNDAM_HISTOGRAM_H
#define GUNDAM_HISTOGRAM_H

#include "Event.h"
#include "Bin.h"
#include "ConfigUtils.h"

#include "GenericToolbox.Loops.h"


class Histogram{

public:
  // structs
  struct BinContent{
    // keeps the contents close together in memory
    double sumWeights{0};
    double sqrtSumSqWeights{0};
  };
  struct BinContext{
    Bin bin{};
    std::vector<Event*> eventPtrList{};
  };

  // const getters
  [[nodiscard]] auto getNbBins() const { return nBins; }
  [[nodiscard]] auto& getBinContentList() const { return binContentList; }
  [[nodiscard]] auto& getBinContextList() const { return binContextList; }

  // mutable getters
  auto& getBinContentList(){ return binContentList; }
  auto& getBinContextList(){ return binContextList; }

  // core
  void build(const JsonType& binningConfig_);
  void throwEventMcError();
  void throwStatError(bool useGaussThrow_ = false);

  // multi-thread
  void updateBinEventList(std::vector<Event>& eventList_, int iThread_ = -1);
  void refillHistogram(int iThread_ = -1);

  // utils
  auto loop(){ return GenericToolbox::Zip(binContentList, binContextList); }
  auto loop(size_t start_, size_t end_){ return GenericToolbox::ZipPartial(start_, end_, binContentList, binContextList); }
  [[nodiscard]] auto loop() const{ return GenericToolbox::Zip(binContentList, binContextList); }
  [[nodiscard]] auto loop(size_t start_, size_t end_) const { return GenericToolbox::ZipPartial(start_, end_, binContentList, binContextList); }

private:
  int nBins{0};
  std::vector<BinContent> binContentList{};
  std::vector<BinContext> binContextList{};

};


#endif //GUNDAM_HISTOGRAM_H
