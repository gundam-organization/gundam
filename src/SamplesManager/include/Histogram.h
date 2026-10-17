//
// Created by Nadrino on 19/10/2024.
//

#ifndef GUNDAM_HISTOGRAM_H
#define GUNDAM_HISTOGRAM_H

#include "BinningResponse.h"
#include "Event.h"
#include "Bin.h"
#include "ConfigUtils.h"

#include "GenericToolbox.Loops.h"


namespace GenericToolbox { class ParallelWorker; }

class Histogram{

public:
  // structs
  struct BinContent{
    // keeps the contents close together in memory
    double sumWeights{0};
    double sqrtSumSqWeights{0};
  };
  struct EventContribution{
    Event* eventPtr{nullptr};
    double fraction{0};
  };
  struct BinContext{
    Bin bin{};
    std::vector<Event*> eventPtrList{};
    std::vector<EventContribution> eventContributionList{};
  };

  // const getters
  [[nodiscard]] const BinningResponse& getBinningResponse() const{ return _binningResponse_; }
  [[nodiscard]] bool isBinningResponseEnabled() const{ return _binningResponse_.isEnabled(); }
  [[nodiscard]] auto& getOutsideBinContent() const{ return _outsideBinContent_; }
  [[nodiscard]] auto getNbBins() const { return nBins; }
  [[nodiscard]] auto& getBinContentList() const { return binContentList; }
  [[nodiscard]] auto& getBinContextList() const { return binContextList; }

  // mutable getters
  auto& getBinContentList(){ return binContentList; }
  auto& getBinContextList(){ return binContextList; }

  // core
  void configureBinningResponse(const ConfigReader& config_);
  void build(const ConfigReader& binningConfig_);
  auto& getOutsideBinContent(){ return _outsideBinContent_; }
  auto& getOutsideContributionList(){ return _outsideContributionList_; }
  // Bin edge variable-index caches must match the event's variable layout.
  [[nodiscard]] int findBinIndex(const Event& event_) const;
  [[nodiscard]] BinningResponse::EventResponse evaluateEventBinningResponse(const Event& event_) const;
  [[nodiscard]] std::vector<BinningResponse::BinFraction> evaluateBinningResponse(const Event& event_) const;
  void throwEventMcError();
  // With a binning response, fluctuates bin contents only; refilling discards the throw.
  void throwStatError(bool useGaussThrow_ = false);

  // multi-thread
  void updateBinEventLists(std::vector<Event>& eventList_, GenericToolbox::ParallelWorker& threadPool_, bool updateBinIndices_ = false);
  void updateBinEventList(std::vector<Event>& eventList_, int iThread_ = -1, int nThreads_ = -1);
  void refillHistogram(int iThread_ = -1);

  // utils
  auto loop(){ return GenericToolbox::Zip(binContentList, binContextList); }
  auto loop(size_t start_, size_t end_){ return GenericToolbox::ZipPartial(start_, end_, binContentList, binContextList); }
  [[nodiscard]] auto loop() const{ return GenericToolbox::Zip(binContentList, binContextList); }
  [[nodiscard]] auto loop(size_t start_, size_t end_) const { return GenericToolbox::ZipPartial(start_, end_, binContentList, binContextList); }

private:
  struct IndexedContribution{
    int binIndex{-1};
    EventContribution contribution;
  };
  struct ResponseWorker{
    BinningResponse::EvaluationBuffer evaluationBuffer;
    std::vector<IndexedContribution> contributions;
    std::vector<size_t> contributionOffsets;
    std::vector<size_t> eventOffsets;
  };

  std::vector<EventContribution>& getContributionList(int binIndex_);
  static void fillBinContent(const std::vector<EventContribution>& contributions_, BinContent& content_);

  BinContent _outsideBinContent_;
  std::vector<EventContribution> _outsideContributionList_;
  std::vector<ResponseWorker> _responseWorkers_;
  BinningResponse _binningResponse_;
  int nBins{0};
  std::vector<BinContent> binContentList{};
  std::vector<BinContext> binContextList{};

};


#endif //GUNDAM_HISTOGRAM_H
