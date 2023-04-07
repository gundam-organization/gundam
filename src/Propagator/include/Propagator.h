//
// Created by Nadrino on 11/06/2021.
//

#ifndef GUNDAM_PROPAGATOR_H
#define GUNDAM_PROPAGATOR_H


#include "DatasetLoader.h"
#include "PlotGenerator.h"
#include "EventTreeWriter.h"
#include "FitSampleSet.h"
#include "FitParameterSet.h"
#include "JsonBaseClass.h"
#include "ParScanner.h"
#include "DialCollection.h"
#include "EventDialCache.h"

#include "GenericToolbox.CycleTimer.h"

#include <vector>
#include <map>
#include <future>

class Propagator : public JsonBaseClass {

public:
  // Setters
  void setShowTimeStats(bool showTimeStats);
  void setThrowAsimovToyParameters(bool throwAsimovToyParameters);
  void setIThrow(int iThrow);
  void setLoadAsimovData(bool loadAsimovData);
  void setParameterInjector(const nlohmann::json &parameterInjector);
  void setGlobalCovarianceMatrix(const std::shared_ptr<TMatrixD> &globalCovarianceMatrix);

  // Const getters
  [[nodiscard]] bool isThrowAsimovToyParameters() const;
  [[nodiscard]] int getIThrow() const;
  [[nodiscard]] double getLlhBuffer() const;
  [[nodiscard]] double getLlhStatBuffer() const;
  [[nodiscard]] double getLlhPenaltyBuffer() const;
  [[nodiscard]] double getLlhRegBuffer() const;
  [[nodiscard]] const EventTreeWriter &getTreeWriter() const;
  [[nodiscard]] const std::shared_ptr<TMatrixD> &getGlobalCovarianceMatrix() const;
  [[nodiscard]] const std::vector<DatasetLoader> &getDataSetList() const;
  [[nodiscard]] const std::vector<FitParameterSet> &getParameterSetsList() const;

  // Non-const getters
  std::shared_ptr<TMatrixD> &getGlobalCovarianceMatrix();
  FitSampleSet &getFitSampleSet();
  PlotGenerator &getPlotGenerator();
  ParScanner& getParScanner(){ return _parScanner_; }
  std::vector<FitParameterSet> &getParameterSetsList();
  std::vector<DatasetLoader> &getDataSetList();

  // Misc getters
  double* getLlhBufferPtr(){ return &_llhBuffer_; }
  double* getLlhStatBufferPtr(){ return &_llhStatBuffer_; }
  double* getLlhPenaltyBufferPtr(){ return &_llhPenaltyBuffer_; }
  double* getLlhRegBufferPtr(){ return &_llhRegBuffer_; }

  [[nodiscard]] const FitParameterSet* getFitParameterSetPtr(const std::string& name_) const;
  [[nodiscard]] FitParameterSet* getFitParameterSetPtr(const std::string& name_);

  // Core
  void updateLlhCache();
  void propagateParametersOnSamples();
  void resetReweight();
  void reweightMcEvents();
  void refillSampleHistograms();
  void throwParametersFromGlobalCovariance();

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

  void initializeThreads();

  // multithreading
  void reweightMcEvents(int iThread_);

private:
  // Parameters
  bool _showTimeStats_{false};
  bool _loadAsimovData_{false};
  bool _debugPrintLoadedEvents_{false};
  int _debugPrintLoadedEventsNbPerSample_{5};
  nlohmann::json _parameterInjector_;

  // Internals
  bool _throwAsimovToyParameters_{false};
  bool _reThrowParSetIfOutOfBounds_{true};
  bool _enableStatThrowInToys_{true};
  bool _gaussStatThrowInToys_{false};
  bool _enableEventMcThrow_{true};
  int _iThrow_{-1};
  double _llhBuffer_{0};
  double _llhStatBuffer_{0};
  double _llhPenaltyBuffer_{0};
  double _llhRegBuffer_{0};
  std::shared_ptr<TMatrixD> _globalCovarianceMatrix_{nullptr};
  std::shared_ptr<TMatrixD> _strippedCovarianceMatrix_{nullptr};
  std::shared_ptr<TMatrixD> _choleskyMatrix_{nullptr};
  std::vector<FitParameter*> _strippedParameterList_{};

  // Sub-layers
  FitSampleSet _fitSampleSet_;
  PlotGenerator _plotGenerator_;
  EventTreeWriter _treeWriter_;
  ParScanner _parScanner_{this};
  std::vector<FitParameterSet> _parameterSetList_;
  std::vector<DatasetLoader> _dataSetList_;

  // Monitoring
  bool _showEventBreakdown_{true};

#ifdef GUNDAM_USING_CACHE_MANAGER
  // Build the precalculated caches.  This is only relevant when using a GPU
  // and must be done after the datasets are loaded.  This returns true if
  // the cache is built.
  bool buildGPUCaches();

  // Prefill the caches using a GPU (if available).  If the GPU is not
  // available, then this is a NOP.  This copies the fit parameter values into
  // the GPU, triggers the appropriate kernel(s), and copies the results into
  // a CPU based cache.  This returns true if the cache is filled.
  bool fillGPUCaches();

  // A map of parameters to the indices that got used by the GPU.
  std::map<const FitParameter*, int> _gpuParameterIndex_;
#endif

  // DEV
#if USE_NEW_DIALS
  // A vector of all the dial collections used by all of the fit samples.
  // Once a dial collection has been added to this vector, it's index becomes
  // the immutable tag for that specific group of dials.
  std::vector<DialCollection> _dialCollections_{};
  EventDialCache _eventDialCache_{};
#endif

public:
  GenericToolbox::CycleTimer dialUpdate;
  GenericToolbox::CycleTimer weightProp;
  GenericToolbox::CycleTimer fillProp;

};


#endif //GUNDAM_PROPAGATOR_H
