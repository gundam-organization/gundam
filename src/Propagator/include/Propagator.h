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
  void setShowTimeStats(bool showTimeStats){ _showTimeStats_ = showTimeStats; }
  void setThrowAsimovToyParameters(bool throwAsimovToyParameters){ _throwAsimovToyParameters_ = throwAsimovToyParameters; }
  void setEnableEigenToOrigInPropagate(bool enableEigenToOrigInPropagate){ _enableEigenToOrigInPropagate_ = enableEigenToOrigInPropagate; }
  void setIThrow(int iThrow){ _iThrow_ = iThrow; }
  void setLoadAsimovData(bool loadAsimovData){ _loadAsimovData_ = loadAsimovData; }
  void setParameterInjectorConfig(const nlohmann::json &parameterInjector){ _parameterInjectorMc_ = parameterInjector; }
  void setGlobalCovarianceMatrix(const std::shared_ptr<TMatrixD> &globalCovarianceMatrix){ _globalCovarianceMatrix_ = globalCovarianceMatrix; }

  // Const getters
  [[nodiscard]] bool isThrowAsimovToyParameters() const { return _throwAsimovToyParameters_; }
  [[nodiscard]] int getIThrow() const { return _iThrow_; }
  [[nodiscard]] double getLlhBuffer() const{ return _llhBuffer_; }
  [[nodiscard]] double getLlhStatBuffer() const{ return _llhStatBuffer_; }
  [[nodiscard]] double getLlhPenaltyBuffer() const{ return _llhPenaltyBuffer_; }
  [[nodiscard]] double getLlhRegBuffer() const{ return _llhRegBuffer_; }
  [[nodiscard]] const EventTreeWriter &getTreeWriter() const{ return _treeWriter_; }
  [[nodiscard]] const std::shared_ptr<TMatrixD> &getGlobalCovarianceMatrix() const{ return _globalCovarianceMatrix_; }
  [[nodiscard]] const std::shared_ptr<TMatrixD> &getStrippedCovarianceMatrix() const{ return _strippedCovarianceMatrix_; }
  [[nodiscard]] const std::vector<DatasetLoader> &getDataSetList() const{ return _dataSetList_; }
  [[nodiscard]] const std::vector<FitParameterSet> &getParameterSetsList() const{ return _parameterSetList_; }
  [[nodiscard]] const std::vector<DialCollection> &getDialCollections() const{ return _dialCollections_; }

  // Non-const getters
  ParScanner& getParScanner(){ return _parScanner_; }
  FitSampleSet &getFitSampleSet(){ return _fitSampleSet_; }
  PlotGenerator &getPlotGenerator(){ return _plotGenerator_; }
  EventDialCache& getEventDialCache(){ return _eventDialCache_; }
  std::shared_ptr<TMatrixD> &getGlobalCovarianceMatrix(){ return _globalCovarianceMatrix_; }
  std::vector<DatasetLoader> &getDataSetList(){ return _dataSetList_; }
  std::vector<FitParameterSet> &getParameterSetsList(){ return _parameterSetList_; }

  // Misc getters
  [[nodiscard]] const double* getLlhBufferPtr() const { return &_llhBuffer_; }
  [[nodiscard]] const double* getLlhStatBufferPtr() const { return &_llhStatBuffer_; }
  [[nodiscard]] const double* getLlhPenaltyBufferPtr() const { return &_llhPenaltyBuffer_; }
  [[nodiscard]] const double* getLlhRegBufferPtr() const { return &_llhRegBuffer_; }
  [[nodiscard]] std::string getLlhBufferSummary() const;
  [[nodiscard]] std::string getParametersSummary( bool showEigen_ = true ) const;
  [[nodiscard]] const FitParameterSet* getFitParameterSetPtr(const std::string& name_) const;
  [[nodiscard]] FitParameterSet* getFitParameterSetPtr(const std::string& name_);
  [[nodiscard]] DatasetLoader* getDatasetLoaderPtr(const std::string& name_);

  // Core
  void updateLlhCache();
  void propagateParametersOnSamples();
  void resetReweight();
  void reweightMcEvents();
  void refillSampleHistograms();

  // Misc
  [[nodiscard]] nlohmann::json exportParameterInjectorConfig() const;
  void injectParameterValues(const nlohmann::json &config_);
  void throwParametersFromGlobalCovariance(bool quietVerbose_ = true);

  // Logger related
  static void muteLogger();
  static void unmuteLogger();

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
  nlohmann::json _parameterInjectorMc_;
  nlohmann::json _parameterInjectorToy_;

  // Internals
  bool _throwAsimovToyParameters_{false};
  bool _throwToyParametersWithGlobalCov_{false};
  bool _reThrowParSetIfOutOfBounds_{true};
  bool _enableStatThrowInToys_{true};
  bool _gaussStatThrowInToys_{false};
  bool _enableEventMcThrow_{true};
  bool _enableEigenToOrigInPropagate_{true};
  int _iThrow_{-1};
  double _llhBuffer_{0};
  double _llhStatBuffer_{0};
  double _llhPenaltyBuffer_{0};
  double _llhRegBuffer_{0};
  std::vector<FitParameter*> _globalCovParList_{};
  std::vector<FitParameter*> _strippedParameterList_{};
  std::shared_ptr<TMatrixD> _globalCovarianceMatrix_{nullptr};
  std::shared_ptr<TMatrixD> _strippedCovarianceMatrix_{nullptr};
  std::shared_ptr<TMatrixD> _choleskyMatrix_{nullptr};

  bool _devSingleThreadReweight_{false};
  bool _devSingleThreadHistFill_{false};

  // Sub-layers
  FitSampleSet _fitSampleSet_;
  PlotGenerator _plotGenerator_;
  EventTreeWriter _treeWriter_;
  ParScanner _parScanner_{this};
  std::vector<FitParameterSet> _parameterSetList_;
  std::vector<DatasetLoader> _dataSetList_;

  // Monitoring
  bool _showEventBreakdown_{true};

  // A vector of all the dial collections used by all the fit samples.
  // Once a dial collection has been added to this vector, it's index becomes
  // the immutable tag for that specific group of dials.
  std::vector<DialCollection> _dialCollections_{};
  EventDialCache _eventDialCache_{};

  // parallel holders
  std::function<void(int)> reweightMcEventsFct;
  std::function<void(int)> refillSampleHistogramsFct;
  std::function<void()> refillSampleHistogramsPostParallelFct;

public:
  GenericToolbox::CycleTimer dialUpdate;
  GenericToolbox::CycleTimer weightProp;
  GenericToolbox::CycleTimer fillProp;

};
#endif //GUNDAM_PROPAGATOR_H

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
