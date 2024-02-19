//
// Created by Nadrino on 11/06/2021.
//

#ifndef GUNDAM_PROPAGATOR_H
#define GUNDAM_PROPAGATOR_H


#include "ParametersManager.h"
#include "EventTreeWriter.h"
#include "DialCollection.h"
#include "EventDialCache.h"
#include "DatasetLoader.h"
#include "PlotGenerator.h"
#include "JsonBaseClass.h"
#include "ParScanner.h"
#include "SampleSet.h"

#include "GenericToolbox.Time.h"

#include <vector>
#include <map>
#include <future>

class Propagator : public JsonBaseClass {


protected:
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  Propagator() = default;

  // Setters
  void setShowTimeStats(bool showTimeStats){ _showTimeStats_ = showTimeStats; }
  void setThrowAsimovToyParameters(bool throwAsimovToyParameters){ _throwAsimovToyParameters_ = throwAsimovToyParameters; }
  void setEnableEigenToOrigInPropagate(bool enableEigenToOrigInPropagate){ _enableEigenToOrigInPropagate_ = enableEigenToOrigInPropagate; }
  void setIThrow(int iThrow){ _iThrow_ = iThrow; }
  void setLoadAsimovData(bool loadAsimovData){ _loadAsimovData_ = loadAsimovData; }
  void setParameterInjectorConfig(const JsonType &parameterInjector){ _parameterInjectorMc_ = parameterInjector; }

  // Const getters
  [[nodiscard]] bool isThrowAsimovToyParameters() const { return _throwAsimovToyParameters_; }
  [[nodiscard]] int getIThrow() const { return _iThrow_; }
  [[nodiscard]] double getLlhBuffer() const{ return _llhBuffer_; }
  [[nodiscard]] double getLlhStatBuffer() const{ return _llhStatBuffer_; }
  [[nodiscard]] double getLlhPenaltyBuffer() const{ return _llhPenaltyBuffer_; }
  [[nodiscard]] double getLlhRegBuffer() const{ return _llhRegBuffer_; }
  [[nodiscard]] const ParametersManager &getParametersManager() const { return _parManager_; }
  [[nodiscard]] const EventTreeWriter &getTreeWriter() const{ return _treeWriter_; }
  [[nodiscard]] const std::vector<DatasetLoader> &getDataSetList() const{ return _dataSetList_; }
  [[nodiscard]] const std::vector<DialCollection> &getDialCollections() const{ return _dialCollections_; }

  // Non-const getters
  ParScanner& getParScanner(){ return _parScanner_; }
  SampleSet &getSampleSet(){ return _fitSampleSet_; }
  ParametersManager &getParametersManager(){ return _parManager_; }
  PlotGenerator &getPlotGenerator(){ return _plotGenerator_; }
  EventDialCache& getEventDialCache(){ return _eventDialCache_; }
  std::vector<DatasetLoader> &getDataSetList(){ return _dataSetList_; }

  // Misc getters
  [[nodiscard]] const double* getLlhBufferPtr() const { return &_llhBuffer_; }
  [[nodiscard]] const double* getLlhStatBufferPtr() const { return &_llhStatBuffer_; }
  [[nodiscard]] const double* getLlhPenaltyBufferPtr() const { return &_llhPenaltyBuffer_; }
  [[nodiscard]] const double* getLlhRegBufferPtr() const { return &_llhRegBuffer_; }
  [[nodiscard]] std::string getLlhBufferSummary() const;
  [[nodiscard]] DatasetLoader* getDatasetLoaderPtr(const std::string& name_);

  // Core
  void updateLlhCache();
  void propagateParametersOnSamples();
  void resetReweight();
  void reweightMcEvents();
  void refillSampleHistograms();

  // Misc
  [[nodiscard]] std::string getSampleBreakdownTableStr() const;

  // Logger related
  static void muteLogger();
  static void unmuteLogger();

  // Deprecated
  [[deprecated("use getSampleSet()")]] SampleSet &getFitSampleSet(){ return _fitSampleSet_; }

protected:
  void initializeThreads();

  // multithreading
  void reweightMcEvents(int iThread_);
  void refillSampleHistogramsFct(int iThread_);
  void refillSampleHistogramsPostParallelFct();

private:
  // Parameters
  bool _showTimeStats_{false};
  bool _loadAsimovData_{false};
  bool _debugPrintLoadedEvents_{false};
  bool _devSingleThreadReweight_{false};
  bool _devSingleThreadHistFill_{false};
  int _debugPrintLoadedEventsNbPerSample_{5};
  JsonType _parameterInjectorMc_;
  JsonType _parameterInjectorToy_;

  // Internals
  bool _throwAsimovToyParameters_{false};
  bool _enableStatThrowInToys_{true};
  bool _gaussStatThrowInToys_{false};
  bool _enableEventMcThrow_{true};
  bool _enableEigenToOrigInPropagate_{true};
  int _iThrow_{-1};
  double _llhBuffer_{0};
  double _llhStatBuffer_{0};
  double _llhPenaltyBuffer_{0};
  double _llhRegBuffer_{0};

  // Sub-layers
  SampleSet _fitSampleSet_;
  ParametersManager _parManager_;
  PlotGenerator _plotGenerator_;
  EventTreeWriter _treeWriter_;
  ParScanner _parScanner_{this};
  std::vector<DatasetLoader> _dataSetList_;

  // Monitoring
  bool _showEventBreakdown_{true};

  // A vector of all the dial collections used by all the fit samples.
  // Once a dial collection has been added to this vector, it's index becomes
  // the immutable tag for that specific group of dials.
  std::vector<DialCollection> _dialCollections_{};
  EventDialCache _eventDialCache_{};

public:
  GenericToolbox::Time::AveragedTimer<10> reweightTimer{};
  GenericToolbox::Time::AveragedTimer<10> refillHistogramTimer{};

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
