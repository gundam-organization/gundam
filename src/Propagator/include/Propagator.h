//
// Created by Nadrino on 11/06/2021.
//

#ifndef GUNDAM_PROPAGATOR_H
#define GUNDAM_PROPAGATOR_H


#include "ParametersManager.h"
#include "DialCollection.h"
#include "EventDialCache.h"
#include "PlotGenerator.h"
#include "JsonBaseClass.h"
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
  [[nodiscard]] bool isEnableStatThrowInToys() const { return _enableStatThrowInToys_; }
  [[nodiscard]] bool isEnableEventMcThrow() const { return _enableEventMcThrow_; }
  [[nodiscard]] bool isGaussStatThrowInToys() const { return _gaussStatThrowInToys_; }
  [[nodiscard]] bool isLoadAsimovData() const { return _loadAsimovData_; }
  [[nodiscard]] bool isShowEventBreakdown() const { return _showEventBreakdown_; }
  [[nodiscard]] bool isDebugPrintLoadedEvents() const { return _debugPrintLoadedEvents_; }
  [[nodiscard]] int getDebugPrintLoadedEventsNbPerSample() const { return _debugPrintLoadedEventsNbPerSample_; }
  [[nodiscard]] int getIThrow() const { return _iThrow_; }
  [[nodiscard]] const EventDialCache& getEventDialCache() const { return _eventDialCache_; }
  [[nodiscard]] const ParametersManager &getParametersManager() const { return _parManager_; }
  [[nodiscard]] const std::vector<DialCollection> &getDialCollectionList() const{ return _dialCollectionList_; }
  [[nodiscard]] const SampleSet &getSampleSet() const { return _sampleSet_; }
  [[nodiscard]] const JsonType &getParameterInjectorMc() const { return _parameterInjectorMc_;; }

  // Non-const getters
  SampleSet &getSampleSet(){ return _sampleSet_; }
  ParametersManager &getParametersManager(){ return _parManager_; }
  PlotGenerator &getPlotGenerator(){ return _plotGenerator_; }
  EventDialCache& getEventDialCache(){ return _eventDialCache_; }
  std::vector<DialCollection> &getDialCollectionList(){ return _dialCollectionList_; }

  // Core
  void propagateParameters();
  void resetReweight();
  void reweightMcEvents();
  void refillMcHistograms();

  // Misc
  [[nodiscard]] std::string getSampleBreakdownTableStr() const;
  void printBreakdowns();

  // Logger related
  static void muteLogger();
  static void unmuteLogger();

protected:
  void initializeThreads();

  // multithreading
  void reweightMcEvents(int iThread_);
  void refillMcHistogramsFct( int iThread_);

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
  bool _showEventBreakdown_{true};
  bool _enableEigenToOrigInPropagate_{true};
  int _iThrow_{-1};

  // Sub-layers
  SampleSet _sampleSet_{};
  PlotGenerator _plotGenerator_{};
  EventDialCache _eventDialCache_{};
  ParametersManager _parManager_{};

  // A vector of all the dial collections used by all the fit samples.
  // Once a dial collection has been added to this vector, it's index becomes
  // the immutable tag for that specific group of dials.
  std::vector<DialCollection> _dialCollectionList_{};

public:
  GenericToolbox::Time::AveragedTimer<10> reweightTimer;
  GenericToolbox::Time::AveragedTimer<10> refillHistogramTimer;

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
