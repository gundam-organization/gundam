//
// Created by Nadrino on 11/06/2021.
//

#ifndef GUNDAM_PROPAGATOR_H
#define GUNDAM_PROPAGATOR_H


#include <DialManager.h>

#include "ParametersManager.h"
#include "DialCollection.h"
#include "EventDialCache.h"
#include "SampleSet.h"

#include "GenericToolbox.Time.h"
#include "GenericToolbox.Thread.h"

#include <vector>
#include <map>
#include <future>

class Propagator : public JsonBaseClass {

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  static void muteLogger();
  static void unmuteLogger();

  Propagator() = default;

  // setters
  void setShowTimeStats(bool showTimeStats){ _showTimeStats_ = showTimeStats; }
  void setEnableEigenToOrigInPropagate(bool enableEigenToOrigInPropagate){ _enableEigenToOrigInPropagate_ = enableEigenToOrigInPropagate; }
  void setIThrow(int iThrow){ _iThrow_ = iThrow; }
  void setParameterInjectorConfig(const JsonType &parameterInjector){ _parameterInjectorMc_ = parameterInjector; }

  // const getters
  [[nodiscard]] auto getIThrow() const { return _iThrow_; }
  [[nodiscard]] auto isDebugPrintLoadedEvents() const { return _debugPrintLoadedEvents_; }
  [[nodiscard]] auto getDebugPrintLoadedEventsNbPerSample() const { return _debugPrintLoadedEventsNbPerSample_; }
  [[nodiscard]] auto& getSampleSet() const { return _sampleSet_; }
  [[nodiscard]] auto& getDialManager() const { return _dialManager_; }
  [[nodiscard]] auto& getEventDialCache() const { return _eventDialCache_; }
  [[nodiscard]] auto& getParametersManager() const { return _parManager_; }
  [[nodiscard]] auto& getDialCollectionList() const{ return _dialManager_.getDialCollectionList(); }
  [[nodiscard]] auto& getParameterInjectorMc() const { return _parameterInjectorMc_; }

  // mutable getters
  auto& getSampleSet(){ return _sampleSet_; }
  auto& getThreadPool(){ return _threadPool_; }
  auto& getDialManager(){ return _dialManager_; }
  auto& getEventDialCache(){ return _eventDialCache_; }
  auto& getParametersManager(){ return _parManager_; }
  auto& getDialCollectionList(){ return _dialManager_.getDialCollectionList(); }

  // Core
  void clearContent();
  void buildDialCache();

  /// Apply the current parameters and wait for it to finish.  This reweights
  /// the events, and refills the histograms.  This is a convenience wrapper
  /// around applyParameters().get().
  void propagateParameters();

  /// Promise to eventually apply the current parameters.  This returns a
  /// future that becomes available after reweighting and the histograms are
  /// refilled.  With The CPU, the parameters are applied synchronously so the
  /// future is basically a "dummy".  When a GPU is used, the promise is
  /// returned before the information is copied from the GPU, so accessing it
  /// may cause a wait.  The future will be valid if the calculation was
  /// successfully started, true if the calculation has completed correctly,
  /// and false if the calculation started correctly, but failed.
  std::future<bool> applyParameters();

  void reweightEvents(bool updateDials = true);

  // misc
  void copyEventsFrom(const Propagator& src_);
  void copyHistBinContentFrom(const Propagator& src_);
  void printConfiguration() const;
  void printBreakdowns() const;
  void writeEventRates(const GenericToolbox::TFilePath& saveDir_) const;
  void writeParameterStateTree(const GenericToolbox::TFilePath& saveDir_) const;

#ifdef GUNDAM_USING_CACHE_MANAGER
  void initializeCacheManager();
#endif

  // public members
  GenericToolbox::Time::AveragedTimer<10> reweightTimer;
  GenericToolbox::Time::AveragedTimer<10> refillHistogramTimer;

protected:
  void initializeThreads();

  // multithreading
  void reweightEvents( int iThread_);
  void refillHistogramsFct( int iThread_);

  void refillHistograms();

private:

  // Parameters
  bool _showTimeStats_{false};
  bool _debugPrintLoadedEvents_{false};
  bool _devSingleThreadReweight_{false};
  bool _devSingleThreadHistFill_{false};
  int _debugPrintLoadedEventsNbPerSample_{5};
  JsonType _parameterInjectorMc_;
  JsonType _parameterInjectorToy_;

  // Internals
  bool _showNbEventParameterBreakdown_{true};
  bool _showNbEventPerSampleParameterBreakdown_{false};
  bool _enableEigenToOrigInPropagate_{true};
  int _iThrow_{-1};

  // Sub-layers
  SampleSet _sampleSet_{};
  EventDialCache _eventDialCache_{};
  ParametersManager _parManager_{};
  DialManager _dialManager_{};

  GenericToolbox::ParallelWorker _threadPool_{};

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
