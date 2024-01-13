//
// Created by Clark McGrew on 25/01/2023.
//

#ifndef GUNDAM_MINIMIZER_BASE_H
#define GUNDAM_MINIMIZER_BASE_H

#include "Propagator.h"
#include "ParameterScanner.h"
#include "LikelihoodInterface.h"
#include "Parameter.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.Utils.h"

#include "TDirectory.h"

#include <vector>
#include <string>


class FitterEngine; // owner

class MinimizerBase : public JsonBaseClass {

protected:
  struct Monitor;

  void readConfigImpl() override;
  void initializeImpl() override;

public:
  // virtual
  virtual void minimize();
  virtual void calcErrors(){}
  virtual void scanParameters( TDirectory* saveDir_ );
  virtual bool isErrorCalcEnabled(){ return false; }
  [[nodiscard]] virtual bool isFitHasConverged() const = 0;

  // c-tor
  explicit MinimizerBase(FitterEngine* owner_) : _owner_(owner_) {}

  // const getters
  int getMinimizerStatus() const { return _minimizerStatus_; }
  const FitterEngine& getOwner() const;
  const Propagator& getPropagator() const;
  const ParameterScanner& getParameterScanner() const;
  const LikelihoodInterface& getLikelihoodInterface() const;

  // mutable getters
  Monitor& getMonitor(){ return _monitor_; }
  FitterEngine& getOwner();
  Propagator& getPropagator();
  ParameterScanner& getParameterScanner();
  LikelihoodInterface& getLikelihoodInterface();

  // core
  int getNbDegreeOfFreedom(){ return getLikelihoodInterface().getNbSampleBins() - _nbFreeParameters_; }

protected:
  void printParameters();

  // config
  bool _throwOnBadLlh_{false};
  bool _useNormalizedFitSpace_{true};

  // internals
  int _minimizerStatus_{-1}; // -1: invalid, 0: success, >0: errors
  int _nbFreeParameters_{0};
  std::vector<Parameter*> _minimizerParameterPtrList_{};

  // monitor
  struct Monitor{
    bool isEnabled{false};
    bool showParameters{false};
    int maxNbParametersPerLine{15};
    int nbEvalLikelihoodCalls{0};

    std::string minimizerTitle{"unset"};
    std::string stateTitleMonitor{};

    GenericToolbox::Time::AveragedTimer<10> evalLlhTimer{};
    GenericToolbox::Time::AveragedTimer<10> externalTimer{};
    GenericToolbox::Time::CycleTimer itSpeed;
    GenericToolbox::Time::CycleCounterClock itSpeedMon{"it"};

    GenericToolbox::VariablesMonitor convergenceMonitor;

    std::unique_ptr<TTree> historyTree{nullptr};

    struct GradientDescentMonitor{
      bool isEnabled{false};
      int lastGradientFall{-2};
      struct GradientStepPoint {
        JsonType parState;
        double llh;
      };
      std::vector<GradientStepPoint> stepPointList{};
    };
    GradientDescentMonitor gradientDescentMonitor{};

    void writeHistoryTree(TDirectory* saveDir_);
  };
  Monitor _monitor_{};

private:
  FitterEngine* _owner_{nullptr};

};

#endif // GUNDAM_MINIMIZER_BASE_H

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
