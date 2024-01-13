//
// Created by Clark McGrew on 25/01/2023.
//

#ifndef GUNDAM_MINIMIZER_BASE_H
#define GUNDAM_MINIMIZER_BASE_H

#include "Propagator.h"
#include "Parameter.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.Utils.h"

#include "TDirectory.h"

#include <vector>
#include <string>


class FitterEngine; // owner

class MinimizerBase : public JsonBaseClass {

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  explicit MinimizerBase(FitterEngine* owner_) : _owner_(owner_) {}

  virtual void minimize() = 0;

  [[nodiscard]] virtual bool isFitHasConverged() const = 0;
  virtual double evalFit( const double* parArray_ );

  /// Set if the calcErrors method should be called by the FitterEngine.
  void setEnablePostFitErrorEval(bool enablePostFitErrorEval_) {_enablePostFitErrorEval_ = enablePostFitErrorEval_;}
  [[nodiscard]] bool isEnablePostFitErrorEval() const {return _enablePostFitErrorEval_;}

protected:
  void summarizeParameters();

protected:

  // config
  bool _throwOnBadLlh_{false};
  bool _useNormalizedFitSpace_{true};
  bool _enablePostFitErrorEval_{true};

  // internals
  FitterEngine* _owner_{nullptr};
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

  };
  Monitor _monitor_{};




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
