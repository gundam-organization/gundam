//
// Created by Clark McGrew 10/01/23
//

#ifndef GUNDAM_LIKELIHOOD_INTERFACE_H
#define GUNDAM_LIKELIHOOD_INTERFACE_H

#include "JointProbability.h"
#include "ParameterSet.h"
#include "Propagator.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Time.h"

#include "Math/Functor.h"

#include <string>
#include <vector>


/*
 The LikelihoodInterface job is to evaluate the Likelihood component of a given pair of data/MC.
 It contains a few buffers that are used externally to monitor the fit status.
*/

class LikelihoodInterface : public JsonBaseClass {

  struct Monitor;
  struct Buffer;

protected:
  // called through public JsonBaseClass::readConfig() and JsonBaseClass::initialize()
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  LikelihoodInterface() = default;
  ~LikelihoodInterface() override = default;

  // setters
  void setUseNormalizedFitSpace(bool useNormalizedFitSpace_) { _useNormalizedFitSpace_ = useNormalizedFitSpace_; }
  void setMonitorGradientDescent(bool monitorGradientDescent_){ _monitorGradientDescent_ = monitorGradientDescent_; } // TODO: relocate?
  void setParameterValidity(const std::string& validity);

  // const getters
  [[nodiscard]] bool hasValidParameterValues() const;
  [[nodiscard]] bool isMonitorGradientDescent() const { return _monitorGradientDescent_; }
  [[nodiscard]] bool getUseNormalizedFitSpace() const { return _useNormalizedFitSpace_; }
  [[nodiscard]] int getNbFitCalls() const {return _nbLlhEvals_; }
  [[nodiscard]] int getNbFitParameters() const { return _nbFitParameters_; }
  [[nodiscard]] int getNbFreePars() const {return _nbFreePars_; }
  [[nodiscard]] int getNbFitBins() const {return _nbFitBins_; }
  const Propagator& getPropagator() const { return _propagator_; }
  const Buffer& getBuffer() const { return _buffer_; }

  // mutable getters
  Propagator& getPropagator(){ return _propagator_; }
  Monitor& getFitMonitor(){ return _monitor_; }
  std::vector<Parameter *> &getMinimizerFitParameterPtr(){ return _minimizerFitParameterPtr_; }
  ROOT::Math::Functor* getFunctor() { return _functor_.get(); }
  ROOT::Math::Functor* getValidFunctor() { return _validFunctor_.get(); }
  GenericToolbox::VariablesMonitor& getConvergenceMonitor(){ return _convergenceMonitor_; } // TODO: relocate?

  double evalLikelihood( const double* parArray_);
  double evalFitValid(const double* parArray_);

  double evalLikelihood();
  double evalStatLikelihood();

  void writeChi2History();
  void saveGradientSteps();

private:
  bool _throwOnBadLlh_{false};
  bool _useNormalizedFitSpace_{true};

  int _nbFitBins_{0};
  int _nbLlhEvals_{0};
  int _nbFreePars_{0};
  int _nbFitParameters_{0};

  Propagator _propagator_{};
  std::vector<Parameter*> _minimizerFitParameterPtr_{};
  std::shared_ptr<JointProbability::JointProbability> _jointProbabilityPtr_{nullptr};

  std::unique_ptr<ROOT::Math::Functor> _functor_{nullptr};
  std::unique_ptr<ROOT::Math::Functor> _validFunctor_{nullptr};
  std::unique_ptr<TTree> _chi2HistoryTree_{nullptr};

  /// A set of flags used by the evalFitValid method to determine the function
  /// validity.  The flaggs are:
  /// "1" -- require valid parameters
  /// "2" -- require in the mirrored range
  /// "4" -- require in the physical range
  int _validFlags_{7}; // TODO: Use enum instead

  struct Buffer{
    double totalLikelihood{std::nan("unset")};
    double statLikelihood{std::nan("unset")};
    double penaltyLikelihood{std::nan("unset")};
    double regulariseLikelihood{std::nan("unset")};

    std::vector<double> statLikelihoodPerSample{};
    std::vector<double> penaltyLikelihoodPerParSet{};
  };
  Buffer _buffer_{};

  struct SpeedMonitor{
    GenericToolbox::Time::CycleTimer _evalFitAvgTimer_;
    GenericToolbox::Time::CycleTimer _outEvalFitAvgTimer_;
    GenericToolbox::Time::CycleTimer _itSpeed_;
    GenericToolbox::Time::CycleClock _itSpeedMon_{"it"};
  };
  SpeedMonitor _speedMonitor_{};

  // Output monitors!
  GenericToolbox::VariablesMonitor _convergenceMonitor_;


  struct Monitor{
    bool isEnabled{false};
    bool showParameters{false};
    int maxNbParametersPerLine{15};
  };
  Monitor _monitor_{};
  /// Parameters to control how the monitor behaves.

  // TODO: relocate too?
  bool _monitorGradientDescent_{false};
  int _lastGradientFall_{-2};
  struct GradientStepPoint {
    JsonType parState;
    double llh;
  };
  std::vector<GradientStepPoint> _gradientMonitor_{};

};

#endif //  GUNDAM_LIKELIHOOD_INTERFACE_H

// An MIT Style License

// Copyright (c) 2022 Clark McGrew

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
