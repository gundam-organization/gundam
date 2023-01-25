//
// Created by Clark McGrew 10/01/23
//

#ifndef GUNDAM_LikelihoodInterface_h
#define GUNDAM_LikelihoodInterface_h

#include "FitParameterSet.h"

#include "GenericToolbox.VariablesMonitor.h"
#include "GenericToolbox.CycleTimer.h"

#include "Math/Functor.h"

class FitterEngine;

/// Wrap the calculation of the likelihood using the propagator into a single
/// place.  This provides an abstract interface that can is provided to the
/// FitterEngine and can be accessed by any MinimizerInterface.  The main
/// access is through the evalFit method which takes an array of floating
/// point values and returns the likelihood.  The meaning of the parameters is
/// defined by the vector of pointers to FitParameter returned by
/// getMinimizerFitParameterPtr.
class LikelihoodInterface {
public:
  LikelihoodInterface(FitterEngine* owner_);
  virtual ~LikelihoodInterface();
  void setOwner(FitterEngine* owner_) {_owner_ = owner_;}

  /// Initialize the likelihood interface.  Must be called after all of the
  /// paramters are set, but before the first function evaluation.
  void initialize();

  /// Calculate the likelihood based on an array of parameters.  Used
  /// to create a functor that can be used by MINIUT or TSimpleMCMC.
  double evalFit(const double* parArray_);

  /// A pointer to a ROOT functor that calls the evalFit method.  The object
  /// referenced by the functor can be handed directly to Minuit.
  ROOT::Math::Functor* evalFitFunctor() {return _functor_.get();}

  /// A vector of the parameters being used in the fit.  This provides
  /// the correspondence between an array of doubles (param[]) and the
  /// pointers to the parameters.
  std::vector<FitParameter *> &getMinimizerFitParameterPtr()
    { return _minimizerFitParameterPtr_; }

  /// Set whether a normalized fit space should be used.  This controls how
  /// the fit parameter array is copied into the propagator parameter
  /// structures.
  void setUseNormalizedFitSpace(bool v) {_useNormalizedFitSpace_ = v;}
  bool getUseNormalizedFitSpace() const {return _useNormalizedFitSpace_;}

  /// Set the minimizer type and algorithm.
  void setMinimizerInfo(const std::string& type, const std::string& algo) {
    _minimizerType_ = type;
    _minimizerAlgo_ = algo;
  }

  /// Set the target EDM for this fitter.  This is only informational in the
  /// LikelihoodInterface, but it appears in the running summaries.  It needs
  /// to be set by the minimizer.
  void setTargetEDM(double v) { _targetEDM_=v; }

  /// Get the total number of parameters expected by the evalFit method.
  int getNbFitParameters() const {return _nbFitParameters_;}

  /// Get the number of parameters that are free in the likelihood
  int getNbFreePars() const {return _nbFreePars_; }

  /// Get the number of times the evalFit function was called.
  int getNbFitCalls() const {return _nbFitCalls_; }

  /// Return the number of sample bins being used in the fit.  This really
  /// belongs to the propagator!
  int getNbFitBins() const {return _nbFitBins_; }

  /// Get the convergence monitor.
  GenericToolbox::VariablesMonitor& getConvergenceMonitor()
    {return _convergenceMonitor_;}

  /// Enable and disable the monitor output.
  void enableFitMonitor(){ _enableFitMonitor_ = true; }
  void disableFitMonitor(){ _enableFitMonitor_ = false; }

  /// Set whether fit parameters should be shown in the monitor output.
  void setShowParametersOnFitMonitor(bool v)
    {_showParametersOnFitMonitor_=v;}

  /// Set the maximum number of parameters to show on a line when parameters
  /// are being show by the monitor.
  void setMaxNbParametersPerLineOnMonitor(int v)
    {_maxNbParametersPerLineOnMonitor_=v;}

  /// Save the chi2 history to the current output file.
  void saveChi2History();

private:
  /// True as soon as this has been initialized.
  bool _isInitialized_{false};

    /// The fitter engion that owns this likelihood.
  FitterEngine* _owner_{nullptr};

  /// A functor that can be called by Minuit or anybody else.  This wraps evalFit.
  std::unique_ptr<ROOT::Math::Functor> _functor_;

  /// A vector of pointers to fit parameters that defined the elements in the
  /// array of parameters passed to evalFit.
  std::vector<FitParameter*> _minimizerFitParameterPtr_{};

  /// The number of calls to the fitter function.
  int _nbFitCalls_{0};

  /// The total number of parameters in the likelihood.
  int _nbFitParameters_{0};

  /// The number of parameters in the likelihood.
  int _nbFreePars_{0};

  /// The number of sample bins being fitted.
  int _nbFitBins_{0};

  /// Flag for if a normalized fit space is being used.
  bool _useNormalizedFitSpace_{true};

  /// The type of minimizer being used (usually minuit2)
  std::string _minimizerType_{"not-set"};

  /// The algorithm being used (usually Migrad).
  std::string _minimizerAlgo_{"not-set"};

  /// The target EDM for the best fit point.  This will have different
  /// meanings for different "minimizers"
  double _targetEDM_{1E-6};

  /// A tree that save the history of the minimization.
  std::unique_ptr<TTree> _chi2HistoryTree_{nullptr};

  // Output monitors!
  GenericToolbox::VariablesMonitor _convergenceMonitor_;
  GenericToolbox::CycleTimer _evalFitAvgTimer_;
  GenericToolbox::CycleTimer _outEvalFitAvgTimer_;
  GenericToolbox::CycleTimer _itSpeed_;

  /// Parameters to control how the monitor behaves.
  bool _enableFitMonitor_{false};
  bool _showParametersOnFitMonitor_{false};
  int _maxNbParametersPerLineOnMonitor_{15};
};

#endif //  GUNDAM_LikelihoodInterface_h

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
