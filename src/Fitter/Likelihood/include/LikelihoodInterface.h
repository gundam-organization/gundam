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
#include "TDirectory.h"

#include <string>
#include <vector>


/*
 The LikelihoodInterface job is to evaluate the Likelihood component of a given pair of data/MC.
 It contains a few buffers that are used externally to monitor the fit status.
*/

class LikelihoodInterface : public JsonBaseClass {

  struct Buffer;

protected:
  // called through public JsonBaseClass::readConfig() and JsonBaseClass::initialize()
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  // const getters
  [[nodiscard]] int getNbFitParameters() const { return _nbFitParameters_; }
  [[nodiscard]] int getNbFreePars() const {return _nbFreePars_; }
  [[nodiscard]] int getNbFitBins() const {return _nbFitBins_; }
  [[nodiscard]] const Buffer& getBuffer() const { return _buffer_; }
  [[nodiscard]] const Propagator& getPropagator() const { return _propagator_; }
  const JointProbability::JointProbability* getJointProbabilityPtr() const { return _jointProbabilityPtr_.get(); }

  // mutable getters
  Buffer& getBuffer() { return _buffer_; }
  Propagator& getPropagator(){ return _propagator_; }

  // core
  void propagateAndEvalLikelihood();
  void scanParameter(Parameter* parPtr_, TDirectory* saveDir_);
  [[nodiscard]] int getNbDof() const { return _nbFitBins_ - _nbFreePars_; }
  double evalLikelihood() const;
  double evalStatLikelihood() const;
  double evalPenaltyLikelihood() const;
  [[nodiscard]] double evalStatLikelihood(const Sample& sample_) const;
  [[nodiscard]] double evalPenaltyLikelihood(const ParameterSet& parSet_) const;
  [[nodiscard]] std::string getSummary() const;


  // mutable core
  void writeChi2History();
  void saveGradientSteps();

  // TODO: to relocate
  double evalFitValid(const double* parArray_);
  void setParameterValidity(const std::string& validity);
  [[nodiscard]] bool hasValidParameterValues() const;

private:
  // internals
  int _nbFitBins_{0};
  int _nbFreePars_{0};
  int _nbFitParameters_{0};

  Propagator _propagator_{};

  std::shared_ptr<JointProbability::JointProbability> _jointProbabilityPtr_{nullptr};

  struct Buffer{
    double totalLikelihood{0};

    double statLikelihood{0};
    double penaltyLikelihood{0};

    void updateTotal(){ totalLikelihood = statLikelihood + penaltyLikelihood; }
    [[nodiscard]] bool isValid() const { return not ( std::isnan(totalLikelihood) or std::isinf(totalLikelihood) ); }
  };
  mutable Buffer _buffer_{};


  // TODO: relocate
  /// A set of flags used by the evalFitValid method to determine the function
  /// validity.  The flaggs are:
  /// "1" -- require valid parameters
  /// "2" -- require in the mirrored range
  /// "4" -- require in the physical range
  int _validFlags_{7}; // TODO: Use enum instead

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
