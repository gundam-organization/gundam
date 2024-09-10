//
// Created by Clark McGrew 10/01/23
//

#ifndef GUNDAM_LIKELIHOOD_INTERFACE_H
#define GUNDAM_LIKELIHOOD_INTERFACE_H

#include "ParameterSet.h"
#include "JointProbability.h"
#include "Propagator.h"
#include "DataSetManager.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Time.h"

#include <memory>
#include <string>


/// Evaluate the likelihood between data and MC.  The calculation is buffered
/// and and updated by the propagateAndEvalLikelihood() method.  The
/// likelihood value for the last calculation is accessible through
/// getLastLikelihood().
///
/// The "likelihood" for GUNDAM is based on the comparision between a "data"
/// and "expected" (i.e. MC) histogram.  The bin-by-bin comparisions are done
/// using the JointProbability class and are based on one of several LLH (Log
/// Likelihood) calculations (e.g. Barlow-Beeston, Icecube, Poissonian).
/// While the value is proportional to the LLH, it is more closely related to
/// the chi-square since we use -2*LLH.
class LikelihoodInterface : public JsonBaseClass  {

public:

  struct Buffer{
    double totalLikelihood{0};

    double statLikelihood{0};
    double penaltyLikelihood{0};

    void updateTotal(){ totalLikelihood = statLikelihood + penaltyLikelihood; }
    [[nodiscard]] bool isValid() const { return not ( std::isnan(totalLikelihood) or std::isinf(totalLikelihood) ); }
  };

protected:
  // called through public JsonBaseClass::readConfig() and JsonBaseClass::initialize()
  void readConfigImpl() override;
  void initializeImpl() override;

public:

  // const getters
  [[nodiscard]] int getNbParameters() const {return _nbParameters_; }
  [[nodiscard]] int getNbSampleBins() const {return _nbSampleBins_; }
  [[nodiscard]] double getLastLikelihood() const { return _buffer_.totalLikelihood; }
  [[nodiscard]] double getLastStatLikelihood() const { return _buffer_.statLikelihood; }
  [[nodiscard]] double getLastPenaltyLikelihood() const { return _buffer_.penaltyLikelihood; }
  [[nodiscard]] const DataSetManager& getDataSetManager() const { return _dataSetManager_; }
  const JointProbability::JointProbabilityBase* getJointProbabilityPtr() const { return _jointProbabilityPtr_.get(); }

  // mutable getters
  Buffer& getBuffer() { return _buffer_; }
  DataSetManager& getDataSetManager(){ return _dataSetManager_; }

  // mutable core
  void propagateAndEvalLikelihood();

  // core
  double evalLikelihood() const;
  double evalStatLikelihood() const;
  double evalPenaltyLikelihood() const;
  [[nodiscard]] double evalStatLikelihood(const Sample& sample_) const;
  [[nodiscard]] double evalPenaltyLikelihood(const ParameterSet& parSet_) const;
  [[nodiscard]] std::string getSummary() const;

  // dev deprecated
  [[deprecated("use getDataSetManager().getPropagator()")]] [[nodiscard]] const Propagator& getPropagator() const { return _dataSetManager_.getPropagator(); }
  [[deprecated("use getDataSetManager().getPropagator()")]] Propagator& getPropagator(){ return _dataSetManager_.getPropagator(); }

private:
  // internals
  int _nbParameters_{0};
  int _nbSampleBins_{0};

  /// Definition of data sets to use for filling the Propagator
  DataSetManager _dataSetManager_{};

  /// Statistical likelihood
  std::shared_ptr<JointProbability::JointProbabilityBase> _jointProbabilityPtr_{nullptr};

  mutable Buffer _buffer_{};
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
// End:
