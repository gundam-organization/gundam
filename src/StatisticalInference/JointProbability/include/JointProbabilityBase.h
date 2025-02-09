//
// Created by Nadrino on 23/06/2022.
//

#ifndef GUNDAM_JOINT_PROBABILITY_BASE_H
#define GUNDAM_JOINT_PROBABILITY_BASE_H

#include "SamplePair.h"

#include <string>

namespace JointProbability{

  class JointProbabilityBase : public JsonBaseClass {

  public:
    // simple rtti, makes the class purely virtual
    [[nodiscard]] virtual std::string getType() const = 0;

    // Joint probability for observing dataVal given a predicted value (with
    // an uncertainty).  The predErr is the uncertainty on the predicted
    // value, not the expected spread for the data around the prediction.
    // This is separated out so that the evaluation is amenable to unit
    // testing.
    virtual double eval(double data_, double pred_, double err_, int bin_) const = 0;

    // Likelihood for a single bin llh from histogram and bin index.  Override this if precalculation is needed.
    [[nodiscard]] virtual double eval( const SamplePair& samplePair_, int bin_ ) const {
      const double dataVal{samplePair_.data->getHistogram().getBinContentList()[bin_].sumWeights};
      const double predVal{samplePair_.model->getHistogram().getBinContentList()[bin_].sumWeights};
      const double predErr{samplePair_.model->getHistogram().getBinContentList()[bin_].sqrtSumSqWeights};
      return eval(dataVal, predVal, predErr, bin_);
    }

    // classic binned llh. Could be overriden to introduce correlations for instance.
    [[nodiscard]] virtual double eval( const SamplePair& samplePair_ ) const{
      double out{0};
      const int nBins{int(samplePair_.model->getHistogram().getNbBins())};
      for( int iBin = 0; iBin < nBins; iBin++ ){ out += this->eval(samplePair_, iBin); }
      return out;
    }

  };
}


#endif // GUNDAM_JOINT_PROBABILITY_BASE_H
