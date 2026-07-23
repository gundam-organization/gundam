//
// Created by Nadrino on 23/06/2022.
//

#ifndef GUNDAM_JOINT_PROBABILITY_BASE_H
#define GUNDAM_JOINT_PROBABILITY_BASE_H

#include "SamplePair.h"

#include <cmath>
#include <limits>
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

    [[nodiscard]] virtual double evalPredictionGradient(double data_, double pred_, double err_, int bin_) const {
      return evalGradientByFiniteDifference(pred_, [&](double predVal_){ return eval(data_, predVal_, err_, bin_); });
    }

    [[nodiscard]] virtual double evalSqrtSumSqWeightsGradient(double data_, double pred_, double err_, int bin_) const {
      return evalGradientByFiniteDifference(err_, [&](double errVal_){ return eval(data_, pred_, errVal_, bin_); });
    }

    // Likelihood for a single bin llh from histogram and bin index.  Override this if precalculation is needed.
    [[nodiscard]] virtual double eval( const SamplePair& samplePair_, int bin_ ) const {
      const double dataVal{samplePair_.data->getHistogram().getBinContentList()[bin_].sumWeights};
      const double predVal{samplePair_.model->getHistogram().getBinContentList()[bin_].sumWeights};
      const double predErr{samplePair_.model->getHistogram().getBinContentList()[bin_].sqrtSumSqWeights};
      return eval(dataVal, predVal, predErr, bin_);
    }

    [[nodiscard]] virtual double evalPredictionGradient( const SamplePair& samplePair_, int bin_ ) const {
      auto& binContent = samplePair_.model->getHistogram().getBinContentList()[bin_];
      return evalGradientByFiniteDifference(binContent.sumWeights, [&](double){ return eval(samplePair_, bin_); });
    }

    [[nodiscard]] virtual double evalSqrtSumSqWeightsGradient( const SamplePair& samplePair_, int bin_ ) const {
      auto& binContent = samplePair_.model->getHistogram().getBinContentList()[bin_];
      return evalGradientByFiniteDifference(binContent.sqrtSumSqWeights, [&](double){ return eval(samplePair_, bin_); });
    }

    // classic binned llh. Could be overriden to introduce correlations for instance.
    [[nodiscard]] virtual double eval( const SamplePair& samplePair_ ) const{
      double out{0};
      const int nBins{int(samplePair_.model->getHistogram().getNbBins())};
      for( int iBin = 0; iBin < nBins; iBin++ ){ out += this->eval(samplePair_, iBin); }
      return out;
    }

  protected:
    template<typename EvalFct>
    [[nodiscard]] static double evalGradientByFiniteDifference(double& value_, const EvalFct& evalFct_){
      const double nominal{value_};
      const double step{std::sqrt(std::numeric_limits<double>::epsilon()) * (std::abs(nominal) + 1.)};

      double gradient{0};
      if( nominal - step > 0 ){
        value_ = nominal + step;
        const double up{evalFct_(value_)};
        value_ = nominal - step;
        const double down{evalFct_(value_)};
        gradient = (up - down) / (2. * step);
      }
      else{
        value_ = nominal + step;
        const double up{evalFct_(value_)};
        value_ = nominal;
        const double center{evalFct_(value_)};
        gradient = (up - center) / step;
      }

      value_ = nominal;
      if( not std::isfinite(gradient) ){ return 0.; }
      return gradient;
    }

  };
}


#endif // GUNDAM_JOINT_PROBABILITY_BASE_H
