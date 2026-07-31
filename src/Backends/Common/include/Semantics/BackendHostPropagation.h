#ifndef GUNDAM_BACKEND_HOST_PROPAGATION_H
#define GUNDAM_BACKEND_HOST_PROPAGATION_H

#include "EngineView.h"
#include "ParameterSnapshot.h"
#include "Semantics/BackendDialSemantics.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace Backends::Semantics {

  inline void calculateEventWeights(std::vector<double>& eventWeights_,
                                    const BackendPropagationView& propagation_,
                                    const ParameterSnapshot& parameters_) {
    eventWeights_.resize(propagation_.events.size());
    for( const auto& event : propagation_.events ){
      eventWeights_[event.resultIndex] = evalEventWeight(propagation_, event, parameters_);
    }
  }

  inline void calculateHistogramsFromEventWeights(std::vector<double>& histSums_,
                                                  std::vector<double>& histSumSquares_,
                                                  const BackendPropagationView& propagation_,
                                                  const std::vector<double>& eventWeights_) {
    histSums_.assign(propagation_.totalBins, 0.);
    histSumSquares_.assign(propagation_.totalBins, 0.);

    for( const auto& event : propagation_.events ){
      if( event.globalBinIndex < 0 ){ continue; }
      const double weight = eventWeights_.at(event.resultIndex);
      histSums_[event.globalBinIndex] += weight;
      histSumSquares_[event.globalBinIndex] += weight * weight;
    }
  }

  inline void calculateHistograms(std::vector<double>& histSums_,
                                  std::vector<double>& histSumSquares_,
                                  const BackendPropagationView& propagation_,
                                  const ParameterSnapshot& parameters_) {
    std::vector<double> eventWeights{};
    calculateEventWeights(eventWeights, propagation_, parameters_);
    calculateHistogramsFromEventWeights(histSums_, histSumSquares_, propagation_, eventWeights);
  }

  inline double calculateLikelihood(const BackendLikelihoodView& likelihood_,
                                    const std::vector<double>& histSums_,
                                    const std::vector<double>& histSumSquares_) {
    double likelihoodValue{0};

    for( const auto& sample : likelihood_.samples ){
      for( int iBin = 0 ; iBin < int(sample.dataSums.size()) ; iBin++ ){
        if( iBin < int(sample.ignoredBins.size()) and sample.ignoredBins[iBin] ){ continue; }
        const int globalBin = sample.binOffset + iBin;
        const double pred = histSums_.at(globalBin);
        const double predErr = std::sqrt(histSumSquares_.at(globalBin));
        likelihoodValue += sample.evalBin(sample.dataSums[iBin], pred, predErr, iBin);
      }
    }

    return likelihoodValue;
  }

}

#endif // GUNDAM_BACKEND_HOST_PROPAGATION_H
