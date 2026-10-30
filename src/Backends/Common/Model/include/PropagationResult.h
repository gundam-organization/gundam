#ifndef GUNDAM_PROPAGATION_RESULT_H
#define GUNDAM_PROPAGATION_RESULT_H
#include "BackendTypes.h"
#include <vector>
namespace Backends {
  struct PropagationResult {
    PropagationToken token{};
    PropagationStatus status{};
    std::vector<double> eventWeights{};
    std::vector<double> histSums{};
    std::vector<double> histSumSquares{};
    std::vector<double> outsideSums{};
    std::vector<double> outsideSumSquares{};
    double likelihood{0};
  };
}
#endif
