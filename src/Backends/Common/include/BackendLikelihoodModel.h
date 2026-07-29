#ifndef GUNDAM_BACKEND_LIKELIHOOD_MODEL_H
#define GUNDAM_BACKEND_LIKELIHOOD_MODEL_H

#include <functional>
#include <vector>

namespace Backends {

  struct BackendLikelihoodSampleRef {
    int binOffset{0};
    std::vector<double> dataSums{};
    std::vector<bool> ignoredBins{};
    std::function<double(double, double, double, int)> evalBin{};
  };

  struct BackendLikelihoodModel {
    std::vector<BackendLikelihoodSampleRef> samples{};

    [[nodiscard]] bool empty() const { return samples.empty(); }
  };

}

#endif // GUNDAM_BACKEND_LIKELIHOOD_MODEL_H
