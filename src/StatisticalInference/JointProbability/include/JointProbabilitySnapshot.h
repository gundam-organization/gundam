#ifndef GUNDAM_JOINT_PROBABILITY_SNAPSHOT_H
#define GUNDAM_JOINT_PROBABILITY_SNAPSHOT_H
#include "JointProbabilityBase.h"

namespace JointProbability {
  // Read-only export for executors. The validated classes remain unchanged.
  struct JointProbabilitySnapshot {
    enum class Kind { Unsupported, Poisson, BarlowBeeston2022, BarlowBeestonLiteGaussian, BarlowBeestonLitePoisson };
    Kind kind{Kind::Unsupported};
    bool usePoissonLikelihood{false};
    bool useNominalVariance{false};
    bool throwIfInfLlh{true};
    double fractionalErrorLimit{1.e150};
    double expectedValueMinimum{-1.};
    std::vector<double> nominalVariances{};
  };
  JointProbabilitySnapshot snapshotJointProbability(const JointProbabilityBase& probability_, const SamplePair& sample_);
}
#endif
