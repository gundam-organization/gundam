//
// Created by Nadrino on 22/02/2024.
//

#ifndef GUNDAM_JOINT_PROBABILITY_H
#define GUNDAM_JOINT_PROBABILITY_H

#include "JointProbabilityBase.h"

/*
 * This is the factory for the joint probability class
 * */


// only valid types (non-pure virtual)
#define ENUM_NAME JointProbabilityType
#define ENUM_FIELDS \
  ENUM_FIELD( PoissonLLH, 0 ) \
  ENUM_FIELD( LeastSquares ) \
  ENUM_FIELD( BarlowLLH ) \
  ENUM_FIELD( BarlowLLH_BANFF_OA2020 ) \
  ENUM_FIELD( BarlowLLH_BANFF_OA2021 ) \
  ENUM_FIELD( BarlowLLH_BANFF_OA2021_SFGD ) \
  ENUM_FIELD( Chi2 ) \
  ENUM_FIELD( Plugin )
#include "GenericToolbox.MakeEnum.h"

namespace JointProbability{

  JointProbabilityBase* makeJointProbability(const std::string& type_);
  JointProbabilityBase* makeJointProbability(JointProbabilityType type_);

}




#endif //GUNDAM_JOINT_PROBABILITY_H
