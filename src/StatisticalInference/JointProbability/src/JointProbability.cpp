//
// Created by Nadrino on 22/02/2024.
//

#include "JointProbability.h"
#include "PoissonLogLikelihood.h"
#include "PluginJointProbability.h"
#include "LeastSquares.h"
#include "ChiSquared.h"
#include "BarlowBeeston.h"
#include "BarlowBeestonBanff2020.h"
#include "BarlowBeestonBanff2022.h"
#include "BarlowBeestonBanff2022Sfgd.h"

#include "Logger.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[JointProbability]");
});

namespace JointProbability{

  JointProbabilityBase* makeJointProbability(const std::string& type_){

    std::string enumTypeStr{type_};

    // TODO: possible backward compatibility
//    if(  ){ }


    auto jType{JointProbabilityType::toEnum( type_, true )};

    if( jType == JointProbabilityType::EnumOverflow  ){
      LogThrow( "Unknown JointProbabilityType: " << type_ );
    }

    return makeJointProbability( jType );
  }
  JointProbabilityBase* makeJointProbability(JointProbabilityType type_){
    std::unique_ptr<JointProbabilityBase> out{nullptr};

    switch( type_.value ){
      case JointProbabilityType::Plugin:
        out = std::make_unique<PluginJointProbability>();
        break;
      case JointProbabilityType::Chi2:
        out = std::make_unique<ChiSquared>();
        break;
      case JointProbabilityType::PoissonLLH:
        out = std::make_unique<PoissonLogLikelihood>();
        break;
      case JointProbabilityType::LeastSquares:
        out = std::make_unique<LeastSquares>();
        break;
      case JointProbabilityType::BarlowLLH:
        out = std::make_unique<BarlowBeeston>();
        break;
      case JointProbabilityType::BarlowLLH_BANFF_OA2020:
        out = std::make_unique<BarlowBeestonBanff2020>();
        break;
      case JointProbabilityType::BarlowLLH_BANFF_OA2021:
        out = std::make_unique<BarlowBeestonBanff2022>();
        break;
      case JointProbabilityType::BarlowLLH_BANFF_OA2021_SFGD:
        out = std::make_unique<BarlowBeestonBanff2022Sfgd>();
        break;
      default:
        LogThrow( "Unknown JointProbabilityType: " << type_.toString() );
    }

    return out.release();
  }

}