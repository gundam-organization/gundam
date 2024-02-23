//
// Created by Clark McGrew 24/1/23
//

#include "LikelihoodInterface.h"
#include "GundamGlobals.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Json.h"
#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[LikelihoodInterface]");
});


void LikelihoodInterface::readConfigImpl(){
  LogWarning << "Configuring LikelihoodInterface..." << std::endl;

  _propagator_.readConfig(
      GenericToolbox::Json::fetchValue( _config_, "propagatorConfig", _propagator_.getConfig() )
  );

  // placeholders
  JsonType configJointProbability;
  std::string jointProbabilityTypeStr;

  GenericToolbox::Json::deprecatedAction(_propagator_.getSampleSet().getConfig(), "llhStatFunction", [&]{
    LogAlert << R"("llhStatFunction" should now be set under "likelihoodInterfaceConfig/jointProbabilityConfig/type".)" << std::endl;
    jointProbabilityTypeStr = GenericToolbox::Json::fetchValue( _propagator_.getSampleSet().getConfig(), "llhStatFunction", jointProbabilityTypeStr );
  });
  GenericToolbox::Json::deprecatedAction(_propagator_.getSampleSet().getConfig(), "llhConfig", [&]{
    LogAlert << R"("llhConfig" should now be set under "likelihoodInterfaceConfig/jointProbabilityConfig".)" << std::endl;
    configJointProbability = GenericToolbox::Json::fetchValue( _propagator_.getSampleSet().getConfig(), "llhConfig", configJointProbability );
  });

  // new config structure
  configJointProbability = GenericToolbox::Json::fetchValue(_config_, "jointProbabilityConfig", configJointProbability);
  jointProbabilityTypeStr = GenericToolbox::Json::fetchValue(configJointProbability, "type", jointProbabilityTypeStr);

  LogInfo << "Using \"" << jointProbabilityTypeStr << "\" JointProbabilityType." << std::endl;
  _jointProbabilityPtr_ = std::shared_ptr<JointProbability::JointProbabilityBase>( JointProbability::makeJointProbability( jointProbabilityTypeStr ) );
  _jointProbabilityPtr_->readConfig( configJointProbability );

  LogWarning << "LikelihoodInterface configured." << std::endl;
}
void LikelihoodInterface::initializeImpl() {
  LogWarning << "Initializing LikelihoodInterface..." << std::endl;

  _propagator_.initialize();
  _jointProbabilityPtr_->initialize();

  LogWarning << "Fetching the effective number of fit parameters..." << std::endl;
  _nbParameters_ = 0;
  for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
    _nbParameters_ += int( parSet.getNbParameters() );
  }

  _nbSampleBins_ = 0;
  for( auto& sample : _propagator_.getSampleSet().getSampleList() ){
    _nbSampleBins_ += int(sample.getBinning().getBinList().size() );
  }

  LogInfo << "LikelihoodInterface initialized." << std::endl;
}

void LikelihoodInterface::propagateAndEvalLikelihood(){
  _propagator_.propagateParameters();
  this->evalLikelihood();
}

double LikelihoodInterface::evalLikelihood() const {
  this->evalStatLikelihood();
  this->evalPenaltyLikelihood();

  _buffer_.updateTotal();
  return _buffer_.totalLikelihood;
}
double LikelihoodInterface::evalStatLikelihood() const {
  _buffer_.statLikelihood = 0.;
  for( auto &sample: _propagator_.getSampleSet().getSampleList()){
    _buffer_.statLikelihood += this->evalStatLikelihood( sample );
  }
  return _buffer_.statLikelihood;
}
double LikelihoodInterface::evalPenaltyLikelihood() const {
  _buffer_.penaltyLikelihood = 0;
  for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
    _buffer_.penaltyLikelihood += this->evalPenaltyLikelihood( parSet );
  }
  return _buffer_.penaltyLikelihood;
}
double LikelihoodInterface::evalStatLikelihood(const Sample& sample_) const {
  return _jointProbabilityPtr_->eval( sample_ );
}
double LikelihoodInterface::evalPenaltyLikelihood(const ParameterSet& parSet_) const {
  if( not parSet_.isEnabled() ){ return 0; }

  double buffer = 0;

  if( parSet_.getPriorCovarianceMatrix() != nullptr ){
    if( parSet_.isEnableEigenDecomp() ){
      for( const auto& eigenPar : parSet_.getEigenParameterList() ){
        if( eigenPar.isFixed() ){ continue; }
        buffer += TMath::Sq( (eigenPar.getParameterValue() - eigenPar.getPriorValue()) / eigenPar.getStdDevValue() ) ;
      }
    }
    else{
      // make delta vector
      parSet_.updateDeltaVector();

      // compute penalty term with covariance
      buffer =
          (*parSet_.getDeltaVectorPtr())
          * ( (*parSet_.getInverseStrippedCovarianceMatrix()) * (*parSet_.getDeltaVectorPtr()) );
    }
  }

  return buffer;
}
[[nodiscard]] std::string LikelihoodInterface::getSummary() const {
  std::stringstream ss;

  this->evalLikelihood(); // make sure the buffer is up-to-date

  ss << "Total likelihood = " << _buffer_.totalLikelihood;
  ss << std::endl << "Stat likelihood = " << _buffer_.statLikelihood;
  ss << " = sum of: " << GenericToolbox::toString(
      _propagator_.getSampleSet().getSampleList(), [&]( const Sample& sample_){
        std::stringstream ssSub;
        ssSub << sample_.getName() << ": ";
        if( sample_.isEnabled() ){ ssSub << this->evalStatLikelihood( sample_ ); }
        else                     { ssSub << "disabled."; }
        return ssSub.str();
      }
  );
  ss << std::endl << "Penalty likelihood = " << _buffer_.penaltyLikelihood;
  ss << " = sum of: " << GenericToolbox::toString(
      _propagator_.getParametersManager().getParameterSetsList(), [&](const ParameterSet& parSet_){
        std::stringstream ssSub;
        ssSub << parSet_.getName() << ": ";
        if( parSet_.isEnabled() ){ ssSub << this->evalPenaltyLikelihood( parSet_ ); }
        else                     { ssSub << "disabled."; }
        return ssSub.str();
      }
  );
  return ss.str();
}

// An MIT Style License

// Copyright (c) 2022 GUNDAM DEVELOPERS

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
