//
// Created by Clark McGrew 24/1/23
//

#include "LikelihoodInterface.h"
#include "GundamGlobals.h"
#include "GundamUtils.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Json.h"
#include "Logger.h"

#include <limits>


LoggerInit([]{
  Logger::setUserHeaderStr("[LikelihoodInterface]");
});


void LikelihoodInterface::readConfigImpl(){

  _propagator_.readConfig(
      GenericToolbox::Json::fetchValue( _config_, "propagatorConfig", _propagator_.getConfig() )
  );

  std::string llhMethod = "PoissonLLH";
  llhMethod = GenericToolbox::Json::fetchValue(_config_, "llhStatFunction", llhMethod);

  // new config structure
  auto configJointProbability = GenericToolbox::Json::fetchValue(_config_, {{"jointProbability"}, {"llhConfig"}}, JsonType());
  llhMethod = GenericToolbox::Json::fetchValue(configJointProbability, "type", llhMethod);

  LogInfo << "Using \"" << llhMethod << "\" LLH function." << std::endl;
  if     ( llhMethod == "Chi2" ){                    _jointProbabilityPtr_ = std::make_shared<JointProbability::Chi2>(); }
  else if( llhMethod == "PoissonLLH" ){              _jointProbabilityPtr_ = std::make_shared<JointProbability::PoissonLLH>(); }
  else if( llhMethod == "BarlowLLH" ) {              _jointProbabilityPtr_ = std::make_shared<JointProbability::BarlowLLH>(); }
  else if( llhMethod == "Plugin" ) {                 _jointProbabilityPtr_ = std::make_shared<JointProbability::JointProbabilityPlugin>(); }
  else if( llhMethod == "BarlowLLH_BANFF_OA2020" ) { _jointProbabilityPtr_ = std::make_shared<JointProbability::BarlowLLH_BANFF_OA2020>(); }
  else if( llhMethod == "BarlowLLH_BANFF_OA2021" ) { _jointProbabilityPtr_ = std::make_shared<JointProbability::BarlowLLH_BANFF_OA2021>(); }
  else if( llhMethod == "LeastSquares" ) { _jointProbabilityPtr_ = std::make_shared<JointProbability::LeastSquaresLLH>(); }
  else if( llhMethod == "BarlowLLH_BANFF_OA2021_SFGD" ) {  _jointProbabilityPtr_ = std::make_shared<JointProbability::BarlowLLH_BANFF_OA2021_SFGD>(); }
  else{ LogThrow("Unknown LLH Method: " << llhMethod); }

  _jointProbabilityPtr_->readConfig(configJointProbability);

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
    if( parSet_.useEigenDecomposition() ){
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

void LikelihoodInterface::propagateAndEvalLikelihood(){
  _propagator_.propagateParameters();
  this->evalLikelihood();
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
