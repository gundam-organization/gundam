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
      GenericToolbox::Json::fetchValue(_config_, "propagatorConfig", _propagator_.getConfig())
  );

}
void LikelihoodInterface::initializeImpl() {
  LogWarning << "Initializing LikelihoodInterface..." << std::endl;

  _propagator_.initialize();

  LogWarning << "Fetching the effective number of fit parameters..." << std::endl;
  _nbFreePars_ = 0;
  _nbParameters_ = 0;
  for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
    _nbParameters_ += int( parSet.getNbParameters() );
    for( auto& par : parSet.getEffectiveParameterList() ){
      _nbParameters_++;
      if( par.isEnabled() and not par.isFixed() ) {
        if( par.isFree() ){ _nbFreePars_++; }
      }
    }
  }

  _nbSampleBins_ = 0;
  for( auto& sample : _propagator_.getSampleSet().getSampleList() ){
    _nbSampleBins_ += int(sample.getBinning().getBinList().size() );
  }

  LogInfo << "LikelihoodInterface initialized." << std::endl;
}

void LikelihoodInterface::saveGradientSteps(){

  if( GundamGlobals::isLightOutputMode() ){
    LogAlert << "Skipping saveGradientSteps as light output mode is fired." << std::endl;
    return;
  }

  LogInfo << "Saving " << _gradientMonitor_.size() << " gradient steps..." << std::endl;

  // make sure the parameter states get restored as we leave
  auto currentParState = _propagator_.getParametersManager().exportParameterInjectorConfig();
  GenericToolbox::ScopedGuard g{
    [&](){
      ParametersManager::muteLogger();
      ParameterSet::muteLogger();
      ParameterScanner::muteLogger();
    },
    [&](){
      _propagator_.getParametersManager().injectParameterValues( currentParState );
      ParametersManager::unmuteLogger();
      ParameterSet::unmuteLogger();
      ParameterScanner::unmuteLogger();
    }
  };

  // load starting point
  auto lastParStep{_owner_->getPreFitParState()};

  std::vector<GraphEntry> globalGraphList;
  for(size_t iGradStep = 0 ; iGradStep < _gradientMonitor_.size() ; iGradStep++ ){
    GenericToolbox::displayProgressBar(iGradStep, _gradientMonitor_.size(), LogInfo.getPrefixString() + "Saving gradient steps...");

    // why do we need to remute the logger at each loop??
    ParameterSet::muteLogger(); Propagator::muteLogger(); ParametersManager::muteLogger();
    _propagator_.getParametersManager().injectParameterValues(_gradientMonitor_[iGradStep].parState );
    _propagator_.updateLlhCache();

    if( not GundamGlobals::isLightOutputMode() ) {
      auto outDir = GenericToolbox::mkdirTFile(_owner_->getSaveDir(), Form("fit/gradient/step_%i", int(iGradStep)));
      GenericToolbox::writeInTFile(outDir, TNamed("parState", GenericToolbox::Json::toReadableString(_gradientMonitor_[iGradStep].parState).c_str()));
      GenericToolbox::writeInTFile(outDir, TNamed("llhState", _propagator_.getLlhBufferSummary().c_str()));
    }

    // line scan from previous point
    _propagator_.getParameterScanner().scanSegment( nullptr, _gradientMonitor_[iGradStep].parState, lastParStep, 8 );
    lastParStep = _gradientMonitor_[iGradStep].parState;

    if( globalGraphList.empty() ){
      // copy
      globalGraphList = _propagator_.getParameterScanner().getGraphEntriesBuf();
    }
    else{
      // current
      auto& grEntries = _propagator_.getParameterScanner().getGraphEntriesBuf();

      for( size_t iEntry = 0 ; iEntry < globalGraphList.size() ; iEntry++ ){
        for(int iPt = 0 ; iPt < grEntries[iEntry].graph.GetN() ; iPt++ ){
          globalGraphList[iEntry].graph.AddPoint( grEntries[iEntry].graph.GetX()[iPt], grEntries[iEntry].graph.GetY()[iPt] );
        }
      }

    }
  }

  if( not globalGraphList.empty() ){
    auto outDir = GenericToolbox::mkdirTFile(_owner_->getSaveDir(), "fit/gradient/global");
    for( auto& gEntry : globalGraphList ){
      gEntry.scanDataPtr->title = "Minimizer path to minimum";
      ParameterScanner::writeGraphEntry(gEntry, outDir);
    }
    GenericToolbox::triggerTFileWrite(outDir);

    outDir = GenericToolbox::mkdirTFile(_owner_->getSaveDir(), "fit/gradient/globalRelative");
    for( auto& gEntry : globalGraphList ){
      if( gEntry.graph.GetN() == 0 ){ continue; }

      double minY{gEntry.graph.GetY()[gEntry.graph.GetN()-1]};
      double maxY{gEntry.graph.GetY()[0]};
      double delta{1E-6*std::abs( maxY - minY )};
      // allow log scale
      minY += delta;

      for( int iPt = 0 ; iPt < gEntry.graph.GetN() ; iPt++ ){
        gEntry.graph.GetY()[iPt] -= minY;
      }
      gEntry.scanDataPtr->title = "Minimizer path to minimum (difference)";
      ParameterScanner::writeGraphEntry(gEntry, outDir);
    }
    GenericToolbox::triggerTFileWrite(outDir);
  }

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

double LikelihoodInterface::evalFitValid(const double* parArray_) {
  double value = this->evalLikelihood(parArray_);
  if (hasValidParameterValues()) return value;
  /// A "Really Big Number".  This is nominally just infinity, but is done as
  /// a defined constant to make the code easier to understand.  This needs to
  /// be an appropriate value to safely represent an impossible chi-squared
  /// value "representing" -log(0.0)/2 and should should be larger than 5E+30.
  const double RBN = std::numeric_limits<double>::infinity();
  return RBN;
}
void LikelihoodInterface::setParameterValidity(const std::string& validity) {
  /// Define the type of validity that needs to be required by
  /// hasValidParameterValues.  This accepts a string with the possible values
  /// being:
  ///
  ///  "range" (default) -- Between the parameter minimum and maximum values.
  ///  "norange"         -- Do not require parameters in the valid range
  ///  "mirror"          -- Between the mirrored values (if parameter has
  ///                       mirroring).
  ///  "nomirror"        -- Do not require parameters in the mirrored range
  ///  "physical"        -- Only physically meaningful values.
  ///  "nophysical"      -- Do not require parameters in the physical range.
  ///
  /// Example: setParameterValidity("range,mirror,physical")

  LogWarning << "Set parameter validity to " << validity << std::endl;

  if      ( GenericToolbox::hasSubStr(validity, "noran") ){ _validFlags_ &= ~0b0001; }
  else if ( GenericToolbox::hasSubStr(validity, "ran")   ){ _validFlags_ |= 0b0001; }

  if (validity.find("nomir") != std::string::npos) _validFlags_ &= ~0b0010;
  else if (validity.find("mir") != std::string::npos) _validFlags_ |= 0b0010;

  if (validity.find("nophy") != std::string::npos) _validFlags_ &= ~0b0100;
  else if (validity.find("phy") != std::string::npos) _validFlags_ |= 0b0100;

  LogWarning << "Set parameter validity to " << validity << " (" << _validFlags_ << ")" << std::endl;
}
bool LikelihoodInterface::hasValidParameterValues() const {
  int invalid = 0;
  for (auto& parSet: _propagator_.getParametersManager().getParameterSetsList()) {
    for (auto& par : parSet.getParameterList()) {
      if ( (_validFlags_ & 0b0001) != 0
          and std::isfinite(par.getMinValue())
          and par.getParameterValue() < par.getMinValue()) [[unlikely]] {
        ++invalid;
      }
      if ((_validFlags_ & 0b0001) != 0
          and std::isfinite(par.getMaxValue())
          and par.getParameterValue() > par.getMaxValue()) [[unlikely]] {
        ++invalid;
      }
      if ((_validFlags_ & 0b0010) != 0
          and std::isfinite(par.getMinMirror())
          and par.getParameterValue() < par.getMinMirror()) [[unlikely]] {
        ++invalid;
      }
      if ((_validFlags_ & 0b0010) != 0
          and std::isfinite(par.getMaxMirror())
          and par.getParameterValue() > par.getMaxMirror()) [[unlikely]] {
        ++invalid;
      }
      if ((_validFlags_ & 0b0100) != 0
          and std::isfinite(par.getMinPhysical())
          and par.getParameterValue() < par.getMinPhysical()) [[unlikely]] {
        ++invalid;
      }
      if ((_validFlags_ & 0b0100) != 0
          and std::isfinite(par.getMaxPhysical())
          and par.getParameterValue() > par.getMaxPhysical()) [[unlikely]] {
        ++invalid;
      }

    }
  }
  return (invalid == 0);
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
