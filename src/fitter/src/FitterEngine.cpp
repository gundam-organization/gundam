//
// Created by Adrien BLANCHET on 11/06/2021.
//

#include <Math/Factory.h>
#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolboxRootExt.h"

#include "JsonUtils.h"
#include "FitterEngine.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[FitterEngine]");
})

FitterEngine::FitterEngine() { this->reset(); }
FitterEngine::~FitterEngine() { this->reset(); }

void FitterEngine::reset() {
  _saveDir_ = nullptr;
  _config_.clear();

  _propagator_.reset();
  _minimizer_.reset();
  _functor_.reset();
  _nb_fit_parameters_ = 0;
}

void FitterEngine::setSaveDir(TDirectory *saveDir) {
  _saveDir_ = saveDir;
}
void FitterEngine::setConfig(const json &config) {
  _config_ = config;
}

void FitterEngine::initialize() {

  if( _config_.empty() ){
    LogError << "Config is empty." << std::endl;
    throw std::runtime_error("config not set.");
  }

  initializePropagator();
//  initializeMinimizer();

}

void FitterEngine::generateSamplePlots(const std::string& saveSubDirPath_){

  LogInfo << __METHOD_NAME__ << std::endl;

  _propagator_.propagateParametersOnSamples();
  _propagator_.fillSampleHistograms();
  _propagator_.getPlotGenerator().generateSamplePlots(
    GenericToolbox::mkdirTFile(_saveDir_, saveSubDirPath_ )
    );

}
void FitterEngine::generateOneSigmaPlots(const std::string& saveSubDirPath_){

  _propagator_.propagateParametersOnSamples();
  _propagator_.fillSampleHistograms();
  _propagator_.getPlotGenerator().generateSamplePlots();

  _saveDir_->cd(); // to put this hist somewhere
  auto refHistList = _propagator_.getPlotGenerator().getHistHolderList(); // current buffer

  // +1 sigma
  for( auto& parSet : _propagator_.getParameterSetsList() ){
    for( auto& par : parSet.getParameterList() ){
      double currentParValue = par.getParameterValue();
      par.setParameterValue( currentParValue + par.getStdDevValue() );
      LogInfo << "+1 sigma on " << parSet.getName() + "/" + par.getTitle() << " -> " << par.getParameterValue() << std::endl;
      _propagator_.propagateParametersOnSamples();
      _propagator_.fillSampleHistograms();

      std::string savePath = saveSubDirPath_;
      if( not savePath.empty() ) savePath += "/";
      savePath += "oneSigma/" + parSet.getName() + "/" + par.getTitle();
      auto* saveDir = GenericToolbox::mkdirTFile(_saveDir_, savePath );
      saveDir->cd();

      _propagator_.getPlotGenerator().generateSamplePlots();

      auto oneSigmaHistList = _propagator_.getPlotGenerator().getHistHolderList();
      _propagator_.getPlotGenerator().generateComparisonPlots( oneSigmaHistList, refHistList, saveDir );
      par.setParameterValue( currentParValue );

      // Since those were not saved, delete manually
      for( auto& hist : oneSigmaHistList ){ delete hist.histPtr; }
      oneSigmaHistList.clear();
    }
  }

  _saveDir_->cd();

  // Since those were not saved, delete manually
  for( auto& refHist : refHistList ){ delete refHist.histPtr; }
  refHistList.clear();

}

double FitterEngine::evalFit(const double* par){
  return 0;
}


void FitterEngine::initializePropagator(){

  _propagator_.setParameterSetConfig(JsonUtils::fetchValue<nlohmann::json>(_config_, "fitParameterSets"));
  _propagator_.setSamplesConfig(JsonUtils::fetchValue<nlohmann::json>(_config_, "samples"));
  _propagator_.setSamplePlotGeneratorConfig(JsonUtils::fetchValue<nlohmann::json>(_config_, "samplePlotGenerator"));

  TFile* f = TFile::Open(JsonUtils::fetchValue<std::string>(_config_, "mc_file").c_str(), "READ");
  _propagator_.setDataTree( f->Get<TTree>("selectedEvents") );
  _propagator_.setMcFilePath(JsonUtils::fetchValue<std::string>(_config_, "mc_file"));

  if( _saveDir_ != nullptr ){
    _propagator_.setSaveDir(GenericToolbox::mkdirTFile(_saveDir_, "propagator"));
  }

  _propagator_.initialize();

  LogTrace << "Counting parameters" << std::endl;
  _nb_fit_parameters_ = 0;
  for( const auto& parSet : _propagator_.getParameterSetsList() ){
    _nb_fit_parameters_ += int(parSet.getNbParameters());
  }
  LogTrace << GET_VAR_NAME_VALUE(_nb_fit_parameters_) << std::endl;

}
void FitterEngine::initializeMinimizer(){

  auto minimizationConfig = JsonUtils::fetchValue<nlohmann::json>(_config_, "fitterSettings");
  minimizationConfig = JsonUtils::fetchValue<nlohmann::json>(minimizationConfig, "minimizerSettings");

  _minimizer_ = std::shared_ptr<ROOT::Math::Minimizer>(
    ROOT::Math::Factory::CreateMinimizer(
      JsonUtils::fetchValue<std::string>(minimizationConfig, "minimizer"),
      JsonUtils::fetchValue<std::string>(minimizationConfig, "algorithm")
    )
  );

  _functor_ = std::shared_ptr<ROOT::Math::Functor>(
    new ROOT::Math::Functor(
      this, &FitterEngine::evalFit, _nb_fit_parameters_
      )
  );

  _minimizer_->SetFunction(*_functor_);
  _minimizer_->SetStrategy(JsonUtils::fetchValue<int>(minimizationConfig, "strategy"));
  _minimizer_->SetPrintLevel(JsonUtils::fetchValue<int>(minimizationConfig, "print_level"));
  _minimizer_->SetTolerance(JsonUtils::fetchValue<double>(minimizationConfig, "tolerance"));
  _minimizer_->SetMaxIterations(JsonUtils::fetchValue<unsigned int>(minimizationConfig, "max_iter"));
  _minimizer_->SetMaxFunctionCalls(JsonUtils::fetchValue<unsigned int>(minimizationConfig, "max_fcn"));

  int iPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){

    for( auto& par : parSet.getParameterList()  ){
      iPar++;

      _minimizer_->SetVariable(
        iPar,
        parSet.getName() + "/" + par.getTitle(),
        par.getParameterValue(),
        0.01
      );
    } // par
  } // parSet

  LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
          << "Number of free parameters   : " << _minimizer_->NFree() << std::endl
          << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree()
          << std::endl;

}