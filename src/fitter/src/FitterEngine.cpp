//
// Created by Nadrino on 11/06/2021.
//

#include <Math/Factory.h>
#include "TGraph.h"

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

#include "JsonUtils.h"
#include "FitterEngine.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[FitterEngine]");
})

FitterEngine::FitterEngine() { this->reset(); }
FitterEngine::~FitterEngine() { this->reset(); }

void FitterEngine::reset() {
  _fitIsDone_ = false;
  _saveDir_ = nullptr;
  _config_.clear();
  _chi2History_.clear();

  _propagator_.reset();
  _minimizer_.reset();
  _functor_.reset();
  _nbFitParameters_ = 0;
  _nbParameters_ = 0;
  _nbFitCalls_ = 0;

  _convergenceMonitor_.reset();
}

void FitterEngine::setSaveDir(TDirectory *saveDir) {
  _saveDir_ = saveDir;
}
void FitterEngine::setConfig(const nlohmann::json &config_) {
  _config_ = config_;
  while( _config_.is_string() ){
    LogWarning << "Forwarding " << __CLASS_NAME__ << " config: \"" << _config_.get<std::string>() << "\"" << std::endl;
    _config_ = JsonUtils::readConfigFile(_config_.get<std::string>());
  }
}

void FitterEngine::initialize() {

  if( _config_.empty() ){
    LogError << "Config is empty." << std::endl;
    throw std::runtime_error("config not set.");
  }

  initializePropagator();
  initializeMinimizer();

  this->fixGhostParameters();

}

void FitterEngine::generateSamplePlots(const std::string& saveDir_){

  LogInfo << __METHOD_NAME__ << std::endl;

  _propagator_.preventRfPropagation(); // Making sure since we need the weight of each event
  _propagator_.propagateParametersOnSamples();
  _propagator_.getPlotGenerator().generateSamplePlots(
    GenericToolbox::mkdirTFile(_saveDir_, saveDir_ )
    );

}
void FitterEngine::generateOneSigmaPlots(const std::string& saveDir_){

  _propagator_.preventRfPropagation(); // Making sure since we need the weight of each event
  _propagator_.propagateParametersOnSamples();
  _propagator_.getPlotGenerator().generateSamplePlots();

  _saveDir_->cd(); // to put this hist somewhere
  auto refHistList = _propagator_.getPlotGenerator().getHistHolderList(); // current buffer

  // +1 sigma
  int iPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){

    for( auto& par : parSet.getParameterList() ){
        iPar++;

        std::string tag;
        if( par.isFixed() ){ tag += "_FIXED"; }

        double currentParValue = par.getParameterValue();
        par.setParameterValue( currentParValue + par.getStdDevValue() );
        LogInfo << "(" << iPar+1 << "/" << _nbParameters_ << ") +1 sigma on " << parSet.getName() + "/" + par.getTitle()
                << " -> " << par.getParameterValue() << std::endl;
        _propagator_.propagateParametersOnSamples();

        std::string savePath = saveDir_;
        if( not savePath.empty() ) savePath += "/";
        savePath += "oneSigma/" + parSet.getName() + "/" + par.getTitle() + tag;
        auto* saveDir = GenericToolbox::mkdirTFile(_saveDir_, savePath );
        saveDir->cd();

        _propagator_.getPlotGenerator().generateSamplePlots();

        auto oneSigmaHistList = _propagator_.getPlotGenerator().getHistHolderList();
        _propagator_.getPlotGenerator().generateComparisonPlots( oneSigmaHistList, refHistList, saveDir );
        par.setParameterValue( currentParValue );
        _propagator_.propagateParametersOnSamples();

        const auto& compHistList = _propagator_.getPlotGenerator().getComparisonHistHolderList();

        // Since those were not saved, delete manually
        for( auto& hist : oneSigmaHistList ){ delete hist.histPtr; }
        oneSigmaHistList.clear();
      }

    if( parSet.isUseEigenDecompInFit() ){
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        double currentParValue = parSet.getEigenParameter(iEigen);
        parSet.setEigenParameter(iEigen, currentParValue + parSet.getEigenSigma(iEigen));
        LogInfo << "(" << iEigen+1 << "/" << parSet.getNbEnabledEigenParameters() << ") +1 sigma on " << parSet.getName() + "/eigen_#" << iEigen
                << " -> " << parSet.getEigenSigma(iEigen) << std::endl;
        parSet.propagateEigenToOriginal();
        _propagator_.propagateParametersOnSamples();

        std::string savePath = saveDir_;
        if( not savePath.empty() ) savePath += "/";
        savePath += "oneSigma/" + parSet.getName() + "/eigen_#" + std::to_string(iEigen);
        auto* saveDir = GenericToolbox::mkdirTFile(_saveDir_, savePath );
        saveDir->cd();

        _propagator_.getPlotGenerator().generateSamplePlots();

        auto oneSigmaHistList = _propagator_.getPlotGenerator().getHistHolderList();
        _propagator_.getPlotGenerator().generateComparisonPlots( oneSigmaHistList, refHistList, saveDir );
        parSet.setEigenParameter(iEigen, currentParValue);
        parSet.propagateEigenToOriginal();
        _propagator_.propagateParametersOnSamples();

        const auto& compHistList = _propagator_.getPlotGenerator().getComparisonHistHolderList();

        // Since those were not saved, delete manually
        for( auto& hist : oneSigmaHistList ){ delete hist.histPtr; }
        oneSigmaHistList.clear();
      }
    }

  }

  _saveDir_->cd();

  // Since those were not saved, delete manually
  for( auto& refHist : refHistList ){ delete refHist.histPtr; }
  refHistList.clear();

}

void FitterEngine::fixGhostParameters(){
  LogInfo << __METHOD_NAME__ << std::endl;

  _propagator_.allowRfPropagation(); // since we don't need the weight of each event (only the Chi2 value)
  updateChi2Cache();

  LogDebug << "Reference χ² = " << _chi2StatBuffer_ << std::endl;
  double baseChi2Stat = _chi2StatBuffer_;

  // +1 sigma
  int iPar = -1;
  int iFitPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){

//    if( parSet.isUseEigenDecompInFit() ){
//      LogWarning << "Skipping " << parSet.getName() << " since eigen decomposition will be used." << std::endl;
//      continue;
//    }

    for( auto& par : parSet.getParameterList() ){
      iPar++;
      if( not parSet.isUseEigenDecompInFit() ) iFitPar++;

      if( par.isFixed() ) continue;

      double currentParValue = par.getParameterValue();
      par.setParameterValue( currentParValue + par.getStdDevValue() );
      LogInfo << "(" << iPar+1 << "/" << _nbParameters_ << ") +1 sigma on " << parSet.getName() + "/" + par.getTitle()
              << " -> " << par.getParameterValue() << std::endl;

      updateChi2Cache();

      double deltaChi2 = _chi2StatBuffer_ - baseChi2Stat;

      if( std::fabs(deltaChi2) < 1E-6 ){
        LogAlert << parSet.getName() + "/" + par.getTitle() << ": Δχ² = " << deltaChi2 << " < " << 1E-6 << ", fixing parameter." << std::endl;
        par.setIsFixed(true); // ignored in the Chi2 computation of the parSet
        if( not parSet.isUseEigenDecompInFit() ) { _minimizer_->FixVariable(iFitPar); }
      }

      par.setParameterValue( currentParValue );
    }

    if( parSet.isUseEigenDecompInFit() ){
      iFitPar += parSet.getNbEnabledEigenParameters();
    }
  }

  updateChi2Cache(); // comeback to old values
  _propagator_.preventRfPropagation();
}
void FitterEngine::scanParameters(int nbSteps_, const std::string &saveDir_) {
  LogInfo << "Performing parameter scans..." << std::endl;
  for( int iPar = 0 ; iPar < _minimizer_->NDim() ; iPar++ ){
    this->scanParameter(iPar, nbSteps_, saveDir_);
  } // iPar
}
void FitterEngine::scanParameter(int iPar, int nbSteps_, const std::string &saveDir_) {

  //Internally Scan performs steps-1, so add one to actually get the number of steps
  //we ask for.
  unsigned int adj_steps = nbSteps_+1;
  auto* x = new double[adj_steps] {};
  auto* y = new double[adj_steps] {};

  LogInfo << "Scanning fit parameter #" << iPar
          << " (" << _minimizer_->VariableName(iPar) << ")." << std::endl;

  _propagator_.allowRfPropagation();
  bool success = _minimizer_->Scan(iPar, adj_steps, x, y);

  if( not success ){
    LogError << "Parameter scan failed." << std::endl;
  }

  TGraph scanGraph(nbSteps_, x, y);

  std::stringstream ss;
  ss << GenericToolbox::replaceSubstringInString(_minimizer_->VariableName(iPar), "/", "_");
  ss << "_TGraph";

  scanGraph.SetTitle(_minimizer_->VariableName(iPar).c_str());

  if( _saveDir_ != nullptr ){
    GenericToolbox::mkdirTFile(_saveDir_, saveDir_)->cd();
    scanGraph.Write( ss.str().c_str() );
  }
  _propagator_.preventRfPropagation();

  delete[] x;
  delete[] y;
}

void FitterEngine::throwParameters(){
  LogInfo << __METHOD_NAME__ << std::endl;

  int iPar = 0;
  for( auto& parSet : _propagator_.getParameterSetsList() ){
    for( auto& par : parSet.getParameterList() ){
      if( par.isEnabled() and not par.isFixed() ){
        par.setParameterValue( par.getPriorValue() + _prng_.Gaus(0, par.getStdDevValue()) );
        _minimizer_->SetVariableValue( iPar, par.getParameterValue() );
      }
    }
    iPar++;
  }

  _propagator_.preventRfPropagation(); // Making sure since we need the weight of each event
  _propagator_.propagateParametersOnSamples();
}

void FitterEngine::fit(){
  LogWarning << __METHOD_NAME__ << std::endl;

  LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
          << "Number of free parameters   : " << _minimizer_->NFree() << std::endl
          << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree()
          << std::endl;

  LogWarning << "-----------------------------" << std::endl;
  LogWarning << "Summary of the fit parameters" << std::endl;
  LogWarning << "-----------------------------" << std::endl;
  int iFitPar = -1;
  for( const auto& parSet : _propagator_.getParameterSetsList() ){
    if( not parSet.isUseEigenDecompInFit() ){
      LogWarning << parSet.getName() << ": " << parSet.getNbParameters() << " parameters" << std::endl;
      Logger::setIndentStr("├─ ");
      for( const auto& par : parSet.getParameterList() ){
        iFitPar++;
        if( par.isEnabled() ){
          if( par.isFixed() ){
            LogInfo << "\033[41m" << "#" << iFitPar << " -> " << parSet.getName() << "/" << par.getTitle() << ": FIXED - Prior: " << par.getParameterValue() <<  "\033[0m" << std::endl;
          }
          else{
            LogInfo << "#" << iFitPar << " -> " << parSet.getName() << "/" << par.getTitle() << " - Prior: " << par.getParameterValue() << std::endl;
          }
        }
        else{
          LogInfo << "\033[43m" << "#" << iFitPar << " -> " << parSet.getName() << "/" << par.getTitle() << ": Disabled" <<  "\033[0m" << std::endl;
        }
      }
      Logger::setIndentStr("");
    }
    else{
      LogWarning << parSet.getName() << ": " << parSet.getNbEnabledEigenParameters() << " eigen parameters" << std::endl;
      Logger::setIndentStr("├─ ");
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        iFitPar++;
        LogInfo << "#" << iFitPar << " -> " << parSet.getName() << "/eigen_#" << iEigen << " - Prior: " << parSet.getEigenParameter(iEigen) << std::endl;
      }
      Logger::setIndentStr("");
    }
  }

  _propagator_.allowRfPropagation(); // if RF are setup -> a lot faster


  LogWarning << "-------------------" << std::endl;
  LogWarning << "Calling minimize..." << std::endl;
  LogWarning << "-------------------" << std::endl;
  _fitUnderGoing_ = true;
  bool _fitHasConverged_ = _minimizer_->Minimize();
  if( _fitHasConverged_ ){
    LogInfo << "Fit converged!" << std::endl
            << "Status code: " << _minimizer_->Status() << std::endl;

    LogWarning << "-------------------" << std::endl;
    LogWarning << "Calling HESSE..." << std::endl;
    LogWarning << "-------------------" << std::endl;
    _fitHasConverged_ = _minimizer_->Hesse();

    if(not _fitHasConverged_){
      LogError  << "Hesse did not converge." << std::endl;
      LogError  << "Failed with status code: " << _minimizer_->Status() << std::endl;
    }
    else{
      LogInfo << "Hesse converged." << std::endl
              << "Status code: " << _minimizer_->Status() << std::endl;
    }
    LogDebug << _convergenceMonitor_.generateMonitorString(); // lasting printout
  }
  else{
    LogError << "Did not converged." << std::endl;
    LogError << _convergenceMonitor_.generateMonitorString(); // lasting printout
  }

  _propagator_.preventRfPropagation(); // since we need the weight of each event
  _propagator_.propagateParametersOnSamples();

  _fitUnderGoing_ = false;
  _fitIsDone_ = true;
}
void FitterEngine::updateChi2Cache(){

  // Propagate on histograms
  _propagator_.propagateParametersOnSamples();

  ////////////////////////////////
  // Compute chi2 stat
  ////////////////////////////////
  _chi2StatBuffer_ = 0; // reset
  double buffer;

  if( not _propagator_.getFitSampleSet().empty() ){
    _chi2StatBuffer_ = _propagator_.getFitSampleSet().evalLikelihood();
  }
  else {
    for( const auto& sample : _propagator_.getSamplesList() ){
      //buffer = _samplesList_.at(sampleContainer)->CalcChi2();
      buffer = sample.CalcLLH();
      //buffer = _samplesList_.at(sampleContainer)->CalcEffLLH();

      _chi2StatBuffer_ += buffer;
    }
  }

  ////////////////////////////////
  // Compute the penalty terms
  ////////////////////////////////
  _chi2PullsBuffer_ = 0;
  _chi2RegBuffer_ = 0;
  for( auto& parSet : _propagator_.getParameterSetsList() ){
    _chi2PullsBuffer_ += parSet.getChi2();
  }

  _chi2Buffer_ = _chi2StatBuffer_ + _chi2PullsBuffer_ + _chi2RegBuffer_;

}
double FitterEngine::evalFit(const double* parArray_){
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  _nbFitCalls_++;

  // Update fit parameter values:
  int iPar = -1;
  for( auto& parSet : _propagator_.getParameterSetsList() ){
    if( not parSet.isUseEigenDecompInFit() ){
      for( auto& par : parSet.getParameterList() ){
        iPar++;
        par.setParameterValue( parArray_[iPar] );
      }
    }
    else{
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        iPar++;
        parSet.setEigenParameter(iEigen, parArray_[iPar]);
      }
      parSet.propagateEigenToOriginal();
    }
  }

  // Compute the Chi2
  updateChi2Cache();

  if( _convergenceMonitor_.isGenerateMonitorStringOk() and _fitUnderGoing_ ){
    std::stringstream ss;
    ss << __METHOD_NAME__ << ": call #" << _nbFitCalls_ << std::endl;
    ss << "Computation time: " << GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__) << std::endl;
    if( not _propagator_.isUseResponseFunctions() ){
      ss << GET_VAR_NAME_VALUE(_propagator_.weightProp) << std::endl;
      ss << GET_VAR_NAME_VALUE(_propagator_.fillProp);
    }
    else{
      ss << GET_VAR_NAME_VALUE(_propagator_.applyRf);
    }
    _convergenceMonitor_.setHeaderString(ss.str());
    _convergenceMonitor_.getVariable("Total").addQuantity(_chi2Buffer_);
    _convergenceMonitor_.getVariable("Stat").addQuantity(_chi2StatBuffer_);
    _convergenceMonitor_.getVariable("Syst").addQuantity(_chi2PullsBuffer_);

    if( _nbFitCalls_ == 1 ){
      LogInfo << _convergenceMonitor_.generateMonitorString();
    }
    else{
      LogInfo << _convergenceMonitor_.generateMonitorString(true);
    }
  }

  // Fill History
  _chi2History_["Total"].emplace_back(_chi2Buffer_);
  _chi2History_["Stat"].emplace_back(_chi2StatBuffer_);
  _chi2History_["Syst"].emplace_back(_chi2PullsBuffer_);

  return _chi2Buffer_;
}

void FitterEngine::writePostFitData() {
  LogInfo << __METHOD_NAME__ << std::endl;

  if( not _fitIsDone_ ){
    LogError << "Can't do " << __METHOD_NAME__ << " while fit has not been called." << std::endl;
    throw std::logic_error("Can't do " + __METHOD_NAME__ + " while fit has not been called.");
  }

  if( _saveDir_ != nullptr ){
    auto* postFitDir = GenericToolbox::mkdirTFile(_saveDir_, "postFit");

    GenericToolbox::mkdirTFile(postFitDir, "chi2")->cd();
    GenericToolbox::convertTVectorDtoTH1D(_chi2History_["Total"], "#chi^{2}(Total) - Converging history")->Write("chi2TotalHistory");
    GenericToolbox::convertTVectorDtoTH1D(_chi2History_["Stat"], "#chi^{2}(Stat) - Converging history")->Write("chi2StatHistory");
    GenericToolbox::convertTVectorDtoTH1D(_chi2History_["Syst"], "#chi^{2}(Syst) - Converging history")->Write("chi2PullsHistory");

    this->generateSamplePlots("postFit/samples");

    auto* errorDir = GenericToolbox::mkdirTFile(postFitDir, "errors");
    const unsigned int nfree = _minimizer_->NFree();
    if(_minimizer_->X() != nullptr){
      double covarianceMatrixArray[_minimizer_->NDim() * _minimizer_->NDim()];
      _minimizer_->GetCovMatrix(covarianceMatrixArray);
      TMatrixDSym fitterCovarianceMatrix(_minimizer_->NDim(), covarianceMatrixArray);
      TH2D* fitterCovarianceMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) &fitterCovarianceMatrix, "fitterCovarianceMatrixTH2D");

      LogInfo << "Fitter covariance matrix is " << fitterCovarianceMatrix.GetNrows() << "x" << fitterCovarianceMatrix.GetNcols() << std::endl;

      std::vector<double> parameterValueList(_minimizer_->X(),      _minimizer_->X()      + _minimizer_->NDim());
      std::vector<double> parameterErrorList(_minimizer_->Errors(), _minimizer_->Errors() + _minimizer_->NDim());

      int parameterIndexOffset = 0;
      for( const auto& parSet : _propagator_.getParameterSetsList() ){

        if( not parSet.isEnabled() ){
          LogWarning << "Skipping disabled parameter set: " << parSet.getName() << std::endl;
          continue;
        }

        LogInfo << "Extracting post-fit errors of parameter set: " << parSet.getName() << std::endl;
        auto* parSetDir = GenericToolbox::mkdirTFile(errorDir, parSet.getName());

        TMatrixD* covMatrix;
        if( not parSet.isUseEigenDecompInFit() ) {
          LogDebug << "Extracting parameters..." << std::endl;
          covMatrix = new TMatrixD(parSet.getParameterList().size(), parSet.getParameterList().size());
          for (const auto &parRow: parSet.getParameterList()) {
            for (const auto &parCol: parSet.getParameterList()) {
              (*covMatrix)[parRow.getParameterIndex()][parCol.getParameterIndex()] =
                fitterCovarianceMatrix[parameterIndexOffset + parRow.getParameterIndex()][parameterIndexOffset +
                                                                                          parCol.getParameterIndex()];
            } // par Y
          } // par X

          for( const auto& par : parSet.getParameterList() ) {
            fitterCovarianceMatrixTH2D->GetXaxis()->SetBinLabel(1 + parameterIndexOffset + par.getParameterIndex(),
                                                                (parSet.getName() + "/" + par.getTitle()).c_str());
            fitterCovarianceMatrixTH2D->GetYaxis()->SetBinLabel(1 + parameterIndexOffset + par.getParameterIndex(),
                                                                (parSet.getName() + "/" + par.getTitle()).c_str());
          }
        }
        else{
          LogDebug << "Extracting eigen parameters..." << std::endl;

          covMatrix = new TMatrixD(parSet.getNbEnabledEigenParameters(), parSet.getNbEnabledEigenParameters());
          for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
            for( int jEigen = 0 ; jEigen < parSet.getNbEnabledEigenParameters() ; jEigen++ ){
              (*covMatrix)[iEigen][jEigen] = fitterCovarianceMatrix[parameterIndexOffset + iEigen][parameterIndexOffset + jEigen];
            }
          }

          auto* covMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) covMatrix, Form("Covariance_Eigen_%s_TH2D", parSet.getName().c_str()));
          auto* corMatrix = GenericToolbox::convertToCorrelationMatrix((TMatrixD*) covMatrix);
          auto* corMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D(corMatrix, Form("Correlation_Eigen_%s_TH2D", parSet.getName().c_str()));

          LogDebug << "Applying labels on eigen histograms..." << std::endl;
          for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
            covMatrixTH2D->GetXaxis()->SetBinLabel(1+iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());
            covMatrixTH2D->GetYaxis()->SetBinLabel(1+iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());
            corMatrixTH2D->GetXaxis()->SetBinLabel(1+iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());
            corMatrixTH2D->GetYaxis()->SetBinLabel(1+iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());

            fitterCovarianceMatrixTH2D->GetXaxis()->SetBinLabel(1+parameterIndexOffset+iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());
            fitterCovarianceMatrixTH2D->GetYaxis()->SetBinLabel(1+parameterIndexOffset+iEigen, (parSet.getName() + "/eigen_#" + std::to_string(iEigen)).c_str());
          }

          GenericToolbox::mkdirTFile(parSetDir, "matrices")->cd();
          covMatrix->Write("Covariance_Eigen_TMatrixD");
          covMatrixTH2D->Write("Covariance_Eigen_TH2D");
          corMatrix->Write("Correlation_Eigen_TMatrixD");
          corMatrixTH2D->Write("Correlation_Eigen_TH2D");

          LogDebug << "Converting eigen to original parameters..." << std::endl;
          auto* originalCovMatrix = new TMatrixD(parSet.getParameterList().size(), parSet.getParameterList().size());
          for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
            for( int jEigen = 0 ; jEigen < parSet.getNbEnabledEigenParameters() ; jEigen++ ){
              (*originalCovMatrix)[iEigen][jEigen] = (*covMatrix)[iEigen][jEigen];
            }
          }

//          (*originalCovMatrix) = (*parSet.getEigenVectors()) * (*originalCovMatrix) * (*parSet.getInvertedEigenVectors());
          (*originalCovMatrix) = (*parSet.getInvertedEigenVectors()) * (*originalCovMatrix) * (*parSet.getEigenVectors());

          for( int iBin = 0 ; iBin < originalCovMatrix->GetNrows() ; iBin++ ){
            for( int jBin = 0 ; jBin < originalCovMatrix->GetNcols() ; jBin++ ){
              if( parSet.getParameterList().at(iBin).isFixed() or parSet.getParameterList().at(jBin).isFixed() ){
                (*originalCovMatrix)[iBin][jBin] = 0;
              }
            }
          }

          covMatrix = originalCovMatrix;

        }

        auto* covMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) covMatrix, Form("Covariance_%s_TH2D", parSet.getName().c_str()));
        auto* corMatrix = GenericToolbox::convertToCorrelationMatrix((TMatrixD*) covMatrix);
        auto* corMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D(corMatrix, Form("Correlation_%s_TH2D", parSet.getName().c_str()));

        LogDebug << "Applying labels on histograms..." << std::endl;
        for( const auto& par : parSet.getParameterList() ){
          covMatrixTH2D->GetXaxis()->SetBinLabel(1+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());
          covMatrixTH2D->GetYaxis()->SetBinLabel(1+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());
          corMatrixTH2D->GetXaxis()->SetBinLabel(1+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());
          corMatrixTH2D->GetYaxis()->SetBinLabel(1+par.getParameterIndex(), (parSet.getName() + "/" + par.getTitle()).c_str());
        }

        LogTrace << "Writing cov matrices..." << std::endl;
        GenericToolbox::mkdirTFile(parSetDir, "matrices")->cd();
        covMatrix->Write("Covariance_TMatrixD");
        covMatrixTH2D->Write("Covariance_TH2D");
        corMatrix->Write("Correlation_TMatrixD");
        corMatrixTH2D->Write("Correlation_TH2D");

        // Parameters
        LogTrace << "Generating parameter plots..." << std::endl;
        GenericToolbox::mkdirTFile(parSetDir, "values")->cd();
        auto* postFitErrorHist = new TH1D("postFitErrors_TH1D", "Post-fit Errors", parSet.getNbParameters(), 0, parSet.getNbParameters());
        auto* preFitErrorHist = new TH1D("preFitErrors_TH1D", "Pre-fit Errors", parSet.getNbParameters(), 0, parSet.getNbParameters());
        for( const auto& par : parSet.getParameterList() ){
          postFitErrorHist->GetXaxis()->SetBinLabel(1 + par.getParameterIndex(), par.getTitle().c_str());
          postFitErrorHist->SetBinContent( 1 + par.getParameterIndex(), par.getParameterValue());
          postFitErrorHist->SetBinError( 1 + par.getParameterIndex(), TMath::Sqrt((*covMatrix)[par.getParameterIndex()][par.getParameterIndex()]));

          preFitErrorHist->GetXaxis()->SetBinLabel(1 + par.getParameterIndex(), par.getTitle().c_str());
          preFitErrorHist->SetBinContent( 1 + par.getParameterIndex(), par.getPriorValue() );
          preFitErrorHist->SetBinError( 1 + par.getParameterIndex(), par.getStdDevValue() );
        }

        LogTrace << "Cosmetics..." << std::endl;
        preFitErrorHist->SetFillColor(kRed-9);
//        preFitErrorHist->SetFillColorAlpha(kRed-9, 0.5);
//        preFitErrorHist->SetFillStyle(4050); // 50 % opaque ?
        preFitErrorHist->SetMarkerStyle(kFullDotLarge);
        preFitErrorHist->SetMarkerColor(kRed-3);
        preFitErrorHist->SetTitle("Pre-fit Errors");

        postFitErrorHist->SetLineColor(9);
        postFitErrorHist->SetLineWidth(2);
        postFitErrorHist->SetMarkerColor(9);
        postFitErrorHist->SetMarkerStyle(kFullDotLarge);
        postFitErrorHist->SetTitle("Post-fit Errors");

        auto* errorsCanvas = new TCanvas(Form("Fit Constraints for %s", parSet.getName().c_str()), Form("Fit Constraints for %s", parSet.getName().c_str()), 800, 600);
        errorsCanvas->cd();
        preFitErrorHist->Draw("E2");
        errorsCanvas->Update(); // otherwise does not display...
        postFitErrorHist->Draw("E SAME");

//        auto* legend = gPad->BuildLegend();
        gPad->SetGridx();
        gPad->SetGridy();

        preFitErrorHist->SetTitle(Form("Pre-fit/Post-fit Comparison for %s", parSet.getName().c_str()));
        errorsCanvas->Write("fitConstraints_TCanvas");

        preFitErrorHist->SetTitle(Form("Pre-fit Errors of %s", parSet.getName().c_str()));
        postFitErrorHist->SetTitle(Form("Post-fit Errors of %s", parSet.getName().c_str()));
        postFitErrorHist->Write();
        preFitErrorHist->Write();


        if( not parSet.isUseEigenDecompInFit() ){
          parameterIndexOffset += int(parSet.getNbParameters());
        }
        else{
          parameterIndexOffset += parSet.getNbEnabledEigenParameters();
        }

      } // parSet

      errorDir->cd();
      fitterCovarianceMatrix.Write("fitterCovarianceMatrix_TMatrixDSym");
      fitterCovarianceMatrixTH2D->Write("fitterCovarianceMatrix_TH2D");
    } // minimizer valid?
  } // save dir?

}

void FitterEngine::initializePropagator(){
  LogDebug << __METHOD_NAME__ << std::endl;

  _propagator_.setConfig(JsonUtils::fetchValue<json>(_config_, "propagatorConfig"));
  _propagator_.setSaveDir(_saveDir_);

//  TFile* f = TFile::Open(JsonUtils::fetchValue<std::string>(_config_, "mc_file").c_str(), "READ");
//  _propagator_.setDataTree( f->Get<TTree>("selectedEvents") );
//  _propagator_.setMcFilePath(JsonUtils::fetchValue<std::string>(_config_, "mc_file"));

  _propagator_.initialize();

}
void FitterEngine::initializeMinimizer(){
  LogDebug << __METHOD_NAME__ << std::endl;

  _nbFitParameters_ = 0;
  _nbParameters_ = 0;
  for( const auto& parSet : _propagator_.getParameterSetsList() ){
    _nbParameters_ += int(parSet.getNbParameters());
    if( not parSet.isUseEigenDecompInFit() ){
      _nbFitParameters_ += int(parSet.getNbParameters());
    }
    else{
      _nbFitParameters_ += parSet.getNbEnabledEigenParameters();
    }
  }

  LogTrace << GET_VAR_NAME_VALUE(_nbParameters_) << std::endl;
  LogTrace << GET_VAR_NAME_VALUE(_nbFitParameters_) << std::endl;


  auto minimizationConfig = JsonUtils::fetchSubEntry(_config_, {"minimizerConfig"});
  if( minimizationConfig.is_string() ){ minimizationConfig = JsonUtils::readConfigFile(minimizationConfig.get<std::string>()); }

  _minimizer_ = std::shared_ptr<ROOT::Math::Minimizer>(
    ROOT::Math::Factory::CreateMinimizer(
      JsonUtils::fetchValue<std::string>(minimizationConfig, "minimizer"),
      JsonUtils::fetchValue<std::string>(minimizationConfig, "algorithm")
    )
  );

  _functor_ = std::shared_ptr<ROOT::Math::Functor>(
    new ROOT::Math::Functor(
      this, &FitterEngine::evalFit, _nbFitParameters_
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

    if( not parSet.isUseEigenDecompInFit() ){
      for( auto& par : parSet.getParameterList()  ){
        iPar++;
        _minimizer_->SetVariable(
          iPar,
          parSet.getName() + "/" + par.getTitle(),
          par.getParameterValue(),
          0.01
        );
//      _minimizer_->SetVariableLimits(); // TODO: IMPLEMENT SetVariableLimits
      } // par
    }
    else{
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        iPar++;
//        LogTrace << GET_VAR_NAME_VALUE(iPar) << " is eigen: " << parSet.getEigenParameter(iEigen) << std::endl;
        _minimizer_->SetVariable(
          iPar,
          parSet.getName() + "/eigen_#" + std::to_string(iPar),
          parSet.getEigenParameter(iEigen),
          0.01
        );
      }
    }

  } // parSet

  _convergenceMonitor_.addDisplayedQuantity("VarName");
  _convergenceMonitor_.addDisplayedQuantity("LastAddedValue");
  _convergenceMonitor_.addDisplayedQuantity("SlopePerCall");

  _convergenceMonitor_.getQuantity("VarName").title = "Chi2";
  _convergenceMonitor_.getQuantity("LastAddedValue").title = "Current Value";
  _convergenceMonitor_.getQuantity("SlopePerCall").title = "Avg. Slope /call";

  _convergenceMonitor_.addVariable("Total");
  _convergenceMonitor_.addVariable("Stat");
  _convergenceMonitor_.addVariable("Syst");

}
