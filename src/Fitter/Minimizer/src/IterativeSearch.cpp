//
// Created by Haowei Zheng on 19/08/2026.
//

#include "IterativeSearch.h"
#include "FitterEngine.h"

#include "GundamGlobals.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Time.h"
#include "Logger.h"

#include "Math/Factory.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>


void IterativeSearch::configureImpl(){
  LogDebugIf(GundamGlobals::isDebug()) << "Configuring IterativeSearch..." << std::endl;

  // read general parameters first
  this->MinimizerBase::configureImpl();

  _config_.defineFields({
    {FieldFlag::MANDATORY, "searchParameters"},
    {"firstPoint"},
    {"nbPoints"},
    {"pointList"},
    {"serpentine"},
    {"warmStart"},
    {"coldRefitThreshold"},
    {"saveParameterVector"},
    {"minimizer"},
    {"algorithm"},
    {"strategy"},
    {"print_level"},
    {"tolerance"},
    {"maxIterations", {"max_iter"}},
    {"maxFcnCalls", {"max_fcn"}},
  });
  _config_.checkConfiguration();

  _config_.fillValue(_firstPoint_, "firstPoint");
  _config_.fillValue(_nbPoints_, "nbPoints");
  _config_.fillValue(_runList_, "pointList");

  _config_.fillValue(_serpentine_, "serpentine");
  _config_.fillValue(_warmStart_, "warmStart");
  _config_.fillValue(_coldRefitThreshold_, "coldRefitThreshold");
  _config_.fillValue(_saveParameterVector_, "saveParameterVector");

  _config_.fillValue(_minimizerType_, "minimizer");
  _config_.fillValue(_minimizerAlgo_, "algorithm");

  _config_.fillValue(_strategy_, "strategy");
  _config_.fillValue(_printLevel_, "print_level");
  _config_.fillValue(_tolerance_, "tolerance");
  _config_.fillValue(_maxIterations_, "maxIterations");
  _config_.fillValue(_maxFcnCalls_, "maxFcnCalls");

  for( auto& parConfig : _config_.loop("searchParameters") ){
    parConfig.defineFields({
      {FieldFlag::MANDATORY, "parameterSetName"},
      {FieldFlag::MANDATORY, "parameterName"},
      {"min"},
      {"max"},
      {"nSteps"},
      {"values"},
    });
    parConfig.checkConfiguration();

    _searchParameterList_.emplace_back();
    auto& searchPar = _searchParameterList_.back();

    parConfig.fillValue(searchPar.parameterSetName, "parameterSetName");
    parConfig.fillValue(searchPar.parameterName, "parameterName");

    if( parConfig.hasField("values") ){
      parConfig.fillValue(searchPar.values, "values");
    }
    else{
      LogThrowIf(not parConfig.hasField("min") or not parConfig.hasField("max") or not parConfig.hasField("nSteps"),
                 searchPar.parameterName << ": needs either \"values\", or all of \"min\", \"max\" and \"nSteps\".");

      double min{parConfig.fetchValue<double>("min")};
      double max{parConfig.fetchValue<double>("max")};
      int nSteps{parConfig.fetchValue<int>("nSteps")};

      LogThrowIf(nSteps < 2, searchPar.parameterName << ": nSteps is " << nSteps << ", needs at least 2.");
      LogThrowIf(max <= min, searchPar.parameterName << ": max (" << max << ") is not above min (" << min << ").");

      // both ends are on the grid
      searchPar.values.reserve(nSteps);
      for( int iStep = 0 ; iStep < nSteps ; iStep++ ){
        searchPar.values.emplace_back( min + (max - min) * iStep / (nSteps - 1) );
      }
    }

    LogThrowIf(searchPar.values.empty(), searchPar.parameterName << ": empty value list.");
  }

  LogThrowIf(_searchParameterList_.empty(), "No search parameter defined.");
}

void IterativeSearch::initializeImpl(){

  // find the search parameters and flag them as fixed, so the base class does not count them as free
  this->resolveSearchParameters();

  this->MinimizerBase::initializeImpl();

  // the search parameters must not be handed to the minimizer
  this->stripSearchParametersFromList();

  LogInfo << "Initializing IterativeSearch..." << std::endl;

  LogInfo << "Tolerance is set to: " << _tolerance_ << std::endl;

  LogInfo << "Defining minimizer as: " << _minimizerType_ << "/" << _minimizerAlgo_ << std::endl;
  _rootMinimizer_ = std::unique_ptr<ROOT::Math::Minimizer>(
      ROOT::Math::Factory::CreateMinimizer(_minimizerType_, _minimizerAlgo_)
  );
  LogThrowIf(_rootMinimizer_ == nullptr, "Could not create minimizer: " << _minimizerType_ << "/" << _minimizerAlgo_);

  if( _minimizerAlgo_.empty() ){
    _minimizerAlgo_ = _rootMinimizer_->Options().MinimizerAlgorithm();
    LogWarning << "Using default minimizer algo: " << _minimizerAlgo_ << std::endl;
  }

  _functor_ = ROOT::Math::Functor(this, &IterativeSearch::evalFit, getMinimizerFitParameterPtr().size());
  _rootMinimizer_->SetFunction( _functor_ );
  _rootMinimizer_->SetStrategy(_strategy_);
  _rootMinimizer_->SetPrintLevel(_printLevel_);
  _rootMinimizer_->SetTolerance(_tolerance_);
  _rootMinimizer_->SetMaxIterations(_maxIterations_);
  _rootMinimizer_->SetMaxFunctionCalls(_maxFcnCalls_);

  // no Hesse: the search never needs the error matrix
  _rootMinimizer_->SetValidError(false);

  // snapshot of the prefit point in fit space
  _prefitValues_.clear();
  _prefitSteps_.clear();
  _prefitValues_.reserve( getMinimizerFitParameterPtr().size() );
  _prefitSteps_.reserve( getMinimizerFitParameterPtr().size() );
  for( auto* parPtr : getMinimizerFitParameterPtr() ){
    LogThrowIf(std::isnan(parPtr->getStepSize()), "No step size provided for: " << parPtr->getFullTitle());

    if( useNormalizedFitSpace() ){
      _prefitValues_.emplace_back( ParameterSet::toNormalizedParValue(parPtr->getParameterValue(), *parPtr) );
      _prefitSteps_.emplace_back( ParameterSet::toNormalizedParRange(parPtr->getStepSize(), *parPtr) );
    }
    else{
      _prefitValues_.emplace_back( parPtr->getParameterValue() );
      _prefitSteps_.emplace_back( parPtr->getStepSize() );
    }
  }

  this->resetMinimizer( _prefitValues_ );

  LogInfo << _searchParameterList_.size() << " search parameters:" << std::endl;
  for( auto& searchPar : _searchParameterList_ ){
    LogInfo << "  " << searchPar.parPtr->getFullTitle() << ": " << searchPar.values.size() << " values from "
            << searchPar.values.front() << " to " << searchPar.values.back() << std::endl;
  }
  LogInfo << "Minimizer parameters: " << getMinimizerFitParameterPtr().size()
          << " (search parameters removed)" << std::endl;

  // the points to visit, and the slice of them this job runs
  this->buildSearchPointList();
  this->buildRunList();
  LogInfo << _searchPointList_.size() << " search points. This job runs " << _runList_.size()
          << " of them, from point " << _runList_.front() << " to " << _runList_.back() << "." << std::endl;

  LogInfo << "IterativeSearch initialized." << std::endl;
}

void IterativeSearch::buildSearchPointList(){

  int nbPoints{1};
  for( auto& searchPar : _searchParameterList_ ){ nbPoints *= int(searchPar.values.size()); }

  _searchPointList_.clear();
  _searchPointList_.reserve( nbPoints );

  std::vector<int> valueIndexList( _searchParameterList_.size(), 0 );
  for( int iPoint = 0 ; iPoint < nbPoints ; iPoint++ ){

    // decode the walk position like a mixed radix number, last search parameter running fastest
    int remainder{iPoint};
    for( int iPar = int(_searchParameterList_.size()) - 1 ; iPar >= 0 ; iPar-- ){
      int nbValues{ int(_searchParameterList_[iPar].values.size()) };
      valueIndexList[iPar] = remainder % nbValues;
      remainder /= nbValues;
    }

    if( _serpentine_ ){
      // reverse a parameter's direction every time the slower ones have advanced an odd number
      // of times, so consecutive points are always neighbours instead of jumping back across a row
      std::vector<int> walk{ valueIndexList };
      int parity{0};
      for( std::size_t iPar = 0 ; iPar + 1 < walk.size() ; iPar++ ){
        parity += walk[iPar];
        if( parity % 2 == 1 ){
          valueIndexList[iPar + 1] = int(_searchParameterList_[iPar + 1].values.size()) - 1 - walk[iPar + 1];
        }
      }
    }

    _searchPointList_.emplace_back();
    auto& point = _searchPointList_.back();
    point.index = iPoint;
    point.values.reserve( _searchParameterList_.size() );
    for( std::size_t iPar = 0 ; iPar < _searchParameterList_.size() ; iPar++ ){
      point.values.emplace_back( _searchParameterList_[iPar].values[valueIndexList[iPar]] );
    }
  }
}

void IterativeSearch::buildRunList(){

  int nbPoints{ int(_searchPointList_.size()) };

  // an explicit list from the config wins
  if( not _runList_.empty() ){
    for( int point : _runList_ ){
      LogThrowIf(point < 0 or point >= nbPoints, "pointList has point " << point << ", the search has " << nbPoints << " points.");
    }
    return;
  }

  LogThrowIf(_firstPoint_ < 0 or _firstPoint_ >= nbPoints,
             "firstPoint is " << _firstPoint_ << ", the search has " << nbPoints << " points.");

  // default: run everything from firstPoint
  if( _nbPoints_ < 0 ){ _nbPoints_ = nbPoints - _firstPoint_; }
  LogThrowIf(_nbPoints_ == 0, "nbPoints is 0. Give a positive nbPoints, or a pointList.");

  int nbToRun{ _nbPoints_ };
  if( _firstPoint_ + nbToRun > nbPoints ){
    LogAlert << "Only " << nbPoints - _firstPoint_ << " points left from " << _firstPoint_
             << ", running those instead of " << nbToRun << "." << std::endl;
    nbToRun = nbPoints - _firstPoint_;
  }

  _runList_.reserve( nbToRun );
  for( int iPoint = 0 ; iPoint < nbToRun ; iPoint++ ){ _runList_.emplace_back( _firstPoint_ + iPoint ); }
}

void IterativeSearch::resolveSearchParameters(){

  for( auto& searchPar : _searchParameterList_ ){
    auto* parSetPtr = getModelPropagator().getParametersManager().getFitParameterSetPtr( searchPar.parameterSetName );
    LogThrowIf(parSetPtr == nullptr, "Could not find parameter set: " << searchPar.parameterSetName);

    // the search moves a physical parameter, which an eigen decomposed set does not expose
    LogThrowIf(parSetPtr->isEnableEigenDecomp(),
               searchPar.parameterSetName << " uses eigen decomposition, so " << searchPar.parameterName << " cannot be searched.");

    searchPar.parPtr = parSetPtr->getParameterPtr( searchPar.parameterName );
    LogThrowIf(searchPar.parPtr == nullptr, "Could not find parameter: " << searchPar.parameterName << " in " << searchPar.parameterSetName);

    LogThrowIf(not searchPar.parPtr->isEnabled(), searchPar.parPtr->getFullTitle() << " is disabled.");

    // parameters with the penalty disabled never reach the minimizer list (see MinimizerBase::initializeImpl)
    LogThrowIf(searchPar.parPtr->isPenaltyDisabled(),
               searchPar.parPtr->getFullTitle() << " has its penalty disabled. A search parameter has to be a minimizer parameter.");

    for( double value : searchPar.values ){
      LogThrowIf(not searchPar.parPtr->isInDomain(value),
                 searchPar.parPtr->getFullTitle() << ": value " << value << " is outside " << searchPar.parPtr->getParameterLimits());
    }

    // the search drives this parameter, not the minimizer
    searchPar.parPtr->setIsFixed( true );
  }
}

void IterativeSearch::stripSearchParametersFromList(){

  auto& parList = getMinimizerFitParameterPtr();

  for( auto& searchPar : _searchParameterList_ ){
    auto it = std::find( parList.begin(), parList.end(), searchPar.parPtr );
    LogThrowIf(it == parList.end(), searchPar.parPtr->getFullTitle() << " is not a minimizer parameter.");
    parList.erase( it );
  }
}

void IterativeSearch::resetMinimizer(const std::vector<double>& startValues_){

  // drops the whole Minuit2 state
  _rootMinimizer_->Clear();

  for( std::size_t iFitPar = 0 ; iFitPar < getMinimizerFitParameterPtr().size() ; iFitPar++ ){
    auto& fitPar = *(getMinimizerFitParameterPtr()[iFitPar]);

    _rootMinimizer_->SetVariable(iFitPar, fitPar.getFullTitle(), startValues_[iFitPar], _prefitSteps_[iFitPar]);

    // strange ROOT parameter setting...
    auto& limits = fitPar.getParameterLimits();
    if( useNormalizedFitSpace() ){
      if     ( limits.hasBothBounds() ){
        _rootMinimizer_->SetVariableLimits(iFitPar,
                                           ParameterSet::toNormalizedParValue(limits.min, fitPar),
                                           ParameterSet::toNormalizedParValue(limits.max, fitPar));
      }
      else if( limits.hasLowerBound() ){
        _rootMinimizer_->SetVariableLowerLimit(iFitPar, ParameterSet::toNormalizedParValue(limits.min, fitPar));
      }
      else if( limits.hasUpperBound() ){
        _rootMinimizer_->SetVariableUpperLimit(iFitPar, ParameterSet::toNormalizedParValue(limits.max, fitPar));
      }
    }
    else{
      if     ( limits.hasBothBounds() ){ _rootMinimizer_->SetVariableLimits(iFitPar, limits.min, limits.max); }
      else if( limits.hasLowerBound() ){ _rootMinimizer_->SetVariableLowerLimit(iFitPar, limits.min); }
      else if( limits.hasUpperBound() ){ _rootMinimizer_->SetVariableUpperLimit(iFitPar, limits.max); }
    }

    // fixed by the user, not by the search (search parameters are not in this list)
    if( fitPar.isFixed() ){ _rootMinimizer_->FixVariable(iFitPar); }
  }
}

void IterativeSearch::bookPointList(){

  // no output file, nothing to book. The fill and write calls check the pointer
  if( getOwner().getSaveDir() == nullptr ){ return; }

  // the tree is attached to the directory, so AutoSave() can flush it while the job runs
  GenericToolbox::mkdirTFile( getOwner().getSaveDir(), "postFit" )->cd();

  _pointListTree_ = new TTree("pointList", "One entry per visited point");

  _pointListTree_->Branch("Point", &_outPoint_, "Point/I");
  _pointListTree_->Branch("LLH", &_outLlh_, "LLH/D");
  _pointListTree_->Branch("LLHStatistical", &_outLlhStat_, "LLHStatistical/D");
  _pointListTree_->Branch("LLHPenalty", &_outLlhPenalty_, "LLHPenalty/D");
  _pointListTree_->Branch("Converged", &_outConverged_, "Converged/O");
  _pointListTree_->Branch("Edm", &_outEdm_, "Edm/D");
  _pointListTree_->Branch("nCalls", &_outNbCalls_, "nCalls/I");

  // the profiled values of this point, in the order of the searchParameters list
  _outProfiledValues_.assign( _searchParameterList_.size(), 0. );
  _pointListTree_->Branch("ProfiledValues", _outProfiledValues_.data(),
                          ("ProfiledValues[" + std::to_string(_outProfiledValues_.size()) + "]/D").c_str());

  // the post-fit values of the other parameters, in minimizer order (the key is in problemDefinition)
  if( _saveParameterVector_ ){
    _outParValues_.assign( getMinimizerFitParameterPtr().size(), 0. );
    _pointListTree_->Branch("PostFitParameterValues", _outParValues_.data(),
                            ("PostFitParameterValues[" + std::to_string(_outParValues_.size()) + "]/D").c_str());
  }
}

void IterativeSearch::writeProblemDefinition(int bestPoint_, double bestLlh_){

  if( getOwner().getSaveDir() == nullptr ){ return; }

  GenericToolbox::mkdirTFile( getOwner().getSaveDir(), "postFit" )->cd();

  auto* problemDefinition = new TTree("problemDefinition", "Profiled parameters and the key for pointList");

  int nProfiledParameters{ int(_searchParameterList_.size()) };
  int nPoints{ int(_searchPointList_.size()) };
  int nSampleBins{ getLikelihoodInterface().getNbSampleBins() };
  int nParameters{ int(getMinimizerFitParameterPtr().size()) };

  // which parameters were profiled, over which values
  std::vector<std::string> profiledParameterName;
  std::vector<std::vector<double>> profiledParameterValues;
  for( auto& searchPar : _searchParameterList_ ){
    profiledParameterName.emplace_back( searchPar.parPtr->getFullTitle() );
    profiledParameterValues.emplace_back( searchPar.values );
  }

  // the key for PostFitParameterValues[]: minimizer parameter order, search parameters already stripped
  std::vector<std::string> parameterName;
  std::vector<double> parameterPrior;
  std::vector<double> parameterSigma;
  for( auto* parPtr : getMinimizerFitParameterPtr() ){
    parameterName.emplace_back( parPtr->getFullTitle() );
    parameterPrior.emplace_back( parPtr->getPriorValue() );
    parameterSigma.emplace_back( parPtr->getStdDevValue() );
  }

  problemDefinition->Branch("nProfiledParameters", &nProfiledParameters, "nProfiledParameters/I");
  problemDefinition->Branch("nPoints", &nPoints, "nPoints/I");
  problemDefinition->Branch("nSampleBins", &nSampleBins, "nSampleBins/I");
  problemDefinition->Branch("BestPointInFile", &bestPoint_, "BestPointInFile/I");
  problemDefinition->Branch("BestLLHInFile", &bestLlh_, "BestLLHInFile/D");
  problemDefinition->Branch("ProfiledParameterName", &profiledParameterName);
  problemDefinition->Branch("ProfiledParameterValues", &profiledParameterValues);
  problemDefinition->Branch("nParameters", &nParameters, "nParameters/I");
  problemDefinition->Branch("ParameterName", &parameterName);
  problemDefinition->Branch("ParameterPrior", &parameterPrior);
  problemDefinition->Branch("ParameterSigma", &parameterSigma);

  problemDefinition->Fill();
  problemDefinition->Write();
}

void IterativeSearch::minimize(){
  // calling the common routine: parameter table and initial likelihood
  this->MinimizerBase::minimize();

  getMonitor().minimizerTitle = _minimizerType_ + "/" + _minimizerAlgo_;

  this->bookPointList();

  double previousLlh{std::nan("unset")};
  double bestLlh{std::numeric_limits<double>::infinity()};
  int bestPoint{-1};
  std::vector<double> bestFitValues( getMinimizerFitParameterPtr().size(), 0 );
  std::vector<double> bestSearchValues( _searchParameterList_.size(), 0 );

  int nbFailedPoints{0};

  GenericToolbox::Time::Timer pointStopWatch;

  for( std::size_t iRun = 0 ; iRun < _runList_.size() ; iRun++ ){
    auto& point = _searchPointList_[ _runList_[iRun] ];

    // move the search parameters onto this point. They are fixed for the minimizer, so only this loop changes them
    std::stringstream ssPoint;
    for( std::size_t iPar = 0 ; iPar < _searchParameterList_.size() ; iPar++ ){
      _searchParameterList_[iPar].parPtr->setParameterValue( point.values[iPar] );
      ssPoint << ( iPar == 0 ? "" : ", " ) << _searchParameterList_[iPar].parameterName << " = " << point.values[iPar];
    }

    // the first point runs on the declaration made at init. With a warm start, Minuit keeps the
    // state of the previous point and starts from there; otherwise go back to the prefit point
    if( not _warmStart_ and iRun != 0 ){ this->resetMinimizer( _prefitValues_ ); }

    getMonitor().stateTitleMonitor = "Search point " + std::to_string(point.index)
                                   + " (" + std::to_string(iRun + 1) + "/" + std::to_string(_runList_.size()) + ")"
                                   + " / " + ssPoint.str();

    int nbCallOffset{ getMonitor().nbEvalLikelihoodCalls };
    pointStopWatch.start();

    getMonitor().isEnabled = true;
    bool hasConverged{ _rootMinimizer_->Minimize() };

    // a warm start can get stuck in the previous point's valley. If the result is much worse
    // than the previous point, redo it from the prefit point
    if( _warmStart_ and not std::isnan(_coldRefitThreshold_) and not std::isnan(previousLlh)
        and _rootMinimizer_->MinValue() > previousLlh + _coldRefitThreshold_ ){
      LogAlert << "Point " << point.index << " landed " << _rootMinimizer_->MinValue() - previousLlh
               << " above the previous point. Refitting from the prefit point." << std::endl;
      this->resetMinimizer( _prefitValues_ );
      hasConverged = _rootMinimizer_->Minimize();
    }
    getMonitor().isEnabled = false;

    pointStopWatch.stop();

    // Minuit's last call is not necessarily the minimum: put the propagator and the llh buffers on X()
    this->evalFit( _rootMinimizer_->X() );

    previousLlh = _rootMinimizer_->MinValue();
    if( not hasConverged ){ nbFailedPoints++; }

    if( previousLlh < bestLlh ){
      bestLlh = previousLlh;
      bestPoint = point.index;
      for( std::size_t iPar = 0 ; iPar < bestFitValues.size() ; iPar++ ){ bestFitValues[iPar] = _rootMinimizer_->X()[iPar]; }
      bestSearchValues = point.values;
    }

    // one entry per point in the pointList tree
    _outPoint_ = point.index;
    _outLlh_ = previousLlh;
    _outLlhStat_ = getLikelihoodInterface().getBuffer().statLikelihood;
    _outLlhPenalty_ = getLikelihoodInterface().getBuffer().penaltyLikelihood;
    _outConverged_ = hasConverged;
    _outEdm_ = _rootMinimizer_->Edm();
    _outNbCalls_ = getMonitor().nbEvalLikelihoodCalls - nbCallOffset;
    // element-wise copy: the branch holds the address of the buffer, it must not be reallocated
    std::copy( point.values.begin(), point.values.end(), _outProfiledValues_.begin() );
    if( _saveParameterVector_ ){
      for( std::size_t iPar = 0 ; iPar < _outParValues_.size() ; iPar++ ){
        auto& fitPar = *(getMinimizerFitParameterPtr()[iPar]);
        double value{ _rootMinimizer_->X()[iPar] };
        if( useNormalizedFitSpace() ){ value = ParameterSet::toRealParValue(value, fitPar); }
        _outParValues_[iPar] = value;
      }
    }
    if( _pointListTree_ != nullptr ){
      _pointListTree_->Fill();
      // a job can run for hours: keep the file readable if it gets killed
      if( iRun % 100 == 99 ){ _pointListTree_->AutoSave("SaveSelf"); }
    }

    LogInfo << "Point " << point.index << " (" << iRun + 1 << "/" << _runList_.size() << "): " << ssPoint.str()
            << " -> llh " << previousLlh
            << " (stat " << getLikelihoodInterface().getBuffer().statLikelihood
            << ", syst " << getLikelihoodInterface().getBuffer().penaltyLikelihood << ")"
            << " in " << getMonitor().nbEvalLikelihoodCalls - nbCallOffset << " calls, "
            << GenericToolbox::toString(pointStopWatch.eval())
            << ( hasConverged ? "" : " NOT CONVERGED" )
            << std::endl;
  }

  if( _pointListTree_ != nullptr ){ _pointListTree_->Write(); }
  this->writeProblemDefinition( bestPoint, bestLlh );

  LogInfo << "Best point: " << bestPoint << ", llh " << bestLlh << " at";
  for( std::size_t iPar = 0 ; iPar < _searchParameterList_.size() ; iPar++ ){
    LogInfo << " " << _searchParameterList_[iPar].parameterName << " = " << bestSearchValues[iPar];
  }
  LogInfo << std::endl;

  if( nbFailedPoints != 0 ){ LogError << nbFailedPoints << " of " << _runList_.size() << " points did not converge." << std::endl; }

  // FitterEngine writes the parameter state next, so leave the propagator on the best point
  for( std::size_t iPar = 0 ; iPar < _searchParameterList_.size() ; iPar++ ){
    _searchParameterList_[iPar].parPtr->setParameterValue( bestSearchValues[iPar] );
  }
  this->evalFit( bestFitValues.data() );

  setMinimizerStatus( nbFailedPoints == 0 ? 0 : 1 );
}
