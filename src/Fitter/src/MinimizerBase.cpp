#include "MinimizerBase.h"
#include "FitterEngine.h"

#include "GenericToolbox.Json.h"
#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[MinimizerBase]");
});


void MinimizerBase::readConfigImpl(){
  LogInfo << "Reading MinimizerBase config..." << std::endl;

  _enablePostFitErrorEval_ = GenericToolbox::Json::fetchValue(_config_, "enablePostFitErrorFit", _enablePostFitErrorEval_);

  bool useNormalizedFitSpace = getLikelihood().getUseNormalizedFitSpace();
  useNormalizedFitSpace = GenericToolbox::Json::fetchValue(_config_, "useNormalizedFitSpace", useNormalizedFitSpace);
  getLikelihood().setUseNormalizedFitSpace(useNormalizedFitSpace);

  bool showParametersOnFitMonitor = getLikelihood().getShowParametersOnFitMonitor();
  showParametersOnFitMonitor = GenericToolbox::Json::fetchValue(_config_, "showParametersOnFitMonitor", showParametersOnFitMonitor);
  getLikelihood().setShowParametersOnFitMonitor(showParametersOnFitMonitor);

  auto maxNbParametersPerLineOnMonitor = getLikelihood().getMaxNbParametersPerLineOnMonitor();
  maxNbParametersPerLineOnMonitor = GenericToolbox::Json::fetchValue(_config_, "maxNbParametersPerLineOnMonitor", maxNbParametersPerLineOnMonitor);
  getLikelihood().setMaxNbParametersPerLineOnMonitor(maxNbParametersPerLineOnMonitor);

  if( GenericToolbox::getTerminalWidth() == 0 ){
    // batch mode
    double monitorBashModeRefreshRateInS = GenericToolbox::Json::fetchValue(_config_, "monitorBashModeRefreshRateInS", double(30.0));
    getConvergenceMonitor().setMaxRefreshRateInMs(monitorBashModeRefreshRateInS * 1000.);
  }
  else{
    int monitorRefreshRateInMs = GenericToolbox::Json::fetchValue(_config_, "monitorRefreshRateInMs", int(5000));
    getConvergenceMonitor().setMaxRefreshRateInMs(monitorRefreshRateInMs);
  }

}
void MinimizerBase::initializeImpl(){
  LogInfo << "Initializing the minimizer..." << std::endl;
  LogThrowIf( _owner_== nullptr, "FitterEngine ptr not set." );
}

void MinimizerBase::scanParameters(TDirectory* saveDir_) {
  LogWarning << "Parameter scanning is not implemented for this minimizer"
             << std::endl;
}

std::vector<Parameter *>& MinimizerBase::getMinimizerFitParameterPtr() {
  return getLikelihood().getMinimizerFitParameterPtr();
}

GenericToolbox::VariablesMonitor &MinimizerBase::getConvergenceMonitor() {
  return getLikelihood().getConvergenceMonitor();
}

Propagator& MinimizerBase::getPropagator() {return owner().getPropagator();}
const Propagator& MinimizerBase::getPropagator() const { return owner().getPropagator(); }

LikelihoodInterface& MinimizerBase::getLikelihood() {return owner().getLikelihood();}
const LikelihoodInterface& MinimizerBase::getLikelihood() const {return owner().getLikelihood();}

void MinimizerBase::printMinimizerFitParameters () {
  // This prints the same set of parameters as are in the vector returned by
  // getMinimizerFitParameterPtr(), but does it by parameter set so that the
  // output is a little more clear.

  LogWarning << std::endl << GenericToolbox::addUpDownBars("Summary of the fit parameters:") << std::endl;
  for( const auto& parSet : getPropagator().getParametersManager().getParameterSetsList() ){

    GenericToolbox::TablePrinter t;
    t.setColTitles({ {"Title"}, {"Starting"}, {"Prior"}, {"StdDev"}, {"Min"}, {"Max"}, {"Status"} });

    auto& parList = parSet.getEffectiveParameterList();
    LogWarning << parSet.getName() << ": " << parList.size() << " parameters" << std::endl;
    if( parList.empty() ) continue;

    for( const auto& par : parList ){
      std::string colorStr;
      std::string statusStr;

      if( not par.isEnabled() ) { statusStr = "Disabled"; colorStr = GenericToolbox::ColorCodes::yellowBackground; }
      else if( par.isFixed() )  { statusStr = "Fixed";    colorStr = GenericToolbox::ColorCodes::redBackground; }
      else                      {
        statusStr = PriorType::PriorTypeEnumNamespace::toString(par.getPriorType(), true) + " Prior";
        if(par.getPriorType()==PriorType::Flat) colorStr = GenericToolbox::ColorCodes::blueBackground;
      }

#ifdef NOCOLOR
      colorStr = "";
#endif

      t.addTableLine({
                         par.getTitle(),
                         std::to_string( par.isValueWithinBounds() ?
                                         par.getParameterValue()
                                         : std::nan("Invalid")),
                         std::to_string( par.getPriorValue() ),
                         std::to_string( par.getStdDevValue() ),
                         std::to_string( par.getMinValue() ),
                         std::to_string( par.getMaxValue() ),
                         statusStr
                     }, colorStr);
    }

    t.printTable();
  }
}

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
