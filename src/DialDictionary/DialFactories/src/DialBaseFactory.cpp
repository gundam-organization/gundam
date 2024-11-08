#include "DialBaseFactory.h"
#include "NormDialBaseFactory.h"
#include "GraphDialBaseFactory.h"
#include "SplineDialBaseFactory.h"
#include "SurfaceDialBaseFactory.h"

#include "RootFormula.h"
#include "CompiledLibDial.h"

#include "Logger.h"

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[DialBaseFactory]"); });
#endif


DialBase* DialBaseFactory::makeDial(const std::string& dialTitle_,
                                    const std::string& dialType_,
                                    const std::string& dialSubType_,
                                    TObject* dialInitializer_,
                                    bool useCachedDial_) {

  // Stuff the created dial into a unique_ptr, so it will be properly deleted
  // in the event of an exception.
  std::unique_ptr<DialBase> dialBase;

  LogExitIf(dialType_.empty(), "Dial type not set.");

  if (dialType_ == "Norm" || dialType_ == "Normalization") {
    NormDialBaseFactory factory;
    dialBase.reset(factory.makeDial(dialTitle_, dialType_, dialSubType_, dialInitializer_, useCachedDial_));
  }
  else if (dialType_ == "Graph") {
    GraphDialBaseFactory factory;
    dialBase.reset(factory.makeDial(dialTitle_, dialType_, dialSubType_, dialInitializer_, useCachedDial_));
  }
  else if (dialType_ == "Spline") {
    SplineDialBaseFactory factory;
    dialBase.reset(factory.makeDial(dialTitle_, dialType_, dialSubType_, dialInitializer_, useCachedDial_));
  }
  else if (dialType_ == "Surface") {
    SurfaceDialBaseFactory factory;
    dialBase.reset(factory.makeDial(dialTitle_, dialType_, dialSubType_, dialInitializer_, useCachedDial_));
  }
#define INCLUDE_DEPRECATED_DIAL_TYPES
#ifdef INCLUDE_DEPRECATED_DIAL_TYPES
  else if (dialType_ == "MonotonicSpline") {
    LogAlertOnce << "DEPRECATED DIAL-TYPE USED: MonotonicSpline will be removed. Instead use:"
                 << std::endl << "  dialType: \"Spline\""
                 << std::endl << "  dialSubType: \"catmull-rom, monotonic\""
                 << std::endl;
    SplineDialBaseFactory factory;
    dialBase.reset(factory.makeDial(dialTitle_, "Spline", "catmull-rom, monotonic",
                           dialInitializer_, useCachedDial_));
  }
  else if (dialType_ == "GeneralSpline") {
    LogAlertOnce << "DEPRECATED DIAL-TYPE USED: GeneralSpline will be removed. Instead use:"
                 << std::endl << "  dialType: \"Spline\""
                 << std::endl << "  dialSubType: \"not-a-knot\""
                 << std::endl;
   SplineDialBaseFactory factory;
    dialBase.reset(factory.makeDial(dialTitle_, "Spline", "not-a-knot", dialInitializer_, useCachedDial_));
  }
  else if (dialType_ == "SimpleSpline") {
    LogAlertOnce << "DEPRECATED DIAL-TYPE USED: SimpleSpline will be removed. Instead use:"
                 << std::endl << "  dialType: \"Spline\""
                 << std::endl << "  dialSubType: \"not-a-knot\""
                 << std::endl;
    SplineDialBaseFactory factory;
    dialBase.reset(factory.makeDial(dialTitle_, "Spline", "knot-a-knot", dialInitializer_, useCachedDial_));
  }
  else if (dialType_ == "LightGraph") {
    LogAlertOnce << "DEPRECATED DIAL-TYPE USED: LightGraph will be removed. Instead use:"
                 << std::endl << "  dialType: \"Graph\""
                 << std::endl << "  dialSubType: \"light\""
                 << std::endl;
    GraphDialBaseFactory factory;
    dialBase.reset(factory.makeDial(dialTitle_, "Graph", "light", dialInitializer_, useCachedDial_));
  }
#endif
  else{
    LogExit("Unrecognized dial type: " << dialType_);
  }

  // Pass the ownership without any constraints!
  return dialBase.release();
}


DialBase* DialBaseFactory::makeDial(const JsonType& config_){
  std::unique_ptr<DialBase> dialBase{nullptr};
  std::string dialType{};

  dialType = GenericToolbox::Json::fetchValue(config_, {{"dialType"}, {"dialsType"}}, dialType);

  if( dialType == "Formula" or dialType == "RootFormula" ){
    dialBase = std::make_unique<RootFormula>();
    auto* rootFormulaPtr{(RootFormula*) dialBase.get()};

    auto formulaConfig(GenericToolbox::Json::fetchValue<JsonType>(config_, "dialConfig"));

    rootFormulaPtr->setFormulaStr( GenericToolbox::Json::fetchValue<std::string>(formulaConfig, "formulaStr") );
  }
  else if( dialType == "CompiledLibDial" ){
    dialBase = std::make_unique<CompiledLibDial>();
    auto* compiledLibDialPtr{(CompiledLibDial*) dialBase.get()};

    auto formulaConfig(GenericToolbox::Json::fetchValue<JsonType>(config_, "dialConfig"));

    bool success = compiledLibDialPtr->loadLibrary( GenericToolbox::Json::fetchValue<std::string>(formulaConfig, "libraryFile") );
    if( not success ){
      LogExit("Could not load CompiledLibDial. " << GenericToolbox::Json::fetchValue(formulaConfig, "messageOnError", std::string("")) );
    }
  }
  else{ LogExit("Unknown dial type: " << dialType); }

  return dialBase.release();
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
