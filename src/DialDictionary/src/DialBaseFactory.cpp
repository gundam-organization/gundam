#include "DialBaseFactory.h"
#include "NormDialBaseFactory.h"
#include "GraphDialBaseFactory.h"
#include "SplineDialBaseFactory.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[DialBaseFactory]");
});

DialBaseFactory::DialBaseFactory() {}
DialBaseFactory::~DialBaseFactory() {}

DialBase* DialBaseFactory::operator () (std::string dialType,
                                        std::string dialSubType,
                                        TObject* dialInitializer,
                                        bool cached) {

  // Stuff the created dial into a unique_ptr, so it will be properly deleted
  // in the event of an exception.
  std::unique_ptr<DialBase> dialBase;

  if (dialType == "Norm" || dialType == "Normalization") {
    NormDialBaseFactory factory;
    dialBase.reset(factory(dialType, dialSubType, dialInitializer,cached));
  }
  else if (dialType == "Graph") {
    GraphDialBaseFactory factory;
    dialBase.reset(factory(dialType, dialSubType, dialInitializer,cached));
  }
  else if (dialType == "Spline") {
    SplineDialBaseFactory factory;
    dialBase.reset(factory(dialType, dialSubType, dialInitializer,cached));
  }
#define INCLUDE_DEPRECATED_DIAL_TYPES
#ifdef INCLUDE_DEPRECATED_DIAL_TYPES
  else if (dialType == "MonotonicSpline") {
    LogWarning << "DEPRECATED DIAL-TYPE USED: MonotonicSpline will be removed"
            << std::endl;
    SplineDialBaseFactory factory;
    dialBase.reset(factory("Spline", "catmull-rom, monotonic",
                           dialInitializer,cached));
  }
  else if (dialType == "GeneralSpline") {
    LogWarning << "DEPRECATED DIAL-TYPE USED: GeneralSpline will be removed"
            << std::endl;
    SplineDialBaseFactory factory;
    dialBase.reset(factory("Spline", "not-a-knot", dialInitializer,cached));
  }
  else if (dialType == "SimpleSpline") {
    LogWarning << "DEPRECATED DIAL-TYPE USED: SimpleSpline will be removed"
            << std::endl;
    SplineDialBaseFactory factory;
    dialBase.reset(factory("Spline", "knot-a-knot", dialInitializer,cached));
  }
  else if (dialType == "LightGraph") {
    LogWarning << "DEPRECATED DIAL-TYPE USED: LightGraph will be removed"
            << std::endl;
    GraphDialBaseFactory factory;
    dialBase.reset(factory("Graph", "light", dialInitializer,cached));
  }
#endif

  // Pass the ownership without any constraints!
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
