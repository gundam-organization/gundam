#include "GraphDialBaseFactory.h"

#include "DialTypes.h"

#include <TGraph.h>

LoggerInit([]{
  Logger::setUserHeaderStr("[GraphFactory]");
});

DialBase* GraphDialBaseFactory::makeDial(const std::string& dialType_,
                                             const std::string& dialSubType_,
                                             TObject* dialInitializer_,
                                             bool useCachedDial_) {

  auto* srcGraph = dynamic_cast<TGraph*>(dialInitializer_);
  LogThrowIf(srcGraph == nullptr, "Graph dial initializer must be a TGraph");

  // Stuff the created dial into a unique_ptr, so it will be properly deleted
  // in the event of an exception.
  std::unique_ptr<DialBase> dialBase;

  if (dialSubType_ == "ROOT") {
    (useCachedDial_ ? ( dialBase = std::make_unique<GraphCache>() ) : ( dialBase = std::make_unique<Graph>() ) );
  }
  else {
    (useCachedDial_ ? ( dialBase = std::make_unique<LightGraphCache>() ) : ( dialBase = std::make_unique<LightGraph>() ) );
  }

  dialBase->buildDial(*srcGraph);

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
