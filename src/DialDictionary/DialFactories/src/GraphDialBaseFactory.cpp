#include "GraphDialBaseFactory.h"

// Explicitly list the headers that are actually needed.  Do not include
// others.
#include "Graph.h"
#include "LightGraph.h"
#include "Shift.h"

#include <TGraph.h>

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[GraphFactory]"); });
#endif

DialBase* GraphDialBaseFactory::makeDial(const std::string& dialTitle_,
                                         const std::string& dialType_,
                                         const std::string& dialSubType_,
                                         TObject* dialInitializer_,
                                         bool useCachedDial_) {

  TGraph* srcGraph = dynamic_cast<TGraph*>(dialInitializer_);

  LogThrowIf(srcGraph == nullptr, "Graph dial initializer must be a TGraph");

  // Stuff the created dial into a unique_ptr, so it will be properly deleted
  // in the event of an exception.
  std::unique_ptr<DialBase> dialBase;

  if (dialSubType_ == "ROOT") {
    // Basic coding: Give a hint to the reader and put likely branch "first".
    // Do we really expect the cached version more than the uncached?
    dialBase = (useCachedDial_) ?
      std::make_unique<GraphCache>():
      std::make_unique<Graph>();
  }
  else if (srcGraph->GetN() < 2) {
    // For one point graph, just use a scale. Do the unique_ptr dance in case
    // there are exceptions.
    double value = srcGraph->GetY()[0];
    if (std::abs(value-1.0) < 2*std::numeric_limits<float>::epsilon()) {
      return nullptr;
    }
    dialBase = std::make_unique<Shift>();
    dialBase->buildDial(value);
    return dialBase.release();
  }
  else {
    dialBase = (useCachedDial_) ?
      std::make_unique<LightGraphCache>():
      std::make_unique<LightGraph>();
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
