#include "GraphDialBaseFactory.h"

#include "Graph.h"
#include "LightGraph.h"

#include "GraphCache.h"
#include "LightGraphCache.h"

#include <TGraph.h>

GraphDialBaseFactory::GraphDialBaseFactory() {}
GraphDialBaseFactory::~GraphDialBaseFactory() {}

DialBase* GraphDialBaseFactory::operator () (std::string dialType,
                                             std::string dialSubType,
                                             TObject* dialInitializer,
                                             bool cached) {

  TGraph* graph = dynamic_cast<TGraph*>(dialInitializer);
  LogThrowIf(!graph, "Graph dial initializer must be a TGraph");

  // Stuff the created dial into a unique_ptr, so it will be properly deleted
  // in the event of an exception.
  std::unique_ptr<DialBase> dialBase;

  if (dialSubType == "ROOT") {
    dialBase.reset((not cached) ? new Graph: new GraphCache);
  }
  else {
    dialBase.reset((not cached) ? new LightGraph: new LightGraphCache);
  }

  dialBase->buildDial(*graph);

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
