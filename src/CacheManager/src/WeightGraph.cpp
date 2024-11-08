#include "CacheWeights.h"
#include "WeightBase.h"
#include "WeightGraph.h"

#include <algorithm>
#include <iostream>
#include <exception>
#include <limits>
#include <cmath>

#include <hemi/hemi_error.h>
#include <hemi/launch.h>
#include <hemi/grid_stride_range.h>

#include "Logger.h"
#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[Cache::Weight::Graph]"); });
#endif

// The constructor
Cache::Weight::Graph::Graph(
    Cache::Weights::Results& weights,
    Cache::Parameters::Values& parameters,
    Cache::Parameters::Clamps& lowerClamps,
    Cache::Parameters::Clamps& upperClamps,
    std::size_t graphs, std::size_t space)
    : Cache::Weight::Base("graph",weights,parameters),
      fLowerClamp(lowerClamps), fUpperClamp(upperClamps),
      fGraphsReserved(graphs), fGraphsUsed(0),
      fGraphSpaceReserved(space), fGraphSpaceUsed(0) {

  LogInfo << "Reserved " << GetName() << " Graphs: "
          << GetGraphsReserved() << std::endl;
  if (GetGraphsReserved() < 1) return;

  fTotalBytes += GetGraphsReserved()*sizeof(int);      // fGraphResult
  fTotalBytes += GetGraphsReserved()*sizeof(short);    // fGraphParameter
  fTotalBytes += (1+GetGraphsReserved())*sizeof(int);  // fGraphIndex

  LogInfo << "Reserved " << GetName()
          << " Graph Data: " << GetGraphSpaceReserved()
          << std::endl;

  fTotalBytes += GetGraphSpaceReserved()*sizeof(WEIGHT_BUFFER_FLOAT);

  LogInfo << "Approximate Memory Size for " << GetName()
          << ": " << fTotalBytes/1E+9
          << " GB" << std::endl;

  try {
    // Get the CPU/GPU memory for the graph index tables.  These are
    // copied once during initialization so do not pin the CPU memory into
    // the page set.
    fGraphResult.reset(new hemi::Array<int>(GetGraphsReserved(),false));
    LogExitIf(not fGraphResult, "GraphResult not allocated");
    fGraphParameter.reset(
        new hemi::Array<short>(GetGraphsReserved(),false));
    LogExitIf(not fGraphParameter, "GraphParameter not allocated");
    fGraphIndex.reset(new hemi::Array<int>(1+GetGraphsReserved(),false));
    LogExitIf(not fGraphIndex, "GraphIndex not allocated");

    // Get the CPU/GPU memory for the graph space.  This is copied once
    // during initialization so do not pin the CPU memory into the page
    // set.
    fGraphSpace.reset(
        new hemi::Array<WEIGHT_BUFFER_FLOAT>(GetGraphSpaceReserved(),false));
    LogExitIf(not fGraphSpace, "GraphSpace not allocated");
  }
  catch (...) {
    LogError << "Uncaught exception in WeightGraph" << std::endl;
    LogExit("WeightGraph -- uncaught exception");
  }

  // Initialize the caches.  Don't try to zero everything since the
  // caches can be huge.
  Reset();
  fGraphIndex->hostPtr()[0] = 0;
}

// The destructor
Cache::Weight::Graph::~Graph() {}

void Cache::Weight::Graph::AddGraph(int resIndex, int parIndex,
                                    const std::vector<double>& graphData) {
  if (resIndex < 0) {
    LogError << "Invalid result index"
             << std::endl;
    LogExit("Negative result index");
  }
  if (fWeights.size() <= resIndex) {
    LogError << "Invalid result index"
             << std::endl;
    LogExit("Result index out of bounds");
  }
  if (parIndex < 0) {
    LogError << "Invalid parameter index"
             << std::endl;
    LogExit("Negative parameter index");
  }
  if (fParameters.size() <= parIndex) {
    LogError << "Invalid parameter index " << parIndex
             << std::endl;
    LogExit("Parameter index out of bounds");
  }
  if (graphData.size() < 2) {
    LogError << "Insufficient points in graph " << graphData.size()
             << std::endl;
    LogExit("Invalid number of graph points");
  }
  int knots = (graphData.size())/2;
  if (15 < knots) {
    LogError << "Up to 15 knots supported by Graph " << knots
             << std::endl;
    LogExit("Invalid number of graph points");
  }

  int newIndex = fGraphsUsed++;
  if (fGraphsUsed > fGraphsReserved) {
    LogError << "Not enough space reserved for graphs"
             << std::endl;
    LogExit("Not enough space reserved for graphs");
  }
  fGraphResult->hostPtr()[newIndex] = resIndex;
  fGraphParameter->hostPtr()[newIndex] = parIndex;
  if (fGraphIndex->hostPtr()[newIndex] != fGraphSpaceUsed) {
    LogError << "Last graph knot index should be at old end of graphs"
             << std::endl;
    LogExit("Problem with control indices");
  }
  int knotIndex = fGraphSpaceUsed;
  fGraphSpaceUsed += graphData.size();
  if (fGraphSpaceUsed > fGraphSpaceReserved) {
    LogError << "Not enough space reserved for graph space"
             << std::endl;
    LogExit("Not enough space reserved for graph space");
  }
  fGraphIndex->hostPtr()[newIndex+1] = fGraphSpaceUsed;
  for (std::size_t i = 0; i<graphData.size(); ++i) {
    fGraphSpace->hostPtr()[knotIndex+i] = graphData.at(i);
  }

}

int Cache::Weight::Graph::GetGraphParameterIndex(int sIndex) {
  if (sIndex < 0) {
    LogExit("Graph index invalid");
  }
  if (GetGraphsUsed() <= sIndex) {
    LogExit("Graph index invalid");
  }
  return fGraphParameter->hostPtr()[sIndex];
}

double Cache::Weight::Graph::GetGraphParameter(int sIndex) {
  int i = GetGraphParameterIndex(sIndex);
  if (i<0) {
    LogExit("Graph parameter index out of bounds");
  }
  if (fParameters.size() <= i) {
    LogExit("Graph parameter index out of bounds");
  }
  return fParameters.hostPtr()[i];
}

int Cache::Weight::Graph::GetGraphKnotCount(int sIndex) {
  if (sIndex < 0) {
    LogExit("Graph index invalid");
  }
  if (GetGraphsUsed() <= sIndex) {
    LogExit("Graph index invalid");
  }
  int k = fGraphIndex->hostPtr()[sIndex+1]-fGraphIndex->hostPtr()[sIndex];
  return k/2;
}

double Cache::Weight::Graph::GetGraphLowerBound(int sIndex) {
  if (sIndex < 0) {
    LogExit("Graph index invalid");
  }
  if (GetGraphsUsed() <= sIndex) {
    LogExit("Graph index invalid");
  }
  int spaceIndex = fGraphIndex->hostPtr()[sIndex];
  return fGraphSpace->hostPtr()[spaceIndex];
}

double Cache::Weight::Graph::GetGraphUpperBound(int sIndex) {
  if (sIndex < 0) {
    LogExit("Graph index invalid");
  }
  if (GetGraphsUsed() <= sIndex) {
    LogExit("Graph index invalid");
  }
  int knotCount = GetGraphKnotCount(sIndex);
  double lower = GetGraphLowerBound(sIndex);
  int spaceIndex = fGraphIndex->hostPtr()[sIndex];
  double step = fGraphSpace->hostPtr()[spaceIndex+1];
  return lower + (knotCount-1)/step;
}

double Cache::Weight::Graph::GetGraphLowerClamp(int sIndex) {
  int i = GetGraphParameterIndex(sIndex);
  if (i<0) {
    LogExit("Graph lower clamp index out of bounds");
  }
  if (fLowerClamp.size() <= i) {
    LogExit("Graph lower clamp index out of bounds");
  }
  return fLowerClamp.hostPtr()[i];
}

double Cache::Weight::Graph::GetGraphUpperClamp(int sIndex) {
  int i = GetGraphParameterIndex(sIndex);
  if (i<0) {
    LogExit("Graph upper clamp index out of bounds");
  }
  if (fUpperClamp.size() <= i) {
    LogExit("Graph upper clamp index out of bounds");
  }
  return fUpperClamp.hostPtr()[i];
}

double Cache::Weight::Graph::GetGraphKnotValue(int sIndex, int knot) {
  if (sIndex < 0) {
    LogExit("Graph index invalid");
  }
  if (GetGraphsUsed() <= sIndex) {
    LogExit("Graph index invalid");
  }
  int spaceIndex = fGraphIndex->hostPtr()[sIndex];
  int count = GetGraphKnotCount(sIndex);
  if (knot < 0) {
    LogExit("Knot index invalid");
  }
  if (count <= knot) {
    LogExit("Knot index invalid");
  }
  return fGraphSpace->hostPtr()[spaceIndex+2+2*knot];
}

double Cache::Weight::Graph::GetGraphKnotPlace(int sIndex, int knot) {
  if (sIndex < 0) {
    LogExit("Graph index invalid");
  }
  if (GetGraphsUsed() <= sIndex) {
    LogExit("Graph index invalid");
  }
  int spaceIndex = fGraphIndex->hostPtr()[sIndex];
  int count = GetGraphKnotCount(sIndex);
  if (knot < 0) {
    LogExit("Knot index invalid");
  }
  if (count <= knot) {
    LogExit("Knot index invalid");
  }
  return fGraphSpace->hostPtr()[spaceIndex+2+2*knot+1];
}

#include "CalculateGraph.h"
#include "CacheAtomicMult.h"

namespace {

  // A function to be used as the kernel on either the CPU or GPU.  This
  // must be valid CUDA coda.
  HEMI_KERNEL_FUNCTION(HEMIGraphsKernel,
                       double* results,
                       const double* params,
                       const double* lowerClamp,
                       const double* upperClamp,
                       const WEIGHT_BUFFER_FLOAT* space,
                       const int* rIndex,
                       const short* pIndex,
                       const int* sIndex,
                       const int NP) {
    for (int i : hemi::grid_stride_range(0,NP)) {
      const int id0 = sIndex[i];
      const int id1 = sIndex[i+1];
      const int dim = id1-id0;
      const double x = params[pIndex[i]];
      const double lClamp = lowerClamp[pIndex[i]];
      const double uClamp = upperClamp[pIndex[i]];

#ifndef HEMI_DEV_CODE
      if (dim>15) std::runtime_error("To many bins in graph");
#endif

      double v = CalculateGraph(x, lClamp,uClamp,&space[id0],dim);

      CacheAtomicMult(&results[rIndex[i]], v);
    }
  }
}

void Cache::Weight::Graph::Reset() {
  // Use the parent reset.
  Cache::Weight::Base::Reset();
  // Reset this class
  fGraphsUsed = 0;
  fGraphSpaceUsed = 0;
}

bool Cache::Weight::Graph::Apply() {
  if (GetGraphsUsed() < 1) return false;

  HEMIGraphsKernel graphsKernel;
  hemi::launch(graphsKernel,
               fWeights.writeOnlyPtr(),
               fParameters.readOnlyPtr(),
               fLowerClamp.readOnlyPtr(),
               fUpperClamp.readOnlyPtr(),
               fGraphSpace->readOnlyPtr(),
               fGraphResult->readOnlyPtr(),
               fGraphParameter->readOnlyPtr(),
               fGraphIndex->readOnlyPtr(),
               GetGraphsUsed()
  );

  return true;
}

// An MIT Style License

// Copyright (c) 2022 Clark McGrew

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
// c-basic-offset:4
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
