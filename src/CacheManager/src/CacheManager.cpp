#include "Logger.h"

#include "CacheManager.h"
#include "CacheParameters.h"
#include "CacheWeights.h"
#include "WeightNormalization.h"
#include "WeightCompactSpline.h"
#include "WeightMonotonicSpline.h"
#include "WeightUniformSpline.h"
#include "WeightGeneralSpline.h"
#include "WeightGraph.h"
#include "CacheIndexedSums.h"

#include "ParameterSet.h"
#include "GundamGlobals.h"

#include "EventDialCache.h"
#include "Norm.h"
#include "GeneralSpline.h"
#include "UniformSpline.h"
#include "CompactSpline.h"
#include "MonotonicSpline.h"
#include "LightGraph.h"
#include "Shift.h"

#include <memory>
#include <vector>
#include <set>

LoggerInit([]{
  Logger::setUserHeaderStr("[Cache::Manager]");
});

Cache::Manager* Cache::Manager::fSingleton = nullptr;
bool Cache::Manager::fUpdateRequired = true;
std::map<const Parameter*, int> Cache::Manager::ParameterMap;

Cache::Manager::Manager(int events, int parameters,
                        int norms,
                        int compactSplines, int compactPoints,
                        int monotonicSplines, int monotonicPoints,
                        int uniformSplines, int uniformPoints,
                        int generalSplines, int generalPoints,
                        int graphs, int graphPoints,
                        int histBins, std::string spaceOption) {
    LogInfo  << "Creating cache manager" << std::endl;

    fTotalBytes = 0;
    try {
        fParameterCache = std::make_unique<Cache::Parameters>(parameters);
        fTotalBytes += fParameterCache->GetResidentMemory();

        fWeightsCache = std::make_unique<Cache::Weights>(events);
        fTotalBytes += fWeightsCache->GetResidentMemory();

        fNormalizations = std::make_unique<Cache::Weight::Normalization>(
                                  fWeightsCache->GetWeights(),
                                  fParameterCache->GetParameters(),
                                  norms);
        fWeightsCache->AddWeightCalculator(fNormalizations.get());
        fTotalBytes += fNormalizations->GetResidentMemory();

        fCompactSplines = std::make_unique<Cache::Weight::CompactSpline>(
                                  fWeightsCache->GetWeights(),
                                  fParameterCache->GetParameters(),
                                  fParameterCache->GetLowerClamps(),
                                  fParameterCache->GetUpperClamps(),
                                  compactSplines, compactPoints,
                                  spaceOption);
        fWeightsCache->AddWeightCalculator(fCompactSplines.get());
        fTotalBytes += fCompactSplines->GetResidentMemory();

        fMonotonicSplines = std::make_unique<Cache::Weight::MonotonicSpline>(
                                  fWeightsCache->GetWeights(),
                                  fParameterCache->GetParameters(),
                                  fParameterCache->GetLowerClamps(),
                                  fParameterCache->GetUpperClamps(),
                                  monotonicSplines, monotonicPoints,
                                  spaceOption);
        fWeightsCache->AddWeightCalculator(fMonotonicSplines.get());
        fTotalBytes += fMonotonicSplines->GetResidentMemory();

        fUniformSplines = std::make_unique<Cache::Weight::UniformSpline>(
                                  fWeightsCache->GetWeights(),
                                  fParameterCache->GetParameters(),
                                  fParameterCache->GetLowerClamps(),
                                  fParameterCache->GetUpperClamps(),
                                  uniformSplines, uniformPoints,
                                  spaceOption);
        fWeightsCache->AddWeightCalculator(fUniformSplines.get());
        fTotalBytes += fUniformSplines->GetResidentMemory();

        fGeneralSplines = std::make_unique<Cache::Weight::GeneralSpline>(
                                  fWeightsCache->GetWeights(),
                                  fParameterCache->GetParameters(),
                                  fParameterCache->GetLowerClamps(),
                                  fParameterCache->GetUpperClamps(),
                                  generalSplines, generalPoints,
                                  spaceOption);
        fWeightsCache->AddWeightCalculator(fGeneralSplines.get());
        fTotalBytes += fGeneralSplines->GetResidentMemory();

        fGraphs = std::make_unique<Cache::Weight::Graph>(
                                  fWeightsCache->GetWeights(),
                                  fParameterCache->GetParameters(),
                                  fParameterCache->GetLowerClamps(),
                                  fParameterCache->GetUpperClamps(),
                                  graphs, graphPoints);
        fWeightsCache->AddWeightCalculator(fGraphs.get());
        fTotalBytes += fGraphs->GetResidentMemory();
        fHistogramsCache = std::make_unique<Cache::IndexedSums>(
                                  fWeightsCache->GetWeights(),
                                  histBins);
        fTotalBytes += fHistogramsCache->GetResidentMemory();

    }
    catch (...) {
        LogError << "Failed to allocate memory, so stopping" << std::endl;
        throw std::runtime_error("Not enough memory available");
    }

    LogInfo << "Approximate cache manager size for"
            << " " << events << " events:"
            << " " << double(GetResidentMemory())/1E+9 << " GB "
            << " (" << GetResidentMemory()/events << " bytes per event)"
            << std::endl;
}

bool Cache::Manager::HasCUDA() {
    return Cache::Parameters::UsingCUDA();
}

bool Cache::Manager::Build(SampleSet& sampleList,
                           EventDialCache& eventDials) {
    LogInfo << "Build the internal caches " << std::endl;

    /// Zero everything before counting the amount of space needed for the
    /// event dials
    int events = 0;
    int compactSplines = 0;
    int compactPoints = 0;
    int monotonicSplines = 0;
    int monotonicPoints = 0;
    int uniformSplines = 0;
    int uniformPoints = 0;
    int generalSplines = 0;
    int generalPoints = 0;
    int graphs = 0;
    int graphPoints = 0;
    int norms = 0;
    int shifts = 0;
    Cache::Manager::ParameterMap.clear();

    /// Find the amount of space needed for the cache.
    std::set<const Parameter*> usedParameters;

    std::map<std::string, int> useCount;
    for (EventDialCache::CacheElem_t& elem : eventDials.getCache()) {
        if (elem.event->getSampleBinIndex() < 0) {
            throw std::runtime_error("Caching event that isn't used");
        }
        ++events;
        for (EventDialCache::DialsElem_t& dialElem : elem.dials) {
            DialInterface* dialInterface = dialElem.interface;
            // This is depending behavior that is not guarranteed, but which
            // is probably valid because of the particular usage.
            // Specifically, it depends on the vector of Parameter objects
            // not being moved.  This happens after the vectors are "closed",
            // so it is probably safe, but this isn't good.  The particular
            // usage is forced do to an API change.
            const Parameter* fp = &(dialInterface->getInputBufferRef()->getParameter());
            usedParameters.insert(fp);
            ++useCount[fp->getFullTitle()];

            DialBase* dial = dialInterface->getDialBaseRef();
            std::string dialType = dial->getDialTypeName();
            if (dialType.find("Norm") == 0) {
                ++norms;
            }
            else if (dialType.find("GeneralSpline") == 0) {
                ++generalSplines;
                generalPoints += dial->getDialData().size();
            }
            else if (dialType.find("UniformSpline") == 0) {
                ++uniformSplines;
                uniformPoints += dial->getDialData().size();
            }
            else if (dialType.find("MonotonicSpline") == 0) {
                ++monotonicSplines;
                monotonicPoints += dial->getDialData().size();
            }
            else if (dialType.find("CompactSpline") == 0) {
                ++compactSplines;
                compactPoints += dial->getDialData().size();
            }
            else if (dialType.find("LightGraph") == 0) {
                ++graphs;
                graphPoints += dial->getDialData().size();
            }
            else if (dialType.find("Shift") == 0) {
                ++shifts;
            }
            else {
                LogError << "Unsupported dial type in CacheManager -- "
                          << dialType
                          << std::endl;
                // throw std::runtime_error("unsupported dial");
            }
        }
    }

    // Count the total number of histogram cells.
    int histCells = 0;
    for(const Sample& sample : sampleList.getSampleList() ){
        if (!sample.getMcContainer().histogram) continue;
        int cells = sample.getMcContainer().histogram->GetNcells();
        LogInfo  << "Add histogram for " << sample.getName()
                << " with " << cells
                << " cells (includes under/over-flows)" << std::endl;
        histCells += cells;
    }

    /// Summarize the space and get the cache memory.
    int parameters = int(usedParameters.size());
    LogInfo  << "Cache for " << events << " events --"
            << " using " << parameters << " parameters"
            << std::endl;
    LogInfo  << "    Compact splines: " << compactSplines
            << " (" << 1.0*compactSplines/events << " per event)"
            << std::endl;
    LogInfo  << "    Monotonic splines: " << monotonicSplines
            << " (" << 1.0*monotonicSplines/events << " per event)"
            << std::endl;
    LogInfo  << "    Uniform Splines: " << uniformSplines
            << " (" << 1.0*uniformSplines/events << " per event)"
            << std::endl;
    LogInfo  << "    General Splines: " << generalSplines
            << " (" << 1.0*generalSplines/events << " per event)"
            << std::endl;
    LogInfo  << "    Graphs: " << graphs
            << " (" << 1.0*graphs/events << " per event)"
            << std::endl;
    LogInfo  << "    Normalizations: " << norms
            <<" ("<< 1.0*norms/events <<" per event)"
            << std::endl;
    LogInfo  << "    Shifts: " << shifts
            <<" ("<< 1.0*shifts/events <<" per event)"
            << std::endl;
    LogInfo  << "    Histogram bins: " << histCells
            << " (" << 1.0*events/histCells << " events per bin)"
            << std::endl;

    if (compactSplines > 0) {
        LogInfo  << "    Compact spline cache uses "
                << compactPoints << " control points --"
                << " (" << 1.0*compactPoints/compactSplines
                << " points per spline)"
                << " for " << compactSplines << " splines"
                << std::endl;
    }
    if (monotonicSplines > 0) {
        LogInfo  << "    Monotonic spline cache uses "
                << monotonicPoints << " control points --"
                << " (" << 1.0*monotonicPoints/monotonicSplines
                << " points per spline)"
                << " for " << monotonicSplines << " splines"
                << std::endl;
    }
    if (uniformSplines > 0) {
        LogInfo  << "    Uniform spline cache uses "
                << uniformPoints << " control points --"
                << " (" << 1.0*uniformPoints/uniformSplines
                << " points per spline)"
                << " for " << uniformSplines << " splines"
                << std::endl;
    }
    if (generalSplines > 0) {
        LogInfo  << "    General spline cache uses "
                << generalPoints << " control points --"
                << " (" << 1.0*generalPoints/generalSplines
                << " points per spline)"
                << " for " << generalSplines << " splines"
                << std::endl;
    }
    if (graphs > 0) {
        LogInfo  << "    Graph cache uses "
                << graphPoints << " control points --"
                << " (" << 1.0*graphPoints/graphs << " points per graph)"
                << std::endl;
    }

    // Try to allocate the Cache::Manager memory (including for the GPU if
    // it's being used).
    if (!Cache::Manager::Get()
        && GundamGlobals::getEnableCacheManager()) {
        LogInfo << "Creating the Cache::Manager" << std::endl;
        if (!Cache::Manager::HasCUDA()) {
            LogInfo << "    GPU Not enabled with Cache::Manager"
                      << std::endl;
        }
        fSingleton = new Manager(events,parameters,
                                 norms,
                                 compactSplines,compactPoints,
                                 monotonicSplines,monotonicPoints,
                                 uniformSplines,uniformPoints,
                                 generalSplines,generalPoints,
                                 graphs, graphPoints,
                                 histCells,
                                 "space");
    }

    // In case the cache isn't allocated (usually because it's turned off on
    // the command line), but this is a safety check.
    if (!Cache::Manager::Get()) {
        LogWarning << "Cache will not be used"
                   << std::endl;
        return false;
    }

    Cache::Manager::UpdateRequired();

    // return Update(sampleList, eventDials);
    return true;
}

void Cache::Manager::UpdateRequired() {
    fUpdateRequired = true;
}


bool Cache::Manager::Update(SampleSet& sampleList,
                            EventDialCache& eventDials) {
    if (not fUpdateRequired) return true;

    // This is the updated that is required!
    fUpdateRequired = false;

    // In case the cache isn't allocated (usually because it's turned off on
    // the command line), but this is a safety check.
    if (!Cache::Manager::Get()) {
        LogWarning << "Cache will not be used"
                   << std::endl;
        return false;
    }

    LogInfo << "Update the internal caches" << std::endl;

    // Initialize the internal caches so they are in the default state.
    Cache::Manager::Get()->GetParameterCache().Reset();
    Cache::Manager::Get()->GetHistogramsCache().Reset();
    Cache::Manager::Get()->GetWeightsCache().Reset();

    int usedResults = 0;

    // Add the dials in the EventDialCache to the internal cache.
    for (EventDialCache::CacheElem_t& elem : eventDials.getCache()) {
        // Skip events that are not in a bin.
        if (elem.event->getSampleBinIndex() < 0) continue;
        PhysicsEvent& event = *elem.event;
        // The reduce index.  This is where to save the results for this
        // event in the cache.
        int resultIndex = usedResults++;

        event.setCacheManagerIndex(resultIndex);
        event.setCacheManagerValuePointer(Cache::Manager::Get()
                                          ->GetWeightsCache()
                                          .GetResultPointer(resultIndex));
        event.setCacheManagerValidPointer(Cache::Manager::Get()
                                          ->GetWeightsCache()
                                          .GetResultValidPointer());
        event.setCacheManagerUpdatePointer(
            [](){Cache::Manager::Get()->GetWeightsCache().GetResult(0);});

        // Get the initial value for this event and save it.
        double initialEventWeight = event.getBaseWeight();

        // Add each dial for the event to the GPU caches.
        for (EventDialCache::DialsElem_t& dialElem : elem.dials) {
            DialInterface* dialInterface = dialElem.interface;
            DialInputBuffer* dialInputs = dialInterface->getInputBufferRef();

            // Check if this dial is used at all.
            if (dialInputs->isMasked()) continue;

            // Make sure all of the used parameters are in the parameter
            // map.
            for (std::size_t i = 0; i < dialInputs->getBufferSize(); ++i) {
                // Find the index (or allocate a new one) for the dial
                // parameter.  This only works for 1D dials.
                const Parameter* fp
                    = &(dialInterface->getInputBufferRef()
                        ->getParameter(i));
                auto parMapIt = Cache::Manager::ParameterMap.find(fp);
                if (parMapIt == Cache::Manager::ParameterMap.end()) {
                    Cache::Manager::ParameterMap[fp]
                        = int(Cache::Manager::ParameterMap.size());
                }
            }

            // Apply the mirroring for the parameters
            if (dialInputs->useParameterMirroring()) {
                for (std::size_t i = 0; i < dialInputs->getBufferSize(); ++i) {
                    const Parameter* fp = &(dialInputs->getParameter(i));
                    const std::pair<double,double>& bounds =
                        dialInputs->getMirrorBounds(i);
                    int parIndex = Cache::Manager::ParameterMap[fp];
                    Cache::Manager::Get()->GetParameterCache()
                        .SetLowerMirror(parIndex,bounds.first);
                    Cache::Manager::Get()->GetParameterCache()
                        .SetUpperMirror(parIndex,bounds.first+bounds.second);
                }
            }

            // Apply the clamps to the parameter range
            for (std::size_t i = 0; i < dialInputs->getBufferSize(); ++i) {
                const Parameter* fp = &(dialInputs->getParameter(i));
                const DialResponseSupervisor* resp
                    = dialInterface->getResponseSupervisorRef();
                int parIndex = Cache::Manager::ParameterMap[fp];
                double minResponse = 0.0;
                if (std::isfinite(resp->getMinResponse())) {
                    minResponse = resp->getMinResponse();
                }
                Cache::Manager::Get()->GetParameterCache()
                    .SetLowerClamp(parIndex,minResponse);
                if (not std::isfinite(resp->getMaxResponse())) continue;
                Cache::Manager::Get()->GetParameterCache()
                    .SetUpperClamp(parIndex,resp->getMaxResponse());
            }

            // Add the dial information to the appropriate caches
            int dialUsed = 0;
            const DialBase* baseDial = dialInterface->getDialBaseRef();
            const Norm* normDial = dynamic_cast<const Norm*>(baseDial);
            if (normDial) {
                ++dialUsed;
                const Parameter* fp = &(dialInputs->getParameter(0));
                int parIndex = Cache::Manager::ParameterMap[fp];
                Cache::Manager::Get()
                    ->fNormalizations
                    ->ReserveNorm(resultIndex,parIndex);
            }
            const CompactSpline* compactSpline
                = dynamic_cast<const CompactSpline*>(baseDial);
            if (compactSpline) {
                ++dialUsed;
                const Parameter* fp = &(dialInputs->getParameter(0));
                int parIndex = Cache::Manager::ParameterMap[fp];
                Cache::Manager::Get()
                    ->fCompactSplines
                    ->AddSpline(resultIndex,parIndex,
                                baseDial->getDialData());
            }
            const MonotonicSpline* monotonicSpline
                = dynamic_cast<const MonotonicSpline*>(baseDial);
            if (monotonicSpline) {
                ++dialUsed;
                const Parameter* fp = &(dialInputs->getParameter(0));
                int parIndex = Cache::Manager::ParameterMap[fp];
                Cache::Manager::Get()
                    ->fMonotonicSplines
                    ->AddSpline(resultIndex,parIndex,
                                baseDial->getDialData());
            }
            const UniformSpline* uniformSpline
                = dynamic_cast<const UniformSpline*>(baseDial);
            if (uniformSpline) {
                ++dialUsed;
                const Parameter* fp = &(dialInputs->getParameter(0));
                int parIndex = Cache::Manager::ParameterMap[fp];
                Cache::Manager::Get()
                    ->fUniformSplines
                    ->AddSpline(resultIndex,parIndex,
                                baseDial->getDialData());
            }
            const GeneralSpline* generalSpline
                = dynamic_cast<const GeneralSpline*>(baseDial);
            if (generalSpline) {
                ++dialUsed;
                const Parameter* fp = &(dialInputs->getParameter(0));
                int parIndex = Cache::Manager::ParameterMap[fp];
                Cache::Manager::Get()
                    ->fGeneralSplines
                    ->AddSpline(resultIndex,parIndex,
                                baseDial->getDialData());
            }
            const LightGraph* lightGraph
                = dynamic_cast<const LightGraph*>(baseDial);
            if (lightGraph) {
                ++dialUsed;
                const Parameter* fp = &(dialInputs->getParameter(0));
                int parIndex = Cache::Manager::ParameterMap[fp];
                Cache::Manager::Get()
                    ->fGraphs
                    ->AddGraph(resultIndex,parIndex,
                               baseDial->getDialData());
            }
            const Shift* shift
                = dynamic_cast<const Shift*>(baseDial);
            if (shift) {
                ++dialUsed;
                initialEventWeight *= shift->evalResponse(DialInputBuffer());
            }

            if (dialUsed != 1) {
                LogError << "Problem with dial: " << dialUsed
                          << std::endl;
                LogError << "Dial Type Name: "
                          << baseDial->getDialTypeName()
                          << std::endl;
                // std::runtime_error("Dial use problem");
            }
        }

        // Set the initial weight for the event.  This is done here since the
        // raw tree weight may get rescaled by "Shift" dials
        Cache::Manager::Get()
            ->GetWeightsCache()
            .SetInitialValue(resultIndex,initialEventWeight);

    }

    LogInfo << "Error checking for cache" << std::endl;

    // Error checking adding the dials to the cache!
    if (usedResults != Cache::Manager::Get()
        ->GetWeightsCache().GetResultCount()) {
        LogError << "Cache Manager -- used Results:     "
                 << usedResults << std::endl;
        LogError << "Cache Manager -- expected Results: "
                 << Cache::Manager::Get()->GetWeightsCache().GetResultCount()
                 << std::endl;
        // throw std::runtime_error("Probable problem putting dials in cache");
    }

    // Add the histogram cells to the cache.  THIS CODE IS SUSPECT!!!!
    LogInfo << "Add this histogram cells to the cache." << std::endl;
    int nextHist = 0;
    for(Sample& sample : sampleList.getSampleList() ) {
        LogInfo  << "Fill cache for " << sample.getName()
                << " with " << sample.getMcContainer().eventList.size()
                << " events" << std::endl;
        std::shared_ptr<TH1> hist(sample.getMcContainer().histogram);
        if (!hist) {
            throw std::runtime_error("missing sample histogram");
        }
        int thisHist = nextHist;
        sample.getMcContainer().setCacheManagerIndex(thisHist);
        sample.getMcContainer().setCacheManagerValuePointer(
            Cache::Manager::Get()->GetHistogramsCache()
            .GetSumsPointer());
        sample.getMcContainer().setCacheManagerValue2Pointer(
            Cache::Manager::Get()->GetHistogramsCache()
            .GetSums2Pointer());
        sample.getMcContainer().setCacheManagerValidPointer(
            Cache::Manager::Get()->GetHistogramsCache()
            .GetSumsValidPointer());
        sample.getMcContainer().setCacheManagerUpdatePointer(
            [](){
                Cache::Manager::Get()->GetHistogramsCache().GetSum(0);
                Cache::Manager::Get()->GetHistogramsCache().GetSum2(0);
            });
        int cells = hist->GetNcells();
        nextHist += cells;
        /// ARE ALL OF THE EVENTS HANDLED?
        for (PhysicsEvent& event
                 : sample.getMcContainer().eventList) {
            int eventIndex = event.getCacheManagerIndex();
            int cellIndex = event.getSampleBinIndex();
            if (cellIndex < 0 || cells <= cellIndex) {
                throw std::runtime_error("Histogram bin out of range");
            }
            int theEntry = thisHist + cellIndex;
            Cache::Manager::Get()->GetHistogramsCache()
                .SetEventIndex(eventIndex,theEntry);
        }
    }

    if (Cache::Manager::Get()->GetHistogramsCache().GetSumCount()
        != nextHist) {
        throw std::runtime_error("Histogram cells are missing");
    }

    return true;
}

bool Cache::Manager::Fill() {
    Cache::Manager* cache = Cache::Manager::Get();
    if (!cache) return false;
    if (fUpdateRequired) {
        LogError << "Fill while an update is required" << std::endl;
        LogThrow("Fill while an update is required");
    }
#define DUMP_FILL_INPUT_PARAMETERS
#ifdef DUMP_FILL_INPUT_PARAMETERS
    do {
        static bool printed = false;
        if (printed) break;
        printed = true;
        for (auto& par : Cache::Manager::ParameterMap ) {
            // This produces a crazy amount of output.
            LogInfo  << "FILL: " << par.second
                    << "/" << Cache::Manager::ParameterMap.size()
                    << " " << par.first->isEnabled()
                    << " " << par.first->getParameterValue()
                    << " (" << par.first->getFullTitle() << ")"
                    << std::endl;
        }
    } while(false);
#endif
    for (auto& par : Cache::Manager::ParameterMap ) {
        cache->GetParameterCache().SetParameter(
            par.second, par.first->getParameterValue());
    }
    cache->GetWeightsCache().Apply();
    cache->GetHistogramsCache().Apply();

#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning CACHE_MANAGER_SLOW_VALIDATION in Cache::Manager::Fill()
    // Returning false means that the event weights will also be calculated
    // using the CPU.
    return false;
#endif
    return true;
}

int Cache::Manager::ParameterIndex(const Parameter* fp) {
    auto parMapIt = Cache::Manager::ParameterMap.find(fp);
    if (parMapIt == Cache::Manager::ParameterMap.end()) return -1;
    return parMapIt->second;
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
