#include "CacheManager.h"
#include "CacheParameters.h"
#include "CacheWeights.h"
#include "WeightNormalization.h"
#include "WeightCompactSpline.h"
#include "WeightUniformSpline.h"
#include "WeightGeneralSpline.h"
#include "CacheIndexedSums.h"

#include "FitParameterSet.h"
#include "Dial.h"
#include "SplineDial.h"
#include "GraphDial.h"
#include "NormalizationDial.h"
#include "GlobalVariables.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

#include <vector>
#include <set>

#include "Logger.h"
LoggerInit([](){
  Logger::setUserHeaderStr("[Cache]");
})

Cache::Manager* Cache::Manager::fSingleton = nullptr;
std::map<const FitParameter*, int> Cache::Manager::ParameterMap;

std::string Cache::Manager::SplineType(const TSpline3* s) {
    const int points = s->GetNp();

    // Manage non-uniform splines by increasing the number of
    // points based on the smallest interval size.
    for (int i = 1; i < points-1; ++i) {
        double x;
        double y;
        s->GetKnot(i-1,x,y);
        double d1 = x;
        s->GetKnot(i,x,y);
        d1 = x - d1;
        double d2 = x;
        s->GetKnot(i+1,x,y);
        d2 = x - d2;
        if (std::abs((d1-d2)/(d1+d2)) > 1E-6) {
            return std::string("generalSpline");
        }
    }

    return std::string("uniformSpline");
}

Cache::Manager::Manager(int events, int parameters,
                        int norms,
                        int compactSplines, int compactPoints,
                        int uniformSplines, int uniformPoints,
                        int generalSplines, int generalPoints,
                        int histBins) {
    LogInfo << "Creating cache manager" << std::endl;

    fTotalBytes = 0;
    try {
        fParameterCache.reset(new Cache::Parameters(parameters));
        fTotalBytes += fParameterCache->GetResidentMemory();

        fWeightsCache.reset(
            new Cache::Weights(fParameterCache->GetParameters(),
                               fParameterCache->GetLowerClamps(),
                               fParameterCache->GetUpperClamps(),
                               events));
        fTotalBytes += fWeightsCache->GetResidentMemory();

        fNormalizations.reset(new Cache::Weight::Normalization(
                                  fWeightsCache->GetWeights(),
                                  fParameterCache->GetParameters(),
                                  norms));
        fWeightsCache->AddWeightCalculator(fNormalizations.get());
        fTotalBytes += fNormalizations->GetResidentMemory();

        fCompactSplines.reset(new Cache::Weight::CompactSpline(
                                  fWeightsCache->GetWeights(),
                                  fParameterCache->GetParameters(),
                                  fParameterCache->GetLowerClamps(),
                                  fParameterCache->GetUpperClamps(),
                                  compactSplines, compactPoints));
        fWeightsCache->AddWeightCalculator(fCompactSplines.get());
        fTotalBytes += fCompactSplines->GetResidentMemory();

        fUniformSplines.reset(new Cache::Weight::UniformSpline(
                                  fWeightsCache->GetWeights(),
                                  fParameterCache->GetParameters(),
                                  fParameterCache->GetLowerClamps(),
                                  fParameterCache->GetUpperClamps(),
                                  uniformSplines, uniformPoints));
        fWeightsCache->AddWeightCalculator(fUniformSplines.get());
        fTotalBytes += fUniformSplines->GetResidentMemory();

        fGeneralSplines.reset(new Cache::Weight::GeneralSpline(
                                  fWeightsCache->GetWeights(),
                                  fParameterCache->GetParameters(),
                                  fParameterCache->GetLowerClamps(),
                                  fParameterCache->GetUpperClamps(),
                                  generalSplines, generalPoints));
        fWeightsCache->AddWeightCalculator(fGeneralSplines.get());
        fTotalBytes += fGeneralSplines->GetResidentMemory();

        fHistogramsCache.reset(new Cache::IndexedSums(
                                  fWeightsCache->GetWeights(),
                                  histBins));
        fTotalBytes += fHistogramsCache->GetResidentMemory();

    }
    catch (...) {
        LogError << "Failed to allocate memory, so stopping" << std::endl;
        throw std::runtime_error("Not enough memory available");
    }

    LogInfo << "Approximate cache manager size for"
            << " " << events << " events:"
            << " " << GetResidentMemory()/1E+9 << " GB "
            << " (" << GetResidentMemory()/events << " bytes per event)"
            << std::endl;
}

bool Cache::Manager::Build(FitSampleSet& sampleList) {
    LogInfo << "Build the cache for Cache::Manager" << std::endl;

    int events = 0;
    int compactSplines = 0;
    int compactPoints = 0;
    int uniformSplines = 0;
    int uniformPoints = 0;
    int generalSplines = 0;
    int generalPoints = 0;
    int graphs = 0;
    int graphPoints = 0;
    int norms = 0;
    Cache::Manager::ParameterMap.clear();

    std::set<const FitParameter*> usedParameters;
    for(const FitSample& sample : sampleList.getFitSampleList() ){
        LogInfo << "Sample " << sample.getName()
                << " with " << sample.getMcContainer().eventList.size()
                << " events" << std::endl;
        for (const PhysicsEvent& event
                 : sample.getMcContainer().eventList) {
            ++events;
            if (event.getSampleBinIndex() < 0) {
                throw std::runtime_error("Caching event that isn't used");
            }
            for (const Dial* dial
                     : event.getRawDialPtrList()) {
                FitParameter* fp = static_cast<FitParameter*>(
                    dial->getAssociatedParameterReference());
                usedParameters.insert(fp);
                const SplineDial* sDial
                    = dynamic_cast<const SplineDial*>(dial);
                if (sDial) {
                    const TSpline3* s = sDial->getSplinePtr();
                    if (!s) throw std::runtime_error("Null spline pointer");
                    std::string splineType = Cache::Manager::SplineType(s);
                    if (splineType == "compactSpline") {
                        ++compactSplines;
                        compactPoints
                            += Cache::Weight::CompactSpline::FindPoints(s);
                    }
                    else if (splineType == "uniformSpline") {
                        ++uniformSplines;
                        uniformPoints
                            += Cache::Weight::UniformSpline::FindPoints(s);
                    }
                    else if (splineType == "generalSpline") {
                        ++generalSplines;
                        generalPoints
                            += Cache::Weight::GeneralSpline::FindPoints(s);
                    }
                    else {
                        LogError << "Not a valid spline type: " << splineType
                                 << std::endl;
                        throw std::runtime_error("Invalid spline type");
                    }
                }
                const GraphDial* gDial
                    = dynamic_cast<const GraphDial*>(dial);
                if (gDial) {
                    ++graphs;
                }
                const NormalizationDial* nDial
                    = dynamic_cast<const NormalizationDial*>(dial);
                if (nDial) {
                    ++norms;
                }
            }
        }
    }

    // Count the total number of histogram cells.
    int histCells = 0;
    for(const FitSample& sample : sampleList.getFitSampleList() ){
        if (!sample.getMcContainer().histogram) continue;
        int cells = sample.getMcContainer().histogram->GetNcells();
        LogInfo << "Add histogram for " << sample.getName()
                << " with " << cells
                << " cells (includes under/over-flows)" << std::endl;
        histCells += cells;
    }

    int parameters = usedParameters.size();
    LogInfo << "Cache for " << events << " events --"
            << " using " << parameters << " parameters"
            << std::endl;
    LogInfo << "    Compact splines: " << compactSplines
            << " (" << 1.0*compactSplines/events << " per event)"
            << std::endl;
    LogInfo << "    Uniform Splines: " << uniformSplines
            << " (" << 1.0*uniformSplines/events << " per event)"
            << std::endl;
    LogInfo << "    General Splines: " << generalSplines
            << " (" << 1.0*generalSplines/events << " per event)"
            << std::endl;
    LogInfo << "    Graphs: " << graphs
            << " (" << 1.0*graphs/events << " per event)"
            << std::endl;
    LogInfo << "    Normalizations: " << norms
            <<" ("<< 1.0*norms/events <<" per event)"
            << std::endl;
    LogInfo << "    Histogram bins: " << histCells
            << " (" << 1.0*events/histCells << " events per bin)"
            << std::endl;

    if (compactSplines > 0) {
        LogInfo << "    Compact spline cache uses "
                << compactPoints << " control points --"
                << " (" << 1.0*compactPoints/compactSplines
                << " points per spline)"
                << " for " << compactSplines << " splines"
                << std::endl;
    }
    if (uniformSplines > 0) {
        LogInfo << "    Uniform spline cache uses "
                << uniformPoints << " control points --"
                << " (" << 1.0*uniformPoints/uniformSplines
                << " points per spline)"
                << " for " << uniformSplines << " splines"
                << std::endl;
    }
    if (generalSplines > 0) {
        LogInfo << "    General spline cache uses "
                << generalPoints << " control points --"
                << " (" << 1.0*generalPoints/generalSplines
                << " points per spline)"
                << " for " << generalSplines << " splines"
                << std::endl;
    }
    if (graphs > 0) {
        LogInfo << "   Graph cache for " << graphPoints << " control points --"
                << " (" << 1.0*graphPoints/graphs << " points per graph)"
                << std::endl;
    }

    // Try to allocate the GPU
    if (!Cache::Manager::Get()
        && GlobalVariables::getEnableEventWeightCache()) {
        fSingleton = new Manager(events,parameters,
                                 norms,
                                 compactSplines,compactPoints,
                                 uniformSplines,uniformPoints,
                                 generalSplines,generalPoints,
                                 histCells);
    }

    // In case the cache isn't allocated (usually because it's turned off on
    // the command line).
    if (!Cache::Manager::Get()) {
        LogInfo << "Cache will not be used"
                << std::endl;
        return false;
    }

    // Add the dials to the cache.
    int usedResults = 0; // Number of cached results that have been used up.
    for(FitSample& sample : sampleList.getFitSampleList() ) {
        LogInfo << "Fill cache for " << sample.getName()
                << " with " << sample.getMcContainer().eventList.size()
                << " events" << std::endl;
        for (PhysicsEvent& event
                 : sample.getMcContainer().eventList) {
            // The reduce index to save the result for this event.
            int resultIndex = usedResults++;
            event.setResultIndex(resultIndex);
            event.setResultPointer(Cache::Manager::Get()
                                   ->GetWeightsCache()
                                   .GetResultPointer(resultIndex));
            event.setResultValidPointer(Cache::Manager::Get()
                                        ->GetWeightsCache()
                                        .GetResultValidPointer());
            Cache::Manager::Get()
                ->GetWeightsCache()
                .SetInitialValue(resultIndex,event.getTreeWeight());
            for (Dial* dial
                     : event.getRawDialPtrList()) {
                if (!dial->isReferenced()) continue;
                FitParameter* fp = static_cast<FitParameter*>(
                    dial->getAssociatedParameterReference());
                std::map<const FitParameter*,int>::iterator parMapIt
                    = Cache::Manager::ParameterMap.find(fp);
                if (parMapIt == Cache::Manager::ParameterMap.end()) {
                    Cache::Manager::ParameterMap[fp]
                        = Cache::Manager::ParameterMap.size();
                }
                int parIndex = Cache::Manager::ParameterMap[fp];
                if (dial->getUseMirrorDial()) {
                    double xLow = dial->getMirrorLowEdge();
                    double xHigh = xLow + dial->getMirrorRange();
                    Cache::Manager::Get()
                        ->GetParameterCache()
                        .SetLowerMirror(parIndex,xLow);
                    Cache::Manager::Get()
                        ->GetParameterCache()
                        .SetUpperMirror(parIndex,xHigh);
                }
                double lowerClamp = dial->getMinDialResponse();
                if (std::isfinite(lowerClamp)) {
                    Cache::Manager::Get()
                        ->GetParameterCache()
                        .SetLowerClamp(parIndex,lowerClamp);
                }
                double upperClamp = dial->getMaxDialResponse();
                if (std::isfinite(upperClamp)) {
                    Cache::Manager::Get()
                        ->GetParameterCache()
                        .SetUpperClamp(parIndex,upperClamp);
                }
                if (lowerClamp > upperClamp) {
                    throw std::runtime_error(
                        "lower and upper clamps reversed");
                }
                int dialUsed = 0;
                if(dial->getDialType() == DialType::Normalization) {
                    ++dialUsed;
                    Cache::Manager::Get()
                        ->fNormalizations
                        ->ReserveNorm(resultIndex,parIndex);
                }
                SplineDial* sDial = dynamic_cast<SplineDial*>(dial);
                if (sDial) {
                    ++dialUsed;
                    const TSpline3* s = sDial->getSplinePtr();
                    if (!s) throw std::runtime_error("Null spline pointer");
                    std::string splineType = Cache::Manager::SplineType(s);
                    int spline = -1;
                    if (splineType == "compactSpline") {
                        Cache::Manager::Get()
                            ->fCompactSplines
                            ->AddSpline(resultIndex,parIndex,sDial);
                    }
                    else if (splineType == "uniformSpline") {
                        Cache::Manager::Get()
                            ->fUniformSplines
                            ->AddSpline(resultIndex,parIndex,sDial);
                    }
                    else if (splineType == "generalSpline") {
                        Cache::Manager::Get()
                            ->fGeneralSplines
                            ->AddSpline(resultIndex,parIndex,sDial);
                    }
                    else {
                        LogError << "Not a valid spline type: " << splineType
                                 << std::endl;
                        throw std::runtime_error("Invalid spline type");
                    }
                }
                if (!dialUsed) throw std::runtime_error("Unused dial");
            }
        }
    }

    // Error checking!
    if (usedResults != Cache::Manager::Get()
        ->GetWeightsCache().GetResultCount()) {
        LogError << "Cache Manager -- used Results:     "
                 << usedResults << std::endl;
        LogError << "Cache Manager -- expected Results: "
                 << Cache::Manager::Get()->GetWeightsCache().GetResultCount()
                 << std::endl;
        throw std::runtime_error("Probable problem putting dials in cache");
    }

    // Add this histogram cells to the cache.
    int nextHist = 0;
    for(FitSample& sample : sampleList.getFitSampleList() ) {
        LogInfo << "Fill cache for " << sample.getName()
                << " with " << sample.getMcContainer().eventList.size()
                << " events" << std::endl;
        std::shared_ptr<TH1> hist(sample.getMcContainer().histogram);
        if (!hist) {
            throw std::runtime_error("missing sample histogram");
        }
        int thisHist = nextHist;
        sample.getMcContainer().setCacheManagerIndex(thisHist);
        int cells = hist->GetNcells();
        nextHist += cells;
        for (PhysicsEvent& event
                 : sample.getMcContainer().eventList) {
            int eventIndex = event.getResultIndex();
            int cellIndex = event.getSampleBinIndex();
            if (cellIndex < 0 || cells <= cellIndex) {
                throw std::runtime_error("Histogram bin out of range");
            }
            int theEntry = thisHist + cellIndex;
            Cache::Manager::Get()->GetHistogramsCache()
                .SetEventIndex(eventIndex,theEntry);
        }
    }

    if (histCells != nextHist) {
        throw std::runtime_error("Histogram cells are missing");
    }

    return true;
}

bool Cache::Manager::Fill() {
    Cache::Manager* cache = Cache::Manager::Get();
    if (!cache) return false;
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

int Cache::Manager::ParameterIndex(const FitParameter* fp) {
    std::map<const FitParameter*,int>::iterator parMapIt
        = Cache::Manager::ParameterMap.find(fp);
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
