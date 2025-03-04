#include "Logger.h"

// Check if global variables need to be defined.
#ifndef __cpp_inline_variables
#define HEMI_COMPILE_DEFINITIONS
#include <hemi/hemi.h>
#endif

#include "CacheManager.h"

#include "CacheParameters.h"
#include "CacheWeights.h"

#include "WeightNormalization.h"
#include "WeightCompactSpline.h"
#include "WeightMonotonicSpline.h"
#include "WeightUniformSpline.h"
#include "WeightGeneralSpline.h"
#include "WeightGraph.h"
#include "WeightBilinear.h"
#include "WeightBicubic.h"
#include "WeightTabulated.h"

#include "ParameterSet.h"
#include "GundamGlobals.h"

#include "EventDialCache.h"
#include "Norm.h"
#include "GeneralSpline.h"
#include "UniformSpline.h"
#include "CompactSpline.h"
#include "MonotonicSpline.h"
#include "LightGraph.h"
#include "Bilinear.h"
#include "Bicubic.h"
#include "Shift.h"
#include "Tabulated.h"

#include <memory>
#include <set>


// static definitions
Cache::Manager::Parameters Cache::Manager::fParameters{};

Cache::Manager::Manager(const Cache::Manager::Configuration& config) {
  LogInfo  << "Creating cache manager" << std::endl;

  fTotalBytes = 0;
  try {
    fParameterCache = std::make_unique<Cache::Parameters>(config.parameters);
    LogThrowIf(not fParameterCache, "Bad ParameterCache alloc");
    fTotalBytes += fParameterCache->GetResidentMemory();

    fWeightsCache = std::make_unique<Cache::Weights>(config.events);
    LogThrowIf(not fWeightsCache, "Bad WeightsCache alloc");
    fTotalBytes += fWeightsCache->GetResidentMemory();

    fNormalizations = std::make_unique<Cache::Weight::Normalization>(
        fWeightsCache->GetWeights(),
        fParameterCache->GetParameters(),
        config.norms);
    LogThrowIf(not fNormalizations, "Bad Normalizations alloc");
    fWeightsCache->AddWeightCalculator(fNormalizations.get());
    fTotalBytes += fNormalizations->GetResidentMemory();

    fCompactSplines = std::make_unique<Cache::Weight::CompactSpline>(
        fWeightsCache->GetWeights(),
        fParameterCache->GetParameters(),
        fParameterCache->GetLowerClamps(),
        fParameterCache->GetUpperClamps(),
        config.compactSplines, config.compactPoints,
        config.spaceOption);
    LogThrowIf(not fCompactSplines, "Bad CompactSplines alloc");
    fWeightsCache->AddWeightCalculator(fCompactSplines.get());
    fTotalBytes += fCompactSplines->GetResidentMemory();

    fMonotonicSplines = std::make_unique<Cache::Weight::MonotonicSpline>(
        fWeightsCache->GetWeights(),
        fParameterCache->GetParameters(),
        fParameterCache->GetLowerClamps(),
        fParameterCache->GetUpperClamps(),
        config.monotonicSplines, config.monotonicPoints,
        config.spaceOption);
    LogThrowIf(not fMonotonicSplines, "Bad MonotonicSplines alloc");
    fWeightsCache->AddWeightCalculator(fMonotonicSplines.get());
    fTotalBytes += fMonotonicSplines->GetResidentMemory();

    fUniformSplines = std::make_unique<Cache::Weight::UniformSpline>(
        fWeightsCache->GetWeights(),
        fParameterCache->GetParameters(),
        fParameterCache->GetLowerClamps(),
        fParameterCache->GetUpperClamps(),
        config.uniformSplines, config.uniformPoints,
        config.spaceOption);
    LogThrowIf(not fUniformSplines, "Bad UniformSplines alloc");
    fWeightsCache->AddWeightCalculator(fUniformSplines.get());
    fTotalBytes += fUniformSplines->GetResidentMemory();

    fGeneralSplines = std::make_unique<Cache::Weight::GeneralSpline>(
        fWeightsCache->GetWeights(),
        fParameterCache->GetParameters(),
        fParameterCache->GetLowerClamps(),
        fParameterCache->GetUpperClamps(),
        config.generalSplines, config.generalPoints,
        config.spaceOption);
    LogThrowIf(not fGeneralSplines, "Bad GeneralSplines alloc");
    fWeightsCache->AddWeightCalculator(fGeneralSplines.get());
    fTotalBytes += fGeneralSplines->GetResidentMemory();

    fGraphs = std::make_unique<Cache::Weight::Graph>(
        fWeightsCache->GetWeights(),
        fParameterCache->GetParameters(),
        fParameterCache->GetLowerClamps(),
        fParameterCache->GetUpperClamps(),
        config.graphs, config.graphPoints);
    LogThrowIf(not fGraphs, "Bad Graphs alloc");
    fWeightsCache->AddWeightCalculator(fGraphs.get());
    fTotalBytes += fGraphs->GetResidentMemory();

    fBilinear = std::make_unique<Cache::Weight::Bilinear>(
        fWeightsCache->GetWeights(),
        fParameterCache->GetParameters(),
        fParameterCache->GetLowerClamps(),
        fParameterCache->GetUpperClamps(),
        config.bilinear, config.bilinearPoints);
    LogThrowIf(not fBilinear, "Bad Bilinear alloc");
    fWeightsCache->AddWeightCalculator(fBilinear.get());
    fTotalBytes += fBilinear->GetResidentMemory();

    fBicubic = std::make_unique<Cache::Weight::Bicubic>(
        fWeightsCache->GetWeights(),
        fParameterCache->GetParameters(),
        fParameterCache->GetLowerClamps(),
        fParameterCache->GetUpperClamps(),
        config.bicubic, config.bicubicPoints);
    LogThrowIf(not fBicubic, "Bad Bicubic alloc");
    fWeightsCache->AddWeightCalculator(fBicubic.get());
    fTotalBytes += fBicubic->GetResidentMemory();

    fTabulated = std::make_unique<Cache::Weight::Tabulated>(
        fWeightsCache->GetWeights(),
        fParameterCache->GetParameters(),
        config.tabulated,
        config.tabulatedPoints,
        config.tables);
    LogThrowIf(not fTabulated, "Bad Tabulated alloc");
    fWeightsCache->AddWeightCalculator(fTabulated.get());
    fTotalBytes += fTabulated->GetResidentMemory();

    fHistogramsCache = std::make_unique<Cache::HistogramSum>(
        fWeightsCache->GetWeights(),
        config.histBins);
    LogThrowIf(not fHistogramsCache, "Bad HistogramsCache alloc");
    fTotalBytes += fHistogramsCache->GetResidentMemory();

  }
  catch (...) {
    LogError << "Failed to allocate memory, so stopping" << std::endl;
    LogThrow("Not enough memory available");
  }

  LogInfo << "Approximate cache manager size for"
          << " " << config.events << " events:"
          << " " << double(GetResidentMemory())/1E+9 << " GB "
          << " (" << GetResidentMemory()/config.events << " bytes per event)"
          << std::endl;
}

bool Cache::Manager::HasCUDA() {
  return Cache::Parameters::UsingCUDA();
}

bool Cache::Manager::HasGPU(bool dump) {
  return Cache::Parameters::HasGPU(dump);
}

bool Cache::Manager::Build(SampleSet& sampleSet,
                           EventDialCache& eventDialCache) {
  if (not IsCacheManagerEnabled()) return false;

  // Make sure that the Cache::Manager hasn't already been built.  Rebuilding
  // isn't supported since we are talking about GB blocks of memory on the GPU
  // and this should only be used with static initialization.
  LogThrowIf(Cache::Manager::Get() != nullptr,
             "Overwriting Cache::Manager");

  LogInfo << "Build the internal caches " << std::endl;

  // Create a "config" variable to hold the configuration that will be used
  // to create the Cache::Manager
  Cache::Manager::Configuration config;

  // Make sure that the parameter map is empty (it should already be empty,
  // but be sure).  The map is between the address of the parameter, and the
  // index of the parameter and is needed since the dials only contain a
  // pointer to the parameter.
  fParameters.ParameterMap.clear();

  /// Keep track of which parameters are used.  This also provides a count
  /// of the parameters.
  std::set<const Parameter*> usedParameters;

  int dialErrorCount = 0;     // This should *stay* zero.
  std::map<std::string, int> useCount;
  for (EventDialCache::CacheEntry& elem : eventDialCache.getCache()) {
    if (elem.event->getIndices().bin < 0) {
      LogThrow("Caching event that isn't used");
    }
    ++config.events;
    for( auto& dialResponseCache : elem.dialResponseCacheList) {
      // This is depending behavior that is not guarranteed, but which
      // is probably valid because of the particular usage.
      // Specifically, it depends on the vector of Parameter objects
      // not being moved.  This happens after the vectors are "closed",
      // so it is probably safe, but this isn't good.  The particular
      // usage is forced do to an API change.
      // Make sure all of the used parameters are in the parameter
      // map.
      for (std::size_t i = 0; i < dialResponseCache.dialInterface->getInputBufferRef()->getBufferSize(); ++i) {
        const Parameter* fp = &(dialResponseCache.dialInterface->getInputBufferRef()->getParameter(i));
        usedParameters.insert(fp);
        ++useCount[fp->getFullTitle()];
      }

      DialBase* dial = dialResponseCache.dialInterface->getDialBaseRef();
      std::string dialType = dial->getDialTypeName();
      if (dialType.find("Norm") == 0) {
        ++config.norms;
      }
      else if (dialType.find("GeneralSpline") == 0) {
        ++config.generalSplines;
        config.generalPoints += dial->getDialData().size();
      }
      else if (dialType.find("UniformSpline") == 0) {
        ++config.uniformSplines;
        config.uniformPoints += dial->getDialData().size();
      }
      else if (dialType.find("MonotonicSpline") == 0) {
        ++config.monotonicSplines;
        config.monotonicPoints += dial->getDialData().size();
      }
      else if (dialType.find("CompactSpline") == 0) {
        ++config.compactSplines;
        config.compactPoints += dial->getDialData().size();
      }
      else if (dialType.find("LightGraph") == 0) {
        ++config.graphs;
        config.graphPoints += dial->getDialData().size();
      }
      else if (dialType.find("Bilinear") == 0) {
        ++config.bilinear;
        config.bilinearPoints += dial->getDialData().size();
      }
      else if (dialType.find("Bicubic") == 0) {
        ++config.bicubic;
        config.bicubicPoints += dial->getDialData().size();
      }
      else if (dialType.find("Shift") == 0) {
        ++config.shifts;
      }
      else if (dialType.find("Tabulated") == 0) {
        ++config.tabulated;
        auto* tabDial = dynamic_cast<Tabulated*>(dial);
        LogThrowIf(tabDial == nullptr, "Tabulated dial is not a Tabulated dial");
        // Add a place holder for this table.  This will be filled
        // with the offset to the table when the weighting is built.
        config.tables[tabDial->getTable()] = 0;
      }
      else {
        LogError << "Unsupported dial type -- "
                 << dialType
                 << std::endl;
        ++dialErrorCount;
      }
    }
  }

  if (dialErrorCount > 0) {
    LogError << "Dial creation errors: "
             << dialErrorCount
             << std::endl;
    LogThrow("Unsupported dial type: Incomplete dial implementation");
  }

  // Finish filling the configuration for the tabulated dials
  {
    config.tabulatedPoints = 0;
    for (auto table : config.tables) {
      table.second = config.tabulatedPoints;
      config.tabulatedPoints += table.first->size();
    }
  }

  // Count the total number of histogram cells.
  config.histBins = 0;
  for(const Sample& sample : sampleSet.getSampleList() ){
    int cells = sample.getHistogram().getNbBins(); // GetNcells() of TH1D
    LogInfo  << "Add histogram for " << sample.getName()
             << " with " << cells
             << " cells (includes under/over-flows)" << std::endl;
    config.histBins += cells;
  }

  /// Summarize the space and get the cache memory.
  config.parameters = int(usedParameters.size());
  LogInfo  << "Cache for " << config.events << " events --"
           << " using " << config.parameters << " parameters"
           << std::endl;
  LogInfo  << "    Compact splines: " << config.compactSplines
           << " (" << 1.0*config.compactSplines/config.events << " per event)"
           << std::endl;
  LogInfo  << "    Monotonic splines: " << config.monotonicSplines
           << " (" << 1.0*config.monotonicSplines/config.events << " per event)"
           << std::endl;
  LogInfo  << "    Uniform Splines: " << config.uniformSplines
           << " (" << 1.0*config.uniformSplines/config.events << " per event)"
           << std::endl;
  LogInfo  << "    General Splines: " << config.generalSplines
           << " (" << 1.0*config.generalSplines/config.events << " per event)"
           << std::endl;
  LogInfo  << "    Graphs: " << config.graphs
           << " (" << 1.0*config.graphs/config.events << " per event)"
           << std::endl;
  LogInfo  << "    Normalizations: " << config.norms
           <<" ("<< 1.0*config.norms/config.events <<" per event)"
           << std::endl;
  LogInfo  << "    Shifts: " << config.shifts
           <<" ("<< 1.0*config.shifts/config.events <<" per event)"
           << std::endl;
  LogInfo  << "    Tabulated: " << config.tabulated
           <<" ("<< 1.0*config.tabulated/config.events <<" per event)"
           << std::endl;
  LogInfo  << "    Bilinear: " << config.bilinear
           <<" ("<< 1.0*config.bilinear/config.events <<" per event)"
           << std::endl;
  LogInfo  << "    Bicubic: " << config.bicubic
           <<" ("<< 1.0*config.bicubic/config.events <<" per event)"
           << std::endl;
  LogInfo  << "    Histogram bins: " << config.histBins
           << " (" << 1.0*config.events/config.histBins << " events per bin)"
           << std::endl;

  if (config.compactSplines > 0) {
    LogInfo  << "    Compact spline cache uses "
             << config.compactPoints << " control points --"
             << " (" << 1.0*config.compactPoints/config.compactSplines
             << " points per spline)"
             << " for " << config.compactSplines << " splines"
             << std::endl;
  }
  if (config.monotonicSplines > 0) {
    LogInfo  << "    Monotonic spline cache uses "
             << config.monotonicPoints << " control points --"
             << " (" << 1.0*config.monotonicPoints/config.monotonicSplines
             << " points per spline)"
             << " for " << config.monotonicSplines << " splines"
             << std::endl;
  }
  if (config.uniformSplines > 0) {
    LogInfo  << "    Uniform spline cache uses "
             << config.uniformPoints << " control points --"
             << " (" << 1.0*config.uniformPoints/config.uniformSplines
             << " points per spline)"
             << " for " << config.uniformSplines << " splines"
             << std::endl;
  }
  if (config.generalSplines > 0) {
    LogInfo  << "    General spline cache uses "
             << config.generalPoints << " control points --"
             << " (" << 1.0*config.generalPoints/config.generalSplines
             << " points per spline)"
             << " for " << config.generalSplines << " splines"
             << std::endl;
  }
  if (config.graphs > 0) {
    LogInfo  << "    Graph cache uses "
             << config.graphPoints << " control points --"
             << " (" << 1.0*config.graphPoints/config.graphs << " points per graph)"
             << std::endl;
  }
  if (config.bilinear > 0) {
    LogInfo  << "    Bilinear cache uses "
             << config.bilinearPoints << " control points --"
             << " (" << 1.0*config.bilinearPoints/config.bilinear << " points per surface)"
             << std::endl;
  }
  if (config.bicubic > 0) {
    LogInfo  << "    Bicubic cache uses "
             << config.bicubicPoints << " control points --"
             << " (" << 1.0*config.bicubicPoints/config.bicubic << " points per surface)"
             << std::endl;
  }

  // Try to allocate the Cache::Manager memory (including for the GPU if
  // it's being used).
  if( Cache::Manager::Get() == nullptr and IsCacheManagerEnabled() ){
    LogInfo << "Creating the Cache::Manager" << std::endl;
    if (!Cache::Manager::HasCUDA()) {
      LogInfo << "    GPU Not enabled with Cache::Manager"
              << std::endl;
    }
    try {
      fParameters.fSingleton = new Manager(config);
      LogThrowIf(not fParameters.fSingleton, "CacheManager Not allocated");
    }
    catch (...) {
      LogError << "Did not allocated cache manager" << std::endl;
      LogThrow("Cache::Manager allocation error");
    }
  }

  // In case the cache isn't allocated (usually because it's turned off on
  // the command line), but this is a safety check.
  if (!Cache::Manager::Get()) {
    LogWarning << "Cache will not be used"
               << std::endl;
    return false;
  }

  LogThrowIf(Cache::Manager::Get()->fSampleSet != nullptr,
             "Cannot change Cache::Manager SampleSet");
  Cache::Manager::Get()->fSampleSet = &sampleSet;

  LogThrowIf(Cache::Manager::Get()->fEventDialCache != nullptr,
             "Cannot change Cache::Manager EventDialCache");
  Cache::Manager::Get()->fEventDialCache = &eventDialCache;

  Cache::Manager::Get()->fUpdateRequired = true;

  Cache::Manager::Get()->Update(*Cache::Manager::Get()->fSampleSet,
                                *Cache::Manager::Get()->fEventDialCache);
  return true;
}

bool Cache::Manager::Update(SampleSet& sampleSet, EventDialCache& eventDialCache) {
  LogThrowIf(
      fSampleSet != &sampleSet,
      "Cannot change Cache::Manager SampleSet");

  LogThrowIf(
      fEventDialCache != &eventDialCache,
      "Cannot change Cache::Manager EventDialCache");

  if (!fUpdateRequired) {
    // This is not the update that you are looking for.
    LogError << "Update called when not required" << std::endl;
    LogThrow("Invalid Cache::Manager::Update()");
  }

  // This is the update that is required!
  fUpdateRequired = false;

  LogInfo << "Update the internal caches" << std::endl;

  // Initialize the internal caches so they are in the default state.
  GetParameterCache().Reset();
  GetHistogramsCache().Reset();
  GetWeightsCache().Reset();

  int usedResults = 0;

  fParameters.fEventWeightFillerList.clear();
  fParameters.fEventWeightFillerList.reserve(eventDialCache.getCache().size());

  // Add the dials in the EventDialCache to the internal cache.
  for (EventDialCache::CacheEntry& elem : eventDialCache.getCache()) {
    // Skip events that are not in a bin.
    if (elem.event->getIndices().bin < 0) continue;
    Event& event = *elem.event;
    // The result index.  This is where to save the results for this
    // event in the cache.
    int resultIndex = usedResults++;

    fParameters.fEventWeightFillerList.emplace_back( elem.event, resultIndex );

    // Get the initial value for this event and save it.
    double initialEventWeight = event.getWeights().base;

    int dialErrorCount = 0;
    // Add each dial for the event to the GPU caches.
    for( auto& dialElem : elem.dialResponseCacheList ){
      DialInputBuffer* dialInputs
          = dialElem.dialInterface->getInputBufferRef();

      // Make sure all the used parameters are in the parameter
      // map.
      for (std::size_t i = 0; i < dialInputs->getBufferSize(); ++i) {
        // Find the index (or allocate a new one) for the dial
        // parameter.
        const Parameter* fp
            = &(dialElem.dialInterface->getInputBufferRef()
                ->getParameter(i));
        auto parMapIt = fParameters.ParameterMap.find(fp);
        if (parMapIt == fParameters.ParameterMap.end()) {
          fParameters.ParameterMap[fp]
              = int(fParameters.ParameterMap.size());
        }
      }

      // Apply the mirroring for the parameters
      for (std::size_t i = 0; i < dialInputs->getBufferSize(); ++i) {
        const Parameter* fp = &(dialInputs->getParameter(i));
        auto& bounds = dialInputs->getMirrorEdges(i);
        if( not std::isnan(bounds.minValue) ){
          int parIndex = fParameters.ParameterMap[fp];
          GetParameterCache().SetLowerMirror(parIndex, bounds.minValue);
          GetParameterCache().SetUpperMirror(parIndex, bounds.minValue+bounds.range);
        }
      }

      // Apply the clamps to the parameter range
      for (std::size_t i = 0; i < dialInputs->getBufferSize(); ++i) {
        const Parameter* fp = &(dialInputs->getParameter(i));
        const DialResponseSupervisor* resp
            = dialElem.dialInterface->getResponseSupervisorRef();
        int parIndex = fParameters.ParameterMap[fp];
        double minResponse = 0.0;
        if (std::isfinite(resp->getMinResponse())) {
          minResponse = resp->getMinResponse();
        }
        GetParameterCache().SetLowerClamp(parIndex,minResponse);
        if (not std::isfinite(resp->getMaxResponse())) continue;
        GetParameterCache().SetUpperClamp(parIndex,resp->getMaxResponse());
      }

      // Add the dial information to the appropriate caches
      int dialUsed = 0;
      auto* baseDial = dialElem.dialInterface->getDialBaseRef();
      auto* normDial = dynamic_cast<const Norm*>(baseDial);
      if (normDial) {
        ++dialUsed;
        const Parameter* fp = &(dialInputs->getParameter(0));
        int parIndex = fParameters.ParameterMap[fp];
        fNormalizations->ReserveNorm(resultIndex,parIndex);
      }
      auto* compactSpline = dynamic_cast<const CompactSpline*>(baseDial);
      if (compactSpline) {
        ++dialUsed;
        const Parameter* fp = &(dialInputs->getParameter(0));
        int parIndex = fParameters.ParameterMap[fp];
        fCompactSplines->AddSpline(resultIndex,parIndex,
                                            baseDial->getDialData());
      }
      auto* monotonicSpline = dynamic_cast<const MonotonicSpline*>(baseDial);
      if (monotonicSpline) {
        ++dialUsed;
        const Parameter* fp = &(dialInputs->getParameter(0));
        int parIndex = fParameters.ParameterMap[fp];
        fMonotonicSplines->AddSpline(resultIndex,parIndex,
                                            baseDial->getDialData());
      }
      auto* uniformSpline = dynamic_cast<const UniformSpline*>(baseDial);
      if (uniformSpline) {
        ++dialUsed;
        const Parameter* fp = &(dialInputs->getParameter(0));
        int parIndex = fParameters.ParameterMap[fp];
        fUniformSplines->AddSpline(resultIndex,parIndex,
                                   baseDial->getDialData());
      }
      auto* generalSpline = dynamic_cast<const GeneralSpline*>(baseDial);
      if (generalSpline) {
        ++dialUsed;
        const Parameter* fp = &(dialInputs->getParameter(0));
        int parIndex = fParameters.ParameterMap[fp];
        fGeneralSplines->AddSpline(resultIndex,parIndex,
                                   baseDial->getDialData());
      }
      auto* lightGraph = dynamic_cast<const LightGraph*>(baseDial);
      if (lightGraph) {
        ++dialUsed;
        const Parameter* fp = &(dialInputs->getParameter(0));
        int parIndex = fParameters.ParameterMap[fp];
        fGraphs->AddGraph(resultIndex,parIndex,baseDial->getDialData());
      }
      auto* bilinear = dynamic_cast<const Bilinear*>(baseDial);
      if (bilinear) {
        ++dialUsed;
        const Parameter* fp1 = &(dialInputs->getParameter(0));
        int parIndex1 = fParameters.ParameterMap[fp1];
        const Parameter* fp2 = &(dialInputs->getParameter(1));
        int parIndex2 = fParameters.ParameterMap[fp2];
        fBilinear->AddData(resultIndex,parIndex1,parIndex2,
                           baseDial->getDialData());
      }
      auto* bicubic = dynamic_cast<const Bicubic*>(baseDial);
      if (bicubic) {
        ++dialUsed;
        const Parameter* fp1 = &(dialInputs->getParameter(0));
        int parIndex1 = fParameters.ParameterMap[fp1];
        const Parameter* fp2 = &(dialInputs->getParameter(1));
        int parIndex2 = fParameters.ParameterMap[fp2];
        fBicubic->AddData(resultIndex,parIndex1,parIndex2,
                          baseDial->getDialData());
      }
      auto* shift = dynamic_cast<const Shift*>(baseDial);
      if (shift) {
        ++dialUsed;
        initialEventWeight *= shift->evalResponse(DialInputBuffer());
      }
      auto* tabulated = dynamic_cast<const Tabulated*>(baseDial);
      if (tabulated) {
        ++dialUsed;
        fTabulated->AddData(resultIndex,
                            tabulated->getTable(),
                            tabulated->getIndex(),
                            tabulated->getFraction());
      }

      if (dialUsed != 1) {
        LogError << "Problem with dial: " << dialUsed
                 << std::endl;
        LogError << "Unsupported Dial Type Name: "
                 << baseDial->getDialTypeName()
                 << std::endl;
        ++dialErrorCount;
      }
    }

    if (dialErrorCount > 0) {
      LogError << "Dial creation errors --"
               << " Unsupported dial types: " << dialErrorCount
               << std::endl;
      LogThrow("Unsupported dial type: Incomplete dial implementation");
    }

    // Set the initial weight for the event.  This is done here since the
    // raw tree weight may get rescaled by "Shift" dials
    GetWeightsCache().SetInitialValue(resultIndex,initialEventWeight);

  }

  LogInfo << "Error checking for cache" << std::endl;

  // Error checking adding the dials to the cache!
  if (usedResults != GetWeightsCache().GetResultCount()) {
    LogError << "Cache Manager -- used Results:     "
             << usedResults << std::endl;
    LogError << "Cache Manager -- expected Results: "
             << GetWeightsCache().GetResultCount()
             << std::endl;
    LogThrow("Probable problem putting dials in cache");
  }

  fParameters.fSampleHistFillerList.clear();
  fParameters.fSampleHistFillerList.reserve(sampleSet.getSampleList().size());

  // Add the histogram cells to the cache.  THIS CODE IS SUSPECT SINCE IT IS
  // SAVING ADDRESSES OF CLASS FIELDS.  This *will* be OK since the fields
  // are not going to be moved, and is needed for a huge win in efficiency,
  // but is officially "dangerous".
  LogInfo << "Add this histogram cells to the cache." << std::endl;
  int nextHist = 0;
  for(Sample& sample : sampleSet.getSampleList() ) {
    LogInfo  << "Fill cache for " << sample.getName()
             << " with " << sample.getEventList().size()
             << " events" << std::endl;
    int thisHistIndexOffset = nextHist;

    fParameters.fSampleHistFillerList.emplace_back( &sample.getHistogram(), thisHistIndexOffset );

    int cells = sample.getHistogram().getNbBins();
    nextHist += cells;

    for( auto& eventFiller : fParameters.fEventWeightFillerList ){
      if( eventFiller.getEventPtr()->getIndices().sample == sample.getIndex() ){
        GetHistogramsCache().SetEventIndex(
            eventFiller.getValueIndex(),
            thisHistIndexOffset + eventFiller.getEventPtr()->getIndices().bin
        );
      }
    }
  }

  if (GetHistogramsCache().GetSumCount() != nextHist) {
    LogThrow("Histogram cells are missing");
  }

  // If the event weight cap has been set, then pass it along
  if (eventDialCache.getGlobalEventReweightCap().isEnabled) {
    double c = eventDialCache.getGlobalEventReweightCap().maxReweight;
    if (std::isfinite(c)) GetHistogramsCache().SetMaximumEventWeight(c);
  }

  // Notify all of the internal caches (mostly the CacheRecursiveSums) that
  // the internal buffers should be update
  GetHistogramsCache().Initialize();

  return true;
}

bool Cache::Manager::Fill() {
  if (fUpdateRequired) {
    LogError << "Fill while an update is required" << std::endl;
    LogThrow("Fill while an update is required");
  }
  LogTraceIf(GundamGlobals::isDebug() ) << "Cache::Manager::Fill -- Fill the GPU cache" << std::endl;
#define DUMP_FILL_INPUT_PARAMETERS
#ifdef DUMP_FILL_INPUT_PARAMETERS
  do {
    static bool printed = false;
    if (printed) break;
    printed = true;
    for (auto& par : fParameters.ParameterMap ) {
      // This produces a crazy amount of output.
      LogInfo  << "FILL: " << par.second
               << "/" << fParameters.ParameterMap.size()
               << " " << par.first->getParameterValue()
               << " (" << par.first->getFullTitle() << ")"
               << " enabled: " << par.first->isEnabled()
               << std::endl;
    }
  } while(false);
#endif
  GetWeightsCache().Invalidate();
  GetHistogramsCache().Invalidate();
  for (auto& par : fParameters.ParameterMap ) {
    if (not par.first->isEnabled()) {
      LogWarning << "WARNING: Disabled parameter: "
                 << par.first->getFullTitle()
                 << std::endl;
      LogWarning << "WARNING: Cache::Manager will not be used"
                 << std::endl;
      return false;
    }
    GetParameterCache().SetParameter(
        par.second, par.first->getParameterValue());
  }
  GetWeightsCache().Apply();
  GetHistogramsCache().Apply();

  return true;
}

bool Cache::Manager::PropagateParameters(){

  bool isSuccess{false};

  // Copy updated information to the device and start the kernels
  {
    auto s{fParameters.cacheFillTimer.scopeTime()};

    // if disabled, leave
    if (Cache::Manager::Get() == nullptr) {
      return false;
    }

    // do the propagation on the device
    isSuccess = Cache::Manager::Get()->Fill();
    if (not isSuccess) return false;
  }

  {
    // This section should be delayed as long as possible, and the current
    // serialization is wasting the GPU parallization.  For efficiency, the
    // logic needs to follow LTS version where the back copy is delayed until
    // the results are actually needed.
    auto s{fParameters.pullFromDeviceTimer.scopeTime()};

    // do we need to copy every event weight to the CPU structures ?
    if( fParameters.fIsEventWeightCopyEnabled ){
      Cache::Manager::CopyEventWeights();
    }

    // do we need to copy bin content to the CPU structures ?
    if( fParameters.fIsHistContentCopyEnabled ){
      Cache::Manager::CopyHistogramContents();
    }
  }
  return true;
}

bool Cache::Manager::CopyEventWeights(){

  if( not Cache::Manager::Get()->GetWeightsCache().IsResultValid() ){
    // Trigger this update
    if( fParameters.fEnableDebugPrintouts ){ LogDebug << "Copy event weights from Device to Host" << std::endl; }
    Cache::Manager::Get()->GetWeightsCache().GetResult(0);
  }

  for( auto& eventFiller : fParameters.fEventWeightFillerList ){
    eventFiller.copyCacheToCpu( Cache::Manager::Get()->GetWeightsCache().GetWeights().hostPtr() );
  }

  return true;
}

bool Cache::Manager::CopyHistogramContents(){

  for( auto& histFiller : fParameters.fSampleHistFillerList ){
    histFiller.copyHistogram(
        Cache::Manager::Get()->GetHistogramsCache().GetSumsPointer(),
        Cache::Manager::Get()->GetHistogramsCache().GetSums2Pointer()
    );
  }

  return true;
}

bool Cache::Manager::ValidateHistogramContents(int quiet){

  int count = 0;
  int failed = 0;
  for( auto& histFiller : fParameters.fSampleHistFillerList ){
    if (not histFiller.validateHistogram(
            (quiet < 2),
            Cache::Manager::Get()->GetHistogramsCache().GetSumsPointer(),
            Cache::Manager::Get()->GetHistogramsCache().GetSums2Pointer()
        )) {
      if (quiet < 3) LogError << "Histogram " << count++ << " FAILED" << std::endl;
      ++failed;
      continue;
    }
    if (quiet < 1) LogError << "Histogram " << count++ << " OK" << std::endl;
  }

  return failed < 1;
}

int Cache::Manager::ParameterIndex(const Parameter* fp) {
  auto parMapIt = fParameters.ParameterMap.find(fp);
  if (parMapIt == fParameters.ParameterMap.end()) return -1;
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
// mode:C++
// c-basic-offset:4
// End:
