/// @file EventVariableResponseCache.cpp
/// @brief Implements EventVariableResponseCache.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in
/// EventVariableResponseCache.h. The comments here describe how each method
/// does its job.

#include "EventVariableResponseCache.h"
#include "EventDialCache.h"
#include "DialCollection.h"
#include "DialInputBuffer.h"
#include "SampleSet.h"

#include "GenericToolbox.Thread.h"
#include "GenericToolbox.Utils.h"
#include "Logger.h"

/// @par Implementation
/// Clears the recorded buffers but keeps `_variableDialBinningSamples_`.
void EventVariableResponseCache::invalidate(){
  _variableStateValid_ = false;
  _variableInputRevisions_.clear();
}

/// @par Implementation
/// A sample is looked at through the variable names of its bin edges. Only
/// samples that were not already dynamic are flagged, so a later call does
/// not mark a sample that the configuration made dynamic.
void EventVariableResponseCache::prepare(SampleSet& sampleSet_, const std::vector<DialCollection>& collections_){
  auto& samples = sampleSet_.getSampleList();
  _variableDialBinningSamples_.resize(samples.size(), false);
  for( const auto& collection : collections_ ){
    if( not collection.isEnabled() or collection.getApplyToVariable().empty() ){ continue; }
    GenericToolbox::addIfNotInVector(collection.getApplyToVariable(), sampleSet_.getEventVariableNameList());
    for( size_t iSample = 0 ; iSample < samples.size() ; ++iSample ){
      auto& sample = samples[iSample];
      for( const auto& context : sample.getHistogram().getBinContextList() ){
        for( const auto& edges : context.bin.getEdgesList() ){
          if( edges.varName == collection.getApplyToVariable() and not sample.isUseDynamicBinning() ){
            LogInfo << "Enabling dynamic binning for sample \"" << sample.getName()
                    << "\": dial targets variable \"" << edges.varName << "\"." << std::endl;
            sample.setUseDynamicBinning(true);
            _variableDialBinningSamples_[iSample] = true;
          }
        }
      }
    }
  }
}

/// @par Implementation
/// invalidate() runs first, so the next update() applies the responses
/// whatever the recorded revisions.
void EventVariableResponseCache::build(std::vector<DialCollection>& collections_){
  invalidate();
  for( auto& collection : collections_ ){
    if( not collection.isEnabled() or collection.getApplyToVariable().empty() ){ continue; }
    for( const auto& buffer : collection.getDialInputBufferList() ){
      _variableInputRevisions_.emplace_back(&buffer, buffer.getRevision());
    }
  }
}

/// @par Implementation
/// The recorded revisions are only updated after the variables and the
/// binning of every sample are updated, so an exception in between leaves
/// the state marked as changed.
void EventVariableResponseCache::update(EventDialCache& eventDialCache_, SampleSet& sampleSet_,
                                        GenericToolbox::ParallelWorker& threadPool_){
  if( not eventDialCache_.hasVariableResponses() ){ return; }
  bool changed = not _variableStateValid_;
  for( const auto& input : _variableInputRevisions_ ){
    if( input.first->getRevision() != input.second ){ changed = true; }
  }
  if( not changed ){
    // Explicit dynamic binning may depend on changes outside variable dials.
    auto& samples = sampleSet_.getSampleList();
    for( size_t iSample = 0 ; iSample < samples.size() ; ++iSample ){
      if( samples[iSample].isUseDynamicBinning() and not _variableDialBinningSamples_[iSample] ){
        samples[iSample].updateEventBinning(threadPool_);
      }
    }
    return;
  }
  threadPool_.runJob([&](int iThread_){
    auto& cache = eventDialCache_.getCache();
    auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(iThread_, threadPool_.getNbThreads(), cache.size());
    for( size_t iEntry = bounds.beginIndex ; iEntry < bounds.endIndex ; ++iEntry ){
      eventDialCache_.applyVariableResponses(cache[iEntry]);
    }
  });
  sampleSet_.updateEventBinning(threadPool_);
  // Commit only after all variables and bin contributions have been updated.
  for( auto& input : _variableInputRevisions_ ){ input.second = input.first->getRevision(); }
  _variableStateValid_ = true;
}
