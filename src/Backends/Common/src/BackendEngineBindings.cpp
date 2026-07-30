#include "BackendEngineBindings.h"

#include "DialInputBuffer.h"
#include "DialInterface.h"
#include "EventDialCache.h"
#include "Event.h"
#include "LikelihoodInterface.h"
#include "Parameter.h"
#include "Sample.h"
#include "SampleSet.h"

#include <algorithm>
#include <utility>

namespace {
  template<typename Container, typename Predicate>
  bool containsIf(const Container& container_, Predicate&& predicate_) {
    return std::find_if(container_.begin(), container_.end(), std::forward<Predicate>(predicate_)) != container_.end();
  }
}

void Backends::BackendEngineBindings::clear() {
  events.clear();
  eventDials.clear();
  samples.clear();
  parameters.clear();
  inputBuffers.clear();
}

void Backends::BackendEngineBindings::build(LikelihoodInterface& likelihoodInterface_) {
  clear();

  auto& sampleSet = const_cast<Propagator&>(likelihoodInterface_.getModelPropagator()).getSampleSet();
  const auto& eventDialCache = likelihoodInterface_.getModelPropagator().getEventDialCache();

  samples.reserve(sampleSet.getSampleList().size());
  for( auto& sample : sampleSet.getSampleList() ){
    BackendSampleBinding sampleBinding;
    sampleBinding.histogram = &sample.getHistogram();
    sampleBinding.sampleIndex = sample.getIndex();
    samples.emplace_back(sampleBinding);
  }

  events.reserve(eventDialCache.getCache().size());
  eventDials.reserve(eventDialCache.getCache().size());

  for( const auto& cacheEntry : eventDialCache.getCache() ){
    if( cacheEntry.event == nullptr ){ continue; }
    if( cacheEntry.event->getIndices().bin < 0 ){ continue; }

    BackendEventBinding eventBinding;
    eventBinding.event = cacheEntry.event;
    events.emplace_back(eventBinding);

    for( const auto& dialResponse : cacheEntry.dialResponseCacheList ){
      BackendDialBinding dialBinding;
      dialBinding.interface = dialResponse.dialInterface;
      eventDials.emplace_back(dialBinding);

      auto* inputBuffer = const_cast<DialInputBuffer*>(dialResponse.dialInterface->getInputBufferRef());
      if( inputBuffer != nullptr ){
        if( not containsIf(inputBuffers, [inputBuffer](const auto& binding_){ return binding_.inputBuffer == inputBuffer; }) ){
          inputBuffers.emplace_back(BackendInputBufferBinding{inputBuffer});
        }
        for( int iInput = 0 ; iInput < inputBuffer->getBufferSize() ; iInput++ ){
          auto* parameter = const_cast<Parameter*>(&inputBuffer->getParameter(iInput));
          if( not containsIf(parameters, [parameter](const auto& binding_){ return binding_.parameter == parameter; }) ){
            parameters.emplace_back(BackendParameterBinding{parameter});
          }
        }
      }
    }
  }
}
