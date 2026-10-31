#ifndef GUNDAM_DIAL_DEPENDENCY_GROUPS_H
#define GUNDAM_DIAL_DEPENDENCY_GROUPS_H

#include "DialDependencies.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace Backends {
  // Build-time metadata only. Executors retain their dial numbering, group
  // flattening order and sparse/dense dispatch policies. Scalar groups use
  // offsets to avoid allocating a separate vector for every parameter.
  template<typename Index> struct DialDependencyGroups {
    using MultiParameters = std::vector<std::size_t>;
    std::vector<std::size_t> parameterOffsets;
    std::vector<Index> parameterDialIndices;
    std::map<MultiParameters, std::vector<Index>> multiParameterDials;
    std::map<std::size_t, std::vector<Index>> externalDials;
  };

  // Sorted input lists preserve repeated references, support arbitrary arity,
  // and keep each dial in exactly one group even when several inputs change.
  // Call after the executor's model validation and any dial remapping.
  template<typename Index = std::uint32_t, typename Dials>
  DialDependencyGroups<Index> buildDialDependencyGroups(const PropagationView& model_, const Dials& dials_) {
    DialDependencyGroups<Index> out;
    out.parameterOffsets.assign(model_.parameterCount + 1, 0);
    typename DialDependencyGroups<Index>::MultiParameters parameters;
    for( std::size_t i = 0; i < dials_.size(); ++i ){
      const auto& dial = dials_[i];
      if( dial.type == BackendDialType::Shift ){ continue; }
      parameters.clear();
      std::size_t parameterCount = 0;
      visitDialDependencies(model_, dial, [&](const auto& input_){
        if( dial.inputCount > 1 ){
          parameters.push_back(input_.parameterIndex);
          ++parameterCount;
        }
        else{ ++out.parameterOffsets.at(input_.parameterIndex + 1); }
      }, [&](std::size_t block_){ out.externalDials[block_].push_back(static_cast<Index>(i)); });
      if( parameterCount == 0 ){ continue; }
      std::sort(parameters.begin(), parameters.end());
      out.multiParameterDials[parameters].push_back(static_cast<Index>(i));
    }
    for( std::size_t i = 1; i < out.parameterOffsets.size(); ++i ){
      out.parameterOffsets[i] += out.parameterOffsets[i - 1];
    }
    out.parameterDialIndices.resize(out.parameterOffsets.back());
    auto positions = out.parameterOffsets;
    for( std::size_t i = 0; i < dials_.size(); ++i ){
      const auto& dial = dials_[i];
      if( dial.type == BackendDialType::Shift or dial.inputCount > 1 ){ continue; }
      visitDialDependencies(model_, dial, [&](const auto& input_){
        out.parameterDialIndices[positions[input_.parameterIndex]++] = static_cast<Index>(i);
      }, [](std::size_t){});
    }
    return out;
  }
}
#endif
