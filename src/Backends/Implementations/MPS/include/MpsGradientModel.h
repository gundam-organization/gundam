#ifndef GUNDAM_MPS_GRADIENT_MODEL_H
#define GUNDAM_MPS_GRADIENT_MODEL_H
#include "MpsGradientTypes.h"
#include "DevicePropagationModel.h"
#include "GradientModel.h"
#include "GradientValidation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Backends {
  inline void validateMpsGradient(const EngineView& view_) {
    validateGradientModel(view_);
  }

  struct MpsGradientModel {
    GradientWork<ObservableScalar> work{};
    std::vector<LikelihoodBin<ObservableScalar>> bins;
    std::vector<MpsGradientInput> inputs;
    std::vector<DeviceDialInput> bindings;
    std::vector<DeviceDialInputRange> ranges;
    std::vector<MpsGradientEvent> events;
    std::vector<DeviceIndex> inputOffsets, indices, parameterOffsets, parameterOccurrences;
    std::size_t scratchSize{0}, contributionCount{0}, externalCount{0};
    void build(const EngineView& view_, const DevicePropagationModel& packed_) {
      *this = MpsGradientModel();
      validateMpsGradient(view_);
      const auto& m = view_.propagation;
      work = {gradientIndex(m.events.size()), 0, gradientIndex(m.totalBins), gradientIndex(m.parameterCount),
        m.hasReweightCap ? ObservableScalar(m.maxReweight) : ObservableScalar(0), DeviceIndex(m.hasReweightCap), 0};
      bins.assign(m.totalBins, {});
      for( auto& bin : bins ) bin.ignored = 1;
      for( const auto& bin : packLikelihood<ObservableScalar>(view_.likelihood, m.totalBins) ) bins.at(bin.globalBin) = bin;
      for( std::size_t i = 0; i < m.dials.size(); ++i ){
        const auto& dial = m.dials[i];
        const auto& device = packed_.dials.at(i);
        ranges.push_back({gradientIndex(bindings.size()), gradientIndex(dial.inputCount)});
        for( std::size_t j = 0; j < dial.inputCount; ++j ){
          const auto& input = m.dialInputs.at(dial.firstInput+j);
          bindings.push_back({DeviceIndex(DialInputSource::Parameter), gradientIndex(input.parameterIndex), DeviceIndex(input.useMirror),
            ObservableScalar(input.mirrorMin), ObservableScalar(input.mirrorRange)});
        }
        inputOffsets.push_back(gradientIndex(inputs.size()));
        std::size_t scratch = 0;
        if( dial.type == BackendDialType::Composite ){
          const auto& program = m.compositePrograms.at(dial.compositeProgram);
          scratch = device.stackOffset;
          for( std::size_t j = 0; j < program.instructionCount; ++j ){
            const auto& node = m.compositeInstructions.at(program.firstInstruction+j).dial;
            if( node.type == BackendDialType::Formula )
              scratch = std::max(scratch, std::size_t(device.stackOffset)+m.formulaPrograms.at(node.formulaProgramIndex).stackSize);
          }
        }
        if( dial.type == BackendDialType::Formula ) scratch = m.formulaPrograms.at(dial.formulaProgramIndex).stackSize;
        const bool producer = dial.type == BackendDialType::ExternalWeight or isTableDial(dial.type);
        for( std::size_t j = 0; j < dial.inputCount; ++j ){
          inputs.push_back({gradientIndex(i), gradientIndex(j), gradientIndex(scratchSize),
            producer ? gradientIndex(externalCount++) : 0});
          scratchSize += scratch;
          gradientIndex(scratchSize);
        }
      }
      inputOffsets.push_back(gradientIndex(inputs.size()));
      work.dials = gradientIndex(inputs.size());
      std::vector<std::vector<DeviceIndex>> byParameter(m.parameterCount);
      for( const auto& event : m.events ){
        const auto& weight = m.eventWeights.at(event.resultIndex);
        events.push_back({gradientIndex(indices.size()), gradientIndex(weight.dialCount),
          gradientIndex(event.resultIndex), gradientIndex(contributionCount), ObservableScalar(weight.baseWeight)});
        const auto addDial = [&](std::size_t index_){
          indices.push_back(gradientIndex(index_));
          const auto& dial = m.dials.at(index_);
          for( std::size_t k = 0; k < dial.inputCount; ++k ){
            byParameter.at(m.dialInputs.at(dial.firstInput+k).parameterIndex).push_back(gradientIndex(contributionCount++));
          }
        };
        for( std::size_t j = 0; j < weight.dialCount; ++j ) addDial(m.eventDialIndices.at(weight.firstDial+j));
        // Observable occurrences follow weight occurrences. The device walks
        // these targets in the same override/multiply/add order, even for a
        // non-histogram observable (whose statistical adjoint is zero).
        for( std::size_t j = 0; j < event.variableResponseCount; ++j ){
          const auto& target = m.variableResponses.at(event.firstVariableResponse+j);
          if( target.overrideDial != std::size_t(-1) ) addDial(target.overrideDial);
          for( std::size_t k = 0; k < target.multiplyDialCount; ++k ) addDial(m.variableDialIndices.at(target.firstMultiplyDial+k));
          for( std::size_t k = 0; k < target.addDialCount; ++k ) addDial(m.variableDialIndices.at(target.firstAddDial+k));
        }
      }
      gradientIndex(contributionCount);
      for( const auto& occurrences : byParameter ){
        parameterOffsets.push_back(gradientIndex(parameterOccurrences.size()));
        parameterOccurrences.insert(parameterOccurrences.end(), occurrences.begin(), occurrences.end());
      }
      parameterOffsets.push_back(gradientIndex(parameterOccurrences.size()));
    }
  };
}
#endif
