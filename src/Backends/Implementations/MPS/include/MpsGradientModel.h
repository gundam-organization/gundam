#ifndef GUNDAM_MPS_GRADIENT_MODEL_H
#define GUNDAM_MPS_GRADIENT_MODEL_H
#include "MpsGradientTypes.h"
#include "DevicePropagationModel.h"
#include "GradientModel.h"
#include "DialProgramValidation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Backends {
  inline void validateMpsGradient(const EngineView& view_) {
    const auto& m = view_.propagation;
    if( m.requiresEventBinning() ) throw std::runtime_error("MPS gradients require fixed, unsmoothed bins without observable-response dials.");
    if( view_.likelihood.empty() ) throw std::runtime_error("MPS gradient requires a statistical likelihood.");
    for( const auto& sample : view_.likelihood.samples ){
      if( sample.portableOptions.kind != LikelihoodKind::BarlowBeestonLiteGaussian )
        throw std::runtime_error("MPS gradients currently require BarlowBeestonLiteGaussian.");
    }
    validateCompositePrograms(m);
    for( const auto& dial : m.dials ){
      switch( dial.type ){
        case BackendDialType::Norm: case BackendDialType::Graph:
        case BackendDialType::CompactSpline: case BackendDialType::UniformSpline:
        case BackendDialType::MonotonicSpline: case BackendDialType::GeneralSpline:
          if( dial.inputCount != 1 ) throw std::runtime_error("MPS scalar gradient requires one input.");
          break;
        case BackendDialType::Bilinear: case BackendDialType::Bicubic:
          Semantics::validateSurfaceDial(m, dial); break;
        case BackendDialType::Formula: validateFormulaDial(m, dial); break;
        case BackendDialType::Shift: case BackendDialType::Composite:
        case BackendDialType::ExternalWeight: break;
        case BackendDialType::Tabulated: case BackendDialType::Kriged: Semantics::validateTableDial(m, dial); break;
        default: throw std::runtime_error("Unsupported MPS gradient dial.");
      }
      for( std::size_t i = 0; i < dial.inputCount; ++i ){
        const auto& input = m.dialInputs.at(dial.firstInput+i);
        if( input.source != DialInputSource::Parameter or input.parameterIndex >= m.parameterCount )
          throw std::runtime_error("MPS gradient requires valid parameter inputs.");
      }
    }
    const auto validateMirror = [](const auto& input_){
      if( input_.useMirror and (not std::isfinite(input_.mirrorRange) or input_.mirrorRange <= 0.) )
        throw std::runtime_error("MPS gradient requires a positive finite mirror range.");
    };
    for( const auto& input : m.dialInputs ) validateMirror(input);
    for( const auto& input : m.compositeInputs ) validateMirror(input);
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
          event.globalBinIndex < 0 ? std::numeric_limits<DeviceIndex>::max() : gradientIndex(event.globalBinIndex),
          gradientIndex(event.resultIndex), gradientIndex(contributionCount), ObservableScalar(weight.baseWeight)});
        for( std::size_t j = 0; j < weight.dialCount; ++j ){
          const auto index = m.eventDialIndices.at(weight.firstDial+j);
          indices.push_back(gradientIndex(index));
          const auto& dial = m.dials.at(index);
          for( std::size_t k = 0; k < dial.inputCount; ++k ){
            byParameter.at(m.dialInputs.at(dial.firstInput+k).parameterIndex).push_back(gradientIndex(contributionCount++));
          }
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
