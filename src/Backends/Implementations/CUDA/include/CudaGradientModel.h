#ifndef GUNDAM_CUDA_GRADIENT_MODEL_H
#define GUNDAM_CUDA_GRADIENT_MODEL_H
#include "CudaGradientSemantics.h"
#include "GradientModel.h"
#include "DialProgramValidation.h"

#include <algorithm>

namespace Backends {
  inline void validateCudaGradient(const EngineView& view_) {
    const auto& m = view_.propagation;
    if( m.requiresEventBinning() ) throw std::runtime_error("CUDA gradients require fixed, unsmoothed bins without observable-response dials.");
    if( view_.likelihood.empty() ) throw std::runtime_error("CUDA gradient requires a statistical likelihood.");
    for( const auto& sample : view_.likelihood.samples )
      if( sample.portableOptions.kind != LikelihoodKind::BarlowBeestonLiteGaussian )
        throw std::runtime_error("CUDA gradients currently require BarlowBeestonLiteGaussian.");
    validateCompositePrograms(m);
    for( const auto& program : m.formulaPrograms ) program.validate();
    for( const auto& dial : m.dials ){
      switch( dial.type ){
        case BackendDialType::Norm: case BackendDialType::Graph:
        case BackendDialType::CompactSpline: case BackendDialType::UniformSpline:
        case BackendDialType::MonotonicSpline: case BackendDialType::GeneralSpline:
          if( dial.inputCount != 1 ) throw std::runtime_error("CUDA scalar gradient requires one input.");
          break;
        case BackendDialType::Bilinear: case BackendDialType::Bicubic: Semantics::validateSurfaceDial(m,dial); break;
        case BackendDialType::Formula: validateFormulaDial(m,dial); break;
        case BackendDialType::Tabulated: case BackendDialType::Kriged: Semantics::validateTableDial(m,dial); break;
        case BackendDialType::Shift: case BackendDialType::Composite: case BackendDialType::ExternalWeight: break;
        default: throw std::runtime_error("Unsupported CUDA gradient dial.");
      }
      for( std::size_t j = 0; j < dial.inputCount; ++j ){
        const auto& input = m.dialInputs.at(dial.firstInput+j);
        if( input.source != DialInputSource::Parameter || input.parameterIndex >= m.parameterCount )
          throw std::runtime_error("CUDA gradient requires valid parameter inputs.");
      }
    }
    // Scalar interpolation descriptors use the shared uint32 wire offsets.
    for( const auto& dial : m.dials ){ gradientIndex(dial.payloadOffset); gradientIndex(dial.payloadSize); }
    for( const auto& instruction : m.compositeInstructions ){
      gradientIndex(instruction.dial.payloadOffset); gradientIndex(instruction.dial.payloadSize);
    }
    const auto mirror = [](const auto& input_){
      if( input_.useMirror && (!std::isfinite(input_.mirrorRange) || input_.mirrorRange <= 0.) )
        throw std::runtime_error("CUDA gradient requires positive finite mirror ranges.");
    };
    for( const auto& input : m.dialInputs ) mirror(input);
    for( const auto& input : m.compositeInputs ) mirror(input);
  }
  struct CudaGradientDirection { DeviceIndex dial, input, scratch, stack, external; };
  // Original dial numbering, independent of value-executor remapping, compact
  // policies and deduplicated program templates. No dense event/parameter map.
  struct CudaGradientModel {
    GradientWork<double> work{};
    std::vector<GradientEvent<double>> events;
    std::vector<LikelihoodBin<double>> bins;
    std::vector<CudaGradientDirection> directions;
    std::vector<DeviceIndex> inputOffsets, eventOffsets, indices, parameterOffsets, parameterOccurrences;
    std::vector<CudaFormulaProgram> formulas;
    std::vector<FormulaInstruction> instructions;
    std::vector<double> constants;
    std::size_t scratchSize{0}, contributionCount{0}, externalCount{0};
    void build(const EngineView& view_) {
      *this = CudaGradientModel(); validateCudaGradient(view_);
      const auto& m = view_.propagation;
      bins.assign(m.totalBins, {}); for( auto& bin : bins ) bin.ignored = 1;
      for( const auto& bin : packLikelihood<double>(view_.likelihood,m.totalBins) ) bins.at(bin.globalBin) = bin;
      for( const auto& f : m.formulaPrograms ){
        formulas.push_back({instructions.size(),f.instructions.size(),constants.size(),f.stackSize,f.nativePlan});
        instructions.insert(instructions.end(),f.instructions.begin(),f.instructions.end());
        constants.insert(constants.end(),f.constants.begin(),f.constants.end());
      }
      for( std::size_t i = 0; i < m.dials.size(); ++i ){
        const auto& dial = m.dials[i];
        inputOffsets.push_back(gradientIndex(directions.size()));
        std::size_t frame = dial.inputCount, stack = 0;
        if( dial.type == BackendDialType::Formula ) stack = m.formulaPrograms.at(dial.formulaProgramIndex).stackSize;
        if( dial.type == BackendDialType::Composite ){
          const auto& p = m.compositePrograms.at(dial.compositeProgram); frame = p.scratchSize();
          for( std::size_t j = 0; j < p.instructionCount; ++j ){
            const auto& node = m.compositeInstructions.at(p.firstInstruction+j).dial;
            if( node.type == BackendDialType::Formula ) stack = std::max(stack,m.formulaPrograms.at(node.formulaProgramIndex).stackSize);
          }
        }
        const bool producer = dial.type == BackendDialType::ExternalWeight || isTableDial(dial.type);
        for( std::size_t j = 0; j < std::max(std::size_t(1),dial.inputCount); ++j ){
          directions.push_back({gradientIndex(i),gradientIndex(j),gradientIndex(scratchSize),gradientIndex(scratchSize+frame),
            producer && j < dial.inputCount ? gradientIndex(externalCount++) : std::numeric_limits<DeviceIndex>::max()});
          scratchSize += frame+stack; gradientIndex(scratchSize);
        }
      }
      inputOffsets.push_back(gradientIndex(directions.size()));
      work = {gradientIndex(m.events.size()),gradientIndex(directions.size()),gradientIndex(m.totalBins),
        gradientIndex(m.parameterCount),m.maxReweight,DeviceIndex(m.hasReweightCap),0};
      std::vector<std::vector<DeviceIndex>> byParameter(m.parameterCount);
      for( const auto& event : m.events ){
        const auto& weights = m.eventWeights.at(event.resultIndex);
        eventOffsets.push_back(gradientIndex(contributionCount));
        events.push_back({gradientIndex(indices.size()),gradientIndex(weights.dialCount),
          event.globalBinIndex < 0 ? std::numeric_limits<DeviceIndex>::max() : gradientIndex(event.globalBinIndex),
          gradientIndex(event.resultIndex),weights.baseWeight});
        for( std::size_t j = 0; j < weights.dialCount; ++j ){
          auto index = m.eventDialIndices.at(weights.firstDial+j); indices.push_back(index);
          const auto& dial = m.dials.at(index);
          for( std::size_t k = 0; k < std::max(std::size_t(1),dial.inputCount); ++k ){
            if( k < dial.inputCount ) byParameter.at(m.dialInputs.at(dial.firstInput+k).parameterIndex).push_back(gradientIndex(contributionCount));
            ++contributionCount;
          }
        }
      }
      gradientIndex(contributionCount);
      for( const auto& entries : byParameter ){
        parameterOffsets.push_back(gradientIndex(parameterOccurrences.size()));
        parameterOccurrences.insert(parameterOccurrences.end(),entries.begin(),entries.end());
      }
      parameterOffsets.push_back(gradientIndex(parameterOccurrences.size()));
    }
  };
}
#endif
