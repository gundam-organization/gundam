#ifndef GUNDAM_BACKEND_ENGINE_DESCRIPTORS_H
#define GUNDAM_BACKEND_ENGINE_DESCRIPTORS_H

#include "DevicePropagationTypes.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace Backends {

#define ENUM_NAME DialInputSource
#define ENUM_FIELDS \
ENUM_FIELD( Parameter, 0 ) \
ENUM_FIELD( EventVariable )
#include "GenericToolbox.MakeEnum.h"

  // POD descriptors shared by host propagation and device-capable semantics.
  // They intentionally contain no owning containers or host-only behaviour.
  struct BackendDialInputDescriptor {
    std::size_t parameterIndex{std::size_t(-1)};
    bool useMirror{false};
    // EventVariable is reserved for a future event-context evaluator. Until
    // then, builders reject it rather than treating its index as a parameter.
    DialInputSource source{DialInputSource::Parameter};
    double mirrorMin{0};
    double mirrorRange{0};

    BackendDialInputDescriptor() = default;
    BackendDialInputDescriptor(std::size_t index_, bool useMirror_ = false,
                               double mirrorMin_ = 0., double mirrorRange_ = 0.,
                               DialInputSource source_ = DialInputSource::Parameter)
        : parameterIndex(index_), useMirror(useMirror_), source(source_),
          mirrorMin(mirrorMin_), mirrorRange(mirrorRange_) {}
  };
  static_assert(sizeof(BackendDialInputDescriptor) == 32);

  struct BackendDialDescriptor {
    BackendDialType type{BackendDialType::Norm};
    std::size_t firstInput{0};
    std::size_t inputCount{0};
    std::size_t payloadOffset{0};
    std::size_t payloadSize{0};
    std::size_t externalBlockIndex{0};
    std::size_t externalWeightIndex{0};
    std::size_t compositeProgram{0};
    std::size_t formulaProgramIndex{0};
    std::size_t firstFormulaInput{0};
    std::size_t firstTableTerm{0};
    std::size_t tableTermCount{0};
    bool allowExtrapolation{false};
    ResponseSmoothing<double> smoothing{};
    // Inner transition thresholds; identical to the hard bounds when disabled.
    double minResponse{0};
    double maxResponse{0};
    bool hasMinResponse{false};
    bool hasMaxResponse{false};
  };

  // Instructions read earlier scratch slots; the result occupies the next slot.
  // Input mirrors belong to edges, including edges entering nested composites.
  struct CompositeInstruction {
    BackendDialDescriptor dial{};
    double inputMin{0.}, inputMax{0.}; // Original spline bounds, without reconstruction.
  };

  struct CompositeProgram {
    std::size_t firstInstruction{0}, instructionCount{0};
    std::size_t inputCount{0}, outputSlot{0};
    [[nodiscard]] std::size_t scratchSize() const { return inputCount + instructionCount; }
  };

  struct BackendTableTerm {
    std::size_t index{0};
    double coefficient{0};
  };

  inline bool isTableDial(BackendDialType type_) {
    return type_ == BackendDialType::Tabulated or type_ == BackendDialType::Kriged;
  }

  struct ExternalWeightBlockDescriptor {
    std::size_t offset{0};
    std::size_t count{0};
  };

  struct BackendVariableResponseDescriptor {
    std::size_t variableIndex{0}; // Absolute index in the event-variable buffer.
    double referenceValue{0};
    std::size_t overrideDial{std::size_t(-1)};
    std::size_t firstMultiplyDial{0};
    std::size_t multiplyDialCount{0};
    std::size_t firstAddDial{0};
    std::size_t addDialCount{0};
  };

  struct BackendEventWeightDescriptor {
    double baseWeight{1};
    std::size_t firstDial{0};
    std::size_t dialCount{0};
  };

}

#endif // GUNDAM_BACKEND_ENGINE_DESCRIPTORS_H
