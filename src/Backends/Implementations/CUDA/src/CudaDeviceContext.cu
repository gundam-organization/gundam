#include "CudaDeviceContext.h"
#include "CudaWeightModel.h"
#include "CudaDialPrograms.h"
#include "CudaBinning.h"
#include "LikelihoodModel.h"
#include "Semantics/BackendLikelihood.h"
#include "BackendTypes.h"
#include "PropagationInputs.h"
#include "EventStateCache.h"
#include "Semantics/BackendVariableResponse.h"

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include "Semantics/BackendDialSemanticsCore.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {
  constexpr unsigned int blockSize = Backends::cudaResponseBlockSize;

  void checkCuda(cudaError_t status_, const char* operation_) {
    if( status_ != cudaSuccess ){
      throw std::runtime_error(std::string("CudaBackend: ") + operation_ + ": " + cudaGetErrorString(status_));
    }
  }

  class DeviceScope {
  public:
    explicit DeviceScope(int deviceIndex_) {
      checkCuda(cudaGetDevice(&_previousDevice_), "cudaGetDevice");
      checkCuda(cudaSetDevice(deviceIndex_), "cudaSetDevice");
    }
    ~DeviceScope() { cudaSetDevice(_previousDevice_); }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
  private:
    int _previousDevice_{0};
  };

  // Buffers are destroyed while their owning context's DeviceScope is active.
  template<typename T> struct DeviceBuffer {
    T* data{nullptr};
    std::size_t size{0};
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if( data ){ cudaFree(data); } }
    void allocate(std::size_t size_) {
      if( size_ > std::numeric_limits<std::size_t>::max() / sizeof(T) ){
        throw std::runtime_error("CudaBackend: buffer size overflow.");
      }
      size = size_;
      if( size ){ checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)), "cudaMalloc"); }
    }
    // Scratch/unpublished buffers: growing discards contents after synchronizing
    // their previous use. Callers write the complete active range afterward.
    void ensureCapacity(std::size_t size_) {
      if( size_ <= size ){ return; }
      DeviceBuffer replacement;
      replacement.allocate(size_);
      std::swap(data, replacement.data);
      std::swap(size, replacement.size);
    }
    void upload(const std::vector<T>& values_, cudaStream_t stream_) {
      allocate(values_.size());
      if( size ){ checkCuda(cudaMemcpyAsync(data, values_.data(), size * sizeof(T), cudaMemcpyHostToDevice, stream_), "model upload"); }
    }
    void read(std::vector<T>& values_, cudaStream_t stream_) const {
      values_.resize(size);
      if( size ){ checkCuda(cudaMemcpyAsync(values_.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost, stream_), "device readback"); }
    }
  };

  template<typename... Buffers>
  std::size_t allocatedBufferBytes(const Buffers&... buffers_) {
    return (std::size_t(0) + ... + (buffers_.size * sizeof(*buffers_.data)));
  }

  struct WeightBuffers {
    DeviceBuffer<Backends::LikelihoodBin<double>> likelihoodBins;
    DeviceBuffer<double> likelihoodPartials, likelihoodTotal;
    DeviceBuffer<Backends::BackendEventWeightDescriptor> events;
    DeviceBuffer<Backends::BackendDialDescriptor> dials;
    DeviceBuffer<Backends::BackendDialInputDescriptor> inputs;
    DeviceBuffer<double> payloads, parameters, responses, previousInputs, previousSecondInputs, weights, sums, sumSquares;
    DeviceBuffer<std::uint32_t> dialIndices, parameterDialIndices;
    DeviceBuffer<Backends::CudaResponseUpdate> responseUpdates, externalResponseUpdates;
    DeviceBuffer<Backends::BackendTableTerm> tableTerms;
    DeviceBuffer<Backends::ExternalWeightBlockDescriptor> externalBlocks;
    DeviceBuffer<double> externalWeights;
    DeviceBuffer<Backends::CudaFormulaProgram> formulaPrograms;
    DeviceBuffer<Backends::FormulaInstruction> formulaInstructions;
    DeviceBuffer<double> formulaConstants, programScratch;
    DeviceBuffer<Backends::FormulaInput> formulaInputs;
    DeviceBuffer<Backends::NativeCompositeTemplate> compositePrograms;
    DeviceBuffer<Backends::NativeCompositeInstance> compositeInstances;
    DeviceBuffer<std::size_t> compositeNodeData;
    DeviceBuffer<Backends::CompositeInstruction> compositeInstructions;
    DeviceBuffer<Backends::BackendDialInputDescriptor> compositeInputs;
    DeviceBuffer<std::size_t> programScratchOffsets;
    Backends::CudaProgramView programView() const {
      return {formulaPrograms.data, formulaInstructions.data, formulaConstants.data, formulaInputs.data,
              compositePrograms.data, compositeInstructions.data, compositeInputs.data, inputs.data,
              payloads.data, programScratchOffsets.data, programScratch.data, compositeInstances.data, compositeNodeData.data};
    }
    std::vector<Backends::CudaExternalGroup> externalGroups{};
    std::vector<std::uint64_t> externalGenerations{}, currentExternalGenerations{};
    Backends::CudaResponseUpdatePlan externalUpdatePlan{};
    std::size_t externalDialCount{0};
    std::vector<std::size_t> parameterDialOffsets{};
    std::vector<std::uint32_t> programDispatchIndices{};
    std::vector<Backends::CudaSurfaceGroup> surfaceGroups{};
    std::vector<Backends::CudaMultiInputGroup> multiInputGroups{};
    std::size_t surfaceOffset{0};
    std::vector<double> previousParameters{};
    Backends::CudaResponseUpdatePlan updatePlan{};
    DeviceBuffer<std::size_t> binOffsets, binEvents;
    DeviceBuffer<int> error;
    DeviceBuffer<Backends::EventView> stateEvents;
    DeviceBuffer<Backends::CudaBinningSample> samples;
    DeviceBuffer<Backends::BinningBin> bins;
    DeviceBuffer<Backends::BinningInterval> intervals;
    DeviceBuffer<Backends::CudaBinningNode> binningNodes;
    DeviceBuffer<Backends::CudaBinningBound> binningBounds;
    DeviceBuffer<Backends::BackendVariableResponseDescriptor> variableResponses;
    DeviceBuffer<std::uint32_t> variableDialIndices;
    DeviceBuffer<double> variables, nextVariables, hostVariables;
    DeviceBuffer<std::uint8_t> hostEdits;
    DeviceBuffer<int> stateBins, nextStateBins, histogramKeys, changedSamples;
    DeviceBuffer<unsigned long long> stateCounters;
    std::size_t cachedContributionCount{0};
    std::vector<std::uint8_t> hostEditFlags{}, changedSampleFlags{};
    std::vector<int> sampleChanges{};
    Backends::EventStateCache stateCache{};
    // EventStateCache borrows the original model owned by CudaBackend.
    const Backends::PropagationView* originalModel{nullptr};
    DeviceBuffer<std::size_t> contributionCounts, contributionOffsets, nextContributionOffsets;
    DeviceBuffer<Backends::CudaBinningResult> binningResults;
    DeviceBuffer<Backends::CudaBinContribution> contributions, nextContributions, sortedContributions;
    std::size_t contributionCount{0}, histogramBins{0};
    bool hasSmoothedBinning{false};
    bool hasEventState{false};
    bool onlyFastPrograms{false};
    bool cachePrimed{false};
    bool hasReweightCap{false};
    bool interleavedEventDials{false};
    double maxReweight{0};
  };

  unsigned int gridSize(std::size_t size_) {
    return static_cast<unsigned int>(std::min<std::size_t>((size_ - 1) / blockSize + 1, 65535));
  }

  template<bool surfaces_, int programMode_>
  __device__ void evaluateResponse(std::size_t i, const Backends::BackendDialDescriptor* dials_,
                                  const Backends::BackendDialInputDescriptor* inputs_, const double* payloads_,
                                  const double* parameters_, double* responses_, double* previousInputs_,
                                  bool reuse_, int* error_, double* previousSecondInputs_, std::size_t surfaceOffset_, Backends::CudaProgramView programs_) {
    using namespace Backends;
    const auto& dial = dials_[i];
    // External/table dials are evaluated by their generation-dependent kernel.
    if( dial.type == BackendDialType::ExternalWeight or dial.type == BackendDialType::Tabulated
        or dial.type == BackendDialType::Kriged ){ return; }
    if constexpr( programMode_ != 0 ){
      if( dial.type == BackendDialType::Formula or dial.type == BackendDialType::Composite ){
        const double response = [&](){
          if constexpr( programMode_ == 2 ){ return evaluateCudaFastProgram(programs_, dial, parameters_); }
          else{ return evaluateCudaProgram(programs_, dial, i, parameters_); }
        }();
        if( not ::isfinite(response) ){ atomicExch(error_, 1); }
        responses_[i] = ::isfinite(response) ? response : 0.;
        return;
      }
    }
    double input = 0.;
    if( dial.type != BackendDialType::Shift ){
      const auto& descriptor = inputs_[dial.firstInput];
      input = Semantics::transformDialInput(descriptor, parameters_[descriptor.parameterIndex]);
    }
    if constexpr( surfaces_ ){
      if( i >= surfaceOffset_ ){
        const auto& second = inputs_[dial.firstInput + 1];
        const double y = Semantics::transformDialInput(second, parameters_[second.parameterIndex]);
        if( not ::isfinite(input) or not ::isfinite(y) ){
          responses_[i] = 0.; atomicExch(error_, 1); return;
        }
        const auto slot = i - surfaceOffset_;
        if( reuse_ and previousInputs_[i] == input and previousSecondInputs_[slot] == y ){ return; }
        const double response = Semantics::clampDialResponse(dial, GundamDeviceMath::EvaluateSurface(
            input, y, payloads_ + dial.payloadOffset, dial.type == BackendDialType::Bicubic, dial.allowExtrapolation));
        if( not ::isfinite(response) ){ atomicExch(error_, 1); }
        responses_[i] = ::isfinite(response) ? response : 0.;
        previousInputs_[i] = input;
        previousSecondInputs_[slot] = y;
        return;
      }
    }
    if( not ::isfinite(input) ){ responses_[i] = 0.; atomicExch(error_, 1); return; }
    if( reuse_ and previousInputs_[i] == input ){ return; }
    const double* payload = dial.payloadSize == 0 ? nullptr : payloads_ + dial.payloadOffset;
    // These interpolation routines select a uniform segment through an int.
    if( dial.type == BackendDialType::CompactSpline or dial.type == BackendDialType::MonotonicSpline
        or dial.type == BackendDialType::UniformSpline ){
      const auto knots = dial.type == BackendDialType::UniformSpline ? (dial.payloadSize - 2) / 2 : dial.payloadSize - 2;
      const double x = dial.allowExtrapolation ? input
          : Semantics::clampValue(input, payload[0], payload[0] + payload[1] * double(knots - 1));
      const double position = (x - payload[0]) / payload[1];
      // UniformSpline also forms 2*ix+7 before clamping the segment index.
      const double limit = dial.type == BackendDialType::UniformSpline ? 1073741800. : 2147483000.;
      if( not ::isfinite(position) or ::fabs(position) > limit ){
        responses_[i] = 0.; atomicExch(error_, 1); return;
      }
    }
    const double response = Semantics::evalDialResponseFromInput(dial, input, payload);
    if( not ::isfinite(response) ){ atomicExch(error_, 1); }
    responses_[i] = ::isfinite(response) ? response : 0.;
    previousInputs_[i] = input;
  }

  template<bool surfaces_, int programMode_>
  __global__ void evaluateResponses(const Backends::BackendDialDescriptor* dials_, std::size_t count_,
                                    const Backends::BackendDialInputDescriptor* inputs_, const double* payloads_,
                                    const double* parameters_, double* responses_, double* previousInputs_,
                                    bool reuse_, int* error_, double* previousSecondInputs_, std::size_t surfaceOffset_, Backends::CudaProgramView programs_) {
    for( std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count_; i += std::size_t(blockDim.x) * gridDim.x ){
      evaluateResponse<surfaces_, programMode_>(i, dials_, inputs_, payloads_, parameters_, responses_, previousInputs_, reuse_, error_,
                                  previousSecondInputs_, surfaceOffset_, programs_);
    }
  }

  template<bool surfaces_, int programMode_>
  __global__ void evaluateSelectedResponses(const Backends::CudaResponseUpdate* updates_,
      std::size_t rangeCount_, std::size_t blockCount_, const std::uint32_t* indices_,
      const Backends::BackendDialDescriptor* dials_, const Backends::BackendDialInputDescriptor* inputs_,
      const double* payloads_, const double* parameters_, double* responses_, double* previousInputs_, int* error_,
      double* previousSecondInputs_, std::size_t surfaceOffset_, Backends::CudaProgramView programs_) {
    __shared__ std::size_t first, end;
    __shared__ bool direct;
    for( std::size_t block = blockIdx.x; block < blockCount_; block += gridDim.x ){
      if( threadIdx.x == 0 ){
        // One binary search per block, rather than per dial/thread.
        std::size_t low = 0, high = rangeCount_;
        while( low + 1 < high ){
          const auto mid = low + (high - low) / 2;
          if( updates_[mid].firstBlock <= block ){ low = mid; }
          else{ high = mid; }
        }
        const auto range = updates_[low];
        if constexpr( programMode_ != 0 ){ direct = range.direct; }
        first = range.firstDial + (block - range.firstBlock) * blockSize;
        end = range.endDial;
      }
      __syncthreads();
      const auto index = first + threadIdx.x;
      if( index < end ){
        std::size_t dial;
        if constexpr( programMode_ != 0 ){ dial = direct ? index : indices_[index]; }
        else{ dial = indices_[index]; }
        evaluateResponse<surfaces_, programMode_>(dial, dials_, inputs_, payloads_, parameters_, responses_, previousInputs_, true, error_,
                                    previousSecondInputs_, surfaceOffset_, programs_);
      }
      __syncthreads(); // Shared bounds must survive until every thread finished.
    }
  }

  __global__ void evaluateExternalResponses(const Backends::CudaResponseUpdate* updates_,
      std::size_t rangeCount_, std::size_t blockCount_, const std::uint32_t* indices_,
      const Backends::BackendDialDescriptor* dials_, const Backends::ExternalWeightBlockDescriptor* blocks_,
      const Backends::BackendTableTerm* terms_, const double* values_, double* responses_, int* error_) {
    __shared__ std::size_t first, end;
    for( std::size_t block = blockIdx.x; block < blockCount_; block += gridDim.x ){
      if( threadIdx.x == 0 ){
        std::size_t low = 0, high = rangeCount_;
        while( low + 1 < high ){
          const auto mid = low + (high - low) / 2;
          if( updates_[mid].firstBlock <= block ){ low = mid; }
          else{ high = mid; }
        }
        const auto range = updates_[low];
        first = range.firstDial + (block - range.firstBlock) * blockSize;
        end = range.endDial;
      }
      __syncthreads();
      const auto index = first + threadIdx.x;
      if( index < end ){
        const auto slot = indices_[index];
        const auto& dial = dials_[slot];
        const auto* values = (dial.type == Backends::BackendDialType::ExternalWeight or dial.tableTermCount)
            ? values_ + blocks_[dial.externalBlockIndex].offset : nullptr;
        double response = 0.;
        if( dial.type == Backends::BackendDialType::ExternalWeight ){
          response = values[dial.externalWeightIndex];
        }
        else{
          // Preserve the native table term order, including signed/repeated terms.
          for( std::size_t j = 0; j < dial.tableTermCount; ++j ){
            const auto& term = terms_[dial.firstTableTerm + j];
            response += term.coefficient * values[term.index];
          }
        }
        response = Backends::Semantics::clampDialResponse(dial, response);
        if( not ::isfinite(response) ){ atomicExch(error_, 1); }
        responses_[slot] = ::isfinite(response) ? response : 0.;
      }
      __syncthreads();
    }
  }

  struct CachedResponses {
    const double* values;
    __host__ __device__ double operator()(std::size_t dial_) const { return values[dial_]; }
  };

  // All lanes participate, including threads with no events. Keep diagnostic
  // counters off the inner bin-search loop and aggregate before global atomics.
  __device__ void addWarpCounter(unsigned long long* counter_, unsigned long long value_) {
    for( int offset = 16; offset; offset /= 2 ){
      value_ += __shfl_down_sync(0xffffffffu, value_, offset);
    }
    if( (threadIdx.x & 31) == 0 and value_ ){ atomicAdd(counter_, value_); }
  }

  __global__ void updateEventState(const Backends::EventView* events_, std::size_t count_,
      const Backends::CudaBinningSample* samples_, std::size_t sampleCount_,
      const Backends::BinningBin* bins_, const Backends::BinningInterval* intervals_,
      const Backends::CudaBinningNode* nodes_, const Backends::CudaBinningBound* bounds_,
      const Backends::BackendVariableResponseDescriptor* targets_, const std::uint32_t* indices_,
      const double* responses_, double* variables_, const double* hostValues_, const std::uint8_t* hostEdits_,
      int* stateBins_, int* keys_, std::size_t* eventSlots_, int* changedSamples_, unsigned long long* counters_,
      int totalBins_, bool updateVariables_, bool force_, bool first_, int* error_) {
    using namespace Backends;
    unsigned long long candidates = 0, rebinned = 0, contributions = 0;
    for( std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count_; i += std::size_t(blockDim.x) * gridDim.x ){
      const auto& event = events_[i];
      const auto slot = event.resultIndex;
      int localBin = event.binIndex, globalBin = event.globalBinIndex;
      if( event.samplePosition < sampleCount_ ){
        const auto& sample = samples_[event.samplePosition];
        bool changed = false;
        if( sample.readHostVariables or force_ ){
          for( std::size_t j = event.firstVariable; j < event.firstVariable + event.variableCount; ++j ){
            if( hostEdits_[j] ){
              changed |= j - event.firstVariable < sample.variableCount and variables_[j] != hostValues_[j];
              variables_[j] = hostValues_[j];
            }
          }
        }
        if( updateVariables_ ){
          for( std::size_t j = event.firstVariableResponse; j < event.firstVariableResponse + event.variableResponseCount; ++j ){
            const auto& target = targets_[j];
            const double value = Semantics::evalVariableResponse(target, indices_, CachedResponses{responses_});
            if( not ::isfinite(value) ){ atomicExch(error_, 1); }
            changed |= target.variableIndex - event.firstVariable < sample.variableCount
                and variables_[target.variableIndex] != value;
            variables_[target.variableIndex] = value;
          }
        }
        if( sample.isDynamic or sample.isSmoothed ){
          localBin = stateBins_[slot];
          if( first_ or force_ or changed or sample.readHostVariables ){
            const double* values = event.variableCount ? variables_ + event.firstVariable : nullptr;
            localBin = Semantics::findCudaBin(sample, bins_, intervals_, values,
                candidates, nodes_, bounds_);
            atomicExch(changedSamples_ + event.samplePosition, 1);
            ++rebinned;
          }
          globalBin = localBin < 0 ? -1 : sample.binOffset + localBin;
          if( localBin >= 0 and not sample.isSmoothed ){ ++contributions; }
        }
      }
      stateBins_[slot] = localBin;
      // Reinitialize in original event order before the stable sort. This keeps
      // each histogram's reduction deterministic, independently of result slots.
      if( keys_ ){
        keys_[i] = globalBin < 0 ? totalBins_ : globalBin;
        eventSlots_[i] = slot;
      }
    }
    addWarpCounter(counters_, rebinned);
    addWarpCounter(counters_ + 1, candidates);
    addWarpCounter(counters_ + 2, contributions);
  }

  __global__ void findBinOffsets(const int* keys_, std::size_t count_, std::size_t* offsets_, std::size_t bins_) {
    for( std::size_t bin = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         bin <= bins_; bin += std::size_t(blockDim.x) * gridDim.x ){
      std::size_t low = 0, high = count_;
      while( low < high ){
        const auto mid = low + (high - low) / 2;
        if( keys_[mid] < int(bin) ){ low = mid + 1; }
        else{ high = mid; }
      }
      offsets_[bin] = low;
    }
  }

  __global__ void countContributions(const Backends::EventView* events_, std::size_t count_,
      const Backends::CudaBinningSample* samples_, std::size_t sampleCount_, const Backends::BinningBin* bins_,
      const Backends::BinningInterval* intervals_, const Backends::CudaBinningNode* nodes_,
      const Backends::CudaBinningBound* bounds_, const double* variables_, const int* stateBins_,
      Backends::CudaBinningResult* results_, std::size_t* counts_, unsigned long long* counters_, int* error_) {
    unsigned long long contributions = 0;
    for( std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count_; i += std::size_t(blockDim.x) * gridDim.x ){
      const auto& event = events_[i];
      std::size_t count = event.globalBinIndex >= 0 ? 1 : 0;
      if( event.samplePosition < sampleCount_ ){
        const auto& sample = samples_[event.samplePosition];
        if( sample.isSmoothed ){
          const auto result = Backends::Semantics::analyzeCudaBinning(sample, bins_, intervals_,
              event.variableCount ? variables_ + event.firstVariable : nullptr, nodes_, bounds_);
          results_[i] = result;
          count = result.isValid ? result.count : 0;
          if( not result.isValid ){ atomicExch(error_, 1); }
          contributions += count;
        }
        else if( sample.isDynamic ){ count = stateBins_[event.resultIndex] >= 0 ? 1 : 0; }
      }
      counts_[i] = count;
    }
    addWarpCounter(counters_ + 2, contributions);
  }

  __global__ void writeContributions(const Backends::EventView* events_, std::size_t count_,
      const Backends::CudaBinningSample* samples_, std::size_t sampleCount_, const Backends::BinningBin* bins_,
      const Backends::BinningInterval* intervals_, const Backends::CudaBinningNode* nodes_,
      const Backends::CudaBinningBound* bounds_, const double* variables_, const int* stateBins_,
      const Backends::CudaBinningResult* results_, const std::size_t* offsets_,
      Backends::CudaBinContribution* contributions_, int histogramBins_, int* error_) {
    for( std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count_; i += std::size_t(blockDim.x) * gridDim.x ){
      const auto first = offsets_[i], count = offsets_[i + 1] - first;
      if( count == 0 ){ continue; }
      const auto& event = events_[i];
      int bin = event.globalBinIndex;
      if( event.samplePosition < sampleCount_ ){
        const auto& sample = samples_[event.samplePosition];
        if( sample.isSmoothed ){
          if( not Backends::Semantics::writeCudaBinning(sample, bins_, intervals_,
              event.variableCount ? variables_ + event.firstVariable : nullptr, results_[i], event.resultIndex,
              histogramBins_ + int(event.samplePosition), contributions_ + first, count, nodes_, bounds_) ){ atomicExch(error_, 1); }
          continue;
        }
        if( sample.isDynamic ){ bin = sample.binOffset + stateBins_[event.resultIndex]; }
      }
      contributions_[first] = {event.resultIndex, bin, 1.};
    }
  }

  struct ContributionLess {
    __host__ __device__ bool operator()(const Backends::CudaBinContribution& first_, const Backends::CudaBinContribution& second_) const {
      return first_.bin < second_.bin;
    }
  };

  __global__ void findContributionOffsets(const Backends::CudaBinContribution* contributions_, std::size_t count_,
      std::size_t* offsets_, std::size_t bins_) {
    for( std::size_t bin = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         bin <= bins_; bin += std::size_t(blockDim.x) * gridDim.x ){
      std::size_t low = 0, high = count_;
      while( low < high ){
        const auto mid = low + (high - low) / 2;
        if( contributions_[mid].bin < int(bin) ){ low = mid + 1; }
        else{ high = mid; }
      }
      offsets_[bin] = low;
    }
  }

  template<bool interleaved_>
  __global__ void reweightEvents(const Backends::BackendEventWeightDescriptor* events_, std::size_t count_,
                                 const std::uint32_t* indices_, const double* responses_, double* weights_,
                                 bool capped_, double cap_, int* error_) {
    for( std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count_; i += std::size_t(blockDim.x) * gridDim.x ){
      const auto& event = events_[i];
      double response = 1.;
      for( std::size_t j = 0; j < event.dialCount; ++j ){
        response *= responses_[indices_[event.firstDial + j * (interleaved_ ? 32 : 1)]];
      }
      if( capped_ and response > cap_ ){ response = cap_; }
      weights_[i] = event.baseWeight * response;
      if( not ::isfinite(weights_[i]) ){ atomicExch(error_, 1); }
    }
  }

  __global__ void evaluateLikelihoodBins(const Backends::LikelihoodBin<double>* bins_, std::size_t count_,
      const double* sums_, const double* squares_, double* partials_) {
    __shared__ double values[256];
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    double value = 0.;
    if( index < count_ ){
      const auto bin = bins_[index];
      if( !bin.ignored ){
        const double error = sqrt(squares_[bin.globalBin]);
        const double variance = bin.options.useNominalVariance ? bin.nominalVariance : error * error;
        value = Backends::LikelihoodSemantics::evaluate<Backends::LikelihoodSemantics::DoubleMath>(
            bin.data, sums_[bin.globalBin], variance, bin.options);
      }
    }
    values[threadIdx.x] = value;
    __syncthreads();
    for( unsigned stride = blockDim.x / 2; stride; stride /= 2 ){
      if( threadIdx.x < stride ){ values[threadIdx.x] += values[threadIdx.x + stride]; }
      __syncthreads();
    }
    if( threadIdx.x == 0 ){ partials_[blockIdx.x] = values[0]; }
  }
  __global__ void finishLikelihood(const double* partials_, std::size_t count_, double* total_) {
    double total = 0.;
    for( std::size_t i = 0; i < count_; ++i ){ total += partials_[i]; }
    *total_ = total;
  }

  template<bool fractional_>
  __global__ void reduceHistograms(const std::size_t* offsets_, const std::size_t* events_,
                                   std::size_t bins_, const double* weights_, double* sums_,
                                   double* squares_, int* error_, const Backends::CudaBinContribution* contributions_) {
    __shared__ double sum[blockSize];
    __shared__ double square[blockSize];
    // No floating-point atomics: each bin has one deterministic reduction tree.
    for( std::size_t bin = blockIdx.x; bin < bins_; bin += gridDim.x ){
      double value = 0., valueSquared = 0.;
      for( std::size_t j = offsets_[bin] + threadIdx.x; j < offsets_[bin + 1]; j += blockDim.x ){
        double weight;
        if constexpr( fractional_ ){
          const auto& contribution = contributions_[j];
          weight = weights_[contribution.event] * contribution.fraction;
        }
        else{ weight = weights_[events_[j]]; }
        value += weight;
        valueSquared += weight * weight;
      }
      sum[threadIdx.x] = value;
      square[threadIdx.x] = valueSquared;
      __syncthreads();
      for( unsigned int stride = blockSize / 2; stride != 0; stride /= 2 ){
        if( threadIdx.x < stride ){
          sum[threadIdx.x] += sum[threadIdx.x + stride];
          square[threadIdx.x] += square[threadIdx.x + stride];
        }
        __syncthreads();
      }
      if( threadIdx.x == 0 ){
        sums_[bin] = sum[0];
        squares_[bin] = square[0];
        if( not ::isfinite(sum[0]) or not ::isfinite(square[0]) ){ atomicExch(error_, 1); }
      }
      __syncthreads();
    }
  }
}

struct Backends::CudaDeviceContext::Impl {
  int deviceIndex{0};
  std::string deviceName{};
  cudaStream_t stream{nullptr};
  std::array<cudaEvent_t, 8> stages{};
  std::unique_ptr<WeightBuffers> buffers{};

  ~Impl() {
    int previousDevice = 0;
    if( cudaGetDevice(&previousDevice) != cudaSuccess or cudaSetDevice(deviceIndex) != cudaSuccess ){ return; }
    if( stream ){ cudaStreamSynchronize(stream); }
    buffers.reset();
    for( auto event : stages ){ if( event ){ cudaEventDestroy(event); } }
    if( stream ){ cudaStreamDestroy(stream); }
    cudaSetDevice(previousDevice);
  }
};

Backends::CudaDeviceContext::CudaDeviceContext(int deviceIndex_) : _impl_(std::make_unique<Impl>()) {
  int deviceCount = 0;
  checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount (run inside a GPU allocation)");
  if( deviceCount == 0 ){ throw std::runtime_error("CudaBackend: no CUDA device visible. Run inside a GPU allocation."); }
  if( deviceIndex_ < 0 or deviceIndex_ >= deviceCount ){
    throw std::runtime_error("CudaBackend: deviceIndex " + std::to_string(deviceIndex_)
                             + " is outside the visible device range [0, " + std::to_string(deviceCount) + ").");
  }
  _impl_->deviceIndex = deviceIndex_;
  DeviceScope scope(deviceIndex_);
  cudaDeviceProp properties{};
  checkCuda(cudaGetDeviceProperties(&properties, deviceIndex_), "cudaGetDeviceProperties");
  _impl_->deviceName = properties.name;
  checkCuda(cudaStreamCreateWithFlags(&_impl_->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  for( auto& event : _impl_->stages ){ checkCuda(cudaEventCreate(&event), "cudaEventCreate"); }
}

Backends::CudaDeviceContext::~CudaDeviceContext() = default;
const std::string& Backends::CudaDeviceContext::getDeviceName() const { return _impl_->deviceName; }

void Backends::CudaDeviceContext::build(const CudaWeightModel& model_, const PropagationView& original_, const LikelihoodView& likelihood_) {
  DeviceScope scope(_impl_->deviceIndex);
  _impl_->buffers.reset();
  auto buffers = std::make_unique<WeightBuffers>();
  auto& b = *buffers;
  try{
    b.events.upload(model_.events, _impl_->stream);
    b.dials.upload(model_.dials, _impl_->stream);
    b.inputs.upload(model_.inputs, _impl_->stream);
    b.payloads.upload(model_.payloads, _impl_->stream);
    b.formulaPrograms.upload(model_.formulaPrograms, _impl_->stream);
    b.formulaInstructions.upload(model_.formulaInstructions, _impl_->stream);
    b.formulaConstants.upload(model_.formulaConstants, _impl_->stream);
    b.formulaInputs.upload(model_.formulaInputs, _impl_->stream);
    b.compositePrograms.upload(model_.compositePrograms, _impl_->stream);
    b.compositeInstances.upload(model_.compositeInstances, _impl_->stream);
    b.compositeNodeData.upload(model_.compositeNodeData, _impl_->stream);
    b.compositeInstructions.upload(model_.compositeInstructions, _impl_->stream);
    b.compositeInputs.upload(model_.compositeInputs, _impl_->stream);
    b.programScratchOffsets.upload(model_.programScratchOffsets, _impl_->stream);
    b.programScratch.allocate(model_.programScratchSize);
    b.dialIndices.upload(model_.eventDialIndices, _impl_->stream);
    b.parameterDialIndices.upload(model_.parameterDialIndices, _impl_->stream);
    b.externalGroups = model_.externalGroups;
    b.externalDialCount = model_.externalDialCount;
    b.externalGenerations.assign(model_.externalWeightBlocks.size(), 0);
    b.currentExternalGenerations.resize(model_.externalWeightBlocks.size());
    b.externalUpdatePlan.ranges.reserve(b.externalGroups.size());
    b.externalResponseUpdates.allocate(b.externalGroups.size());
    b.externalBlocks.upload(model_.externalWeightBlocks, _impl_->stream);
    b.tableTerms.upload(model_.tableTerms, _impl_->stream);
    b.externalWeights.allocate(model_.externalWeightCount);
    b.parameterDialOffsets = model_.parameterDialOffsets;
    if( not model_.formulaPrograms.empty() or not model_.compositePrograms.empty() ){ b.programDispatchIndices = model_.parameterDialIndices; }
    b.previousParameters.resize(model_.parameterCount);
    b.surfaceGroups = model_.surfaceGroups;
    b.multiInputGroups = model_.multiInputGroups;
    b.surfaceOffset = model_.surfaceOffset;
    b.updatePlan.ranges.reserve(model_.parameterCount + b.surfaceGroups.size() + b.multiInputGroups.size());
    b.responseUpdates.allocate(model_.parameterCount + b.surfaceGroups.size() + b.multiInputGroups.size());
    b.histogramBins = model_.binOffsets.size() - 1;
    b.hasSmoothedBinning = model_.hasSmoothedBinning;
    const auto histogramCount = b.histogramBins + (b.hasSmoothedBinning ? model_.samples.size() : 0);
    if( b.hasSmoothedBinning ){ b.binOffsets.allocate(histogramCount + 1); }
    else{ b.binOffsets.upload(model_.binOffsets, _impl_->stream); }
    b.hasEventState = model_.hasEventState;
    b.onlyFastPrograms = model_.onlyFastPrograms;
    if( b.hasEventState ){
      b.originalModel = &original_;
      b.stateCache.build(original_);
      b.stateEvents.upload(model_.stateEvents, _impl_->stream);
      b.samples.upload(model_.samples, _impl_->stream);
      b.bins.upload(model_.bins, _impl_->stream);
      b.intervals.upload(model_.intervals, _impl_->stream);
      b.binningNodes.upload(model_.binningNodes, _impl_->stream);
      b.binningBounds.upload(model_.binningBounds, _impl_->stream);
      b.variableResponses.upload(model_.variableResponses, _impl_->stream);
      b.variableDialIndices.upload(model_.variableDialIndices, _impl_->stream);
      b.variables.upload(model_.variableValues, _impl_->stream);
      b.nextVariables.allocate(model_.variableValues.size());
      b.hostVariables.allocate(model_.variableValues.size());
      b.hostEdits.allocate(model_.variableValues.size());
      b.hostEditFlags.resize(model_.variableValues.size());
      b.stateBins.allocate(model_.events.size());
      b.nextStateBins.allocate(model_.events.size());
      if( b.stateBins.size ){ checkCuda(cudaMemsetAsync(b.stateBins.data, 0xff, b.stateBins.size * sizeof(int), _impl_->stream), "initialize event bins"); }
      if( b.hasSmoothedBinning ){
        b.contributionCounts.allocate(model_.events.size() + 1);
        b.contributionOffsets.allocate(model_.events.size() + 1);
        b.nextContributionOffsets.allocate(model_.events.size() + 1);
        b.binningResults.allocate(model_.events.size());
      }
      else{
        b.histogramKeys.allocate(model_.events.size());
        b.binEvents.allocate(model_.events.size());
      }
      b.changedSamples.allocate(model_.samples.size());
      b.sampleChanges.resize(model_.samples.size());
      b.changedSampleFlags.resize(model_.samples.size());
      b.stateCounters.allocate(3);
    }
    else{ b.binEvents.upload(model_.binEvents, _impl_->stream); }
    b.parameters.allocate(model_.parameterCount);
    b.responses.allocate(model_.dials.size());
    b.previousInputs.allocate(model_.dials.size());
    b.previousSecondInputs.allocate(model_.dials.size() - model_.surfaceOffset);
    b.weights.allocate(model_.events.size());
    b.sums.allocate(histogramCount);
    b.sumSquares.allocate(b.sums.size);
    b.error.allocate(1);
    const auto likelihoodBins = packLikelihood<double>(likelihood_, original_.totalBins);
    b.likelihoodBins.upload(likelihoodBins, _impl_->stream);
    b.likelihoodPartials.allocate((likelihoodBins.size() + 255) / 256);
    const std::vector<double> initialLikelihood{0.};
    b.likelihoodTotal.upload(initialLikelihood, _impl_->stream);
    b.hasReweightCap = model_.hasReweightCap;
    b.maxReweight = model_.maxReweight;
    b.interleavedEventDials = model_.eventDialStride == 32;
    checkCuda(cudaStreamSynchronize(_impl_->stream), "model upload completion");
  }
  catch(...){ cudaStreamSynchronize(_impl_->stream); throw; }
  _impl_->buffers = std::move(buffers);
}

void Backends::CudaDeviceContext::propagate(const PropagationInputs& inputs_, bool recomputeResponses_,
                                           BackendTimingSummary& timing_) {
  DeviceScope scope(_impl_->deviceIndex);
  if( not _impl_->buffers ){ throw std::runtime_error("CudaBackend device model has not been built."); }
  auto& b = *_impl_->buffers;
  const auto& parameters_ = inputs_.parameters.values;
  const auto& externalWeights_ = inputs_.externalWeights;
  if( parameters_.size() != b.parameters.size ){ throw std::runtime_error("CUDA parameter count mismatch."); }
  if( externalWeights_.size() != b.externalGenerations.size() ){ throw std::runtime_error("CUDA external block count mismatch."); }
  int error = 0;
  std::array<unsigned long long, 3> stateCounters{};
  std::size_t stagedContributionCount = 0;
  const auto stream = _impl_->stream;
  auto record = [&](std::size_t i_){ checkCuda(cudaEventRecord(_impl_->stages[i_], stream), "cudaEventRecord"); };
  try{
    const bool reuse = b.cachePrimed and not recomputeResponses_;
    if( b.hasEventState ){ b.stateCache.prepare(inputs_); }
    const bool updateState = b.hasEventState and b.stateCache.needsUpdate(inputs_);
    const auto preparationStart = std::chrono::steady_clock::now();
    if( reuse ){ b.updatePlan.select(b.parameterDialOffsets, b.previousParameters, parameters_, b.surfaceGroups, b.multiInputGroups, b.programDispatchIndices); }
    for( std::size_t i = 0; i < externalWeights_.size(); ++i ){
      b.currentExternalGenerations[i] = externalWeights_[i].generation;
    }
    b.externalUpdatePlan.selectExternal(b.externalGroups, b.externalGenerations, b.currentExternalGenerations, not reuse);
    // Dense updates retain the original contiguous traversal. Sparse dispatch
    // avoids a full dial scan and uploads only one range per changed parameter.
    const bool directPrograms = not b.programDispatchIndices.empty() and not b.updatePlan.ranges.empty()
        and std::all_of(b.updatePlan.ranges.begin(), b.updatePlan.ranges.end(), [](const auto& range_){ return range_.direct; });
    const bool sparse = reuse and (b.updatePlan.dialCount <= (b.dials.size - b.externalDialCount) / 2
        or (directPrograms and b.updatePlan.dialCount < b.dials.size - b.externalDialCount));
    const auto candidates = sparse ? b.updatePlan.dialCount : b.dials.size - b.externalDialCount;
    timing_.cacheCandidateCount = candidates + b.externalUpdatePlan.dialCount;
    timing_.cachePreparationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - preparationStart).count();
    record(0);
    if( sparse and not b.updatePlan.ranges.empty() ){
      checkCuda(cudaMemcpyAsync(b.responseUpdates.data, b.updatePlan.ranges.data(),
          b.updatePlan.ranges.size() * sizeof(CudaResponseUpdate), cudaMemcpyHostToDevice, stream), "response update ranges");
    }
    if( not b.externalUpdatePlan.ranges.empty() ){
      checkCuda(cudaMemcpyAsync(b.externalResponseUpdates.data, b.externalUpdatePlan.ranges.data(),
          b.externalUpdatePlan.ranges.size() * sizeof(CudaResponseUpdate), cudaMemcpyHostToDevice, stream), "external response ranges");
    }
    const auto externalUploadStart = std::chrono::steady_clock::now();
    for( std::size_t i = 0; i < externalWeights_.size(); ++i ){
      const auto& block = externalWeights_[i];
      if( b.cachePrimed and b.externalGenerations[i] == block.generation ){ continue; }
      if( block.count ){
        checkCuda(cudaMemcpyAsync(b.externalWeights.data + block.destinationOffset, block.values,
            block.count * sizeof(double), cudaMemcpyHostToDevice, stream), "external weight upload");
      }
      timing_.externalWeightUploadBytes += block.count * sizeof(double);
      ++timing_.externalWeightUploadBlocks;
    }
    timing_.externalWeightUploadSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - externalUploadStart).count();
    checkCuda(cudaMemsetAsync(b.error.data, 0, sizeof(int), stream), "reset execution error");
    if( b.parameters.size ){
      checkCuda(cudaMemcpyAsync(b.parameters.data, parameters_.data(), b.parameters.size * sizeof(double),
                                cudaMemcpyHostToDevice, stream), "parameter upload");
    }
    record(1);
    auto launchResponses = [&](auto surfaceTag_, auto directTag_){
      constexpr bool withSurfaces = decltype(surfaceTag_)::value;
      constexpr int programMode = decltype(directTag_)::value;
      if( sparse and candidates ){
        const auto blocks = static_cast<unsigned int>(std::min<std::size_t>(b.updatePlan.blockCount, 65535));
        evaluateSelectedResponses<withSurfaces, programMode><<<blocks, blockSize, 0, stream>>>(b.responseUpdates.data,
            b.updatePlan.ranges.size(), b.updatePlan.blockCount, b.parameterDialIndices.data,
            b.dials.data, b.inputs.data, b.payloads.data, b.parameters.data,
            b.responses.data, b.previousInputs.data, b.error.data, b.previousSecondInputs.data, b.surfaceOffset, b.programView());
        checkCuda(cudaGetLastError(), "selected dial response kernel launch");
      }
      else if( not sparse and candidates ){
        evaluateResponses<withSurfaces, programMode><<<gridSize(b.dials.size), blockSize, 0, stream>>>(b.dials.data, b.dials.size,
            b.inputs.data, b.payloads.data, b.parameters.data, b.responses.data, b.previousInputs.data,
            reuse, b.error.data, b.previousSecondInputs.data, b.surfaceOffset, b.programView());
        checkCuda(cudaGetLastError(), "dial response kernel launch");
      }
    };
    auto launch = [&](auto directTag_){
      if( b.surfaceOffset < b.dials.size ){ launchResponses(std::true_type{}, directTag_); }
      else{ launchResponses(std::false_type{}, directTag_); }
    };
    if( b.onlyFastPrograms ){ launch(std::integral_constant<int, 2>{}); }
    else if( b.formulaPrograms.size or b.compositePrograms.size ){ launch(std::integral_constant<int, 1>{}); }
    else{ launch(std::integral_constant<int, 0>{}); }
    if( b.externalUpdatePlan.dialCount ){
      const auto blocks = static_cast<unsigned int>(std::min<std::size_t>(b.externalUpdatePlan.blockCount, 65535));
      evaluateExternalResponses<<<blocks, blockSize, 0, stream>>>(b.externalResponseUpdates.data,
          b.externalUpdatePlan.ranges.size(), b.externalUpdatePlan.blockCount, b.parameterDialIndices.data,
          b.dials.data, b.externalBlocks.data, b.tableTerms.data, b.externalWeights.data, b.responses.data, b.error.data);
      checkCuda(cudaGetLastError(), "external/table response kernel launch");
    }
    record(2);
    if( b.events.size ){
      if( b.interleavedEventDials ){
        reweightEvents<true><<<gridSize(b.events.size), blockSize, 0, stream>>>(b.events.data, b.events.size,
            b.dialIndices.data, b.responses.data, b.weights.data, b.hasReweightCap, b.maxReweight, b.error.data);
      }
      else{
        reweightEvents<false><<<gridSize(b.events.size), blockSize, 0, stream>>>(b.events.data, b.events.size,
            b.dialIndices.data, b.responses.data, b.weights.data, b.hasReweightCap, b.maxReweight, b.error.data);
      }
      checkCuda(cudaGetLastError(), "event weight kernel launch");
    }
    record(3);
    if( updateState ){
      // Publish observable state only after the whole propagation succeeds.
      if( b.variables.size ){
        checkCuda(cudaMemcpyAsync(b.nextVariables.data, b.variables.data, b.variables.size * sizeof(double),
                                  cudaMemcpyDeviceToDevice, stream), "stage observable values");
      }
      if( b.stateBins.size ){
        checkCuda(cudaMemcpyAsync(b.nextStateBins.data, b.stateBins.data, b.stateBins.size * sizeof(int),
                                  cudaMemcpyDeviceToDevice, stream), "stage event bins");
      }
      std::fill(b.hostEditFlags.begin(), b.hostEditFlags.end(), 0);
      if( not inputs_.hostVariables.empty() ){
        for( std::size_t i = 0; i < b.hostEditFlags.size(); ++i ){
          b.hostEditFlags[i] = b.stateCache.hostVariableChanged(i, inputs_);
        }
        if( b.hostVariables.size ){
          checkCuda(cudaMemcpyAsync(b.hostVariables.data, inputs_.hostVariables.data(),
              b.hostVariables.size * sizeof(double), cudaMemcpyHostToDevice, stream), "host observable upload");
        }
      }
      if( b.hostEdits.size ){
        checkCuda(cudaMemcpyAsync(b.hostEdits.data, b.hostEditFlags.data(), b.hostEdits.size,
                                  cudaMemcpyHostToDevice, stream), "host observable edit mask");
      }
      if( b.changedSamples.size ){ checkCuda(cudaMemsetAsync(b.changedSamples.data, 0, b.changedSamples.size * sizeof(int), stream), "reset sample changes"); }
      checkCuda(cudaMemsetAsync(b.stateCounters.data, 0, 3 * sizeof(unsigned long long), stream), "reset state counters");
      if( b.events.size ){
        updateEventState<<<gridSize(b.events.size), blockSize, 0, stream>>>(b.stateEvents.data, b.events.size,
            b.samples.data, b.samples.size, b.bins.data, b.intervals.data,
            b.binningNodes.data, b.binningBounds.data, b.variableResponses.data,
            b.variableDialIndices.data, b.responses.data, b.nextVariables.data, b.hostVariables.data, b.hostEdits.data,
            b.nextStateBins.data, b.histogramKeys.data, b.binEvents.data, b.changedSamples.data, b.stateCounters.data,
            int(b.histogramBins), b.stateCache.variableInputsChanged(), inputs_.forceBinningUpdate, not b.cachePrimed, b.error.data);
        checkCuda(cudaGetLastError(), "observable response and strict binning kernel launch");
      }
    }
    record(4);
    if( updateState and b.hasSmoothedBinning ){
      if( b.events.size ){
        countContributions<<<gridSize(b.events.size), blockSize, 0, stream>>>(b.stateEvents.data, b.events.size,
            b.samples.data, b.samples.size, b.bins.data, b.intervals.data,
            b.binningNodes.data, b.binningBounds.data, b.nextVariables.data, b.nextStateBins.data,
            b.binningResults.data, b.contributionCounts.data, b.stateCounters.data, b.error.data);
        checkCuda(cudaGetLastError(), "fractional contribution count kernel launch");
      }
      checkCuda(cudaMemsetAsync(b.contributionCounts.data + b.events.size, 0, sizeof(std::size_t), stream), "contribution count sentinel");
      thrust::exclusive_scan(thrust::cuda::par.on(stream), b.contributionCounts.data,
          b.contributionCounts.data + b.events.size + 1, b.nextContributionOffsets.data);
      checkCuda(cudaMemcpyAsync(&stagedContributionCount, b.nextContributionOffsets.data + b.events.size,
                                sizeof(std::size_t), cudaMemcpyDeviceToHost, stream), "contribution extent readback");
      checkCuda(cudaMemcpyAsync(&error, b.error.data, sizeof(int), cudaMemcpyDeviceToHost, stream), "binning validation readback");
      checkCuda(cudaStreamSynchronize(stream), "contribution count completion");
      if( error ){ throw std::runtime_error("CudaBackend: invalid fractional binning response or nonfinite result."); }
      b.nextContributions.ensureCapacity(stagedContributionCount);
      b.sortedContributions.ensureCapacity(stagedContributionCount);
      if( b.events.size ){
        writeContributions<<<gridSize(b.events.size), blockSize, 0, stream>>>(b.stateEvents.data, b.events.size,
            b.samples.data, b.samples.size, b.bins.data, b.intervals.data,
            b.binningNodes.data, b.binningBounds.data, b.nextVariables.data, b.nextStateBins.data,
            b.binningResults.data, b.nextContributionOffsets.data, b.nextContributions.data, int(b.histogramBins), b.error.data);
        checkCuda(cudaGetLastError(), "fractional contribution write kernel launch");
      }
      if( stagedContributionCount ){
        checkCuda(cudaMemcpyAsync(b.sortedContributions.data, b.nextContributions.data,
            stagedContributionCount * sizeof(CudaBinContribution), cudaMemcpyDeviceToDevice, stream), "contribution index staging");
        thrust::stable_sort(thrust::cuda::par.on(stream), b.sortedContributions.data,
            b.sortedContributions.data + stagedContributionCount, ContributionLess{});
      }
      findContributionOffsets<<<gridSize(b.sums.size + 1), blockSize, 0, stream>>>(
          b.sortedContributions.data, stagedContributionCount, b.binOffsets.data, b.sums.size);
      checkCuda(cudaGetLastError(), "fractional histogram index kernel launch");
    }
    else if( updateState ){
      if( b.events.size ){
        thrust::stable_sort_by_key(thrust::cuda::par.on(stream), b.histogramKeys.data,
            b.histogramKeys.data + b.events.size, b.binEvents.data);
      }
      findBinOffsets<<<gridSize(b.sums.size + 1), blockSize, 0, stream>>>(
          b.histogramKeys.data, b.events.size, b.binOffsets.data, b.sums.size);
      checkCuda(cudaGetLastError(), "dynamic histogram index kernel launch");
    }
    record(5);
    if( b.sums.size ){
      const auto blocks = static_cast<unsigned int>(std::min<std::size_t>(b.sums.size, 65535));
      if( b.hasSmoothedBinning ){
        reduceHistograms<true><<<blocks, blockSize, 0, stream>>>(b.binOffsets.data, nullptr,
            b.sums.size, b.weights.data, b.sums.data, b.sumSquares.data, b.error.data, b.sortedContributions.data);
      }
      else{
        reduceHistograms<false><<<blocks, blockSize, 0, stream>>>(b.binOffsets.data, b.binEvents.data,
            b.sums.size, b.weights.data, b.sums.data, b.sumSquares.data, b.error.data, nullptr);
      }
      checkCuda(cudaGetLastError(), "histogram kernel launch");
    }
    record(6);
    if( b.likelihoodBins.size ){
      evaluateLikelihoodBins<<<static_cast<unsigned>(b.likelihoodPartials.size), 256, 0, stream>>>(
          b.likelihoodBins.data, b.likelihoodBins.size, b.sums.data, b.sumSquares.data, b.likelihoodPartials.data);
      finishLikelihood<<<1, 1, 0, stream>>>(b.likelihoodPartials.data, b.likelihoodPartials.size, b.likelihoodTotal.data);
      checkCuda(cudaGetLastError(), "likelihood kernel launch");
    }
    record(7);
    checkCuda(cudaMemcpyAsync(&error, b.error.data, sizeof(int), cudaMemcpyDeviceToHost, stream), "execution error readback");
    if( updateState ){
      if( b.changedSamples.size ){
        checkCuda(cudaMemcpyAsync(b.sampleChanges.data(), b.changedSamples.data,
            b.sampleChanges.size() * sizeof(int), cudaMemcpyDeviceToHost, stream), "sample changes readback");
      }
      checkCuda(cudaMemcpyAsync(stateCounters.data(), b.stateCounters.data, 3 * sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost, stream), "event state counters readback");
    }
    checkCuda(cudaStreamSynchronize(stream), "propagation completion");
    if( error ){ throw std::runtime_error("CudaBackend: nonfinite or out-of-range numerical result."); }
    if( updateState ){
      for( std::size_t i = 0; i < b.sampleChanges.size(); ++i ){ b.changedSampleFlags[i] = b.sampleChanges[i] != 0; }
      b.stateCache.commit(inputs_, b.changedSampleFlags);
      std::swap(b.variables.data, b.nextVariables.data);
      std::swap(b.stateBins.data, b.nextStateBins.data);
      if( b.hasSmoothedBinning ){
        std::swap(b.contributions.data, b.nextContributions.data);
        std::swap(b.contributions.size, b.nextContributions.size);
        std::swap(b.contributionOffsets.data, b.nextContributionOffsets.data);
        b.contributionCount = stagedContributionCount;
      }
      timing_.rebinnedEventCount = stateCounters[0];
      timing_.binCandidateCount = stateCounters[1];
      b.cachedContributionCount = stateCounters[2];
    }
    std::copy(parameters_.begin(), parameters_.end(), b.previousParameters.begin());
    b.externalGenerations = b.currentExternalGenerations;
    b.cachePrimed = true;
    auto elapsed = [&](std::size_t first_, std::size_t last_){
      float milliseconds = 0.;
      checkCuda(cudaEventElapsedTime(&milliseconds, _impl_->stages[first_], _impl_->stages[last_]), "CUDA stage timing");
      return double(milliseconds) * 1.e-3;
    };
    timing_.binContributionCount = b.cachedContributionCount;
    timing_.contributionBufferBytes = (b.contributions.size + b.nextContributions.size + b.sortedContributions.size)
        * sizeof(CudaBinContribution);
    timing_.parameterUploadSeconds = elapsed(0, 1);
    timing_.cachedDialStageSeconds = elapsed(1, 2);
    timing_.eventWeightsStageSeconds = elapsed(2, 3);
    timing_.variableStageSeconds = updateState ? elapsed(3, 4) : 0.;
    timing_.binningStageSeconds = updateState ? elapsed(4, 5) : 0.;
    timing_.histogramStageSeconds = elapsed(5, 6);
    timing_.likelihoodDeviceSeconds = elapsed(6, 7);
    timing_.devicePropagationSeconds = elapsed(0, 7);
  }
  catch(...){ b.cachePrimed = false; b.stateCache.invalidate(); cudaStreamSynchronize(stream); throw; }
}

double Backends::CudaDeviceContext::readLikelihood() {
  DeviceScope scope(_impl_->deviceIndex);
  double total = 0.;
  checkCuda(cudaMemcpyAsync(&total, _impl_->buffers->likelihoodTotal.data, sizeof(double),
                            cudaMemcpyDeviceToHost, _impl_->stream), "likelihood total readback");
  checkCuda(cudaStreamSynchronize(_impl_->stream), "likelihood readback completion");
  if( std::isnan(total) ){ throw std::runtime_error("CudaBackend: invalid statistical likelihood."); }
  return total;
}

void Backends::CudaDeviceContext::readWeights(std::vector<double>& weights_) {
  DeviceScope scope(_impl_->deviceIndex);
  try{
    _impl_->buffers->weights.read(weights_, _impl_->stream);
    checkCuda(cudaStreamSynchronize(_impl_->stream), "weight readback completion");
  }
  catch(...){ cudaStreamSynchronize(_impl_->stream); throw; }
}

void Backends::CudaDeviceContext::readHistograms(std::vector<double>& sums_, std::vector<double>& sumSquares_,
    std::vector<double>& outsideSums_, std::vector<double>& outsideSquares_) {
  DeviceScope scope(_impl_->deviceIndex);
  const auto& b = *_impl_->buffers;
  try{
    sums_.resize(b.histogramBins);
    sumSquares_.resize(b.histogramBins);
    outsideSums_.assign(b.samples.size, 0.);
    outsideSquares_.assign(b.samples.size, 0.);
    if( b.histogramBins ){
      checkCuda(cudaMemcpyAsync(sums_.data(), b.sums.data, b.histogramBins * sizeof(double),
                                cudaMemcpyDeviceToHost, _impl_->stream), "histogram readback");
      checkCuda(cudaMemcpyAsync(sumSquares_.data(), b.sumSquares.data, b.histogramBins * sizeof(double),
                                cudaMemcpyDeviceToHost, _impl_->stream), "histogram squares readback");
    }
    if( b.hasSmoothedBinning and b.samples.size ){
      checkCuda(cudaMemcpyAsync(outsideSums_.data(), b.sums.data + b.histogramBins, b.samples.size * sizeof(double),
                                cudaMemcpyDeviceToHost, _impl_->stream), "outside histogram readback");
      checkCuda(cudaMemcpyAsync(outsideSquares_.data(), b.sumSquares.data + b.histogramBins, b.samples.size * sizeof(double),
                                cudaMemcpyDeviceToHost, _impl_->stream), "outside histogram squares readback");
    }
    checkCuda(cudaStreamSynchronize(_impl_->stream), "histogram readback completion");
  }
  catch(...){ cudaStreamSynchronize(_impl_->stream); throw; }
}

std::size_t Backends::CudaDeviceContext::readEventState(EventState& state_) {
  DeviceScope scope(_impl_->deviceIndex);
  auto& b = *_impl_->buffers;
  std::vector<std::size_t> offsets;
  std::vector<CudaBinContribution> contributions;
  try{
    b.variables.read(state_.variables, _impl_->stream);
    b.stateBins.read(state_.bins, _impl_->stream);
    if( b.hasSmoothedBinning ){
      b.contributionOffsets.read(offsets, _impl_->stream);
      contributions.resize(b.contributionCount);
      if( b.contributionCount ){
        checkCuda(cudaMemcpyAsync(contributions.data(), b.contributions.data,
            b.contributionCount * sizeof(CudaBinContribution), cudaMemcpyDeviceToHost, _impl_->stream), "fractional event state readback");
      }
    }
    checkCuda(cudaStreamSynchronize(_impl_->stream), "event state readback completion");
    state_.contributions.assign(b.events.size, {});
    const auto& model = *b.originalModel;
    for( std::size_t i = 0; i < model.events.size(); ++i ){
      const auto& event = model.events[i];
      if( event.samplePosition >= model.samples.size() ){ continue; }
      const auto& sample = model.samples[event.samplePosition];
      if( not sample.useDynamicBinning and not sample.binning.isSmoothed ){ continue; }
      auto& target = state_.contributions[event.resultIndex];
      if( b.hasSmoothedBinning ){
        for( std::size_t j = offsets[i]; j < offsets[i + 1]; ++j ){
          const auto& source = contributions[j];
          target.push_back({source.bin >= int(b.histogramBins) ? -1 : source.bin - sample.binOffset, source.fraction});
        }
      }
      else if( state_.bins[event.resultIndex] >= 0 ){
        target.push_back({state_.bins[event.resultIndex], 1.});
      }
    }
    b.stateCache.publishRevisions(state_);
    return state_.variables.size() * sizeof(double) + state_.bins.size() * sizeof(int)
        + offsets.size() * sizeof(std::size_t) + contributions.size() * sizeof(CudaBinContribution);
  }
  catch(...){ cudaStreamSynchronize(_impl_->stream); throw; }
}

void Backends::CudaDeviceContext::invalidate() {
  if( _impl_->buffers ){
    _impl_->buffers->cachePrimed = false;
    _impl_->buffers->stateCache.invalidate();
  }
}

Backends::BackendDeviceView Backends::CudaDeviceContext::getDeviceView() const {
  BackendDeviceView out;
  out.device = _impl_.get();
  const auto& b = *_impl_->buffers;
  out.eventWeights = b.weights.data;
  out.eventWeightsBytes = b.weights.size * sizeof(double);
  out.histSums = b.sums.data;
  out.histSumSquares = b.sumSquares.data;
  out.histogramBytes = b.histogramBins * sizeof(double);
  return out;
}

Backends::BackendResourceUsage Backends::CudaDeviceContext::getResourceUsage() const {
  BackendResourceUsage usage;
  if( not _impl_->buffers ){ return usage; }
  usage.available = true;
  usage.deviceName = _impl_->deviceName;
  const auto& b = *_impl_->buffers;
  usage.allocatedBytes = allocatedBufferBytes(
      b.likelihoodBins, b.likelihoodPartials, b.likelihoodTotal,
      b.formulaPrograms, b.formulaInstructions, b.formulaConstants, b.formulaInputs,
      b.compositePrograms, b.compositeInstances, b.compositeNodeData, b.compositeInstructions, b.compositeInputs, b.programScratchOffsets, b.programScratch,
      b.events, b.dials, b.inputs, b.payloads, b.parameters,
      b.responses, b.previousInputs, b.previousSecondInputs, b.weights, b.sums,
      b.sumSquares, b.dialIndices, b.parameterDialIndices, b.responseUpdates, b.externalResponseUpdates,
      b.tableTerms, b.externalBlocks, b.externalWeights, b.binOffsets, b.binEvents,
      b.error, b.stateEvents, b.samples, b.bins, b.intervals,
      b.binningNodes, b.binningBounds, b.variableResponses, b.variableDialIndices, b.variables,
      b.nextVariables, b.hostVariables, b.hostEdits, b.stateBins, b.nextStateBins,
      b.histogramKeys, b.changedSamples, b.stateCounters, b.contributionCounts, b.contributionOffsets,
      b.nextContributionOffsets, b.binningResults, b.contributions, b.nextContributions, b.sortedContributions);
  // Monitoring is best effort: a telemetry query must not abort a fit.
  try{
    DeviceScope scope(_impl_->deviceIndex);
    usage.hasDeviceMemoryInfo = cudaMemGetInfo(&usage.freeDeviceBytes, &usage.totalDeviceBytes) == cudaSuccess;
  }
  catch( const std::runtime_error& ){}
  return usage;
}
