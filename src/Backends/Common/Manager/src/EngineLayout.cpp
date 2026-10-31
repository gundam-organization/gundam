#include "EngineLayout.h"
#include "BinningGeometryBuilder.h"
#include "FormulaCompiler.h"

#include "Bilinear.h"
#include "Bicubic.h"
#include "CompactSpline.h"
#include "CompositeDial.h"
#include "DialResponseSupervisor.h"
#include "EventDialCache.h"
#include "EventFormula.h"
#include "ExternalWeightDispatcher.h"
#include "Tabulated.h"
#include "Kriged.h"
#include "GeneralSpline.h"
#include "Graph.h"
#include "Histogram.h"
#include "LikelihoodInterface.h"
#include "JointProbabilitySnapshot.h"
#include "MonotonicSpline.h"
#include "Norm.h"
#include "Parameter.h"
#include "RootFormula.h"
#include "Sample.h"
#include "SampleSet.h"
#include "Shift.h"
#include "UniformSpline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
#include <stdexcept>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {
  using CompositePayloadMap = std::unordered_map<const DialBase*, std::pair<std::size_t, std::size_t>>;

  using FormulaProgramMap = std::unordered_map<const RootFormula*, std::size_t>;

  bool compileFormula(Backends::PropagationView& model_, Backends::BackendDialDescriptor& dial_,
                      const DialBase* base_, FormulaProgramMap& programs_, const std::string& path_) {
    const auto* eventFormula = dynamic_cast<const EventFormula*>(base_);
    const auto* formula = dynamic_cast<const RootFormula*>(eventFormula == nullptr ? base_ : eventFormula->getFormula());
    if( formula == nullptr ){ return false; }
    dial_.type = Backends::BackendDialType::Formula;
    const auto inserted = programs_.emplace(formula, model_.formulaPrograms.size());
    if( inserted.second ){
      if( formula->getInputCount() < 0 ){
        throw std::runtime_error("Formula inputs have not been bound at " + path_);
      }
      const auto& root = formula->getFormula();
      const auto& indices = formula->getParameterInputIndices();
      std::vector<std::string> names(formula->getInputCount());
      std::map<std::string, double> constants;
      for( int i = 0; i < root.GetNpar(); ++i ){
        if( indices.at(i) >= 0 ){ names.at(indices[i]) = root.GetParName(i); }
        else{ constants.emplace(root.GetParName(i), root.GetParameter(i)); }
      }
      model_.formulaPrograms.push_back(Backends::FormulaCompiler::compile(
          formula->getFormulaStr(), names, constants));
    }
    dial_.formulaProgramIndex = inserted.first->second;
    dial_.firstFormulaInput = model_.formulaInputs.size();
    for( int i = 0; i < formula->getInputCount(); ++i ){
      if( eventFormula == nullptr ){ model_.formulaInputs.push_back({i, 0.}); }
      else{
        const auto& source = eventFormula->getDefinition().inputs.at(i);
        model_.formulaInputs.push_back({source.parameterIndex,
            source.eventIndex < 0 ? 0. : eventFormula->getEventValues().at(source.eventIndex)});
      }
    }
    return true;
  }

  void packSupervisor(Backends::BackendDialDescriptor& dial_, const DialResponseSupervisor& supervisor_) {
    dial_.hasMinResponse = not std::isnan(supervisor_.getMinResponse());
    dial_.hasMaxResponse = not std::isnan(supervisor_.getMaxResponse());
    auto& smoothing = dial_.smoothing;
    smoothing.minResponse = supervisor_.getMinResponse();
    smoothing.maxResponse = supervisor_.getMaxResponse();
    smoothing.minWidth = supervisor_.getMinSmoothingWidth();
    smoothing.maxWidth = supervisor_.getMaxSmoothingWidth();
    smoothing.minInverseWidth = smoothing.minWidth > 0. ? 0.5 / smoothing.minWidth : 0.;
    smoothing.maxInverseWidth = smoothing.maxWidth > 0. ? 0.5 / smoothing.maxWidth : 0.;
    dial_.minResponse = smoothing.minResponse + smoothing.minWidth;
    dial_.maxResponse = smoothing.maxResponse - smoothing.maxWidth;
  }

  std::size_t compileComposite(Backends::PropagationView& model_, const CompositeDial& composite_,
                               CompositePayloadMap& payloads_, FormulaProgramMap& formulas_) {
    using namespace Backends;
    CompositeProgram program;
    program.firstInstruction = model_.compositeInstructions.size();
    program.inputCount = composite_.getInputCount();
    auto emit = [&](CompositeInstruction instruction_, const std::vector<BackendDialInputDescriptor>& inputs_) {
      instruction_.dial.firstInput = model_.compositeInputs.size();
      instruction_.dial.inputCount = inputs_.size();
      model_.compositeInputs.insert(model_.compositeInputs.end(), inputs_.begin(), inputs_.end());
      model_.compositeInstructions.push_back(instruction_);
      return program.inputCount + model_.compositeInstructions.size() - program.firstInstruction - 1;
    };
    std::function<std::size_t(const CompositeDial&, const std::vector<std::size_t>&, const std::string&)> flatten;
    flatten = [&](const CompositeDial& graph_, const std::vector<std::size_t>& arguments_, const std::string& path_) {
      LogThrowIf(not graph_.isBound(), "Unbound composite at " << path_);
      LogThrowIf(arguments_.size() != std::size_t(graph_.getInputCount()), "Composite input count mismatch at " << path_);
      std::vector<std::size_t> results(graph_.getNodes().size());
      for( auto index : graph_.getEvaluationOrder() ){
        const auto& node = graph_.getNodes().at(index);
        const auto path = path_ + "/" + node.name;
        LogThrowIf(node.dial == nullptr, "Missing composite dial at " << path);
        std::vector<BackendDialInputDescriptor> inputs;
        for( const auto& input : node.inputs ){
          // EventFormula already owns the loaded (and mirrored) event values.
          // Its ordinary buffer contains only parameter/response inputs.
          if( not input.eventVariable.empty() ){
            LogThrowIf(dynamic_cast<const EventFormula*>(node.dial.get()) == nullptr,
                       "Missing bound event formula at " << path);
            continue;
          }
          const auto slot = input.bufferIndex >= 0 ? arguments_.at(input.bufferIndex) : results.at(input.nodeIndex);
          const auto& mirror = node.mirrorEdges.at(inputs.size());
          inputs.emplace_back(slot, not std::isnan(mirror.minValue), mirror.minValue, mirror.range);
        }
        CompositeInstruction instruction;
        auto& dial = instruction.dial;
        packSupervisor(dial, node.supervisor);
        if( const auto* nested = dynamic_cast<const CompositeDial*>(node.dial.get()) ){
          // Materialize the outer node's mirrored arguments before entering its
          // graph. Retain its supervisor as a distinct instruction on exit.
          std::vector<std::size_t> arguments;
          for( const auto& input : inputs ){ arguments.push_back(emit(CompositeInstruction{}, {input})); }
          const auto output = flatten(*nested, arguments, path);
          results[index] = emit(instruction, {BackendDialInputDescriptor(output)});
          continue;
        }
        const auto* base = node.dial.get();
        const GenericToolbox::Range* bounds = nullptr;
        if( compileFormula(model_, dial, base, formulas_, path) ){}
        else if( dynamic_cast<const Norm*>(base) ){ dial.type = BackendDialType::Norm; }
        else if( dynamic_cast<const Shift*>(base) ){ dial.type = BackendDialType::Shift; }
        else if( const auto* spline = dynamic_cast<const CompactSpline*>(base) ){
          dial.type = BackendDialType::CompactSpline; bounds = &spline->getSplineBounds();
        }
        else if( const auto* spline = dynamic_cast<const UniformSpline*>(base) ){
          dial.type = BackendDialType::UniformSpline; bounds = &spline->getSplineBounds();
        }
        else if( const auto* spline = dynamic_cast<const MonotonicSpline*>(base) ){
          dial.type = BackendDialType::MonotonicSpline; bounds = &spline->getSplineBounds();
        }
        else if( const auto* spline = dynamic_cast<const GeneralSpline*>(base) ){
          dial.type = BackendDialType::GeneralSpline; bounds = &spline->getSplineBounds();
        }
        else if( dynamic_cast<const Graph*>(base) ){ dial.type = BackendDialType::Graph; }
        else if( dynamic_cast<const Bilinear*>(base) ){ dial.type = BackendDialType::Bilinear; }
        else if( dynamic_cast<const Bicubic*>(base) ){ dial.type = BackendDialType::Bicubic; }
        else{ throw std::runtime_error("Unsupported dial type in backend composite at " + path + ": " + base->getDialTypeName()); }
        const bool surface = dial.type == BackendDialType::Bilinear or dial.type == BackendDialType::Bicubic;
        LogThrowIf(dial.type != BackendDialType::Shift and dial.type != BackendDialType::Formula and inputs.size() != (surface ? 2u : 1u),
                   "Invalid input count in backend composite at " << path);
        dial.allowExtrapolation = base->getAllowExtrapolation();
        if( bounds ){ instruction.inputMin = bounds->min; instruction.inputMax = bounds->max; }
        if( dial.type != BackendDialType::Norm and dial.type != BackendDialType::Formula ){
          auto inserted = payloads_.emplace(base, std::make_pair(model_.dialPayloads.size(), 0));
          if( inserted.second ){
            if( const auto* shift = dynamic_cast<const Shift*>(base) ){ model_.dialPayloads.push_back(shift->getShiftValue()); }
            else{
              const auto& data = base->getDialData();
              model_.dialPayloads.insert(model_.dialPayloads.end(), data.begin(), data.end());
            }
            inserted.first->second.second = model_.dialPayloads.size() - inserted.first->second.first;
          }
          dial.payloadOffset = inserted.first->second.first;
          dial.payloadSize = inserted.first->second.second;
        }
        results[index] = emit(instruction, inputs);
      }
      return results.at(graph_.getOutputIndex());
    };
    std::vector<std::size_t> arguments(program.inputCount);
    for( std::size_t i = 0; i < arguments.size(); ++i ){ arguments[i] = i; }
    program.outputSlot = flatten(composite_, arguments, "Composite");
    program.instructionCount = model_.compositeInstructions.size() - program.firstInstruction;
    model_.compositePrograms.push_back(program);
    return model_.compositePrograms.size() - 1;
  }

  void buildEngineView(Backends::EngineView& view_, Backends::EngineBindings& bindings_, LikelihoodInterface& likelihoodInterface_) {
    auto& propagation = view_.propagation;
    auto& likelihood = view_.likelihood;

    auto& sampleSet = const_cast<Propagator&>(likelihoodInterface_.getModelPropagator()).getSampleSet();
    const auto& eventDialCache = likelihoodInterface_.getModelPropagator().getEventDialCache();

    propagation.samples.reserve(sampleSet.getSampleList().size());

    std::map<int, std::size_t> samplePositionMap;
    int binOffset{0};
    for( auto& sample : sampleSet.getSampleList() ){
      Backends::SampleView sampleRef;
      sampleRef.sampleIndex = sample.getIndex();
      sampleRef.binOffset = binOffset;
      sampleRef.binCount = sample.getHistogram().getNbBins();
      sampleRef.useDynamicBinning = sample.isUseDynamicBinning();
      sampleRef.readHostVariables = sample.isUseDynamicBinning() and eventDialCache.hasVariableResponses()
          and not likelihoodInterface_.getModelPropagator().getEventVariableResponseCache().isAutomaticBinningSample(propagation.samples.size());
      if( sampleRef.useDynamicBinning or sample.getHistogram().isBinningResponseEnabled() ){
        sampleRef.binning = Backends::buildBinningGeometry(sample.getHistogram());
        sampleRef.variableNames = sampleRef.binning.variableNames;
      }
      samplePositionMap[sampleRef.sampleIndex] = propagation.samples.size();
      propagation.samples.emplace_back(std::move(sampleRef));
      binOffset += sample.getHistogram().getNbBins();
    }
    propagation.totalBins = binOffset;
    propagation.hasVariableResponses = eventDialCache.hasVariableResponses();
    propagation.hasReweightCap = eventDialCache.getGlobalEventReweightCap().isEnabled;
    propagation.maxReweight = eventDialCache.getGlobalEventReweightCap().maxReweight;
    // Every event in a sample uses the same compact variable schema. Binning
    // coordinates come first; auxiliary dial targets follow.
    for( const auto& entry : eventDialCache.getCache() ){
      if( entry.event == nullptr ){ continue; }
      auto& sample = propagation.samples.at(samplePositionMap.at(entry.event->getIndices().sample));
      for( const auto& response : entry.variableResponseCacheList ){
        const auto& name = entry.event->getVariables().getNameListPtr()->at(response.variableIndex);
        if( std::find(sample.variableNames.begin(), sample.variableNames.end(), name) == sample.variableNames.end() ){
          sample.variableNames.emplace_back(name);
        }
      }
    }

    propagation.events.reserve(eventDialCache.getCache().size());
    propagation.eventWeights.reserve(eventDialCache.getCache().size());
    std::unordered_map<const Parameter*, std::size_t> parameterIndexMap{};
    for( std::size_t i = 0; i < bindings_.parameters.size(); ++i ){
      parameterIndexMap.emplace(bindings_.parameters[i].parameter, i);
    }
    std::unordered_map<const DialInterface*, std::uint32_t> dialIndexMap{};
    // Count the flat buffers before filling them. Reuse this map during
    // compilation, preserving the first-encounter order of the dial indices.
    // This avoids geometric vector growth and its temporary payload copies.
    constexpr auto uncompiledDial = std::numeric_limits<std::uint32_t>::max();
    std::size_t payloadCount{0};
    std::size_t eventDialCount{0};
    for( const auto& entry : eventDialCache.getCache() ){
      if( entry.event == nullptr ){ continue; }
      for( const auto& response : entry.dialResponseCacheList ){
        if( not response.appliesToVariable ){ ++eventDialCount; }
        const auto* interface = response.dialInterface;
        LogThrowIf(interface == nullptr, "Null DialInterface while building EngineView.");
        if( not dialIndexMap.emplace(interface, uncompiledDial).second ){ continue; }
        const auto* dial = interface->getDialBaseRef();
        LogThrowIf(dial == nullptr, "Null DialBase while building EngineView.");
        if( dynamic_cast<const CompositeDial*>(dial) != nullptr
            or dynamic_cast<const RootFormula*>(dial) != nullptr
            or dynamic_cast<const EventFormula*>(dial) != nullptr ){ continue; }
        if( dynamic_cast<const Shift*>(dial) != nullptr ){ ++payloadCount; }
        else if( dynamic_cast<const Norm*>(dial) == nullptr
                 and dynamic_cast<const ExternalWeightDispatcher*>(dial) == nullptr
                 and dynamic_cast<const Tabulated*>(dial) == nullptr
                 and dynamic_cast<const Kriged*>(dial) == nullptr ){
          payloadCount += dial->getDialData().size();
        }
      }
    }
    LogThrowIf(dialIndexMap.size() >= uncompiledDial, "Too many dials for EngineView indices.");
    propagation.dials.reserve(dialIndexMap.size());
    propagation.dialPayloads.reserve(payloadCount);
    propagation.eventDialIndices.reserve(eventDialCount);
    // Dial collections share their input buffers. Keep each input sequence
    // once, including its parameter order and mirror settings.
    std::unordered_map<const DialInputBuffer*, std::size_t> inputOffsetMap{};
    std::unordered_map<const ExternalWeightBuffer*, std::size_t> externalSourceIndices{};
    std::unordered_map<const std::vector<double>*, std::size_t> tableSourceIndices{};
    CompositePayloadMap compositePayloads;
    FormulaProgramMap formulaPrograms;
    std::size_t externalWeightCount{0};
    std::unordered_set<const DialInputBuffer*> variableInputBuffers;

    for( const auto& cacheEntry : eventDialCache.getCache() ){
      if( cacheEntry.event == nullptr ){ continue; }

      Backends::EventView eventRef;
      Backends::BackendEventWeightDescriptor weightRef;
      eventRef.sampleIndex = cacheEntry.event->getIndices().sample;
      eventRef.binIndex = cacheEntry.event->getIndices().bin;
      eventRef.samplePosition = samplePositionMap.at(eventRef.sampleIndex);
      const auto& sample = propagation.samples.at(eventRef.samplePosition);
      eventRef.globalBinIndex = eventRef.binIndex < 0 ? -1 : sample.binOffset + eventRef.binIndex;
      weightRef.baseWeight = cacheEntry.event->getWeights().base;
      weightRef.firstDial = propagation.eventDialIndices.size();

      eventRef.resultIndex = propagation.events.size();

      std::vector<std::uint32_t> allDialIndices;
      for( const auto& dialResponse : cacheEntry.dialResponseCacheList ){
        const auto* interface = dialResponse.dialInterface;
        LogThrowIf(interface == nullptr, "Null DialInterface while building EngineView.");
        if( dialResponse.appliesToVariable ){
          const auto* buffer = interface->getInputBufferRef();
          if( buffer != nullptr and variableInputBuffers.insert(buffer).second ){
            bindings_.variableInputBuffers.emplace_back(buffer);
          }
        }
        auto cachedDialIt = dialIndexMap.find(interface);
        if( cachedDialIt->second != uncompiledDial ){
          allDialIndices.emplace_back(cachedDialIt->second);
          continue;
        }

        Backends::BackendDialDescriptor dialRef;
        const auto* dialBase = interface->getDialBaseRef();
        LogThrowIf(dialBase == nullptr, "Null DialBase while building EngineView.");
        const auto* inputBuffer = interface->getInputBufferRef();
        auto inputOffset = inputOffsetMap.emplace(inputBuffer, propagation.dialInputs.size());
        dialRef.firstInput = inputOffset.first->second;
        dialRef.inputCount = inputBuffer == nullptr ? 0 : std::size_t(inputBuffer->getBufferSize());
        dialRef.payloadOffset = propagation.dialPayloads.size();

        if( auto* supervisor = interface->getResponseSupervisorRef() ; supervisor != nullptr ){
          dialRef.hasMinResponse = not std::isnan(supervisor->getMinResponse());
          dialRef.hasMaxResponse = not std::isnan(supervisor->getMaxResponse());
          auto& smoothing = dialRef.smoothing;
          smoothing.minResponse = supervisor->getMinResponse();
          smoothing.maxResponse = supervisor->getMaxResponse();
          smoothing.minWidth = supervisor->getMinSmoothingWidth();
          smoothing.maxWidth = supervisor->getMaxSmoothingWidth();
          smoothing.minInverseWidth = smoothing.minWidth > 0. ? 0.5 / smoothing.minWidth : 0.;
          smoothing.maxInverseWidth = smoothing.maxWidth > 0. ? 0.5 / smoothing.maxWidth : 0.;
          dialRef.minResponse = smoothing.minResponse + smoothing.minWidth;
          dialRef.maxResponse = smoothing.maxResponse - smoothing.maxWidth;
        }

        if( compileFormula(propagation, dialRef, dialBase, formulaPrograms, "Formula") ){}
        else if( const auto* composite = dynamic_cast<const CompositeDial*>(dialBase) ){
          dialRef.type = Backends::BackendDialType::Composite;
          dialRef.compositeProgram = compileComposite(propagation, *composite, compositePayloads, formulaPrograms);
        }
        else if( auto* external = dynamic_cast<const ExternalWeightDispatcher*>(dialBase) ){
          const auto& source = external->getWeightSource();
          LogThrowIf(source == nullptr or external->getWeightIndex() >= source->size(),
                     "Invalid external weight source while building EngineView.");
          auto inserted = externalSourceIndices.emplace(source.get(), bindings_.externalWeights.size());
          if( inserted.second ){
            bindings_.externalWeights.push_back({source});
            propagation.externalWeightBlocks.push_back({externalWeightCount, source->size()});
            externalWeightCount += source->size();
          }
          dialRef.type = Backends::BackendDialType::ExternalWeight;
          dialRef.externalBlockIndex = inserted.first->second;
          dialRef.externalWeightIndex = external->getWeightIndex();
        }
        else if( dynamic_cast<const Tabulated*>(dialBase) != nullptr
                 or dynamic_cast<const Kriged*>(dialBase) != nullptr ){
          const auto* tabulated = dynamic_cast<const Tabulated*>(dialBase);
          const auto* kriged = dynamic_cast<const Kriged*>(dialBase);
          const auto* table = tabulated != nullptr ? tabulated->getTable() : kriged->getTable();
          LogThrowIf(table == nullptr, "Null table while building EngineView.");
          auto inserted = tableSourceIndices.emplace(table, bindings_.externalWeights.size());
          if( inserted.second ){
            bindings_.externalWeights.push_back({nullptr, table});
            propagation.externalWeightBlocks.push_back({externalWeightCount, table->size()});
            externalWeightCount += table->size();
          }
          dialRef.externalBlockIndex = inserted.first->second;
          dialRef.firstTableTerm = propagation.tableTerms.size();
          if( tabulated != nullptr ){
            dialRef.type = Backends::BackendDialType::Tabulated;
            LogThrowIf(tabulated->getIndex() < 0 or table->size() < 2
                       or std::size_t(tabulated->getIndex()) >= table->size() - 1,
                       "Invalid Tabulated table index.");
            const auto index = std::size_t(tabulated->getIndex());
            propagation.tableTerms.push_back({index, tabulated->getFraction()});
            propagation.tableTerms.push_back({index + 1, 1. - tabulated->getFraction()});
          }
          else{
            dialRef.type = Backends::BackendDialType::Kriged;
            for( const auto& weight : kriged->getWeights() ){
              LogThrowIf(weight.first < 0 or std::size_t(weight.first) >= table->size(),
                         "Invalid Kriged table index.");
              propagation.tableTerms.push_back({std::size_t(weight.first), weight.second});
            }
          }
          dialRef.tableTermCount = propagation.tableTerms.size() - dialRef.firstTableTerm;
        }
        else if( dynamic_cast<const Norm*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::Norm;
        }
        else if( dynamic_cast<const Shift*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::Shift;
          propagation.dialPayloads.emplace_back(dialBase->evalResponse(DialInputBuffer()));
        }
        else if( dynamic_cast<const CompactSpline*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::CompactSpline;
          dialRef.allowExtrapolation = dialBase->getAllowExtrapolation();
          const auto& data = dialBase->getDialData();
          propagation.dialPayloads.insert(propagation.dialPayloads.end(), data.begin(), data.end());
        }
        else if( dynamic_cast<const UniformSpline*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::UniformSpline;
          dialRef.allowExtrapolation = dialBase->getAllowExtrapolation();
          const auto& data = dialBase->getDialData();
          propagation.dialPayloads.insert(propagation.dialPayloads.end(), data.begin(), data.end());
        }
        else if( dynamic_cast<const MonotonicSpline*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::MonotonicSpline;
          dialRef.allowExtrapolation = dialBase->getAllowExtrapolation();
          const auto& data = dialBase->getDialData();
          propagation.dialPayloads.insert(propagation.dialPayloads.end(), data.begin(), data.end());
        }
        else if( dynamic_cast<const GeneralSpline*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::GeneralSpline;
          dialRef.allowExtrapolation = dialBase->getAllowExtrapolation();
          const auto& data = dialBase->getDialData();
          propagation.dialPayloads.insert(propagation.dialPayloads.end(), data.begin(), data.end());
        }
        else if( dynamic_cast<const Bilinear*>(dialBase) != nullptr or dynamic_cast<const Bicubic*>(dialBase) != nullptr ){
          dialRef.type = dynamic_cast<const Bilinear*>(dialBase) != nullptr
              ? Backends::BackendDialType::Bilinear : Backends::BackendDialType::Bicubic;
          LogThrowIf(dialRef.inputCount != 2, "Surface dials require exactly two inputs.");
          dialRef.allowExtrapolation = dialBase->getAllowExtrapolation();
          const auto& data = dialBase->getDialData();
          propagation.dialPayloads.insert(propagation.dialPayloads.end(), data.begin(), data.end());
        }
        else if( dynamic_cast<const Graph*>(dialBase) != nullptr ){
          dialRef.type = Backends::BackendDialType::Graph;
          dialRef.allowExtrapolation = dialBase->getAllowExtrapolation();
          const auto& data = dialBase->getDialData();
          propagation.dialPayloads.insert(propagation.dialPayloads.end(), data.begin(), data.end());
        }
        else{
          LogThrow("Unsupported dial type for EngineView: " << dialBase->getDialTypeName());
        }

        dialRef.payloadSize = propagation.dialPayloads.size() - dialRef.payloadOffset;
        const auto dialIndex = std::uint32_t(propagation.dials.size());
        propagation.dials.emplace_back(dialRef);
        cachedDialIt->second = dialIndex;
        allDialIndices.emplace_back(dialIndex);

        if( inputBuffer == nullptr or not inputOffset.second ){ continue; }

        for( int iInput = 0 ; iInput < inputBuffer->getBufferSize() ; iInput++ ){
          const auto* parPtr = &inputBuffer->getParameter(iInput);
          auto parameterIndexIt = parameterIndexMap.find(parPtr);
          if( parameterIndexIt == parameterIndexMap.end() ){
            parameterIndexIt = parameterIndexMap.emplace(parPtr, parameterIndexMap.size()).first;
          }

          Backends::BackendDialInputDescriptor inputRef;
          inputRef.parameterIndex = parameterIndexIt->second;
          const auto& mirrorEdges = inputBuffer->getMirrorEdges(iInput);
          inputRef.useMirror = not std::isnan(mirrorEdges.minValue);
          inputRef.mirrorMin = mirrorEdges.minValue;
          inputRef.mirrorRange = mirrorEdges.range;
          propagation.dialInputs.emplace_back(inputRef);
        }
      }

      for( std::size_t i = 0; i < allDialIndices.size(); ++i ){
        if( not cacheEntry.dialResponseCacheList[i].appliesToVariable ){
          propagation.eventDialIndices.emplace_back(allDialIndices[i]);
        }
      }
      weightRef.dialCount = propagation.eventDialIndices.size() - weightRef.firstDial;
      eventRef.firstVariable = propagation.variableValues.size();
      eventRef.variableCount = sample.variableNames.size();
      const auto& variables = cacheEntry.event->getVariables();
      for( const auto& name : sample.variableNames ){
        int index = variables.findVarIndex(name);
        bool isTarget = std::any_of(cacheEntry.variableResponseCacheList.begin(), cacheEntry.variableResponseCacheList.end(),
                                    [&](const auto& target_){ return target_.variableIndex == index; });
        bindings_.variables.push_back({cacheEntry.event, index, propagation.variableValues.size(), sample.readHostVariables, isTarget});
        propagation.variableValues.emplace_back(variables.getVarList()[index].getVarAsDouble());
      }
      eventRef.firstVariableResponse = propagation.variableResponses.size();
      for( const auto& target : cacheEntry.variableResponseCacheList ){
        Backends::BackendVariableResponseDescriptor response;
        const auto& name = variables.getNameListPtr()->at(target.variableIndex);
        response.variableIndex = eventRef.firstVariable
            + std::size_t(std::find(sample.variableNames.begin(), sample.variableNames.end(), name) - sample.variableNames.begin());
        response.referenceValue = target.referenceValue;
        if( target.overrideDial != std::size_t(-1) ){ response.overrideDial = allDialIndices.at(target.overrideDial); }
        response.firstMultiplyDial = propagation.variableDialIndices.size();
        for( auto dial : target.multiplyDials ){ propagation.variableDialIndices.emplace_back(allDialIndices.at(dial)); }
        response.multiplyDialCount = propagation.variableDialIndices.size() - response.firstMultiplyDial;
        response.firstAddDial = propagation.variableDialIndices.size();
        for( auto dial : target.addDials ){ propagation.variableDialIndices.emplace_back(allDialIndices.at(dial)); }
        response.addDialCount = propagation.variableDialIndices.size() - response.firstAddDial;
        propagation.variableResponses.emplace_back(response);
      }
      eventRef.variableResponseCount = propagation.variableResponses.size() - eventRef.firstVariableResponse;
      propagation.events.emplace_back(eventRef);
      propagation.eventWeights.emplace_back(weightRef);
    }

    propagation.parameterCount = parameterIndexMap.size();
    propagation.variableInputBufferCount = bindings_.variableInputBuffers.size();

    likelihood.samples.reserve(likelihoodInterface_.getSamplePairList().size());
    binOffset = 0;
    for( const auto& samplePair : likelihoodInterface_.getSamplePairList() ){
      Backends::LikelihoodSampleView sampleRef;
      sampleRef.binOffset = binOffset;
      sampleRef.dataSums.reserve(samplePair.data->getHistogram().getNbBins());
      sampleRef.ignoredBins.reserve(samplePair.data->getHistogram().getNbBins());

      const auto& dataBinContentList = samplePair.data->getHistogram().getBinContentList();
      const auto& modelBinContentList = samplePair.model->getHistogram().getBinContentList();
      for( int iBin = 0 ; iBin < samplePair.data->getHistogram().getNbBins() ; iBin++ ){
        sampleRef.dataSums.emplace_back(dataBinContentList[iBin].sumWeights);
        sampleRef.ignoredBins.emplace_back(
            likelihoodInterface_.getJointProbabilityPtr()->isIgnoreBinsWithZeroPredictionAtPrior()
            and modelBinContentList[iBin].sumWeights == 0.
        );
      }

      auto* jointProbability = likelihoodInterface_.getJointProbabilityPtr();
      const auto snapshot = JointProbability::snapshotJointProbability(*jointProbability, samplePair);
      auto& options = sampleRef.portableOptions;
      using Kind = JointProbability::JointProbabilitySnapshot::Kind;
      if( snapshot.kind == Kind::Poisson ){ options.kind = Backends::LikelihoodKind::Poisson; }
      else if( snapshot.kind == Kind::BarlowBeeston2022 ){ options.kind = Backends::LikelihoodKind::BarlowBeeston2022; }
      else if( snapshot.kind == Kind::BarlowBeestonLiteGaussian ){ options.kind = Backends::LikelihoodKind::BarlowBeestonLiteGaussian; }
      else if( snapshot.kind == Kind::BarlowBeestonLitePoisson ){ options.kind = Backends::LikelihoodKind::BarlowBeestonLitePoisson; }
      options.usePoissonLikelihood = snapshot.usePoissonLikelihood;
      options.useNominalVariance = snapshot.useNominalVariance;
      options.throwIfInfLlh = snapshot.throwIfInfLlh;
      options.fractionalErrorLimit = snapshot.fractionalErrorLimit;
      options.expectedValueMinimum = snapshot.expectedValueMinimum;
      sampleRef.nominalVariances = snapshot.nominalVariances;
      sampleRef.evalBin = [jointProbability, samplePair](double data_, double pred_, double err_, int bin_){
        return jointProbability->evalWithPrediction(samplePair, data_, pred_, err_, bin_);
      };

      likelihood.samples.emplace_back(std::move(sampleRef));
      binOffset += samplePair.model->getHistogram().getNbBins();
    }
  }
}

void Backends::EngineLayout::clear() {
  view.clear();
  bindings.clear();
}

void Backends::EngineLayout::build(LikelihoodInterface& likelihoodInterface_) {
  clear();
  bindings.build(likelihoodInterface_);
  buildEngineView(view, bindings, likelihoodInterface_);
}
