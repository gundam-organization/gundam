#include "CompositeDial.h"
#include "DialCollection.h"
#include "Bicubic.h"
#include "Bilinear.h"
#include "RootFormula.h"

#include "Logger.h"

#include <cmath>
#include <exception>
#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>

CompositeDial::CompositeDial(ConfigReader config_, const std::vector<std::string>& parameterNames_)
    : CompositeDial(std::move(config_), int(parameterNames_.size()), [&](const JsonType& input_){
        const auto name = input_.at("parameterName").get<std::string>();
        const auto position = std::find(parameterNames_.begin(), parameterNames_.end(), name);
        return position == parameterNames_.end() ? -1 : int(std::distance(parameterNames_.begin(), position));
      }){}

CompositeDial::CompositeDial(ConfigReader config_, int inputCount_, const std::function<int(const JsonType&)>& findInput_)
    : _inputCount_(inputCount_){
  config_.clearFields();
  config_.defineFields({{"nodes"}, {"output"}});
  config_.checkConfiguration();
  auto nodes = config_.fetchValue<ConfigReader>("nodes");
  LogThrowIf(not nodes.getConfig().is_array() or nodes.getConfig().empty(),
             "Composite: nodes must be a non-empty list");
  std::map<std::string, int> nodeIndices;
  for( auto nodeConfig : nodes.loop() ){
    DialCollection::prepareConfig(nodeConfig);
    const auto name = nodeConfig.fetchValue<std::string>("name");
    LogThrowIf(name.empty() or nodeIndices.count(name), "Composite: empty or duplicate node name: " << name);
    nodeIndices[name] = int(_definition_->nodes.size());
    Node node;
    node.name = name;
    node.config = std::make_shared<ConfigReader>(nodeConfig);
    _definition_->nodes.emplace_back(std::move(node));
  }
  const auto output = config_.fetchValue<std::string>("output");
  LogThrowIf(not nodeIndices.count(output), "Composite: unknown output node: " << output);
  _outputIndex_ = nodeIndices.at(output);
  for( auto& node : _definition_->nodes ){
    const auto inputs = node.config->fetchValue<ConfigReader>("dialInputList");
    const bool nested = node.config->fetchValue<DialCollection::DialType>("dialType") == DialCollection::DialType::Composite;
    LogThrowIf(not inputs.getConfig().is_array() or (inputs.getConfig().empty() and not nested),
               "Composite node " << node.name << ": dialInputList must be a non-empty list");
    for( auto source : inputs.loop() ){
      source.defineFields({{"name"}, {"parameterName"}, {"parameterIndex"}, {"parameterSetName"},
                           {"dialResponse"}, {"eventVariable"}});
      source.checkConfiguration();
      const int sourceCount = int(source.hasField("parameterName")) + int(source.hasField("parameterIndex"))
                              + int(source.hasField("dialResponse")) + int(source.hasField("eventVariable"));
      LogThrowIf(sourceCount != 1, "Composite node " << node.name << ": expected exactly one input source");
      Input input;
      if( node.config->fetchValue<DialCollection::DialType>("dialType") == DialCollection::DialType::Formula ){
        input.name = source.fetchValue("name", std::string());
        LogThrowIf(input.name.empty(), "Composite Formula input requires a name");
      }
      if( source.hasField("parameterName") or source.hasField("parameterIndex") ){
        input.bufferIndex = findInput_(source.getConfig());
        LogThrowIf(input.bufferIndex < 0 or input.bufferIndex >= _inputCount_,
                   "Composite: unresolved parameter input for node " << node.name);
      }
      else if( source.hasField("dialResponse") ){
        const auto dependency = source.fetchValue<std::string>("dialResponse");
        LogThrowIf(not nodeIndices.count(dependency),
                   "Composite node " << node.name << ": unknown dialResponse: " << dependency);
        input.nodeIndex = nodeIndices.at(dependency);
      }
      else{
        input.eventVariable = source.fetchValue<std::string>("eventVariable");
        input.eventVariable = GenericToolbox::trimString(input.eventVariable, " ");
        if( input.eventVariable.size() >= 2 and input.eventVariable.front() == '[' and input.eventVariable.back() == ']' ){
          input.eventVariable = input.eventVariable.substr(1, input.eventVariable.size() - 2);
        }
        LogThrowIf(input.eventVariable.empty(), "Composite node " << node.name << ": eventVariable must not be empty");
      }
      node.inputs.emplace_back(input);
    }
  }
  std::vector<int> state(_definition_->nodes.size(), 0);
  std::function<void(int)> visit = [&](int index_){
    LogThrowIf(state[index_] == 1, "Composite: dependency cycle at node " << _definition_->nodes[index_].name);
    if( state[index_] == 2 ){ return; }
    state[index_] = 1;
    for( const auto& input : _definition_->nodes[index_].inputs ){
      if( input.nodeIndex >= 0 ){ visit(input.nodeIndex); }
    }
    state[index_] = 2;
    _definition_->evaluationOrder.emplace_back(index_);
  };
  visit(_outputIndex_);
  // Reject disconnected nodes rather than loading and updating unused factories.
  LogThrowIf(_definition_->evaluationOrder.size() != _definition_->nodes.size(), "Composite: every node must contribute to output");
}

std::shared_ptr<const CompositeDial::NodeSettings> CompositeDial::NodeSettingsCache::get(
    const DialInterface& interface_, size_t inputCount_,
    const std::optional<DialInputBuffer::ParameterReference::MirrorEdges>& mirror_){
  const auto& supervisor = *interface_.getResponseSupervisorRef();
  const auto& inputs = interface_.getInputBufferRef()->getInputParameterIndicesList();
  LogThrowIf(not mirror_ and inputs.size() != inputCount_, "Composite node input buffer size mismatch");
  // Configured nodes use collection-wide mirroring, including response slots.
  // A directly bound interface can provide individual mirrors for each slot.
  const auto mirrorAt = [&](size_t index_) -> const auto& {
    return mirror_ ? *mirror_ : inputs[index_].mirrorEdges;
  };
  const auto equal = [](double left_, double right_){
    return left_ == right_ or (std::isnan(left_) and std::isnan(right_));
  };
  std::lock_guard<std::mutex> lock(_mutex_);
  for( const auto& settings : _entries_ ){
    if( not settings->supervisor.isEquivalentTo(supervisor) or settings->mirrorEdges.size() != inputCount_ ){ continue; }
    bool matches = true;
    for( size_t index = 0; index < inputCount_; ++index ){
      const auto& stored = settings->mirrorEdges[index];
      const auto& requested = mirrorAt(index);
      if( not equal(stored.minValue, requested.minValue) or not equal(stored.range, requested.range) ){
        matches = false;
        break;
      }
    }
    if( matches ){ return settings; }
  }
  auto settings = std::make_shared<NodeSettings>();
  settings->supervisor = supervisor;
  settings->mirrorEdges.reserve(inputCount_);
  for( size_t index = 0; index < inputCount_; ++index ){ settings->mirrorEdges.emplace_back(mirrorAt(index)); }
  _entries_.emplace_back(settings);
  return settings;
}

void CompositeDial::bind(const std::function<const DialInterface*(int)>& findInterface_){
  _nodeBindings_.resize(_definition_->nodes.size());
  for( size_t index = 0; index < _definition_->nodes.size(); ++index ){
    const auto& node = _definition_->nodes[index];
    auto& binding = _nodeBindings_[index];
    const auto* interface = findInterface_(node.collectionIndex);
    LogThrowIf(interface == nullptr or interface->getDialBaseRef() == nullptr,
               "Composite: node " << node.name << " has no dial for this event (dataset, condition, bin or empty dial)");
    if( const auto* composite = dynamic_cast<const CompositeDial*>(interface->getDialBaseRef());
        composite != nullptr and not composite->isBound() ){
      auto bound = std::make_shared<CompositeDial>(*composite);
      bound->bind(findInterface_);
      binding.dial = std::move(bound);
    }
    else{ binding.dial = interface->getDial().dialPtr; }
    const auto parameterInputCount = std::count_if(node.inputs.begin(), node.inputs.end(),
        [](const Input& input_){ return input_.eventVariable.empty(); });
    binding.settings = node.settingsCache->get(*interface, parameterInputCount, node.mirrorEdges);
    const auto type = node.config->fetchValue<std::string>("dialType");
    const bool surface = dynamic_cast<const Bilinear*>(binding.dial.get()) != nullptr
                         or dynamic_cast<const Bicubic*>(binding.dial.get()) != nullptr;
    LogThrowIf((surface or type == "Surface") and node.inputs.size() != 2,
               "Composite node " << node.name << ": Surface requires exactly two inputs");
    LogThrowIf((type == "Normalization" or type == "Norm" or type == "Graph" or type == "Spline")
               and node.inputs.size() != 1,
               "Composite node " << node.name << ": " << type << " requires exactly one input");
    if( const auto* formula = dynamic_cast<const RootFormula*>(binding.dial.get()) ){
      LogThrowIf(formula->getInputCount() != node.inputs.size(),
                 "Composite node " << node.name << ": named input count mismatch");
    }
  }
  _isBound_ = true;
}

DialInputBuffer CompositeDial::makeNodeInputs(
    const Node& node_, const NodeBinding& binding_, const DialInputBuffer& input_, const std::vector<double>& responses_,
    std::vector<double>* mirrorGradients_) const {
  std::vector<double> values;
  values.reserve(node_.inputs.size());
  size_t iParameter = 0;
  for( const auto& source : node_.inputs ){
    // EventFormula supplies its loaded event values itself.
    if( not source.eventVariable.empty() ){ continue; }
    double value = source.bufferIndex >= 0 ? input_.getInputBuffer()[source.bufferIndex]
                                          : responses_[source.nodeIndex];
    const auto& edges = binding_.settings->mirrorEdges[iParameter++];
    double slope = 1.;
    if( not std::isnan(edges.minValue) ){
      const double raw = std::fmod(value - edges.minValue, 2 * edges.range);
      value = std::abs(raw);
      slope = raw < 0 ? -1. : 1.;
      if( value > edges.range ){
        value = 2 * edges.range - value;
        slope = -slope;
      }
      value += edges.minValue;
    }
    values.emplace_back(value);
    if( mirrorGradients_ != nullptr ){ mirrorGradients_->emplace_back(slope); }
  }
  DialInputBuffer buffer;
  buffer.setInputValues(std::move(values));
  return buffer;
}

double CompositeDial::evalResponse(const DialInputBuffer& input_) const {
  LogThrowIf(not _isBound_, "Composite: graph has not been bound to an event");
  LogThrowIf(input_.getBufferSize() != _inputCount_, "Composite: input count mismatch");
  std::vector<double> responses(_definition_->nodes.size());
  for( int index : _definition_->evaluationOrder ){
    const auto& node = _definition_->nodes[index];
    const auto& binding = _nodeBindings_[index];
    const auto buffer = makeNodeInputs(node, binding, input_, responses);
    responses[index] = binding.settings->supervisor.process(binding.dial->evalResponse(buffer));
  }
  return responses[_outputIndex_];
}

bool CompositeDial::hasAnalyticGradient() const {
  LogThrowIf(not _isBound_, "Composite: analytic gradient requires an event-bound graph");
  for( int index : _definition_->evaluationOrder ){
    const auto& node = _definition_->nodes[index];
    const auto& binding = _nodeBindings_[index];
    try{
      LogThrowIf(binding.dial == nullptr or not binding.dial->hasAnalyticGradient(),
                 "Unsupported analytic gradient");
      for( const auto& edges : binding.settings->mirrorEdges ){
        LogThrowIf(not std::isnan(edges.minValue) and
                   (not std::isfinite(edges.minValue) or not std::isfinite(edges.range) or edges.range <= 0.),
                   "Invalid mirror bounds for composite gradient");
      }
    }
    catch( const std::exception& error ){
      // LogThrow stringifies its stream expression for what(), losing the
      // evaluated node name. Preserve the complete path for callers too.
      const std::string message = "Composite node " + node.name + ": " + error.what();
      LogError << message << std::endl;
      throw std::runtime_error(message);
    }
  }
  return true;
}

std::vector<double> CompositeDial::evalGradients(const DialInputBuffer& input_) const {
  LogThrowIf(not _isBound_, "Composite: graph has not been bound to an event");
  LogThrowIf(input_.getBufferSize() != _inputCount_, "Composite: input count mismatch");
  // All scratch is call-local: bound curves and graphs may be shared by workers.
  std::vector<double> responses(_definition_->nodes.size());
  std::vector<DialInputBuffer> inputs(_definition_->nodes.size());
  std::vector<std::vector<double>> mirrorGradients(_definition_->nodes.size());
  std::vector<double> supervisorGradients(_definition_->nodes.size());
  for( int index : _definition_->evaluationOrder ){
    const auto& node = _definition_->nodes[index];
    const auto& binding = _nodeBindings_[index];
    inputs[index] = makeNodeInputs(node, binding, input_, responses, &mirrorGradients[index]);
    const double raw = binding.dial->evalResponse(inputs[index]);
    responses[index] = binding.settings->supervisor.process(raw);
    supervisorGradients[index] = binding.settings->supervisor.evalDerivative(raw);
  }
  std::vector<double> adjoints(_definition_->nodes.size(), 0.);
  std::vector<double> result(_inputCount_, 0.);
  adjoints[_outputIndex_] = 1.;
  for( auto position = _definition_->evaluationOrder.rbegin(); position != _definition_->evaluationOrder.rend(); ++position ){
    const int index = *position;
    if( supervisorGradients[index] == 0. or adjoints[index] == 0. ){ continue; }
    const auto& node = _definition_->nodes[index];
    const auto& binding = _nodeBindings_[index];
    const auto local = binding.dial->evalGradients(inputs[index]);
    LogThrowIf(local.size() != mirrorGradients[index].size(),
               "Composite node " << node.name << ": gradient size mismatch");
    size_t iParameter = 0;
    for( const auto& source : node.inputs ){
      if( not source.eventVariable.empty() ){ continue; }
      const double contribution = adjoints[index] * supervisorGradients[index] * local[iParameter] * mirrorGradients[index][iParameter];
      ++iParameter;
      if( source.bufferIndex >= 0 ){ result[source.bufferIndex] += contribution; }
      else{ adjoints[source.nodeIndex] += contribution; }
    }
  }
  return result;
}

double CompositeDial::evalGradient(const DialInputBuffer& input_, int iInput_) const {
  LogThrowIf(iInput_ < 0 or iInput_ >= _inputCount_, "Composite: invalid gradient input index " << iInput_);
  return evalGradients(input_)[iInput_];
}

std::string CompositeDial::getSummary() const {
  std::stringstream ss;
  ss << "Composite: " << _definition_->nodes.size() << " nodes, output=" << _definition_->nodes[_outputIndex_].name;
  return ss.str();
}
