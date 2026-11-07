#include "CompositeDial.h"
#include "DialCollection.h"
#include "Bicubic.h"
#include "Bilinear.h"
#include "RootFormula.h"

#include "Logger.h"

#include <cmath>
#include <algorithm>
#include <map>
#include <sstream>

CompositeDial::CompositeDial(ConfigReader config_, const std::vector<std::string>& parameterNames_): _inputCount_(int(parameterNames_.size())){
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
    nodeIndices[name] = int(_nodes_.size());
    Node node;
    node.name = name;
    node.config = std::make_shared<ConfigReader>(nodeConfig);
    _nodes_.emplace_back(std::move(node));
  }
  const auto output = config_.fetchValue<std::string>("output");
  LogThrowIf(not nodeIndices.count(output), "Composite: unknown output node: " << output);
  _outputIndex_ = nodeIndices.at(output);
  for( auto& node : _nodes_ ){
    const auto inputs = node.config->fetchValue<ConfigReader>("dialInputList");
    const bool nested = node.config->fetchValue<DialCollection::DialType>("dialType") == DialCollection::DialType::Composite;
    LogThrowIf(not inputs.getConfig().is_array() or (inputs.getConfig().empty() and not nested),
               "Composite node " << node.name << ": dialInputList must be a non-empty list");
    for( auto source : inputs.loop() ){
      source.defineFields({{"name"}, {"parameterName"}, {"dialResponse"}, {"eventVariable"}});
      source.checkConfiguration();
      const int sourceCount = int(source.hasField("parameterName")) + int(source.hasField("dialResponse"))
                              + int(source.hasField("eventVariable"));
      LogThrowIf(sourceCount != 1,
                 "Composite node " << node.name << ": specify exactly one of parameterName, dialResponse and eventVariable");
      Input input;
      if( node.config->fetchValue<DialCollection::DialType>("dialType") == DialCollection::DialType::Formula ){
        input.name = source.fetchValue("name", std::string());
        LogThrowIf(input.name.empty(), "Composite Formula input requires a name");
      }
      if( source.hasField("parameterName") ){
        input.parameterName = source.fetchValue<std::string>("parameterName");
        const auto position = std::find(parameterNames_.begin(), parameterNames_.end(), input.parameterName);
        LogThrowIf(position == parameterNames_.end(), "Composite: unresolved parameter: " << input.parameterName);
        input.bufferIndex = int(std::distance(parameterNames_.begin(), position));
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
  std::vector<int> state(_nodes_.size(), 0);
  std::function<void(int)> visit = [&](int index_){
    LogThrowIf(state[index_] == 1, "Composite: dependency cycle at node " << _nodes_[index_].name);
    if( state[index_] == 2 ){ return; }
    state[index_] = 1;
    for( const auto& input : _nodes_[index_].inputs ){
      if( input.nodeIndex >= 0 ){ visit(input.nodeIndex); }
    }
    state[index_] = 2;
    _evaluationOrder_.emplace_back(index_);
  };
  visit(_outputIndex_);
  // Reject disconnected nodes rather than loading and updating unused factories.
  LogThrowIf(_evaluationOrder_.size() != _nodes_.size(), "Composite: every node must contribute to output");
}

void CompositeDial::bind(const std::function<const DialInterface*(int)>& findInterface_){
  for( auto& node : _nodes_ ){
    const auto* interface = findInterface_(node.collectionIndex);
    LogThrowIf(interface == nullptr or interface->getDialBaseRef() == nullptr,
               "Composite: node " << node.name << " has no dial for this event (dataset, condition, bin or empty dial)");
    if( const auto* composite = dynamic_cast<const CompositeDial*>(interface->getDialBaseRef());
        composite != nullptr and not composite->isBound() ){
      auto bound = std::make_shared<CompositeDial>(*composite);
      bound->bind(findInterface_);
      node.dial = std::move(bound);
    }
    else{ node.dial = interface->getDialBaseRef()->clone(); }
    node.supervisor = *interface->getResponseSupervisorRef();
    node.mirrorEdges.clear();
    for( const auto& ref : interface->getInputBufferRef()->getInputParameterIndicesList() ){
      node.mirrorEdges.emplace_back(ref.mirrorEdges);
    }
    const auto parameterInputCount = std::count_if(node.inputs.begin(), node.inputs.end(),
        [](const Input& input_){ return input_.eventVariable.empty(); });
    LogThrowIf(node.mirrorEdges.size() != parameterInputCount,
               "Composite node " << node.name << ": input buffer size mismatch");
    const auto type = node.config->fetchValue<std::string>("dialType");
    const bool surface = dynamic_cast<const Bilinear*>(node.dial.get()) != nullptr
                         or dynamic_cast<const Bicubic*>(node.dial.get()) != nullptr;
    LogThrowIf((surface or type == "Surface") and node.inputs.size() != 2,
               "Composite node " << node.name << ": Surface requires exactly two inputs");
    LogThrowIf((type == "Normalization" or type == "Norm" or type == "Graph" or type == "Spline")
               and node.inputs.size() != 1,
               "Composite node " << node.name << ": " << type << " requires exactly one input");
    if( const auto* formula = dynamic_cast<const RootFormula*>(node.dial.get()) ){
      LogThrowIf(formula->getInputCount() != node.inputs.size(),
                 "Composite node " << node.name << ": named input count mismatch");
    }
  }
  _isBound_ = true;
}

double CompositeDial::evalResponse(const DialInputBuffer& input_) const {
  LogThrowIf(not _isBound_, "Composite: graph has not been bound to an event");
  LogThrowIf(input_.getBufferSize() != _inputCount_, "Composite: input count mismatch");
  // Scratch storage belongs to the call: shared curves remain safe when
  // different events are evaluated concurrently.
  std::vector<double> responses(_nodes_.size());
  DialInputBuffer buffer;
  for( int index : _evaluationOrder_ ){
    const auto& node = _nodes_[index];
    std::vector<double> values;
    values.reserve(node.inputs.size());
    size_t iParameter = 0;
    for( const auto& source : node.inputs ){
      // EventFormula supplies its own event inputs; this buffer contains
      // only the parameter/response inputs, in their original relative order.
      if( not source.eventVariable.empty() ){ continue; }
      double value = source.bufferIndex >= 0 ? input_.getInputBuffer()[source.bufferIndex] : responses[source.nodeIndex];
      const auto& edges = node.mirrorEdges[iParameter++];
      if( not std::isnan(edges.minValue) ){
        value = std::abs(std::fmod(value - edges.minValue, 2 * edges.range));
        if( value > edges.range ){ value = 2 * edges.range - value; }
        value += edges.minValue;
      }
      values.emplace_back(value);
    }
    buffer.setInputValues(std::move(values));
    responses[index] = node.supervisor.process(node.dial->evalResponse(buffer));
  }
  return responses[_outputIndex_];
}

std::string CompositeDial::getSummary() const {
  std::stringstream ss;
  ss << "Composite: " << _nodes_.size() << " nodes, output=" << _nodes_[_outputIndex_].name;
  return ss.str();
}
