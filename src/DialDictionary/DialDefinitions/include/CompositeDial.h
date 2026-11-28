#ifndef GUNDAM_COMPOSITE_DIAL_H
#define GUNDAM_COMPOSITE_DIAL_H

#include "DialBase.h"
#include "DialInterface.h"
#include "ConfigUtils.h"

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// The collection holds an unbound graph. EventDialCache binds each node to
// the ordinary dial selected for that event by DataDispenser.
class CompositeDial : public DialBase {
public:
  struct Input {
    std::string name;
    int bufferIndex{-1};
    int nodeIndex{-1};
    std::string eventVariable;
  };
  struct NodeSettings {
    DialResponseSupervisor supervisor;
    std::vector<DialInputBuffer::ParameterReference::MirrorEdges> mirrorEdges;
  };
  class NodeSettingsCache {
  public:
    std::shared_ptr<const NodeSettings> get(
        const DialInterface& interface_,
        const std::optional<DialInputBuffer::ParameterReference::MirrorEdges>& configuredMirrorEdges_,
        size_t parameterInputCount_);
  private:
    std::mutex _mutex_;
    std::vector<std::shared_ptr<const NodeSettings>> _entries_;
  };
  struct Node {
    std::string name;
    std::shared_ptr<const ConfigReader> config;
    std::vector<Input> inputs;
    int collectionIndex{-1};
    std::optional<DialInputBuffer::ParameterReference::MirrorEdges> configuredMirrorEdges;
    std::shared_ptr<NodeSettingsCache> settingsCache{std::make_shared<NodeSettingsCache>()};
  };
  struct NodeBinding {
    std::shared_ptr<const DialBase> dial;
    std::shared_ptr<const NodeSettings> settings;
  };

  CompositeDial(ConfigReader config_, const std::vector<std::string>& parameterNames_);
  CompositeDial(ConfigReader config_, int inputCount_, const std::function<int(const JsonType&)>& findInput_);
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<CompositeDial>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return "Composite"; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  [[nodiscard]] bool hasAnalyticGradient() const override;
  [[nodiscard]] double evalGradient(const DialInputBuffer& input_, int iInput_) const override;
  [[nodiscard]] std::vector<double> evalGradients(const DialInputBuffer& input_) const override;
  [[nodiscard]] std::string getSummary() const override;

  auto& getNodes(){
    if( _definition_.use_count() != 1 ){ _definition_ = std::make_shared<Definition>(*_definition_); }
    return _definition_->nodes;
  }
  [[nodiscard]] const auto& getNodes() const { return _definition_->nodes; }
  [[nodiscard]] const auto& getEvaluationOrder() const { return _definition_->evaluationOrder; }
  [[nodiscard]] int getOutputIndex() const { return _outputIndex_; }
  [[nodiscard]] bool isBound() const { return _isBound_; }
  [[nodiscard]] int getInputCount() const { return _inputCount_; }
  auto& getNodeBindings(){ return _nodeBindings_; }
  [[nodiscard]] const auto& getNodeBindings() const { return _nodeBindings_; }
  void bind(const std::function<const DialInterface*(int)>& findInterface_);

private:
  [[nodiscard]] DialInputBuffer makeNodeInputs(const Node& node_, const NodeBinding& binding_, const DialInputBuffer& input_,
      const std::vector<double>& responses_, std::vector<double>* mirrorGradients_ = nullptr) const;
  int _inputCount_{0};
  int _outputIndex_{-1};
  bool _isBound_{false};
  // Definition copies detach only during collection setup. Event bindings keep
  // their own curves and response settings while sharing the complete graph.
  struct Definition {
    std::vector<Node> nodes;
    std::vector<int> evaluationOrder;
  };
  std::shared_ptr<Definition> _definition_{std::make_shared<Definition>()};
  std::vector<NodeBinding> _nodeBindings_;
};

#endif
