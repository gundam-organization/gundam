#ifndef GUNDAM_COMPOSITE_DIAL_H
#define GUNDAM_COMPOSITE_DIAL_H

#include "DialBase.h"
#include "DialInterface.h"
#include "ConfigUtils.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

// The collection holds an unbound graph. EventDialCache binds each node to
// the ordinary dial selected for that event by DataDispenser.
class CompositeDial : public DialBase {
public:
  struct Input {
    std::string name;
    int bufferIndex{-1};
    std::string parameterName;
    int nodeIndex{-1};
    std::string eventVariable;
  };
  struct Node {
    std::string name;
    std::shared_ptr<const ConfigReader> config;
    std::vector<Input> inputs;
    int collectionIndex{-1};
  };
  struct NodeBinding {
    std::shared_ptr<const DialBase> dial;
    DialResponseSupervisor supervisor;
    std::vector<DialInputBuffer::ParameterReference::MirrorEdges> mirrorEdges;
  };

  CompositeDial(ConfigReader config_, const std::vector<std::string>& parameterNames_);
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<CompositeDial>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return "Composite"; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  [[nodiscard]] std::string getSummary() const override;

  auto& getNodes(){
    if( _definition_.use_count() != 1 ){ _definition_ = std::make_shared<Definition>(*_definition_); }
    return _definition_->nodes;
  }
  [[nodiscard]] const auto& getNodes() const { return _definition_->nodes; }
  [[nodiscard]] bool isBound() const { return _isBound_; }
  [[nodiscard]] int getInputCount() const { return _inputCount_; }
  void bind(const std::function<const DialInterface*(int)>& findInterface_);

private:
  int _inputCount_{0};
  int _outputIndex_{-1};
  bool _isBound_{false};
  // Collection setup detaches on modification; event copies share the graph.
  struct Definition {
    std::vector<Node> nodes;
    std::vector<int> evaluationOrder;
  };
  std::shared_ptr<Definition> _definition_{std::make_shared<Definition>()};
  std::vector<NodeBinding> _nodeBindings_;
};

#endif
