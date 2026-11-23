/// @file CompositeDial.h
/// @brief Declares CompositeDial, a dial whose response is computed from the
///        responses of other dials (its nodes).
/// @ingroup DialDictionary

#ifndef GUNDAM_COMPOSITE_DIAL_H
#define GUNDAM_COMPOSITE_DIAL_H

#include "DialBase.h"
#include "DialInterface.h"
#include "ConfigUtils.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

/// The collection holds an unbound graph. EventDialCache binds each node to
/// the ordinary dial selected for that event by DataDispenser.
///
/// A dial is a response function of one or more fit parameters (see
/// DialBase). A composite dial is a small graph of dials, its nodes: each
/// node is an ordinary dial definition whose inputs can be fit parameters,
/// event variables (formula nodes only) or the responses of other nodes. The
/// response of the node named by `output` is the response of the composite.
///
/// DialCollection::initializeDialsWithDefinition() creates one CompositeDial
/// for a `dialType: Composite` collection, from its `options` block:
/// - `nodes`: a non-empty list of dial definitions, each with a unique
///   `name`, a `dialType`, and a `dialInputList` whose entries each give
///   exactly one of `parameterName`, `dialResponse` (the name of another
///   node) and `eventVariable`; inputs of a Formula node also need a `name`.
/// - `output`: the name of the node whose response is returned.
///
/// DialManager::configureImpl() creates one child DialCollection per node,
/// so that node dials are built, loaded and updated like any other dials,
/// and stores its index in Node::collectionIndex. The collection's own dial
/// is never evaluated: for each event, EventDialCache::buildReferenceCache()
/// copies it and calls bind() with the dials chosen for that event in the
/// child collections.
///
/// Not supported by Cache::Manager, which disables itself for the run when
/// it finds one.
///
/// @par Thread safety
/// evalResponse() keeps its working values in local variables, so several
/// threads may evaluate the same bound dial at once.
/// @ingroup DialDictionary
class CompositeDial : public DialBase {
public:
  /// One input of a node. Exactly one of `bufferIndex`, `nodeIndex` and
  /// `eventVariable` is set.
  struct Input {
    std::string name; ///< Formula input name; only set for Formula nodes.
    int bufferIndex{-1}; ///< Position of the parameter in the composite's input buffer; -1 if not a parameter.
    std::string parameterName; ///< Name of the parameter; empty if not a parameter.
    int nodeIndex{-1}; ///< Node whose response is this input; -1 if not a node response.
    std::string eventVariable; ///< Event variable read by a formula node, without brackets; empty otherwise.
  };
  /// One node of the graph: an ordinary dial and its inputs.
  struct Node {
    std::string name; ///< Unique node name, used by `dialResponse` and `output`.
    std::shared_ptr<const ConfigReader> config; ///< The node's dial definition (prepared with DialCollection::prepareConfig()).
    std::vector<Input> inputs; ///< The node's inputs, in `dialInputList` order.
    int collectionIndex{-1}; ///< Index of the child collection holding this node's dials; set by DialManager::configureImpl().
    std::shared_ptr<const DialBase> dial; ///< Dial chosen for the bound event; null until bind().
    DialResponseSupervisor supervisor; ///< Copy of the child collection's response clamp, applied to this node's response.
    std::vector<DialInputBuffer::ParameterReference::MirrorEdges> mirrorEdges; ///< Mirror edges of the child collection, one per input that is not an event variable.
  };

  /// Reads the nodes and the output, resolves every input, and fixes the
  /// order in which the nodes are evaluated.
  /// @param config_ The composite's `options` block, with `nodes` and
  ///        `output`.
  /// @param parameterNames_ Names of the parameters in the composite's input
  ///        buffer, in buffer order (the `dialInputList` that
  ///        DialCollection::prepareCompositeInputs() builds).
  /// @post The nodes are not bound yet: evalResponse() throws until bind().
  /// @warning Throws if `nodes` is empty, if a node name is empty or
  ///          repeated, if `output` or a `dialResponse` names no node, if an
  ///          input does not give exactly one source, if a parameter is not
  ///          in `parameterNames_`, if the nodes form a cycle, or if a node
  ///          does not contribute to the output.
  CompositeDial(ConfigReader config_, const std::vector<std::string>& parameterNames_);

  /// Copies this composite, including any bound node dials, which the copy
  /// shares.
  /// @return A new CompositeDial.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<CompositeDial>(*this); }

  /// Gives the type name of this dial.
  /// @return `"Composite"`.
  [[nodiscard]] std::string getDialTypeName() const override { return "Composite"; }

  /// Evaluates the nodes in dependency order and returns the output node's
  /// response.
  ///
  /// Each node gets its parameter inputs from `input_` and its node inputs
  /// from the responses computed before it, after the node's mirror edges
  /// are applied; event-variable inputs are read by the node's own
  /// EventFormula. Each node response is clamped by the node's supervisor.
  /// @param input_ Buffer with one value per composite parameter, in the
  ///        order given to the constructor.
  /// @return The clamped response of the output node. The composite
  ///         collection's own supervisor then clamps it again.
  /// @pre bind() was called, and the buffer size equals getInputCount();
  ///      throws otherwise.
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  /// Describes the composite.
  /// @return `"Composite: N nodes, output=NAME"`.
  [[nodiscard]] std::string getSummary() const override;

  /// Gives mutable access to the nodes. DialManager::configureImpl() uses it
  /// to read the node definitions and set Node::collectionIndex.
  /// @return The node list.
  auto& getNodes(){ return _nodes_; }

  /// Tells whether bind() was called.
  /// @return True once the node dials are set.
  [[nodiscard]] bool isBound() const { return _isBound_; }

  /// Gives the number of parameter inputs of the composite.
  /// @return The size of the parameter list given to the constructor.
  [[nodiscard]] int getInputCount() const { return _inputCount_; }

  /// Sets the dial of every node for one event.
  ///
  /// For each node, copies the dial of the interface that `findInterface_`
  /// returns for the node's collection (binding a copy of a nested
  /// composite first), its response supervisor, and the mirror edges of its
  /// input buffer, then checks that the number of inputs suits the dial
  /// type. Called by EventDialCache::buildReferenceCache() on a copy of the
  /// collection's dial.
  /// @param findInterface_ Gives the interface selected for this event in a
  ///        collection, from its index.
  /// @post isBound() is true.
  /// @warning Throws if a node has no dial for the event, if the input count
  ///          does not match the node's input buffer, or if a Surface,
  ///          Norm, Graph, Spline or Formula node has the wrong number of
  ///          inputs.
  void bind(const std::function<const DialInterface*(int)>& findInterface_);

private:
  /// Number of parameter inputs (size of the constructor's parameter list).
  int _inputCount_{0};
  /// Index of the output node in `_nodes_`.
  int _outputIndex_{-1};
  /// True once bind() has set the node dials.
  bool _isBound_{false};
  /// The nodes, in configuration order.
  std::vector<Node> _nodes_;
  /// Node indices in evaluation order: every node comes after the nodes it
  /// reads, and the output node comes last.
  std::vector<int> _evaluationOrder_;
};

#endif
