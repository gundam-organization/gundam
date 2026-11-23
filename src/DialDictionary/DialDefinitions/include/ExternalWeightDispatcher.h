/// @file ExternalWeightDispatcher.h
/// @brief Declares ExternalWeightDispatcher, the dial that returns one weight
///        computed by an external worker.
/// @ingroup DialDictionary

#ifndef GUNDAM_EXTERNAL_WEIGHT_DISPATCHER_H
#define GUNDAM_EXTERNAL_WEIGHT_DISPATCHER_H

#include "DialBase.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

/// Dial whose response is one entry of a list of weights filled by an
/// ExternalWeightWorker.
///
/// A dial is a response function of one or more fit parameters (see
/// DialBase). For `dialType: ExternalWeight` collections, the weights are
/// computed outside the dial, by the worker of the collection's
/// ExternalWeightDialFactory, each time the parameters change. The
/// factory creates one dispatcher per event (ExternalWeightDialFactory::makeDial())
/// or one per bin (ExternalWeightDialFactory::makeBinnedDial()); each one
/// only reads its entry of the shared weight list.
///
/// Not supported by Cache::Manager, which disables itself for the run when
/// it finds an external-weight dial.
/// @ingroup DialDictionary
class ExternalWeightDispatcher : public DialBase {

public:
  /// Not available: a dispatcher needs a weight list and an index.
  ExternalWeightDispatcher() = delete;
  /// Links the dispatcher to one entry of a weight list.
  /// @param weightList_ The worker's weight list, shared with every
  ///        dispatcher of the collection.
  /// @param eventIndex_ Position of this dial's weight: the event's
  ///        registration index, or the bin index for binned weights.
  ExternalWeightDispatcher(std::shared_ptr<const std::vector<double>> weightList_, std::size_t eventIndex_)
      : _weightList_(std::move(weightList_)), _eventIndex_(eventIndex_) {}

  /// Copies this dial; the copy reads the same weight list.
  /// @return A new ExternalWeightDispatcher.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<ExternalWeightDispatcher>(*this); }
  /// Gives the type name of this dial. Cache::Manager::Build() recognises
  /// external-weight dials by the start of this name.
  /// @return `"ExternalWeightDispatcher"`.
  [[nodiscard]] std::string getDialTypeName() const override { return {"ExternalWeightDispatcher"}; }

  /// Returns the stored weight, whatever the parameter values.
  /// @param input_ Not used: the worker has already used the parameters.
  /// @return Entry `eventIndex` of the weight list.
  /// @warning Throws if there is no weight list or the index is out of
  ///          range.
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  /// Describes the dial.
  /// @return `"ExternalWeightDispatcher eventIndex=N"`.
  [[nodiscard]] std::string getSummary() const override;

private:
  /// Weights written by the worker after each evaluation.
  std::shared_ptr<const std::vector<double>> _weightList_{nullptr};
  /// Position of this dial's weight in `_weightList_`.
  std::size_t _eventIndex_{std::size_t(-1)};

};

#endif // GUNDAM_EXTERNAL_WEIGHT_DISPATCHER_H
