//
// Created by Nadrino on 07/03/2025.
//

/// @file DialManager.h
/// @brief Declares DialManager, the owner of every DialCollection used by a
///        Propagator.
/// @ingroup DialDictionary

#ifndef DIALMANAGER_H
#define DIALMANAGER_H

#include "ParametersManager.h"
#include "DialCollection.h"

#include "ConfigUtils.h"

#include <vector>

#include "GenericToolbox.Time.h"

/// Builds, owns and updates all the DialCollection objects of one Propagator.
///
/// A dial is the response function that turns fit-parameter values into a
/// weight for an event or a bin (a normalisation, a spline, a graph, a
/// formula, a lookup table, ...). Dials are grouped into DialCollection
/// objects: one collection per dial-set definition found in the
/// ParametersManager configuration. DialManager is the single place where
/// those collections are created and where they are refreshed before each
/// reweighting pass.
///
/// @par Ownership
/// Each Propagator owns exactly one DialManager (`Propagator::_dialManager_`),
/// so a fit has two: one for the model Propagator and one for the data
/// Propagator. DialManager owns its DialCollection objects by value. It does
/// **not** own the ParametersManager; it only keeps a raw pointer to it.
///
/// @par Life cycle
/// 1. setParametersManager() — give it the parameters to build dials for.
/// 2. configure() — create one DialCollection per dial-set definition,
///    plus one per node of each composite dial (see configureImpl()).
/// 3. initialize() — link each collection's dials to their input buffers.
/// 4. Data loading: DataDispenser fills the event-by-event collections, then
///    shrinkDialContainers() trims them to the number of dials created.
/// 5. Propagator::buildDialCache() links events to dials, then calls
///    invalidateInputBuffers().
/// 6. Every fit iteration: updateDialState() refreshes the dial inputs from
///    the current parameter values, before events are reweighted.
/// 7. Before data is loaded again, Propagator::clearContent() calls
///    setParametersManager() and clearEventByEventDials().
///
/// @par Thread safety
/// Not thread-safe: call its methods from one thread at a time. During data
/// loading, DataDispenser threads write into the collections directly
/// (through each collection's atomic slot counter), not through this class.
/// During reweighting, worker threads only read the dials.
///
/// @see DialCollection, EventDialCache, Propagator, ParametersManager
/// @ingroup DialDictionary
class DialManager : public JsonBaseClass {

protected:
  /// Creates one DialCollection per dial-set definition of every enabled
  /// ParameterSet. Called by JsonBaseClass::configure().
  ///
  /// A ParameterSet can define its dials in one of two ways:
  /// - **Set-wide** (`dialSetDefinitions` on the ParameterSet): one
  ///   collection per definition, covering the whole set. The collection
  ///   holds one dial per parameter, in parameter order.
  /// - **Per parameter** (`dialSetDefinitions` on each Parameter): one
  ///   collection per definition of each enabled parameter. All dials of
  ///   such a collection read the same parameter.
  ///
  /// The two forms are exclusive: if the set-wide list is not empty, the
  /// per-parameter definitions of that set are ignored.
  ///
  /// Each enabled `Composite` collection (see CompositeDial) then gets one
  /// child collection per node, appended after all the others and marked as
  /// a node of its parent (DialCollection::setIsCompositeNode(),
  /// DialCollection::setCompositeParentIndex()). The children are loaded and
  /// updated like any other collection, but only the parent's CompositeDial
  /// applies their response to events. A node that is itself a composite
  /// gets its own children in the same way.
  ///
  /// @pre setParametersManager() was called with a non-null pointer;
  ///      otherwise the job stops with an error.
  /// @post Any previous collections are discarded. The collection at
  ///       position `i` of getDialCollectionList() has index `i`. Throws if
  ///       a composite node is disabled or does not suit its dial type (for
  ///       example, the wrong number of inputs).
  /// @warning Side effect: an enabled parameter with no dial definition is
  ///          **disabled** (Parameter::setIsEnabled(false)) and an alert is
  ///          logged, because it could never affect the likelihood.
  void configureImpl() override;

  /// Initializes every DialCollection (DialCollection::initialize()), which
  /// connects each dial to its input buffer and response supervisor. Called
  /// by JsonBaseClass::initialize().
  /// @pre configure() has been called.
  void initializeImpl() override;

public:
  /// Creates an empty manager. Call setParametersManager() and configure()
  /// before use.
  DialManager() = default;

  // setters

  /// Sets the ParametersManager whose parameter sets the dials are built
  /// for, and points every existing DialCollection at its list of parameter
  /// sets.
  ///
  /// @param parameterManagerPtr_ Non-owning pointer. The ParametersManager
  ///        must outlive this DialManager and must not be moved while the
  ///        dials exist, since every DialCollection keeps a pointer to its
  ///        list of parameter sets.
  /// @note Copying a Propagator copies this raw pointer, so the copy still
  ///       points at the original's parameters until this is called again.
  ///       Propagator::clearContent() calls it before each reload.
  void setParametersManager(ParametersManager* parameterManagerPtr_){
    _parametersManagerPtr_ = parameterManagerPtr_;
    for( auto& collection : _dialCollectionList_ ){
      collection.setParameterSetList(&_parametersManagerPtr_->getParameterSetsList());
    }
  }


  // const getters

  /// Gives read-only access to all dial collections, in creation order.
  /// @return The collection list. The position of a collection in it equals
  ///         DialCollection::getIndex().
  [[nodiscard]] auto& getDialCollectionList() const{ return _dialCollectionList_; }

  // mutable getters

  /// Gives mutable access to all dial collections. Used while loading data
  /// (to add event-by-event dials) and to build the EventDialCache.
  /// @return The collection list.
  /// @warning Adding or removing collections invalidates the collection
  ///          indices and any pointer held by the EventDialCache.
  auto& getDialCollectionList(){ return _dialCollectionList_; }

  // main methods

  /// Trims every event-by-event DialCollection to the number of dials that
  /// were actually created.
  ///
  /// While loading, DataDispenser reserves one dial slot per selected event
  /// for each event-by-event collection, then fills slots from several
  /// threads. Many slots stay empty (for example, events a dial does not
  /// apply to). This releases them and re-links the remaining dials to their
  /// input buffers (DialCollection::resizeContainers()).
  ///
  /// @pre Data loading into the owning Propagator is finished.
  /// @note Call before Propagator::buildDialCache(): shrinking reallocates
  ///       the dial storage, so pointers taken earlier would dangle.
  void shrinkDialContainers();

  /// Prepares every collection for a new data load
  /// (DialCollection::resetForLoad()), then invalidates all input buffers.
  ///
  /// Reset event interfaces and stateful factories while retaining static definitions.
  /// Collections built by a dial factory (tabulated, kriged, external
  /// weight) are re-created from their configuration, event-by-event
  /// collections lose their dials, and binned dials are kept. Called by
  /// Propagator::clearContent().
  void clearEventByEventDials();

  /// Forces every dial to be re-evaluated at the next updateDialState().
  ///
  /// Each DialInputBuffer caches the parameter values it last saw and only
  /// flags its dials for re-evaluation when a value changes. This resets
  /// those cached values to NaN so the next comparison always reports a
  /// change. Called by Propagator::buildDialCache() after the event-to-dial
  /// links are rebuilt.
  void invalidateInputBuffers();

  /// Brings every dial up to date with the current parameter values. Called
  /// once per likelihood evaluation, before events are reweighted
  /// (Propagator::applyParameters(), Propagator::reweightEvents()).
  ///
  /// For each collection this:
  /// 1. copies the current parameter values into its input buffers and flags
  ///    the dials whose inputs changed (DialCollection::updateInputBuffers());
  /// 2. runs the collection's update callbacks (DialCollection::update()).
  ///    Only tabulated and kriged collections register one: it recomputes
  ///    their lookup table for the new parameter values.
  ///
  /// The time spent here is recorded in getUpdateTimer().
  /// @pre The ParametersManager has already converted eigen-space values
  ///      back to the original parameters, if eigen decomposition is used.
  void updateDialState();

  /// Logs a table listing every dial collection with its title, dial type and
  /// options. Event-by-event collections are highlighted in blue, and a
  /// separator line marks each change of parameter set. Called by
  /// Propagator::printConfiguration().
  void printSummaryTable() const;

  /// Gives the timer that measures updateDialState(). Shown in the
  /// minimizer's monitor table.
  /// @return Moving average of the time spent in updateDialState() over its
  ///         last 10 calls.
  auto& getUpdateTimer() const {return _updateTimer_;}

private:
  /// Non-owning pointer to the parameters the dials are built for. Set by
  /// setParametersManager(); must be non-null before configure().
  ParametersManager* _parametersManagerPtr_{nullptr};

  /// All dial collections, owned by value. Index `i` in this vector is the
  /// collection's own index (DialCollection::getIndex()), which the
  /// EventDialCache uses to find it.
  std::vector<DialCollection> _dialCollectionList_;

  /// Averages the duration of updateDialState() over the last 10 calls.
  GenericToolbox::Time::AveragedTimer<10> _updateTimer_;
};



#endif //DIALMANAGER_H
