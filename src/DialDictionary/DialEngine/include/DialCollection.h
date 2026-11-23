//
// Created by Adrien Blanchet on 29/11/2022.
//

/// @file DialCollection.h
/// @brief Declares DialCollection, the group of dials built from one
///        dial-set definition.
/// @ingroup DialDictionary

#ifndef GUNDAM_DIALCOLLECTION_H
#define GUNDAM_DIALCOLLECTION_H

#include "DialBase.h"
#include "EventFormula.h"
#include "DialUtils.h"
#include "DialInterface.h"
#include "DialInputBuffer.h"
#include "DialResponseSupervisor.h"
#include "SampleSet.h"

#include "GenericToolbox.Wrappers.h"

#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <mutex>
#include <unordered_map>

/// Holds the dials built from one dial-set definition, with the input
/// buffers and response clamps they need to be evaluated.
///
/// A dial is a response function of one or more fit parameters: it turns the
/// current parameter values into a weight factor for an event. A
/// DialCollection holds every dial created from one entry of a
/// `dialSetDefinitions` list, plus what those dials need:
/// - one DialInterface per dial (it holds the dial and points to the dial's
///   input buffer, response supervisor and, for binned dials, its bin);
/// - the DialInputBuffer objects that copy the parameter values the dials
///   read;
/// - the DialResponseSupervisor that clamps the dial responses;
/// - for binned dials, the BinSet that decides which dial an event gets.
///
/// DialManager::configureImpl() creates one collection per definition. A
/// collection is either **set-wide** (no supervised parameter: unless a
/// `dialInputList` is given, dial `i` reads parameter `i` of the set) or
/// **per parameter** (every dial reads the supervised parameter).
///
/// Dials are created at one of two moments:
/// - **During configuration** (configureImpl()): normalisation, formula,
///   compiled-library, composite, binned graph/spline/surface and binned
///   external-weight dials. Events share them: an event gets the dial of the
///   bin it falls in, or the single dial when there is no binning. Bins
///   whose dials are identical share one dial object
///   (deduplicateBinnedDials()).
/// - **While loading data** (isEventByEvent() is true): DataDispenser
///   creates one dial per event, either from data read from the tree
///   (hasDialBranchData(): a TGraph/TSpline3 per event, or a pair of TArray
///   branches; stored with storeEventDial(), which lets identical dials share
///   one interface) or, for tabulated, kriged and unbinned external-weight
///   dials, with the factory stored as collection data
///   (getCollectionData()), in a slot claimed with getNextDialFreeSlot().
///
/// A collection with `applyToVariable` does not weight events: its dial
/// responses change that event variable, combined as set by
/// `variableResponseMode` (EventDialCache::applyVariableResponses()). A
/// collection that is a node of a composite dial (isCompositeNode()) is
/// loaded like the others, but only its parent's CompositeDial uses it.
///
/// @par DialType values
/// The kind of dial is read from the `dialType` key. Names must match
/// exactly; an unknown name stops the job.
/// - `Unset` (0): no type; getSummary() then omits the type.
/// - `Norm` (default; alternative name `Normalization`): the response is the
///   parameter value (Norm).
/// - `Graph`: piecewise-linear interpolation through the points of a TGraph
///   or the knots of a TSpline3 (Graph).
/// - `Spline`: cubic spline through the knots of a TGraph or TSpline3; the
///   implementation is chosen from the options (makeSplineDial()).
/// - `Surface`: 2D interpolation on a TH2, `Bilinear` or `Bicubic`
///   (makeSurfaceDial()).
/// - `Formula` (alternative name `RootFormula`): a ROOT TFormula of the
///   input parameters (RootFormula).
/// - `CompiledLibDial`: a function loaded from a shared library
///   (CompiledLibDial).
/// - `Tabulated`: event-by-event lookup in a table that an external library
///   refills at each update (TabulatedDialFactory).
/// - `Kriged`: event-by-event weighted sum of table entries; an external
///   library refills the table at each update (KrigedDialFactory).
/// - `ExternalWeight` (alternative name `ExternalWeightResponse`): weights
///   computed by an external worker from event variables, per event or per
///   bin (ExternalWeightDialFactory).
/// - `Composite`: a response computed from the responses of other dials,
///   its nodes (CompositeDial).
///
/// @par VariableResponseMode values
/// How the responses of a collection with `applyToVariable` change the
/// event variable, starting from its value when the events were loaded:
/// - `Multiply` (default): the value is multiplied by the response.
/// - `Add`: the response is added to the value.
/// - `Override`: the response replaces the loaded value; at most one such
///   dial per variable and event. `Multiply` and `Add` responses then apply
///   to it.
///
/// @par Life cycle
/// 1. DialManager::configureImpl() constructs it with the list of parameter
///    sets, calls setIndex(), setSupervisedParameterSetIndex(), for
///    per-parameter definitions setSupervisedParameterIndex(), then
///    configure() with the definition.
/// 2. initialize() initializes external-weight factories, lets identical
///    binned dials share one object (deduplicateBinnedDials()), and links
///    every dial to its input buffer, response supervisor and bin
///    (setupDialInterfaceReferences()).
/// 3. Data loading: DataDispenser skips disabled collections and those for
///    which isDatasetValid() is false, asks for the event variables the
///    collection needs, adds one empty slot per event to each event-by-event
///    collection, and fills the slots.
/// 4. DialManager::shrinkDialContainers() calls resizeContainers() to drop
///    the unused slots. EventDialCache::buildReferenceCache() then takes
///    references to the dial interfaces, found by getIndex() and the
///    interface index. The GPU Cache::Manager reads the dials through the
///    EventDialCache, not through this class.
/// 5. Every likelihood evaluation, DialManager::updateDialState() calls
///    updateInputBuffers(), then update().
/// 6. Before data is loaded again, DialManager::clearEventByEventDials()
///    calls resetForLoad().
///
/// @par Ownership
/// DialManager owns its collections by value. A collection owns its dials
/// (through DialInterface), its input buffers, response supervisors and bin
/// set, and shares its collection data through std::shared_ptr (copies of
/// the collection share it). It does not own the parameter sets: it keeps a
/// raw pointer to the ParametersManager's list.
///
/// @par Thread safety
/// DataDispenser calls getNextDialFreeSlot() (an atomic counter),
/// storeEventDial() (which locks a mutex) and makeDial(const TObject*) from
/// several threads at once, and each thread writes only to the slots it
/// claimed. Call everything else from one thread.
///
/// @warning Each DialInterface keeps raw pointers into this collection's
///          input buffers, response supervisors and bins. Run
///          setupDialInterfaceReferences() again after any of these
///          containers or the interface list changes size, and after the
///          collection is copied.
///
/// @see DialManager, DialInterface, DialInputBuffer, DialResponseSupervisor,
///      EventDialCache, DataDispenser
/// @ingroup DialDictionary
class DialCollection : public JsonBaseClass {

public:
  /// @cond
  // DialType: the kind of dial a collection builds (see "DialType values" in
  // the class documentation for the meaning of each value).
  // GenericToolbox.MakeEnum.h generates the enum struct from these macros:
  // ENUM_FIELDS lists the values (Unset = 0, then in order), and ENUM_DICT
  // lists alternative names accepted when reading a string ("Normalization"
  // for Norm, "RootFormula" for Formula, "ExternalWeightResponse" for
  // ExternalWeight). MAKE_ENUM_JSON_INTERFACE adds the
  // from_json() that lets the configuration fill a DialType from a string.
  // The cond/endcond lines hide these helper macros from Doxygen: they are
  // inputs of GenericToolbox.MakeEnum.h, which #undefs them after use.
  // Note: MakeEnum.h does not #undef ENUM_DICT, so ENUM_DICT stays defined
  // in every file that includes this header.
#define ENUM_NAME DialType
#define ENUM_FIELDS \
  ENUM_FIELD( Unset, 0 ) \
  ENUM_FIELD( Norm ) \
  ENUM_FIELD( Graph ) \
  ENUM_FIELD( Spline ) \
  ENUM_FIELD( Surface ) \
  ENUM_FIELD( Formula ) \
  ENUM_FIELD( CompiledLibDial ) \
  ENUM_FIELD( Tabulated ) \
  ENUM_FIELD( Kriged ) \
  ENUM_FIELD( ExternalWeight ) \
  ENUM_FIELD( Composite )
#define ENUM_DICT \
   ENUM_DICT_ENTRY("Normalization", "Norm") \
   ENUM_DICT_ENTRY("RootFormula", "Formula") \
   ENUM_DICT_ENTRY("ExternalWeightResponse", "ExternalWeight")
#include "GenericToolbox.MakeEnum.h"
  /// @endcond
  MAKE_ENUM_JSON_INTERFACE(DialType);

  /// @cond
  // VariableResponseMode: how a dial whose collection sets `applyToVariable`
  // changes that event variable (see "VariableResponseMode values" in the
  // class documentation). Generated by GenericToolbox.MakeEnum.h in the same
  // way as DialType above.
#define ENUM_NAME VariableResponseMode
#define ENUM_FIELDS \
  ENUM_FIELD( Multiply ) \
  ENUM_FIELD( Add ) \
  ENUM_FIELD( Override )
#include "GenericToolbox.MakeEnum.h"
  /// @endcond
  MAKE_ENUM_JSON_INTERFACE(VariableResponseMode);

  /// Declares the keys a dial-set definition may use, then checks
  /// @p config_ against them.
  ///
  /// Alternative key names accepted: `printDialsSummary` (for
  /// `printDialSummary`), `type` and `dialsType` (`dialType`), `dialSubType`
  /// and `dialConfig` (`options`), `dialLeafName` and `treeExpression`
  /// (`dialBranchData`), `minimumSplineResponse` (`minDialResponse`) and
  /// `binningFilePath` (`binning`).
  ///
  /// Unknown keys only produce a warning. The job stops if the deprecated
  /// key `parameterLimits` is present (limits belong in the parameter
  /// definition), or if two names of the same key carry different values
  /// (ConfigReader::checkConfiguration()).
  ///
  /// @param config_ The definition to prepare. Its previous key
  ///        declarations are cleared first.
  /// @note Only reading a key declared here works:
  ///       ConfigReader::fetchValue() stops the job for an undeclared name.

  static void prepareConfig(ConfigReader &config_);

public:
  /// Not available: a collection always needs the list of parameter sets.
  DialCollection() = delete;

  /// Creates an empty collection whose dials will read parameters from
  /// @p targetParameterSetListPtr. Call the setters, then configure() and
  /// initialize().
  ///
  /// @param targetParameterSetListPtr Non-owning pointer to the
  ///        ParametersManager's list of parameter sets. The list must stay
  ///        alive and must not be reallocated while the collection exists,
  ///        since the input buffers keep this pointer too.
  explicit DialCollection(std::vector<ParameterSet> *targetParameterSetListPtr): _parameterSetListPtr_(targetParameterSetListPtr) {}

  /// A base class for any specialized DialCollection data needed by a
  /// particular implementation.  For example, the Tabulated dials use this to
  /// keep track of the external libraries needed to fill the tables.
  ///
  /// DialFactoryBase derives from it, so the TabulatedDialFactory or
  /// KrigedDialFactory made by initializeDialsWithTabulation() or
  /// initializeDialsWithKriging() is stored as collection data. It carries
  /// no data itself: getCollectionData() casts it back to the concrete type.
  /// @ingroup DialDictionary
  struct CollectionData {
    /// Creates empty collection data.
    CollectionData() = default;
    /// Destroys the derived data correctly when it is deleted through the
    /// base pointer held in `_dialCollectionData_`.
    virtual ~CollectionData() = default;
  };

  // setters

  /// Sets the position of this collection in DialManager's list.
  /// @param index Position in DialManager::getDialCollectionList(). Called by
  ///        DialManager::configureImpl(); initialize() throws if it is still
  ///        -1.
  void setIndex(int index){ _index_ = index; }

  /// Makes this a per-parameter collection: every dial reads the given
  /// parameter of the supervised set.
  /// @param supervisedParameterIndex Index of the parameter in the supervised
  ///        ParameterSet (Parameter::getParameterIndex()). Leave unset (-1)
  ///        for a set-wide collection.
  /// @pre Call before configure(): a `dialsDefinitions` lookup needs it.
  void setSupervisedParameterIndex(int supervisedParameterIndex){ _supervisedParameterIndex_ = supervisedParameterIndex; }

  /// Sets the ParameterSet that the dials of this collection read from.
  /// @param supervisedParameterSetIndex Index in the parameter-set list given
  ///        to the constructor.
  /// @pre Call before configure(): a `dialInputList` can't be read without
  ///      it, and initialize() throws if it is not set.
  void setSupervisedParameterSetIndex(int supervisedParameterSetIndex){ _supervisedParameterSetIndex_ = supervisedParameterSetIndex; }

  /// Marks this collection as a node of a composite dial. Set by
  /// DialManager::configureImpl() on the child collections it creates.
  /// @param isCompositeNode_ True for a node.
  /// @pre Call before configure(): it changes the default minimum response
  ///      and the handling of flat responses.
  void setIsCompositeNode(bool isCompositeNode_){ _isCompositeNode_ = isCompositeNode_; }

  /// Sets the composite collection this node belongs to.
  /// @param index_ Position of the parent in DialManager's list.
  void setCompositeParentIndex(int index_){ _compositeParentIndex_ = index_; }

  /// Points the collection at a (new) list of parameter sets. Called by
  /// DialManager::setParametersManager(), for example after a Propagator
  /// was copied.
  /// @param parameterSetList_ Non-owning pointer to the ParametersManager's
  ///        list. Existing input buffers are re-pointed by the next
  ///        setupDialInterfaceReferences().
  void setParameterSetList(std::vector<ParameterSet>* parameterSetList_){ _parameterSetListPtr_ = parameterSetList_; }

  // const getters

  /// Gives the index of the parameter set the dials read from. Used by
  /// DialManager::printSummaryTable() to group collections by set.
  /// @return Index in the parameter-set list, or -1 if not set.
  auto getSupervisedParameterSetIndex() const{ return _supervisedParameterSetIndex_; }

  /// Query if this has one DialBaseObject per event, or if DialBaseObjects are
  /// shared between events.  Used to decide how to attach to events in
  /// DataDispenser.
  /// @return True for collections whose dials are created per event while
  ///         loading data: branch data (`dialBranchData`), tabulated,
  ///         kriged, and unbinned external-weight collections.
  /// @note DataDispenser uses this flag to add one dial slot per event for
  ///       factory collections. Which dial an event gets is decided from
  ///       hasDialBranchData() and getDialType() instead.
  [[nodiscard]] bool isEventByEvent() const {return _isEventByEvent_;}

  /// Raw TGraph memoization is valid only for immutable, shareable definitions.
  /// ROOT splines and event-dependent factories retain their construction path.
  ///
  /// DataDispenser uses it to decide whether an event whose TGraph is
  /// identical to the last one it built a dial from, for this collection,
  /// can reuse that dial's interface without building a new dial (option
  /// `enableDialInputCache`).
  /// @return True for Graph and Spline collections without the `ROOT`
  ///         option that have a single input buffer and a single response
  ///         supervisor.
  [[nodiscard]] bool canReuseGraphInput() const{
    return (_dialType_ == DialType::Graph or _dialType_ == DialType::Spline)
        and _dialOptions_.find("ROOT") == std::string::npos
        and _dialInputBufferList_.size() == 1 and _dialResponseSupervisorList_.size() == 1;
  }

  /// Check if the dial is enabled during this run
  /// @return False if the definition sets `isEnabled: false`, or if
  ///         configure() found no usable definition. DataDispenser skips
  ///         disabled collections.

  [[nodiscard]] bool isEnabled() const{ return _isEnabled_; }

  /// Gives the description of a formula that reads event variables.
  /// @return The definition built by configureImpl() when a Formula
  ///         collection's `dialInputList` has `eventVariable` entries;
  ///         null otherwise. EventDialCache makes one EventFormula per event
  ///         from it, and DataDispenser loads the variables it lists.
  [[nodiscard]] const auto& getEventFormulaDefinition() const{ return _eventFormulaDefinition_; }

  /// Gives the mirror edges that an EventFormula applies to the event
  /// variables it reads. Its parameter inputs come from the collection's
  /// input buffer, which is already mirrored.
  /// @return `mirrorLowEdge` and the mirror width when `useMirrorDial` is
  ///         set; a NaN lower edge (no mirroring) otherwise.
  [[nodiscard]] DialInputBuffer::ParameterReference::MirrorEdges getEventInputMirrorEdges() const{
    return {_useMirrorDial_ ? _mirrorLowEdge_ : std::nan("unset"), _mirrorRange_};
  }

  /// Tells whether this collection is a node of a composite dial.
  /// @return True if setIsCompositeNode(true) was called. EventDialCache
  ///         does not attach the dials of a node to events directly.
  [[nodiscard]] bool isCompositeNode() const{ return _isCompositeNode_; }

  /// Gives the composite collection this node belongs to. DataDispenser
  /// walks these indices up and skips a node whose parent or grandparent
  /// is disabled or does not apply to the dataset.
  /// @return Position of the parent in DialManager's list; -1 if this is
  ///         not a node.
  [[nodiscard]] int getCompositeParentIndex() const{ return _compositeParentIndex_; }

  /// Check if the DialBase should be allowed to calculate an extrapolation out
  /// outside of the defined parameter bounds.
  /// @return The `allowDialExtrapolation` option (default true).
  /// @note Not called anywhere: makeDial(const TObject*) reads the member
  ///       directly.
  [[nodiscard]] bool isAllowDialExtrapolation() const{ return _allowDialExtrapolation_; }

  /// The location in the cache for this dialCollection that can be used to
  /// indentify the collection.
  /// @return Position in DialManager::getDialCollectionList(), or -1 before
  ///         setIndex(). EventDialCache stores it to find the collection.
  [[nodiscard]] int getIndex() const{ return _index_; }

  /// If it exists, then this is the name of a leaf in the input file
  /// containing data to weight the event in the entry.  This is empty if the
  /// dial is not event-by-event, or if the dial is not based on a "spline"
  /// (e.g. it might be an tabulated event-by-event dial).
  /// @return The `dialBranchData` value when it is a single expression
  ///         (alternative names `dialLeafName`, `treeExpression`); empty if
  ///         not set or if it is a pair of TArray branches.
  /// @note DataDispenser uses hasDialBranchData(), not isEventByEvent(), to
  ///       decide that a dial is read from the tree for every event.
  [[nodiscard]] const std::string &getDialLeafName() const{ return _dialLeafName_; }

  /// Tells whether the dials are read from the input tree, one per event.
  /// @return True if `dialBranchData` names a branch expression holding a
  ///         TGraph or TSpline3 per event (getDialLeafName()), or a pair of
  ///         TArray branches (getDialParameterValuesBranch(),
  ///         getDialResponsesBranch()). DataDispenser builds these dials
  ///         while loading.
  [[nodiscard]] bool hasDialBranchData() const{ return not _dialLeafName_.empty() or not _dialParameterValuesBranch_.empty(); }
  /// Gives the branch holding the parameter values of each event's graph,
  /// when `dialBranchData` is a pair of TArray branches.
  /// @return The `parameterValuesTArray` branch name; empty otherwise.
  [[nodiscard]] const std::string& getDialParameterValuesBranch() const{ return _dialParameterValuesBranch_; }
  /// Gives the branch holding the responses at those parameter values, when
  /// `dialBranchData` is a pair of TArray branches.
  /// @return The `weightValuesTArray` branch name; empty otherwise.
  [[nodiscard]] const std::string& getDialResponsesBranch() const{ return _dialResponsesBranch_; }

  /// Gives the binning that selects which dial applies to an event.
  /// @return The bin set; empty if the collection is not binned. Bin `i`
  ///         goes with dial `i` of getDialInterfaceList(), or with the only
  ///         dial when there is a single bin.

  [[nodiscard]] const BinSet &getDialBinSet() const{ return _dialBinSet_; }

  /// Gives the dataset-name patterns this collection is restricted to.
  /// @return The `applyOnDataSets` patterns; empty means every dataset.
  /// @note Not called anywhere: isDatasetValid() does the matching.
  [[nodiscard]] const std::vector<std::string> &getDataSetNameList() const{ return _dataSetNameList_; }

  /// A formula string to decide if the dial should be applied to an event.
  /// The dial should be applied if this returns a non-zero value.
  /// @return The `applyCondition` expression, or the one built from
  ///         `applyConditions` (see readParametersFromConfig()); empty when
  ///         no condition is set. Event variables appear in it as named
  ///         TFormula parameters (for example `[C] > 0`); DataDispenser loads
  ///         each one as an event variable before it evaluates the formula.
  [[nodiscard]] const std::string &getApplyConditionStr() const{ return _applyConditionStr_; }

  /// Gives the kind of dial this collection builds.
  /// @return The `dialType` value (Norm if not set).
  [[nodiscard]] auto& getDialType() const{ return _dialType_; }

  /// Gives the dial options string, such as the spline flavour.
  /// @return The `options` value (alternative names `dialSubType`,
  ///         `dialConfig`); empty if not set, and for Formula,
  ///         CompiledLibDial, ExternalWeight and Composite collections, whose
  ///         `options` is a configuration block.
  [[nodiscard]] auto& getDialOptions() const{ return _dialOptions_; }

  /// Gives the event variable that this collection's dials change instead
  /// of the event weight.
  /// @return The `applyToVariable` value; empty for an ordinary collection.
  [[nodiscard]] const std::string& getApplyToVariable() const{ return _applyToVariable_; }

  /// Gives how the responses change that variable.
  /// @return The `variableResponseMode` value; `Multiply` by default.
  [[nodiscard]] VariableResponseMode getVariableResponseMode() const{ return _variableResponseMode_; }

  // mutable getters

  /// Gives mutable access to the dial binning. DataDispenser uses it to
  /// store, in each bin edge, the index of its event variable.
  /// @return The bin set; empty if the collection is not binned.
  BinSet &getDialBinSet(){ return _dialBinSet_; }

  /// One interface per DialBase.  The interface groups the input buffer,
  /// response supervisor, dialBase (what actually calculates the weight) into
  /// a single object and provide the methods to calculate the dial weight.
  /// @return The interface list. DataDispenser and storeEventDial() grow it
  ///         for event-by-event collections and write dials into its slots;
  ///         EventDialCache::buildReferenceCache() keeps references to its
  ///         elements, so do not resize it after that.
  std::vector<DialInterface> &getDialInterfaceList(){ return _dialInterfaceList_; }

  /// The different sets of parameters to use for the interfaces.  There is a
  /// single set of inputs for each DialBase in the collection, or a different
  /// set of parameters for each DialBase (i.e. the number of DialInputBuffer
  /// objects is 1, or _dialBaseList_.size()).  set, but for
  /// @return The input buffers. The tabulated and kriged update callbacks
  ///         pass the first one to their factory.
  /// @note `_dialBaseList_` no longer exists: the number of buffers is 1 or
  ///       the size of getDialInterfaceList().
  std::vector<DialInputBuffer> &getDialInputBufferList(){ return _dialInputBufferList_; }

  // non-trivial getters

  /// Tells whether this collection applies to a dataset. Called by
  /// DataDispenser when it selects the collections for a dataset.
  /// @param datasetName_ Name of the dataset being loaded.
  /// @return True if `applyOnDataSets` is empty, or if one of its patterns
  ///         matches the name. Matching ignores case and accepts `*`
  ///         wildcards.
  [[nodiscard]] bool isDatasetValid(const std::string& datasetName_) const;

  /// Gives a readable name for the collection, used in logs and in
  /// DialManager::printSummaryTable().
  /// @return The full title of the supervised parameter for a per-parameter
  ///         collection, the set name for a set-wide one, or
  ///         `UnsetParameterSuperVision` if no set index is given.
  std::string getTitle() const;

  /// Builds a description of the collection: title, dial type, options, leaf
  /// name and definition ranges, each only when set. Used in error messages.
  /// @param shallow_ If false, also adds one line per dial
  ///        (DialInterface::getSummary()), except for event-by-event
  ///        collections.
  /// @return The description.
  std::string getSummary(bool shallow_ = true) const;

  /// Gives the parameter that every dial reads, for a per-parameter
  /// collection.
  /// @return Pointer into the parameter-set list, or nullptr for a set-wide
  ///         collection or when the set index is not set.
  Parameter* getSupervisedParameter() const;

  /// Gives the parameter set the dials read from.
  /// @return Pointer into the parameter-set list, or nullptr if the set
  ///         index is not set.
  ParameterSet* getSupervisedParameterSet() const;

  // core

  /// Deletes every dial, resets the free-slot counter to 0 and empties the
  /// index used to share identical event dials. Input buffers, response
  /// supervisors and the bin set are kept. Called by resetForLoad() for
  /// event-by-event collections.
  void clear();

  /// Preserve static definitions, but discard state belonging to the previous load.
  ///
  /// A collection with collection data (a dial factory) is rebuilt from its
  /// configuration, keeping its index, supervised set and parameter and
  /// composite-node settings, and is initialized again. Otherwise an
  /// event-by-event collection is cleared (clear()), and an enabled one has
  /// its interface references set up again (setupDialInterfaceReferences()),
  /// which also repairs them after the Propagator was copied. In every case
  /// the input buffers are invalidated. Called by
  /// DialManager::clearEventByEventDials().
  void resetForLoad();

  /// Called after the DialCollection has been fully initialized, and reclaims
  /// any unused space.
  ///
  /// Trims the interface list to the number of slots claimed with
  /// getNextDialFreeSlot(), then re-links every interface
  /// (setupDialInterfaceReferences()). Also logs how many event dials were
  /// shared by storeEventDial() and releases the sharing index. Called by
  /// DialManager::shrinkDialContainers() for event-by-event collections.
  /// @pre Data loading is finished: no thread is still claiming slots.
  /// @warning On a collection whose dials were not stored through
  ///          getNextDialFreeSlot() (binned or normalisation dials), the
  ///          counter is 0 and this deletes every dial.
  void resizeContainers();

  /// After the DialCollection is fully initialized, setup all the pointers
  /// in the DialInterface objects.  This is used after the size of the
  /// DialCollection has changed to fix any pointer issues.
  ///
  /// If no input buffer exists yet (no `dialInputList`), creates them: one
  /// per dial for a set-wide collection, buffer `i` reading parameter `i` of
  /// the set; a single one reading the supervised parameter otherwise. Then
  /// gives every buffer the parameter-set list, applies the mirror edges
  /// when `useMirrorDial` is set, initialises the buffers, and points each
  /// interface to its input buffer, bin and response supervisor.
  ///
  /// @pre setSupervisedParameterSetIndex() was called; throws otherwise.
  /// @warning Stops the job if a set-wide collection that has no input
  ///          buffer yet does not hold exactly one dial per parameter.
  ///          Throws if the number of input buffers, bins (when there are
  ///          bins) or supervisors is neither 1 nor the number of dials.
  void setupDialInterfaceReferences();

  /// Keep one interface per bin while sharing identical immutable curves.
  ///
  /// For a binned collection, makes the bins whose dials have identical
  /// content (same type, extrapolation setting and data) share one dial
  /// object (DialInterface::shareDial()), and logs how many were shared.
  /// Graph, spline, Shift and Norm dials are compared; other types are left
  /// alone. Does nothing for event-by-event collections or collections
  /// without bins. Called by initializeImpl().
  void deduplicateBinnedDials();

  /// Update the dial collection.  Most dial collections don't need an update,
  /// but this provides a handle for collections that precalculate values prior
  /// to doing the event reweighting.  An example of that would be the
  /// Tabulated dials (which can be used in implement things like oscillation
  /// weights).  Update callbacks are std::function\<void(DialCollection*)\>, and
  /// can be added with addUpdate();
  ///
  /// Called by DialManager::updateDialState() once per likelihood
  /// evaluation.
  /// @pre updateInputBuffers() has run: the tabulated and kriged callbacks
  ///      read the first input buffer.

  void update();

  /// Add a dial collection update callback.  These are called in the order
  /// that they are added.  They are activated by the "update()" method.
  /// @param callback Function to call; it receives this collection, so it
  ///        does not need to capture a pointer to it.
  void addUpdate(const std::function<void(DialCollection* dc)>& callback);

  /// Check if the dial will need to be recalculated.  A recalculation
  /// happens when a parameter value has changed since the last calculation.
  /// This also checks if a dial has been masked, and flags that the
  /// Dial calculation should be ignored.  That can only happen before the
  /// reweight calculation has been "frozen".
  ///
  /// Calls DialInputBuffer::update() on every input buffer: it copies the
  /// current parameter values (mirrored if requested) and flags the buffer
  /// when a value changed. Called by DialManager::updateDialState().
  /// @note DialInputBuffer::update() does not check any mask: it only copies
  ///       the values and sets the change flag.
  void updateInputBuffers();

  /// Loading only: store a branch dial, reusing an identical lightweight dial.
  /// Returns an interface index; no interface pointers may be retained while loading.
  /// Optionally report whether the stored definition has a finite, shareable
  /// signature. Norm/Shift signatures do not use the spline payload API.
  ///
  /// DataDispenser calls it, from several threads, for each event dial read
  /// from the tree. When the collection has a single input buffer and a
  /// single response supervisor, a dial with the same type, extrapolation
  /// setting and data as one already stored is not kept: the index of the
  /// existing interface is returned instead. Otherwise the dial is moved into
  /// the next free slot, growing the interface list if needed.
  /// @param dial_ The new dial; must not be null (throws otherwise).
  /// @param hasFiniteSignature_ If not null, set to true when the dial can be
  ///        shared and all the values that identify it are finite.
  /// @return Index in getDialInterfaceList() of the interface that holds the
  ///         dial, new or reused.
  size_t storeEventDial(std::unique_ptr<DialBase> dial_, bool* hasFiniteSignature_ = nullptr);

  /// Return the next preallocated slot. Thread safe for event dial factories.
  /// @return Index of the claimed slot in getDialInterfaceList(). Slots are
  ///         handed out in increasing order from 0.
  /// @pre The interface list is large enough: the index is not checked here.
  ///      DataDispenser adds one slot per event beforehand for factory
  ///      collections; storeEventDial() grows the list itself.
  size_t getNextDialFreeSlot(){ return _dialFreeSlot_++; }

  /// Gives the number of slots claimed so far.
  /// @return The index the next getNextDialFreeSlot() call will return.
  /// @note Not called anywhere.
  size_t getDialFreeSlotIndex() const { return _dialFreeSlot_.getValue(); }

  /// Provide access to a collection data.  The ownership is retained by
  /// the collection.
  /// @tparam T Expected type of the data, for example TabulatedDialFactory,
  ///         or DialFactoryBase for any dial factory.
  /// @param i Index in the collection-data list, in order of creation.
  /// @return The data cast to `T`, or nullptr if it is not a `T`.
  /// @warning @p i is not bounds-checked.
  template <typename T>
  T* getCollectionData(int i=0) const {return dynamic_cast<T*>(_dialCollectionData_[i].get());}

  /// Add an extra leaf name needed for this dial
  /// @param leaf Name of an event variable that DataDispenser must load for
  ///        this collection. Used for the binning or weight variables of
  ///        tabulated and kriged dials.
  void addExtraLeafName(const std::string& leaf) {_globalDialExtraLeafNames_.emplace_back(leaf);}

  /// Gives the extra event variables this collection needs.
  /// @return The names added with addExtraLeafName(). DataDispenser requests
  ///         them as event variables when it loads the data.
  std::vector<std::string>& getExtraLeafNames() {return _globalDialExtraLeafNames_;}

  /// Sets every cached value of every input buffer to NaN, so that the next
  /// updateInputBuffers() flags all dials for re-evaluation. Called by
  /// DialManager::invalidateInputBuffers().
  void invalidateCachedInputBuffers(){ for( auto& inputBuffer : _dialInputBufferList_ ){ inputBuffer.invalidateBuffers(); }}

  /// Logs the title of the collection.
  /// @note Not called anywhere: DialManager::printSummaryTable() lists the
  ///       collections instead.
  void printConfiguration() const;

  // methods to generate dials with factory

  /// Creates a normalisation dial.
  /// @return A new Norm dial (its response is the input parameter value).
  ///         Never nullptr.
  std::unique_ptr<DialBase> makeDial() const;

  /// Creates a dial of type getDialType() from a ROOT object. Used for
  /// binned dials (initializeBinnedDials()) and, from several DataDispenser
  /// threads at once, for event-by-event dials read from the tree.
  ///
  /// Graph and Spline dials may be replaced by a simpler one: no dial for a
  /// response that is always 1 (except in a composite node), a Shift dial for
  /// any other flat response, and a Graph dial for two points
  /// (makeDialShortCircuit()). Points on one straight line give a two-point
  /// Graph, or a Norm dial if the line is y = x and extrapolation is
  /// allowed (makeGraphDial(const std::vector<DialUtils::DialPoint>&)); a
  /// non-ROOT spline whose knots and slopes all fit one line is turned into
  /// such a graph too.
  ///
  /// @param src_ A TGraph or TSpline3 for Graph and Spline, a TH2 for Surface.
  ///        May be nullptr.
  /// @return The dial, with extrapolation allowed as set by
  ///         `allowDialExtrapolation`; nullptr if @p src_ is nullptr, has no
  ///         usable point, or gives a response that is always 1.
  /// @warning Throws for any other dial type.
  std::unique_ptr<DialBase> makeDial(const TObject* src_) const;

  /// Creates a formula or compiled-library dial from a dial definition.
  /// @param config_ The definition. Its `dialType` must be `Formula`,
  ///        `RootFormula` or `CompiledLibDial`. Its `options` block holds
  ///        `formulaStr` for a formula, or `libraryFile` (and optionally
  ///        `messageOnError`) for a compiled library. A formula also needs a
  ///        `dialInputList` whose entries each have a `name`, in the
  ///        definition or in this collection's configuration.
  /// @param bin_ For a binned formula, the bin the dial is made for: the
  ///        names of its variables are accepted as formula constants (see
  ///        RootFormula::setInputNames()). nullptr otherwise.
  /// @return The dial; never nullptr.
  /// @warning Throws for any other `dialType`, if a formula has no named
  ///          `dialInputList`, and if the library can't be loaded.
  std::unique_ptr<DialBase> makeDial(const ConfigReader& config_, const Bin* bin_ = nullptr) const;

protected:
  /// Reads the definition given to configure() and creates the dials that
  /// do not depend on events. Called by JsonBaseClass::configure(), from
  /// DialManager::configureImpl().
  ///
  /// Uses initializeNormDialsWithParBinning() when `parametersBinningPath` is
  /// set, initializeDialsWithDefinition() otherwise. If neither finds a
  /// usable definition, the collection is disabled with an alert. If
  /// `dialInputList` is given, its entries choose the parameters of the
  /// supervised set that the dials read, in order, through a single input
  /// buffer.
  ///
  /// @pre setSupervisedParameterSetIndex() was called, and
  ///      setSupervisedParameterIndex() too for a per-parameter definition.
  /// @post Event-by-event collections hold no dial yet.
  void configureImpl() override;

  /// Links every dial to its input buffer, response supervisor and bin
  /// (setupDialInterfaceReferences()). Called by JsonBaseClass::initialize(),
  /// from DialManager::initializeImpl().
  /// @pre setIndex() was called; throws otherwise.
  /// @warning Disabled collections are not skipped. A set-wide collection
  ///          left with no dial and no input buffer (for example with
  ///          `isEnabled: false`) fails the dial-count check of
  ///          setupDialInterfaceReferences() and stops the job.
  void initializeImpl() override;

  /// Checks the nodes of a `Composite` definition and rewrites their inputs.
  ///
  /// Every node input must give exactly one of `parameterName`,
  /// `parameterIndex`, `dialResponse` and `eventVariable`. Parameter inputs
  /// given by index are rewritten by name, and the collection's own
  /// `dialInputList` is set to the distinct parameters used by its nodes,
  /// nested composites included. Called by configureImpl() for a composite
  /// that is not itself a node.
  /// @param config_ The composite definition, modified in place.
  /// @warning Throws if the definition has its own `dialInputList`, if an
  ///          input uses the removed `inputIndex` key, or if a parameter is
  ///          not in the supervised set.
  void prepareCompositeInputs(ConfigReader& config_);
  /// Builds one normalisation dial per bin of `parametersBinningPath`, if
  /// that key is set. The parameter binning is then also the event binning:
  /// an event in bin `i` gets dial `i` which, in a set-wide collection, reads
  /// parameter `i`.
  /// @return False if `parametersBinningPath` is absent (nothing is done);
  ///         true otherwise.
  /// @post One response supervisor, with minimum `minDialResponse` (default
  ///       0) and maximum `maxDialResponse`.
  bool initializeNormDialsWithParBinning();

  /// Builds the dials described by the definition or, if it has a
  /// `dialsDefinitions` list, by the entry for the supervised parameter
  /// (fetchDialsDefinition()).
  ///
  /// By dial type: Composite gives one CompositeDial; Norm one Norm dial;
  /// Formula one RootFormula dial, or one per bin of `binning`;
  /// CompiledLibDial one dial; Tabulated and Kriged set up their factory and
  /// become event-by-event; ExternalWeight sets up its factory and is
  /// event-by-event unless its weights are binned. Any other type with a
  /// `binning` or `dialsBinningDefinition` gets one dial per bin from a file
  /// (initializeBinnedDials()); with `dialBranchData` instead, it becomes
  /// event-by-event and DataDispenser creates the dials. Ends by adding one
  /// response supervisor (minimum `minDialResponse`, maximum
  /// `maxDialResponse`).
  ///
  /// @return False only if the definition is empty (no matching
  ///         `dialsDefinitions` entry). True otherwise, also when the
  ///         definition has `isEnabled: false`: then no dial and no
  ///         supervisor are created, and the collection stays enabled.
  /// @warning Throws if no case matches the definition, if both `binning`
  ///          and `dialsBinningDefinition` are given, if
  ///          `dialsBinningDefinition` is used with a type other than Graph
  ///          or Spline, or if a composite has a binning or branch data.
  bool initializeDialsWithDefinition();

  /// Builds one dial per bin from ROOT objects stored in a file.
  ///
  /// The bins come from `binning` or, with `dialsBinningDefinition`, from
  /// branches of the dial tree itself: one bin per tree entry, each variable
  /// given by a `valueBranch` or by `minValueBranch` and `maxBranchValue`.
  /// The dials come from `dialsFilePath` and either `dialsList` (path of a
  /// TObjArray with one object per bin, in bin order) or `dialsTreePath` (a
  /// TTree with one TGraph or TSpline3 per entry, in the branch named by
  /// `dialsBranch`). `binnedDialsPath` (`file.root:tree:branch`) gives the
  /// file, tree and branch at once. Without `dialsBranch`, the branch is
  /// `Graph` or `Spline` after the dial type, with a deprecation warning.
  /// Bins whose object gives no dial (a response always 1) are dropped.
  /// @param dialsDefinition The dial definition.
  /// @return False if neither `binning` nor `dialsBinningDefinition` is
  ///         given, or if neither `dialsList` nor a tree is given; true
  ///         otherwise.
  /// @warning Throws if both binning keys are given, if no file is given or
  ///          the sources conflict, if the file can't be opened, or if the
  ///          number of dials does not match the number of bins.
  bool initializeBinnedDials(const ConfigReader& dialsDefinition);

  /// Sets up a Tabulated collection: creates its TabulatedDialFactory from
  /// `tableConfig`, registers the table's binning variables as extra leaves
  /// and registers an update callback that refills the table from the
  /// first input buffer. DataDispenser later creates one dial per event
  /// with the factory.
  /// @param dialsDefinition_ Definition holding `tableConfig`. (Named after
  ///        the parameter of the definition in DialCollection.cpp, which
  ///        Doxygen checks against.)
  /// @return Always true. The factory stops the job itself on error.
  bool initializeDialsWithTabulation(const ConfigReader& dialsDefinition);

  /// Sets up a Kriged collection: creates its KrigedDialFactory from
  /// `tableConfig`, registers the weight variables as extra leaves and
  /// registers an update callback that refills the table from the first
  /// input buffer. DataDispenser later creates one dial per event with the
  /// factory.
  /// @param dialsDefinition_ Definition holding `tableConfig`. (Named after
  ///        the parameter of the definition in DialCollection.cpp, which
  ///        Doxygen checks against.)
  /// @return Always true. The factory stops the job itself on error.
  bool initializeDialsWithKriging(const ConfigReader& dialsDefinition);

  /// Sets up an ExternalWeight collection: creates its
  /// ExternalWeightDialFactory from the `options` block, registers the
  /// factory's input variables as extra leaves, creates one dial per bin when
  /// the weights are binned, and registers an update callback that passes
  /// the first input buffer to the factory.
  /// @param dialsDefinition_ Definition holding `options`. (Named after the
  ///        parameter of the definition in DialCollection.cpp.)
  /// @return Always true; throws if `options` is missing.
  bool initializeDialsWithExternalWeight(const ConfigReader& dialsDefinition);

  /// Reads `applyToVariable` and `variableResponseMode` (default
  /// `Multiply`). Called by configureImpl() and readParametersFromConfig().
  /// @param config_ The definition.
  /// @warning Throws if `applyToVariable` is empty, if the mode is not
  ///          Multiply, Add or Override, or if a mode is given without
  ///          `applyToVariable`.
  void readVariableResponseConfig(const ConfigReader& config_);

  /// Reads the options common to all dial types into the members, and
  /// builds the apply-condition expression.
  ///
  /// Options read: `applyToVariable` and `variableResponseMode` (through
  /// readVariableResponseConfig()), `printDialSummary`, `dialType`, `options`
  /// (only for types that take it as a string), `dialBranchData`,
  /// `minDialResponse`, `maxDialResponse`, `useMirrorDial`, `mirrorLowEdge`,
  /// `mirrorHighEdge`, `allowDialExtrapolation`, `applyCondition`,
  /// `definitionRange`, `knotResponseLimits` and `mirrorDefinitionRange`.
  /// The default minimum response is 0, except for composite nodes and
  /// variable dials, which have none. An `applyConditions` list, if present,
  /// replaces `applyCondition`: string entries are used as they are,
  /// structured entries (`exp` with `allowedRanges`, `allowedValues`,
  /// `excludedRanges`, `excludedValues`) are turned into an expression, and
  /// all entries are joined with AND.
  ///
  /// @param config_ The definition (already prepared with prepareConfig()).
  /// @warning Throws if a condition range has min >= max, if
  ///          `knotResponseLimits` has min > max, if `dialBranchData` is
  ///          malformed (or a TArray pair is used with a type other than
  ///          Spline or Graph), or if mirroring is on and `mirrorHighEdge`
  ///          is below `mirrorLowEdge`.
  /// @warning The expressions built for `excludedRanges` and
  ///          `excludedValues`, and for entries that combine allowed and
  ///          excluded fields, do not match their names; see the
  ///          implementation notes.
  void readParametersFromConfig(const ConfigReader &config_);

  /// Finds the entry of a `dialsDefinitions` list that belongs to the
  /// supervised parameter.
  /// @param definitionsList_ The `dialsDefinitions` list.
  /// @return The entry whose `name` (or `parameterName`) equals the
  ///         parameter name or, if the parameter has no name, the entry at
  ///         the parameter's index. Empty if none matches.
  /// @pre Per-parameter collection (setSupervisedParameterIndex() called).
  ///      Throws if the set index is not set.
  /// @warning The parameter index is not checked: in a set-wide collection
  ///          it is -1 and indexes the parameter list out of range.
  ConfigReader fetchDialsDefinition(const ConfigReader &definitionsList_) const;

  // factory

  /// Builds a piecewise-linear dial from a TGraph (its points) or a TSpline3
  /// (its knots).
  /// @param src_ Source object; may be nullptr.
  /// @return A Graph dial, or what makeDialShortCircuit() gives (nullptr,
  ///         Shift or Graph). nullptr if @p src_ is nullptr.
  std::unique_ptr<DialBase> makeGraphDial(const TObject* src_) const;

  /// Builds a cubic-spline dial from a TGraph or TSpline3, with the
  /// implementation chosen from getDialOptions().
  ///
  /// The options are searched as substrings. Spline kind: `akima`,
  /// `catmull` (Catmull-Rom), `natural`, `not-a-knot` (the default), `pixar`
  /// (Catmull-Rom) or `ROOT`; when several are present, the later one in
  /// this list wins. `monotonic` asks for a monotonic spline and
  /// `uniformity(TOL)` sets the relative tolerance for uniform knot
  /// spacing. The result is a RootSpline for `ROOT`; for Catmull-Rom, a
  /// MonotonicSpline if monotonic, else a CompactSpline; otherwise a
  /// UniformSpline if the knots are uniform, else a GeneralSpline.
  ///
  /// @param src_ Source object; may be nullptr.
  /// @return The dial, or what makeDialShortCircuit() gives for up to two
  ///         points or a flat response. nullptr if @p src_ is nullptr.
  /// @warning Throws if `uniformity(` has no closing parenthesis, or if
  ///          Catmull-Rom knots are not uniform and a non-default tolerance
  ///          was given. With the default tolerance, non-uniform Catmull-Rom
  ///          knots are only logged as an error.
  std::unique_ptr<DialBase> makeSplineDial(const TObject* src_) const;

  /// Builds a 2D surface dial from a TH2.
  /// @param src_ Source histogram; may be nullptr.
  /// @return A Bilinear or Bicubic dial when getDialOptions() is exactly
  ///         `Bilinear` or `Bicubic`; nullptr if @p src_ is nullptr.
  /// @warning Throws if @p src_ is not a TH2 or the options are anything
  ///          else.
  std::unique_ptr<DialBase> makeSurfaceDial(const TObject* src_) const;

  /// Prepares a point list for building a dial: drops the points outside
  /// `definitionRange` (when it has a bound) and sorts the points by
  /// increasing x.
  /// @param pointList_ Points to clean; modified in place.
  /// @warning Stops the job if `mirrorDefinitionRange` is set: it is not
  ///          implemented.
  void checkDialPointList(std::vector<DialUtils::DialPoint>& pointList_) const;

  /// Handles point lists that do not need a full graph or spline.
  ///
  /// No point gives no dial. One point, or a flat list (DialUtils::isFlat()),
  /// gives no dial if the value is 1 within two float epsilons, and a Shift
  /// dial with that value otherwise. Two points give a Graph dial. While
  /// `_verboseShortCircuit_` is set, the reason is stored in
  /// `_verboseShortCircuitStr_`.
  ///
  /// @param pointList_ Sorted points (see checkDialPointList()).
  /// @param dial_ Output. Set to nullptr, then to the replacement dial if
  ///        there is one.
  /// @return True if the list was handled, and @p dial_ (possibly nullptr)
  ///         is the dial to use. False for three or more points that are not
  ///         flat.
  bool makeDialShortCircuit(const std::vector<DialUtils::DialPoint>& pointList_, std::unique_ptr<DialBase>& dial_) const;

  /// Builds a piecewise-linear Graph dial from points.
  /// @param pointList_ Sorted points (see checkDialPointList()).
  /// @return A Graph dial; for zero or one point, what makeDialShortCircuit()
  ///         gives (nullptr or a Shift dial).
  std::unique_ptr<DialBase> makeGraphDial(const std::vector<DialUtils::DialPoint>& pointList_) const;

private:
  /// Doubles the capacity of the event-dial sharing index (at least 16
  /// slots) and re-inserts its entries. Called by storeEventDial(), with
  /// `_eventDialMutex_` held, when the index is three-quarters full.
  void growEventDialIndex();

  // parameters

  /// True when the dials are created per event while loading data (branch
  /// data, tabulated, kriged and unbinned external-weight dials).
  bool _isEventByEvent_{false};
  /// True for a child collection that DialManager::configureImpl() creates
  /// for a node of a composite dial. Its dials are only used through the
  /// parent's CompositeDial, a flat response of 1 is kept as a dial, and
  /// there is no default minimum response.
  bool _isCompositeNode_{false};
  /// Index of the parent composite collection; -1 if this is not a node.
  int _compositeParentIndex_{-1};
  /// False if the definition sets `isEnabled: false` or is not usable.
  bool _isEnabled_{true};
  /// `useMirrorDial`: fold each input value back into the range from
  /// `_mirrorLowEdge_` to `_mirrorHighEdge_` by reflection at its edges
  /// before the dials see it (DialInputBuffer::update()).
  bool _useMirrorDial_{false};
  /// `printDialSummary` option.
  /// @note Read from the configuration but never used.
  bool _enableDialsSummary_{false};
  /// `allowDialExtrapolation`: passed to every dial that
  /// makeDial(const TObject*) builds.
  bool _allowDialExtrapolation_{true};
  /// Position in DialManager's collection list; -1 until setIndex().
  int _index_{-1};
  /// `minDialResponse` of the definition: the minimum of the response
  /// supervisor. Defaults to 0, or to NaN (no minimum) for composite nodes and
  /// variable dials (readParametersFromConfig()).
  double _minDialResponse_{std::nan("unset")};
  /// `maxDialResponse` of the definition; NaN means no maximum. Used as the
  /// default when the top-level definition has no `maxDialResponse`.
  double _maxDialResponse_{std::nan("unset")};
  /// `mirrorLowEdge`: lower edge of the mirroring range, in parameter units.
  double _mirrorLowEdge_{std::nan("unset")};
  /// `mirrorHighEdge`: upper edge of the mirroring range, in parameter units.
  double _mirrorHighEdge_{std::nan("unset")};
  /// Width of the mirroring range (`_mirrorHighEdge_` - `_mirrorLowEdge_`);
  /// only set when mirroring is enabled.
  double _mirrorRange_{std::nan("unset")};
  /// Event condition from `applyCondition`, or built from `applyConditions`;
  /// empty if there is none.
  std::string _applyConditionStr_{};
  /// `dialBranchData` given as one expression (alternative names
  /// `dialLeafName`, `treeExpression`): tree leaf or expression holding one
  /// dial object (TGraph, TSpline3 or TClonesArray of them) per event.
  std::string _dialLeafName_{};
  /// `dialBranchData/parameterValuesTArray`: branch with the parameter
  /// values of each event's graph; empty unless `dialBranchData` is a block.
  std::string _dialParameterValuesBranch_{};
  /// `dialBranchData/weightValuesTArray`: branch with the responses at those
  /// parameter values; empty unless `dialBranchData` is a block.
  std::string _dialResponsesBranch_{};

  /// `applyToVariable`: event variable changed by the dial responses; empty
  /// for an ordinary (weighting) collection.
  std::string _applyToVariable_{};
  /// `variableResponseMode`: how the responses change that variable.
  VariableResponseMode _variableResponseMode_{VariableResponseMode::Multiply};

  /// Kind of dial this collection builds (`dialType`).
  DialType _dialType_{DialType::Norm}; // Graph, Spline...
  /// Dial options (`options`, alternative names `dialSubType`,
  /// `dialConfig`), read by makeSplineDial() and makeSurfaceDial(). Not set
  /// for types whose `options` is a configuration block.
  std::string _dialOptions_{}; // monotonic, catmull-rom...
  /// `applyOnDataSets`: dataset-name patterns; empty means every dataset.
  std::vector<std::string> _dataSetNameList_{};
  /// Extra event variables needed to build the dials (addExtraLeafName()).
  std::vector<std::string> _globalDialExtraLeafNames_{};
  /// `definitionRange`: points of graph and spline inputs outside this range
  /// are dropped (checkDialPointList()).
  GenericToolbox::Range _definitionRange_{std::nan("unset"),std::nan("unset")};
  /// `knotResponseLimits`: the knot values of graph and spline inputs are
  /// clamped to this range before the dial is built (checkDialPointList()).
  GenericToolbox::Range _knotResponseLimits_{};
  /// `mirrorDefinitionRange`: not implemented; checkDialPointList() stops the
  /// job if it is set.
  GenericToolbox::Range _mirrorDefinitionRange_{std::nan("unset"),std::nan("unset")};

  // internal

  /// When true, makeDialShortCircuit() stores why it replaced a dial. Only
  /// set while initializeBinnedDials() builds dials from a `dialsList`.
  bool _verboseShortCircuit_{false};
  /// Reason stored by makeDialShortCircuit(); mutable because that method is
  /// const.
  mutable std::string _verboseShortCircuitStr_{};
  /// Parameter that every dial reads (per-parameter collection); -1 for a
  /// set-wide collection.
  int _supervisedParameterIndex_{-1};
  /// Parameter set the dials read from; -1 until set.
  int _supervisedParameterSetIndex_{-1};
  /// Event binning of a binned collection: bin `i` selects dial `i`. Empty
  /// for unbinned and event-by-event collections.
  BinSet _dialBinSet_{};

  /// One interface per DialBase.  The interface groups the input buffer,
  /// response supervisor, dialBase (what actually calculates the weight) into
  /// a single object and provide the methods to calculate the dial weight.
  std::vector<DialInterface> _dialInterfaceList_{};

  /// The different sets of parameters to use for the interfaces.  There is a
  /// single set of inputs for each DialBase in the collection, or a different
  /// set of parameters for each DialBase (i.e. the number of DialInputBuffer
  /// objects is 1, or _dialBaseList_.size()).  set, but for
  /// @note `_dialBaseList_` no longer exists: the number of buffers is 1 or
  ///       the size of `_dialInterfaceList_`.
  std::vector<DialInputBuffer> _dialInputBufferList_{};
  /// Inputs of a Formula collection that reads event variables; null
  /// otherwise (see getEventFormulaDefinition()).
  std::shared_ptr<const EventFormula::Definition> _eventFormulaDefinition_{};

  /// The response supervisors (which apply an upper and lower clamp).  There
  /// is a single supervisor for all the interfaces, or one for each
  /// DialBase object in the collection (just like for the
  /// _dialInputBufferList_.
  /// @note The current code creates at most one supervisor per collection.
  std::vector<DialResponseSupervisor> _dialResponseSupervisorList_{};

  /// Next free slot of `_dialInterfaceList_` for event-by-event dials. A
  /// mutex-protected counter shared by the data-loading threads.
  GenericToolbox::Atomic<size_t> _dialFreeSlot_{0};
  /// Guards the sharing index during storeEventDial(). Not copied with the
  /// collection.
  GenericToolbox::NoCopyWrapper<std::mutex> _eventDialMutex_{};
  /// Flat, open-addressed loading index: avoid a heap allocation per unique dial.
  /// The full signature is still compared before two interfaces are shared.
  ///
  /// One slot of the hash table that storeEventDial() uses to find an
  /// identical event dial.
  struct EventDialIndexEntry{
    size_t hash{0}; ///< Hash of the dial's type, extrapolation setting and data.
    size_t interfaceIndex{size_t(-1)}; ///< Interface holding the dial; `size_t(-1)` for an empty slot.
  };
  /// The sharing index, sized to a power of two so that a slot is found by
  /// masking the hash; collisions go to the next slot. Empty outside data
  /// loading.
  std::vector<EventDialIndexEntry> _eventDialIndex_{};
  /// Number of used slots in `_eventDialIndex_`.
  size_t _eventDialIndexSize_{0};
  /// Number of event dials not stored because an identical one existed;
  /// logged by resizeContainers().
  size_t _reusedEventDials_{0};

  /// A pointer to dial specific data
  ///
  /// Holds the factory of a tabulated or kriged collection. Shared between
  /// copies of the collection.
  std::vector<std::shared_ptr<DialCollection::CollectionData>>  _dialCollectionData_;

  /// The callbacks for this dial collection
  ///
  /// update() calls them in the order they were added.
  std::vector<std::function<void(DialCollection*)>> _dialCollectionCallbacks_;

  // external refs

  /// Non-owning pointer to the ParametersManager's list of parameter sets,
  /// given to the constructor.
  std::vector<ParameterSet>* _parameterSetListPtr_{nullptr};

};
#endif //GUNDAM_DIALCOLLECTION_H

// Local Variables:
// mode:c++
// c-basic-offset:2
// End:
