//
// Created by Adrien Blanchet on 29/11/2022.
//

/// @file DialCollection.cpp
/// @brief Implements DialCollection: reads a dial-set definition, builds its
///        dials and links them to their input buffers.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in DialCollection.h. The
/// comments here describe how each method does its job.

#include "DialCollection.h"

#include "TabulatedDialFactory.h"
#include "KrigedDialFactory.h"
#include "ExternalWeightDialFactory.h"
#include "CompositeDial.h"

#include "RootFormula.h"
#include "Shift.h"
#include "RootGraph.h"
#include "Graph.h"
#include "Norm.h"
#include "MonotonicSpline.h"
#include "CompactSpline.h"
#include "UniformSpline.h"
#include "GeneralSpline.h"
#include "RootSpline.h"
#include "Bilinear.h"
#include "Bicubic.h"
#include "CompiledLibDial.h"

#include "DialUtils.h"
#include "GundamGlobals.h"

#include "Logger.h"

#include <TBranch.h>
#include <TClass.h>
#include <TLeaf.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <typeinfo>

/// @par Implementation
/// Clears any earlier declarations, so the reader ends up with exactly the
/// keys listed here, declares every key with its alternative names, then runs
/// ConfigReader::checkConfiguration(). `parameterLimits` is declared only so
/// that the check can flag it as deprecated.
void DialCollection::prepareConfig(ConfigReader &config_){
  config_.clearFields();
  config_.defineFields({
      {"name"},
      {"isEnabled"},
      {"applyOnDataSets"},
      {"dialInputList"},
      {"printDialSummary", {"printDialsSummary"}},
      {"dialType", {"type", "dialsType"}},
      {"options", {"dialSubType", "dialConfig"}},
      {"dialBranchData", {"dialLeafName", "treeExpression"}},
      {"minDialResponse", {"minimumSplineResponse"}},
      {"maxDialResponse"},
      {"useMirrorDial"},
      {"mirrorLowEdge"},
      {"mirrorHighEdge"},
      {"allowDialExtrapolation"},
      {"applyCondition"},
      {"applyToVariable"},
      {"variableResponseMode"},
      {"definitionRange"},
      {"knotResponseLimits"},
      {"mirrorDefinitionRange"},
      {"applyConditions"},
      {"parametersBinningPath"},
      {"binning", {"binningFilePath"}},
      {"binnedDialsPath"},
      {"dialsFilePath"},
      {"dialsList"},
      {"dialsTreePath"},
      {"dialsBranch"},
      {"dialsBinningDefinition"},
      {"dialsDefinitions"},
      {"tableConfig"},
      {FieldFlag::DEPRECATED, "parameterLimits", "\"parameterLimits\" should be set in the parameter definition section. Not the dial definition. Support has been removed."},
    });
  config_.checkConfiguration();
}
/// @par Implementation
/// Calls prepareConfig() on the definition and on every node, recursing into
/// nested composites, then collects the parameter inputs. The collected list
/// keeps one entry per parameter name, in order of first use, so that the
/// composite's input buffer has a stable layout.
void DialCollection::prepareCompositeInputs(ConfigReader& config_){
  prepareConfig(config_);
  LogThrowIf(config_.hasField("dialInputList"),
             "Composite: declare parameterName or parameterIndex in nodes, without an outer dialInputList");
  auto options = config_.fetchValue<ConfigReader>("options");
  options.defineFields({{"nodes"}, {"output"}});
  options.checkConfiguration();
  auto nodes = options.fetchValue<ConfigReader>("nodes");
  JsonType normalizedNodes = JsonType::array();
  JsonType parameters = JsonType::array();
  // Infer a stable, deduplicated buffer layout for cache invalidation. Nested
  // graphs expose their inferred dependencies to their enclosing graph.
  for( auto node : nodes.loop() ){
    prepareConfig(node);
    if( node.fetchValue<DialType>("dialType") == DialType::Composite ){
      prepareCompositeInputs(node);
    }
    auto inputs = node.fetchValue<JsonType>("dialInputList");
    LogThrowIf(not inputs.is_array(), "Composite node dialInputList must be a list");
    for( auto& input : inputs ){
      LogThrowIf(input.contains("inputIndex"),
                 "Composite: inputIndex has been removed; use parameterName or parameterIndex");
      const int sources = int(input.contains("parameterName")) + int(input.contains("parameterIndex"))
                          + int(input.contains("dialResponse")) + int(input.contains("eventVariable"));
      LogThrowIf(sources != 1,
                 "Composite input: specify exactly one of parameterName, parameterIndex, dialResponse and eventVariable");
      if( not input.contains("parameterName") and not input.contains("parameterIndex") ){ continue; }
      auto* parameterSet = getSupervisedParameterSet();
      std::string name;
      if( input.contains("parameterIndex") ){
        const auto& index = input.at("parameterIndex");
        LogThrowIf(not index.is_number_integer() or index < 0 or index >= parameterSet->getParameterList().size(),
                   "Composite: parameterIndex outside the supervised parameter set");
        name = parameterSet->getParameterList().at(index.get<int>()).getName();
        input.erase("parameterIndex");
      }
      else{
        name = input.at("parameterName").get<std::string>();
        LogThrowIf(parameterSet->getParameterPtr(name) == nullptr, "Could not find parameter: " << name);
      }
      input["parameterName"] = name;
      const JsonType reference = {{"parameterName", name}};
      if( std::find(parameters.begin(), parameters.end(), reference) == parameters.end() ){
        parameters.emplace_back(reference);
      }
    }
    node.getConfig()["dialInputList"] = std::move(inputs);
    normalizedNodes.emplace_back(node.getConfig());
  }
  options.getConfig()["nodes"] = std::move(normalizedNodes);
  config_.getConfig()[config_.getConfigEntry("options").first] = options.getConfig();
  config_.getConfig()["dialInputList"] = std::move(parameters);
}

/// @par Implementation
/// A composite that is not itself a node has its inputs prepared first
/// (prepareCompositeInputs()), and the variable-response options are read
/// before the dials are built. The order of the two builders matters: a
/// definition with `parametersBinningPath` always gives binned normalisation
/// dials, whatever its other keys. The `dialInputList` block runs after the
/// dials are built and even if no builder applied. It creates a single
/// DialInputBuffer, so setupDialInterfaceReferences() later skips its
/// automatic buffer setup and connects every dial to this buffer. Each
/// parameter entry names its parameter with `parameterIndex` (range-checked),
/// `parameterName`, or `name` when `parameterName` is absent; an entry with
/// none of these keeps parameter index -1, which
/// DialInputBuffer::initialise() rejects during initialize().
///
/// For a Formula collection, an entry may instead give an `eventVariable`
/// (surrounding brackets are removed). Such an entry is not added to the
/// buffer. The parameter and event-variable entries together form an
/// EventFormula::Definition, kept only if at least one event variable is
/// used; EventDialCache then builds one EventFormula per event from it.
void DialCollection::configureImpl() {
  prepareConfig(_config_);

  _config_.fillValue(_dataSetNameList_, "applyOnDataSets");
  _config_.fillValue(_isEnabled_, "isEnabled");
  // A disabled definition leaves the collection empty: no dial, no buffer.
  if( not _isEnabled_ ){ return; }

  if( not _isCompositeNode_ and _config_.fetchValue<DialType>("dialType", DialType::Unset) == DialType::Composite ){
    prepareCompositeInputs(_config_);
  }

  readVariableResponseConfig(_config_);

  // Dials are directly defined with a binning file?
  if     ( initializeNormDialsWithParBinning() ){}
  else if( initializeDialsWithDefinition() ){}
  else {
    LogAlert << std::endl << "No valid definition for DialCollection. Disabling." << std::endl;
    _isEnabled_ = false;
  }

  // if "dialInputList" is not present, it will be setup during the initialize sequence.
  if( _config_.hasField("dialInputList") ){
    auto dialInputList = _config_.fetchValue<ConfigReader>("dialInputList");

    LogThrowIf(_supervisedParameterSetIndex_ == -1, "Can't initialize dialInputList without setting _supervisedParameterSetIndex_");

    _dialInputBufferList_.emplace_back();
    _dialInputBufferList_.back().setParSetRef( _parameterSetListPtr_ );

    auto formulaDefinition = std::make_shared<EventFormula::Definition>();

    // add the ref of each selected parameter
    for( auto& dialInput : dialInputList.getConfig() ){
      EventFormula::Input source; // where this formula input's value comes from (buffer slot or event variable)
      if( _dialType_ == DialType::Formula ){
        const int sourceCount = int(dialInput.contains("parameterName")) + int(dialInput.contains("parameterIndex"))
                                + int(dialInput.contains("eventVariable"));
        LogThrowIf(sourceCount != 1,
                   "Formula input must specify exactly one of parameterName, parameterIndex and eventVariable");
      }
      if( GenericToolbox::Json::doKeyExist(dialInput, "eventVariable") ){
        LogThrowIf(_dialType_ != DialType::Formula, "eventVariable inputs are only supported by Formula dials");
        auto name = GenericToolbox::Json::fetchValue<std::string>(dialInput, "eventVariable");
        name = GenericToolbox::trimString(name, " ");
        if( name.size() >= 2 and name.front() == '[' and name.back() == ']' ){
          name = name.substr(1, name.size() - 2);
        }
        LogThrowIf(name.empty(), "Formula eventVariable must not be empty");
        GenericToolbox::addIfNotInVector(name, formulaDefinition->eventVariables);
        source.eventIndex = GenericToolbox::findElementIndex(name, formulaDefinition->eventVariables);
        formulaDefinition->inputs.emplace_back(source);
        continue;
      }
      source.parameterIndex = int(_dialInputBufferList_.back().getInputSize());
      formulaDefinition->inputs.emplace_back(source);
      DialInputBuffer::ParameterReference p; // the parameter this entry reads
      p.parSetIndex = _supervisedParameterSetIndex_;

      const std::string parameterKey = dialInput.contains("parameterName") ? "parameterName" : "name";
      if( dialInput.contains("parameterIndex") ){
        const auto& index = dialInput.at("parameterIndex");
        LogThrowIf(not index.is_number_integer() or index < 0
                   or index >= _parameterSetListPtr_->at(_supervisedParameterSetIndex_).getParameterList().size(),
                   "Dial input: parameterIndex outside the supervised parameter set");
        p.parIndex = index.get<int>();
      }
      else if( GenericToolbox::Json::doKeyExist(dialInput, parameterKey) ){
        auto parName = GenericToolbox::Json::fetchValue<std::string>(dialInput, parameterKey);
        auto* parPtr{_parameterSetListPtr_->at( _supervisedParameterSetIndex_ ).getParameterPtr( parName )};
        LogThrowIf(parPtr == nullptr, "Could not find parameter: " << parName);
        p.parIndex = parPtr->getParameterIndex();
      }

      _dialInputBufferList_.back().addParameterReference( p );
    }
    if( not formulaDefinition->eventVariables.empty() ){ _eventFormulaDefinition_ = std::move(formulaDefinition); }

  }

}
/// @par Implementation
/// Checks that the index was set, initializes the external-weight factories,
/// lets identical binned dials share one object (deduplicateBinnedDials()),
/// then delegates to setupDialInterfaceReferences(). Event-by-event
/// collections hold no dial
/// yet at this point; their interfaces are linked again by
/// resizeContainers() once the data is loaded.
void DialCollection::initializeImpl() {
  LogThrowIf(_index_==-1, "Index not set.");

  for( auto& collectionData : _dialCollectionData_ ){
    if( auto* externalWeight = dynamic_cast<ExternalWeightDialFactory*>(collectionData.get()) ){
      externalWeight->initialize();
    }
  }

  this->deduplicateBinnedDials();
  this->setupDialInterfaceReferences();
}

// non-trivial getters
/// @par Implementation
/// Lower-cases the dataset name and each pattern, then uses
/// GenericToolbox::isMatching(), which turns `*` into a regex wildcard and
/// escapes the other regex characters.
bool DialCollection::isDatasetValid(const std::string& datasetName_) const{
  if( _dataSetNameList_.empty() ){ return true; }

  auto datasetNameLowerCase = GenericToolbox::toLowerCase(datasetName_);
  bool matching = std::any_of( _dataSetNameList_.begin(), _dataSetNameList_.end(), [&](const std::string& datasetPattern){
    return GenericToolbox::isMatching(datasetNameLowerCase, GenericToolbox::toLowerCase(datasetPattern));
  });
  return matching;
}

/// @par Implementation
/// Tries the supervised parameter first, then the supervised set, so a
/// per-parameter collection is named after its parameter.
std::string DialCollection::getTitle() const {

  auto* parPtr{this->getSupervisedParameter()};
  if( parPtr != nullptr ){ return parPtr->getFullTitle(); }

  auto* parSetPtr{this->getSupervisedParameterSet()};
  if( parSetPtr != nullptr ){ return parSetPtr->getName(); }

  return {"UnsetParameterSuperVision"};
}
/// @par Implementation
/// Appends each part only when it carries information (type not Unset,
/// non-empty strings, ranges with a bound). The event-by-event test sits
/// inside the dial loop, so for those collections the loop runs but prints
/// nothing.
std::string DialCollection::getSummary(bool shallow_) const{
  std::stringstream ss; // the summary being built
  ss << "DialCollection: ";
  ss << this->getTitle();
  if( _dialType_ != DialType::Unset ){ ss << " / " << _dialType_; }
  if( not _dialOptions_.empty() ){ ss << ":\"" << _dialOptions_ << "\""; }
  if( not _dialLeafName_.empty() ){ ss << " / dialBranchData:" << _dialLeafName_; }
  if( not _dialParameterValuesBranch_.empty() ){
    ss << " / dialBranchData:{parameterValuesTArray:" << _dialParameterValuesBranch_
       << ", weightValuesTArray:" << _dialResponsesBranch_ << "}";
  }
  if( not _applyToVariable_.empty() ){ ss << " / variable:" << _applyToVariable_ << " / mode:" << _variableResponseMode_; }
  if( _definitionRange_.hasBound() ){ ss << " / definitionRange:" << _definitionRange_; }
  if( _knotResponseLimits_.hasBound() ){ ss << " / knotResponseLimits:" << _knotResponseLimits_; }
  if( _mirrorDefinitionRange_.hasBound() ){ ss << " / mirrorDefinitionRange:" << _mirrorDefinitionRange_; }

  if( not shallow_ ){
    // print parameters
    for( auto& dialInterface : _dialInterfaceList_ ){
      if( not isEventByEvent() ){
        ss << std::endl << "  " << dialInterface.getSummary(shallow_);
      }
    }
  }

  return ss.str();
}

/// @par Implementation
/// Returns nullptr when the set is missing or the parameter index is
/// negative.
/// @note The upper bound test uses `>` instead of `>=`: an index equal to
///       the number of parameters passes it, and `.at()` then throws
///       std::out_of_range instead of the function returning nullptr.
Parameter* DialCollection::getSupervisedParameter() const {
  auto* parSetPtr = this->getSupervisedParameterSet();
  if( parSetPtr == nullptr ) return nullptr;
  if( _supervisedParameterIndex_ < 0 ) return nullptr;
  if( _supervisedParameterIndex_ > parSetPtr->getParameterList().size() ) return nullptr;
  return &parSetPtr->getParameterList().at(_supervisedParameterIndex_);
}

/// @par Implementation
/// Returns nullptr when the set index is negative. `_parameterSetListPtr_`
/// is used without a null check.
/// @note As in getSupervisedParameter(), the upper bound test uses `>`: an
///       index equal to the list size makes `.at()` throw
///       std::out_of_range.
ParameterSet* DialCollection::getSupervisedParameterSet() const{
  if( _supervisedParameterSetIndex_ < 0 ) return nullptr;
  if( _supervisedParameterSetIndex_ > _parameterSetListPtr_->size() ) return nullptr;
  return &_parameterSetListPtr_->at(_supervisedParameterSetIndex_);
}

// core
namespace {
  /// Tells whether a list of points lies on one straight line.
  /// @param points_ Points sorted by `x`.
  /// @param checkSlopes_ If true, the `slope` of every point must also equal
  ///        the slope of the line.
  /// @return True if there are at least two points, the last `x` is larger
  ///         than the first, and every value (and slope) matches the line
  ///         through the first and last points to within 64 double epsilons
  ///         (relative to the larger magnitude, at least 1).
  bool isAffine(const std::vector<DialUtils::DialPoint>& points_, bool checkSlopes_){
    if( points_.size() < 2 or not (points_.back().x > points_.front().x) ){ return false; }
    const double slope = DialUtils::getSlope(points_.front(), points_.back());
    if( not std::isfinite(slope) ){ return false; }
    auto equal = [](double lhs_, double rhs_){
      return std::isfinite(lhs_) and std::isfinite(rhs_)
          and std::abs(lhs_ - rhs_) <= 64 * std::numeric_limits<double>::epsilon()
              * std::max(1.0, std::max(std::abs(lhs_), std::abs(rhs_)));
    };
    for( const auto& point : points_ ){
      if( not equal(point.y, points_.front().y + slope * (point.x - points_.front().x)) ){ return false; }
      if( checkSlopes_ and not equal(point.slope, slope) ){ return false; }
    }
    return true;
  }

  // Include all evaluation state, including the exact upper bound (which can
  // differ from the value reconstructed from a uniform grid's rounded step).
  /// Fills the signature of a spline dial of type `T`.
  /// @tparam T One of the spline classes with getDialData() and
  ///         getSplineBounds().
  /// @param dial_ The dial to describe.
  /// @param signature_ Replaced by the dial data followed by the lower and
  ///        upper knot bounds.
  /// @return False if `dial_` is not a `T`.
  template<typename T> bool appendSplineSignature(const DialBase& dial_, std::vector<double>& signature_){
    const auto* spline = dynamic_cast<const T*>(&dial_);
    if( spline == nullptr ){ return false; }
    signature_ = spline->getDialData();
    signature_.push_back(spline->getSplineBounds().min);
    signature_.push_back(spline->getSplineBounds().max);
    return true;
  }

  /// Describes the content of a dial by a list of numbers, so that two dials
  /// with equal lists (and the same type and extrapolation setting) give the
  /// same response and can share one object.
  /// @param dial_ The dial to describe.
  /// @param signature_ Filled with the spline data and bounds, the graph
  ///        data, the Shift value, or nothing for a Norm.
  /// @return False for every other dial type, which is never shared.
  bool getDialSignature(const DialBase& dial_, std::vector<double>& signature_){
    if( appendSplineSignature<CompactSpline>(dial_, signature_)
        or appendSplineSignature<MonotonicSpline>(dial_, signature_)
        or appendSplineSignature<UniformSpline>(dial_, signature_)
        or appendSplineSignature<GeneralSpline>(dial_, signature_) ){ return true; }
    if( dynamic_cast<const Graph*>(&dial_) != nullptr ){
      signature_ = dial_.getDialData();
      return true;
    }
    if( const auto* shift = dynamic_cast<const Shift*>(&dial_) ){
      signature_ = {shift->getShiftValue()};
      return true;
    }
    return dynamic_cast<const Norm*>(&dial_) != nullptr;
  }

  /// Hashes a dial's type, extrapolation setting and signature.
  /// @param dial_ The dial.
  /// @param signature_ Its signature (getDialSignature()).
  /// @return A hash used to find candidate dials; equal hashes do not prove
  ///         equal dials (see matchesDialSignature()).
  size_t getDialSignatureHash(const DialBase& dial_, const std::vector<double>& signature_){
    size_t hash = typeid(dial_).hash_code();
    auto combineHash = [&](size_t value_){ hash ^= value_ + 0x9e3779b9 + (hash << 6) + (hash >> 2); };
    combineHash(std::hash<bool>{}(dial_.getAllowExtrapolation()));
    for( double value : signature_ ){ combineHash(std::hash<double>{}(value)); }
    return hash;
  }

  /// Tells whether a stored dial has the same content as a new one.
  /// @param candidate_ The stored dial.
  /// @param dial_ The new dial.
  /// @param signature_ The new dial's signature.
  /// @return True if the types and extrapolation settings match and the
  ///         candidate's signature equals `signature_` exactly.
  bool matchesDialSignature(const DialBase& candidate_, const DialBase& dial_,
                            const std::vector<double>& signature_){
    if( typeid(candidate_) != typeid(dial_)
        or candidate_.getAllowExtrapolation() != dial_.getAllowExtrapolation() ){ return false; }
    std::vector<double> candidateSignature;
    return getDialSignature(candidate_, candidateSignature) and candidateSignature == signature_;
  }
}

/// @par Implementation
/// A hash of each dial's signature finds candidates in a multimap; a
/// candidate is shared only after its full signature compares equal. The
/// first dial with a given content keeps its object, and later ones share
/// it.
void DialCollection::deduplicateBinnedDials(){
  if( isEventByEvent() or _dialBinSet_.getBinList().empty() ){ return; }
  std::unordered_multimap<size_t, size_t> index;
  std::vector<double> signature;
  size_t reused{0};
  for( size_t iDial = 0; iDial < _dialInterfaceList_.size(); ++iDial ){
    auto& interface = _dialInterfaceList_[iDial];
    const auto* dial = interface.getDialBaseRef();
    signature.clear();
    if( dial == nullptr or not getDialSignature(*dial, signature) ){ continue; }
    const auto hash = getDialSignatureHash(*dial, signature);
    const auto matches = index.equal_range(hash);
    bool found{false};
    for( auto match = matches.first; match != matches.second; ++match ){
      const auto& candidate = _dialInterfaceList_[match->second];
      if( matchesDialSignature(*candidate.getDialBaseRef(), *dial, signature) ){
        interface.shareDial(candidate);
        ++reused;
        found = true;
        break;
      }
    }
    if( not found ){ index.emplace(hash, iDial); }
  }
  if( reused != 0 ){
    LogInfo << getTitle() << ": shared " << reused << " binned dial definitions; kept "
            << _dialInterfaceList_.size() << " bin interfaces." << std::endl;
  }
}

/// @par Implementation
/// Linear probing: each entry is re-inserted at its hash masked by the new
/// size, or at the next free slot after it.
void DialCollection::growEventDialIndex(){
  // Power-of-two capacity permits masking; keep at least one empty slot so
  // unsuccessful lookups always terminate. Called under _eventDialMutex_.
  std::vector<EventDialIndexEntry> expanded(std::max(size_t(16), 2 * _eventDialIndex_.size()));
  for( const auto& entry : _eventDialIndex_ ){
    if( entry.interfaceIndex == size_t(-1) ){ continue; }
    size_t slot = entry.hash & (expanded.size() - 1);
    while( expanded[slot].interfaceIndex != size_t(-1) ){
      slot = (slot + 1) & (expanded.size() - 1);
    }
    expanded[slot] = entry;
  }
  _eventDialIndex_.swap(expanded);
}

/// @par Implementation
/// The signature is computed before the lock is taken. Under
/// `_eventDialMutex_`, a match is looked up by linear probing in the
/// open-addressed index; on a match, the new dial is dropped and the count of
/// shared dials increases. Otherwise the index is grown when it is
/// three-quarters full, the dial is moved into a new slot
/// (getNextDialFreeSlot()), and, if it can be shared, recorded in the index.
size_t DialCollection::storeEventDial(std::unique_ptr<DialBase> dial_, bool* hasFiniteSignature_){
  LogThrowIf(dial_ == nullptr, "Cannot store a null event dial");
  std::vector<double> signature;
  // Interface sharing is only valid when every interface uses the same inputs
  // and response supervisor. ROOT splines and event-dependent factories stay unique.
  bool canShare = _dialInputBufferList_.size() == 1
      and _dialResponseSupervisorList_.size() == 1
      and getDialSignature(*dial_, signature);
  if( hasFiniteSignature_ != nullptr ){
    *hasFiniteSignature_ = canShare and std::all_of(signature.begin(), signature.end(),
        [](double value_){ return std::isfinite(value_); });
  }
  const size_t hash = getDialSignatureHash(*dial_, signature);

  std::unique_lock<std::mutex> lock(_eventDialMutex_);
  if( canShare and not _eventDialIndex_.empty() ){
    size_t slot = hash & (_eventDialIndex_.size() - 1);
    while( _eventDialIndex_[slot].interfaceIndex != size_t(-1) ){
      const auto& entry = _eventDialIndex_[slot];
      if( entry.hash == hash ){
        const auto* candidate = _dialInterfaceList_[entry.interfaceIndex].getDialBaseRef();
        if( matchesDialSignature(*candidate, *dial_, signature) ){
          ++_reusedEventDials_;
          return entry.interfaceIndex;
        }
      }
      slot = (slot + 1) & (_eventDialIndex_.size() - 1);
    }
  }
  if( canShare and _eventDialIndexSize_ >= _eventDialIndex_.size() * 3 / 4 ){
    growEventDialIndex();
  }
  const size_t index = getNextDialFreeSlot();
  if( _dialInterfaceList_.size() <= index ){ _dialInterfaceList_.resize(index + 1); }
  _dialInterfaceList_[index].getDial().dialPtr = std::move(dial_);
  if( canShare ){
    size_t slot = hash & (_eventDialIndex_.size() - 1);
    while( _eventDialIndex_[slot].interfaceIndex != size_t(-1) ){
      slot = (slot + 1) & (_eventDialIndex_.size() - 1);
    }
    _eventDialIndex_[slot] = {hash, index};
    ++_eventDialIndexSize_;
  }
  return index;
}

/// @par Implementation
/// A factory collection is rebuilt by configuring a fresh collection and
/// moving it into this one, so that factory state (event inputs,
/// shared-memory buffers, external workers) is dropped instead of being
/// shared with a copied Propagator.
void DialCollection::resetForLoad(){
  if( not _dialCollectionData_.empty() ){
    // Factories may own event inputs, shared-memory buffers or external workers.
    // Recreate them rather than retaining state shared with a copied propagator.
    DialCollection fresh(_parameterSetListPtr_);
    fresh.setIndex(_index_);
    fresh.setSupervisedParameterSetIndex(_supervisedParameterSetIndex_);
    fresh.setSupervisedParameterIndex(_supervisedParameterIndex_);
    // Factory nodes must remain children when recreated for a new load.
    // Otherwise their response is also applied as an independent event weight.
    fresh.setIsCompositeNode(_isCompositeNode_);
    fresh.setCompositeParentIndex(_compositeParentIndex_);
    fresh.configure(_config_);
    *this = std::move(fresh);
    initialize();
  }
  else{
    if( isEventByEvent() ){ clear(); }
    // Also repairs references after copying the model propagator to data.
    if( isEnabled() ){ setupDialInterfaceReferences(); }
  }
  invalidateCachedInputBuffers();
}

/// @par Implementation
/// `shrink_to_fit()` releases the memory of the interface list, which can
/// be large for event-by-event collections (one entry per event). The
/// sharing index is released by swapping it with an empty vector.
void DialCollection::clear(){
  _dialInterfaceList_.clear();
  _dialInterfaceList_.shrink_to_fit();
  _dialFreeSlot_.setValue(0);
  std::vector<EventDialIndexEntry>().swap(_eventDialIndex_);
  _eventDialIndexSize_ = 0;
  _reusedEventDials_ = 0;
}

/// @par Implementation
/// Slots are handed out from 0 upwards, so the first `_dialFreeSlot_`
/// entries are exactly the filled ones and a plain resize keeps them all.
/// The sharing index is only needed while loading, so it is released first.
/// The interfaces added by DataDispenser have no buffer or supervisor
/// pointer yet, which is why setupDialInterfaceReferences() runs again.
void DialCollection::resizeContainers(){
  if( _reusedEventDials_ != 0 ){
    LogInfo << getTitle() << ": reused " << _reusedEventDials_
            << " event dials; stored " << _dialFreeSlot_.getValue() << " interfaces." << std::endl;
  }
  std::vector<EventDialIndexEntry>().swap(_eventDialIndex_);
  _eventDialIndexSize_ = 0;
  _reusedEventDials_ = 0;
  LogDebugIf(GundamGlobals::isDebug()) << "Resizing containers of the dial collection \"" << this->getTitle() << "\" from "
                                       << _dialInterfaceList_.size() << " to " << _dialFreeSlot_.getValue() << std::endl;

  _dialInterfaceList_.resize(_dialFreeSlot_.getValue());
  _dialInterfaceList_.shrink_to_fit();
  this->setupDialInterfaceReferences();
}

/// @par Implementation
/// A plain loop on the calling thread. A collection has at most one buffer
/// per parameter of its supervised set, so the loop is short.
void DialCollection::updateInputBuffers(){
  for( auto& inputBuffer : _dialInputBufferList_ ){ inputBuffer.update(); }
}

/// @par Implementation
/// The buffers are created only when the list is empty, so a second call
/// (from resizeContainers()) or a `dialInputList` buffer made by
/// configureImpl() is kept. Calling DialInputBuffer::initialise() again is
/// harmless: it only recomputes sizes and re-checks the references. The
/// mirror edges are written into every parameter reference of every
/// buffer. The interface loop then matches each container by size: one
/// shared element, or element `iDial` when the sizes are equal.
/// @note The upper bound test on the set index uses `>` instead of `>=`;
///       an index equal to the list size passes it, and the later `.at()`
///       calls then throw std::out_of_range.
/// @note In the interface loop, the second "Input buffers" comment heads the
///       block that sets the bin references, not the input buffers.
void DialCollection::setupDialInterfaceReferences(){
  LogThrowIf(_supervisedParameterSetIndex_==-1, "par set index not set.");
  LogThrowIf(_supervisedParameterSetIndex_>_parameterSetListPtr_->size(), "invalid selected parset index: " << _supervisedParameterSetIndex_);

  // set it up is not already done
  if( _dialInputBufferList_.empty() ){
    if( _supervisedParameterIndex_ == -1 ){
      // one dial interface per parameter
      if (_dialInterfaceList_.size() != _parameterSetListPtr_->at(_supervisedParameterSetIndex_).getParameterList().size()) {
        LogError << "Nb of dial base don't match the number of parameters of the selected set: nDials="
                 << _dialInterfaceList_.size() << " != " << "nPars="
                 << _parameterSetListPtr_->at(_supervisedParameterSetIndex_).getParameterList().size()
                 << std::endl << "is the defined dial binning matching the number of parameters?" << std::endl;
        LogExit("Bad dial definition");
      }

      _dialInputBufferList_.resize(_dialInterfaceList_.size());
      for( int iDial = 0 ; iDial < int(_dialInterfaceList_.size()) ; iDial++ ){
        DialInputBuffer::ParameterReference p{}; // dial iDial reads parameter iDial of the set
        p.parSetIndex = _supervisedParameterSetIndex_;
        p.parIndex = iDial;
        _dialInputBufferList_[iDial].addParameterReference(p);
      }
    }
    else{
      // one parameter for every dial of the collection
      _dialInputBufferList_.resize(1);
      DialInputBuffer::ParameterReference p{}; // the supervised parameter, shared by every dial
      p.parSetIndex = _supervisedParameterSetIndex_;
      p.parIndex = _supervisedParameterIndex_;
      _dialInputBufferList_.back().addParameterReference(p);
    }
  }

  for( auto& inputBuffer : _dialInputBufferList_ ){ inputBuffer.setParSetRef( _parameterSetListPtr_ ); }

  if( _useMirrorDial_ ){
    for( auto& inputBuffer : _dialInputBufferList_ ){
      for( auto & inputParRef : inputBuffer.getInputParameterIndicesList() ){
        inputParRef.mirrorEdges.minValue = _mirrorLowEdge_;
        inputParRef.mirrorEdges.range = _mirrorRange_;
      }
    }
  }

  // initialize the input buffers
  for( auto& inputBuffer : _dialInputBufferList_ ){ inputBuffer.initialise(); }

  // Initializing dial interfaces:
  for( size_t iDial = 0 ; iDial < _dialInterfaceList_.size() ; iDial++ ){
    // Input buffers
    if( _dialInputBufferList_.size() == 1 ){
      _dialInterfaceList_[iDial].setInputBufferRef( &_dialInputBufferList_[0] );
    }
    else if( _dialInputBufferList_.size() == _dialInterfaceList_.size() ){
      _dialInterfaceList_[iDial].setInputBufferRef( &_dialInputBufferList_[iDial] );
    }
    else{
      LogThrow("DEV: size mismatch between input buffers and dial interfaces."
                   << std::endl << "interface = " << _dialInterfaceList_.size()
                   << std::endl << "input = " << _dialInputBufferList_.size()
      );
    }

    // Input buffers
    if( not _dialBinSet_.getBinList().empty() ){
      if( _dialBinSet_.getBinList().size() == 1 ){
        _dialInterfaceList_[iDial].setDialBinRef( &_dialBinSet_.getBinList()[0] );
      }
      else if( _dialBinSet_.getBinList().size() == _dialInterfaceList_.size() ){
        _dialInterfaceList_[iDial].setDialBinRef( &_dialBinSet_.getBinList()[iDial] );
      }
      else{
        LogThrow("DEV: size mismatch between bins and dial interfaces."
                     << std::endl << "interface = " << _dialInterfaceList_.size()
                     << std::endl << "bins = " << _dialBinSet_.getBinList().size()
        );
      }
    }

    // Supervisor reference
    if( _dialResponseSupervisorList_.size() == 1 ){
      _dialInterfaceList_[iDial].setResponseSupervisorRef( &_dialResponseSupervisorList_[0] );
    }
    else if( _dialResponseSupervisorList_.size() == _dialInterfaceList_.size() ){
      _dialInterfaceList_[iDial].setResponseSupervisorRef( &_dialResponseSupervisorList_[iDial] );
    }
    else{
      LogThrow("DEV: size mismatch between response supervisors and dial interfaces."
                   << std::endl << "interface = " << _dialInterfaceList_.size()
                   << std::endl << "supervisor = " << _dialResponseSupervisorList_.size()
      );
    }
  }
}

// init protected
/// @par Implementation
/// `variableResponseMode` is read with ConfigReader::fillEnum(), which keeps
/// `Multiply` when the key is absent; an empty string is rejected first.
void DialCollection::readVariableResponseConfig(const ConfigReader& config_){
  if( config_.hasField("applyToVariable") ){
    _applyToVariable_ = config_.fetchValue<std::string>("applyToVariable");
  }
  if( config_.hasField("variableResponseMode") ){
    LogThrowIf(config_.fetchValue<std::string>("variableResponseMode").empty(), "variableResponseMode must not be empty");
  }
  config_.fillEnum(_variableResponseMode_, "variableResponseMode");
  LogThrowIf(_variableResponseMode_ != VariableResponseMode::Multiply
             and _variableResponseMode_ != VariableResponseMode::Add
             and _variableResponseMode_ != VariableResponseMode::Override,
             "Invalid variableResponseMode: expected Multiply, Add, or Override");
  LogThrowIf(config_.hasField("applyToVariable") and _applyToVariable_.empty(), "applyToVariable must not be empty");
  LogThrowIf(config_.hasField("variableResponseMode") and _applyToVariable_.empty(),
             "variableResponseMode requires applyToVariable");
}

/// @par Implementation
/// Each `fillValue()` keeps the member unchanged when the key is absent, so
/// the defaults in the header apply. The exception is `minDialResponse`,
/// whose default is set here: 0, except for composite nodes and variable
/// dials, which may return negative values and get no minimum (NaN).
///
/// `options` is only read as a string for dial types that use it as one;
/// Formula, CompiledLibDial, ExternalWeight and Composite collections keep
/// a configuration block there, read by their own builders. `dialBranchData`
/// is either a branch expression (`_dialLeafName_`) or a block with
/// `parameterValuesTArray` and `weightValuesTArray`, allowed only for Spline
/// and Graph dials.
///
/// For `applyConditions`, a structured entry with expression `x` is
/// translated as follows:
/// - `allowedRanges` [a, b): `(x >= a && x < b)`, OR-ed together;
/// - `allowedValues` v: `x == v`, OR-ed together and with the ranges;
/// - `excludedRanges` [a, b): `x < a && x >= b`, AND-ed to the rest;
/// - `excludedValues` v: `x == v`, AND-ed to the rest.
///
/// The entries are then wrapped in parentheses and AND-ed, and the result
/// overwrites any `applyCondition` string. The string is kept as it is;
/// DataDispenser compiles it when it loads the data.
///
/// @warning The `excludedRanges` term `x < a && x >= b` can never be true
///          when a < b (which is enforced), so an entry with
///          `excludedRanges` is always false and the dial is never applied.
///          The `excludedValues` term `x == v` keeps only events equal to
///          `v` instead of excluding them (and is never true for two
///          different values).
/// @warning When an entry has both allowed and excluded fields, the code
///          wraps the allowed part with `ssCondEntry.str("(" + ... + ")")`.
///          That call moves the stream's write position back to the start,
///          so the following ` && ...` text overwrites the beginning of the
///          expression instead of being appended.
/// @note The two "allowed regions are linked with OR" comments next to the
///       excluded terms are copies: those terms are joined with `&&`.
/// @note The `parameterLimits` block at the end is not reached in practice:
///       prepareConfig() declares the key deprecated, and
///       ConfigReader::checkConfiguration() already stops the job when the
///       key is present.
void DialCollection::readParametersFromConfig(const ConfigReader &config_) {
  readVariableResponseConfig(config_);
  // globals for the dialSet
  config_.fillValue(_enableDialsSummary_, "printDialSummary");
  config_.fillValue(_dialType_, "dialType");
  if( _dialType_ != DialType::ExternalWeight and _dialType_ != DialType::Formula
      and _dialType_ != DialType::CompiledLibDial and _dialType_ != DialType::Composite ){
    config_.fillValue(_dialOptions_, "options");
  }
  if( config_.hasField("dialBranchData") ){
    auto branchData = config_.fetchValue<ConfigReader>("dialBranchData");
    _dialLeafName_.clear();
    _dialParameterValuesBranch_.clear();
    _dialResponsesBranch_.clear();
    if( branchData.getConfig().is_string() ){
      _dialLeafName_ = branchData.getConfig().get<std::string>();
      LogThrowIf(_dialLeafName_.empty(), "dialBranchData must not be empty");
    }
    else{
      LogThrowIf(not branchData.getConfig().is_object(),
                 "dialBranchData must be a branch expression or {parameterValuesTArray, weightValuesTArray}");
      branchData.defineFields({{"parameterValuesTArray"}, {"weightValuesTArray"}});
      branchData.checkConfiguration();
      _dialParameterValuesBranch_ = branchData.fetchValue<std::string>("parameterValuesTArray");
      _dialResponsesBranch_ = branchData.fetchValue<std::string>("weightValuesTArray");
      LogThrowIf(_dialParameterValuesBranch_.empty() or _dialResponsesBranch_.empty(),
                 "dialBranchData requires non-empty parameterValuesTArray and weightValuesTArray branch names");
      LogThrowIf(_dialType_ != DialType::Spline and _dialType_ != DialType::Graph,
                 "TArray dialBranchData requires dialType Spline or Graph");
    }
  }
// Only final event weights are nonnegative by default. Variable responses
  // (including multipliers) and intermediate composite responses may be negative.
  _minDialResponse_ = not _isCompositeNode_ and _applyToVariable_.empty()
      ? 0. : std::nan("unset");
  config_.fillValue(_minDialResponse_, "minDialResponse");
  config_.fillValue(_maxDialResponse_, "maxDialResponse");
  config_.fillValue(_useMirrorDial_, "useMirrorDial");
  config_.fillValue(_mirrorLowEdge_, "mirrorLowEdge");
  config_.fillValue(_mirrorHighEdge_, "mirrorHighEdge");
  config_.fillValue(_allowDialExtrapolation_, "allowDialExtrapolation");
  config_.fillValue(_applyConditionStr_, "applyCondition");
  config_.fillValue(_definitionRange_, "definitionRange");
  config_.fillValue(_knotResponseLimits_, "knotResponseLimits");
  LogThrowIf(_knotResponseLimits_.hasBothBounds() and _knotResponseLimits_.min > _knotResponseLimits_.max,
             "knotResponseLimits: min must not exceed max");
  config_.fillValue(_mirrorDefinitionRange_, "mirrorDefinitionRange");

  if( config_.hasField("applyConditions") ){
    std::vector<std::string> conditionsList;

    for (auto &condEntry: config_.loop("applyConditions")) {
      condEntry.defineFields({
        {"exp", {"expression", "var", "variable"}},
        {"allowedRanges"},
        {"allowedValues"},
        {"excludedRanges"},
        {"excludedValues"},
      });

      if (condEntry.getConfig().is_string()) {
        conditionsList.emplace_back(condEntry.getConfig().get<std::string>());
      }
      else if (condEntry.getConfig().is_structured()) {
        auto expression = condEntry.fetchValue<std::string>("exp"); // the constrained variable
        std::stringstream ssCondEntry; // condition being built for this entry

        // allowedRanges
        {
          std::vector<GenericToolbox::Range> allowedRanges;
          condEntry.fillValue(allowedRanges, "allowedRanges");
          if (not allowedRanges.empty()) {
            std::vector<std::string> allowedRangesCond;
            for (auto &allowedRange: allowedRanges) {
              LogThrowIf(allowedRange.min >= allowedRange.max,
                         "Invalid range bounds: min(" << allowedRange.min << ") max(" << allowedRange.max << ")")
              std::stringstream condSs; // one term of the condition
              condSs << "(" << expression << " >= " << allowedRange.min;
              condSs << " && " << expression << " < " << allowedRange.max << ")";
              allowedRangesCond.emplace_back(condSs.str());
            }
            ssCondEntry << GenericToolbox::joinVectorString(allowedRangesCond, " || ");
          }
        }

        // allowedValues
        {
          std::vector<double> allowedValues{};
          condEntry.fillValue(allowedValues, "allowedValues");
          if (not allowedValues.empty()) {
            std::vector<std::string> allowedValuesCond;
            for (auto &allowedValue: allowedValues) {
              std::stringstream condSs; // one term of the condition
              condSs << expression << " == " << allowedValue;
              allowedValuesCond.emplace_back(condSs.str());
            }
            if (not ssCondEntry.str().empty()) ssCondEntry << " || "; // allowed regions are linked with "OR"
            ssCondEntry << GenericToolbox::joinVectorString(allowedValuesCond, " || ");
          }
        }

        std::vector<GenericToolbox::Range> excludedRanges;
        std::vector<double> excludedValues;
        condEntry.fillValue(excludedRanges, "excludedRanges");
        condEntry.fillValue(excludedValues, "excludedValues");
        if (not excludedRanges.empty() or not excludedValues.empty()) {
          if (not ssCondEntry.str().empty()) {
            // exclusion ranges are linked with &&: they are supposed to prevail
            ssCondEntry.str("(" + ssCondEntry.str() + ")");
            // after that no parenthesis needed since only && will be used
            // (str(s) also rewinds the write position: see the warning above)
          }

          {
            if (not excludedRanges.empty()) {
              std::vector<std::string> excludedRangesCond;
              for (auto &excludedRange: excludedRanges) {
                LogThrowIf(excludedRange.min >= excludedRange.max,
                           "Invalid range bounds: min(" << excludedRange.min << ") max(" << excludedRange.max
                                                        << ")")
                std::stringstream condSs; // one term of the condition
                condSs << expression << " < " << excludedRange.min << " && ";
                condSs << expression << " >= " << excludedRange.max;
                excludedRangesCond.emplace_back(condSs.str());
              }
              if (not ssCondEntry.str().empty()) ssCondEntry << " && "; // allowed regions are linked with "OR"
              ssCondEntry << GenericToolbox::joinVectorString(excludedRangesCond, " && ");
            }
          }

          {
            if (not excludedValues.empty()) {
              std::vector<std::string> excludedValuesCond;
              for (auto &excludedValue: excludedValues) {
                std::stringstream condSs; // one term of the condition
                condSs << expression << " == " << excludedValue;
                excludedValuesCond.emplace_back(condSs.str());
              }
              if (not ssCondEntry.str().empty()) ssCondEntry << " && "; // allowed regions are linked with "OR"
              ssCondEntry << GenericToolbox::joinVectorString(excludedValuesCond, " && ");
            }
          }
        }

        LogThrowIf(ssCondEntry.str().empty(), "Could not parse condition entry: " << condEntry)
        conditionsList.emplace_back(ssCondEntry.str());
      } else {
        LogThrow("Could not recognise condition entry: " << condEntry);
      }
    }

    LogThrowIf(conditionsList.empty(), "No apply condition was recognised.")
    _applyConditionStr_ = "( ";
    _applyConditionStr_ += GenericToolbox::joinVectorString(conditionsList, " ) && ( ");
    _applyConditionStr_ += " )";
  }

  if (_useMirrorDial_) {
    _mirrorRange_ = _mirrorHighEdge_ - _mirrorLowEdge_;
    LogThrowIf(_mirrorRange_ < 0, GET_VAR_NAME_VALUE(_mirrorHighEdge_) << " < " << GET_VAR_NAME_VALUE(_mirrorLowEdge_))
  }

  // safety: many users ask why do my parameter limits aren't taken into account? -> because the parameters limits should be
  // handled in the parameter definition section
  if( config_.hasField("parameterLimits") ){
    LogError << "\"parameterLimits\" should be set in the parameter definition section. Not the dial definition." << std::endl;
    LogError << "Please move it to the appropriate section in: ";
    if( _parameterSetListPtr_ != nullptr and _supervisedParameterSetIndex_ != -1 ){

      if( _supervisedParameterIndex_ != -1 ){
        LogError << (*_parameterSetListPtr_)[_supervisedParameterSetIndex_].getParameterList()[_supervisedParameterIndex_].getFullTitle();
      }
      else{
        LogError << (*_parameterSetListPtr_)[_supervisedParameterSetIndex_].getName();
      }

    }
    else{
      LogError << "[handled parameter unspecified]";
    }

    LogError << std::endl;
    LogThrow("parameterLimits should be specified in the parameter definition.");
  }
}
/// @par Implementation
/// Reads the common options from the top-level definition, configures the
/// bin set from `parametersBinningPath`, then adds one Norm dial per bin;
/// the loop variable is unused, the loop only counts the bins. The dial
/// type is not forced: dials are Norm even if `dialType` says otherwise.
/// DialInterface::setDial() copies the wrapper it gets, so each Norm dial is
/// cloned once on the way in.
bool DialCollection::initializeNormDialsWithParBinning() {

  if( not _config_.hasField("parametersBinningPath") ){
    return false; // not defined
  }

  // Get global parameters from the main config
  this->readParametersFromConfig(_config_);

  // Read the binning
  LogDebugIf(GundamGlobals::isDebug()) << "Defining binned dials for " << getTitle() << std::endl;
  _dialBinSet_ = BinSet();
  _dialBinSet_.setName("parameterBinning");
  _dialBinSet_.configure( _config_.fetchValue<ConfigReader>("parametersBinningPath") );

  // By default use min dial response for norm dials
  _dialResponseSupervisorList_.resize( 1 );
  _dialResponseSupervisorList_[0].setMinResponse( _minDialResponse_ );
  _dialResponseSupervisorList_[0].setMaxResponse( _config_.fetchValue("maxDialResponse", _maxDialResponse_) );

  _dialInterfaceList_.reserve( _dialBinSet_.getBinList().size() );
  for(const auto & bin : _dialBinSet_.getBinList()) {
    _dialInterfaceList_.emplace_back();
    _dialInterfaceList_.back().setDial( DialBaseObject(this->makeDial()) );
  }

  return true;
}

/// @par Implementation
/// The TabulatedDialFactory constructor does the heavy work: it opens the
/// library, calls its init function and allocates the table. The factory
/// is moved into `_dialCollectionData_`; the update callback captures its
/// index there rather than a pointer, and gets the collection as argument,
/// so it stays valid when the collection is copied or moved. The binning
/// variables become extra leaves so that DataDispenser loads them for
/// TabulatedDialFactory::makeDial(). The callback reads the first input
/// buffer without checking that one exists; initialize() creates it. Since
/// DataDispenser fetches the factory as collection data 0, the factory must
/// be the first collection data, which holds as each collection adds one.
/// @note The configuration comment inside this function is partly
///       outdated. TabulatedDialFactory reads `tableConfig` as one block
///       (not a list), the variables key is `binningVariables` (not
///       `variables`), and every key except `bins` is mandatory, including
///       `initFunction`. The init function's return value is used as the
///       table size, and a value below 1 is treated as a failure (not "0 for
///       success"). The update function's return value is not checked.
bool DialCollection::initializeDialsWithTabulation(const ConfigReader& dialsDefinition_) {
  // Initialize the Tabulated type.  That yaml for this is
  // tableConfig:
  //    - name: <table-name>                A table name passed to the library.
  //      libraryPath: <path-to-library>    Location of the library
  //      initFunction: <init-func-name>    Function called for initialization
  //      updateFunction: <update-func-name>    Function called to update table
  //      binningFunction: <bin-func-name>  Function to find bin index
  //      initArguments: [<arg1>, ...]      List of argument strings (e.g.
  //                                           input file names)
  //      bins:  <number>                   Number of bins in the table
  //      variables: [<var1>, <var2>, ... ] Variables used for binning the
  //                                           table "X" coordinate by the
  //                                           binning function.
  //
  //
  // If initFunction is provided it will be called with the signature:
  //
  //    extern "C"
  //    int initFunc(char* name, int argc, char* argv[], int bins)
  //
  //        name -- The name of the table
  //        argc -- number of arguments
  //        argv -- argument strings (0 is table name)
  //                arguments 1+ are defined by the library
  //        bins -- The size of the table.
  //
  // The function should return 0 for success, and any other value for failure
  //
  // The updateFunction signature is:
  //
  //    extern "C"
  //    int updateFunc(char* name,
  //                 double table[], int bins,
  //                 double par[], int npar)
  //
  //        name  -- table name
  //        table -- address of the table to update
  //        bins  -- The size of the table
  //        par   -- The parameters.  Must match parameters
  //                   define in the dial definition.
  //        npar  -- number of parameters
  //
  // The function should return 0 for success, and any other value for failure
  //
  // The table will be filled with "bins" values calculated with uniform
  // spacing between "low" and "high".  If bins is one, there must be one
  // value calculated for "low", if bins is two or more, then the first point
  // is located at "low", and the last point is located at "high".  The step
  // between the bins is (high-low)/(bins-1).  Examples:
  //
  //    bins = 2, low = 1.0, high = 6.0
  //       values calculated at 1.0 and 6.0
  //
  //    bins = 3, low = 1.0, high = 6.0
  //       values calculated at 1.0, 3.5, and 6.0
  //
  //    extern "C"
  //    double binFunc(char* name, int nvar, double varv[], int bins);
  //        name -- table name
  //        nvar -- number of (truth) variables used to find bin
  //        varv -- array of (truth) variables used to find bin
  //        bins -- The number of bins in the table.
  //
  // The function should return a double giving the fractional bin number
  // greater or equal to zero and LESS THAN the maximum number of bins.  The
  // integer part determines the index of the value below the value to be
  // interpolated, and the fractional part determines the interpolation
  // between the indexed bin, and the next.
  //
  // The code should be compiled with
  // gcc -fPIC -rdynamic --shared -o libLibraryName.so

  // Create a unique copy of this dial data so that it gets deleted if
  // there is a problem during initialization.
  std::unique_ptr<TabulatedDialFactory> tabulated
      = std::make_unique<TabulatedDialFactory>(dialsDefinition_);

  // Save the new object (the move releases the pointer).
  _dialCollectionData_.emplace_back(std::move(tabulated));

  // Get the index of the new dial collection data entry.  This is "back()",
  // but the index will be needed for the update closure, so use that instead.
  int index = _dialCollectionData_.size()-1;

  for (const std::string& var :
       getCollectionData<TabulatedDialFactory>(index)->getBinningVariables()) {
    addExtraLeafName(var);
  }

  addUpdate(
      [index](DialCollection* dc){
        dc->getCollectionData<TabulatedDialFactory>(index)
        ->updateTable(dc->getDialInputBufferList().front());
      });

  return true;
}
/// @par Implementation
/// Same steps as initializeDialsWithTabulation(), with a KrigedDialFactory
/// and its weight variables as extra leaves.
/// @note The configuration comment inside this function was copied from the
///       tabulated case and is partly outdated. KrigedDialFactory reads
///       `tableConfig` as one block with the mandatory keys `name`,
///       `libraryPath`, `initFunction`, `initArguments`, `updateFunction`,
///       `weightFunction` and `weightVariables` (there is no
///       `binningFunction` or `variables` key), plus the optional
///       `maxWeights`, `weightNormalization` and `bins`. The weight function
///       is called through a pointer returning `int`, not `double`. As for
///       tabulated dials, the init function returns the table size (below 1
///       is a failure).
bool DialCollection::initializeDialsWithKriging(const ConfigReader& dialsDefinition_){
  // Initialize the Tabulated type.  That yaml for this is
  // tableConfig:
  //    - name: <table-name>                A table name passed to the library.
  //      libraryPath: <path-to-library>    Location of the library
  //      initFunction: <init-func-name>    Function called for initialization
  //      updateFunction: <update-func-name>    Function called to update table
  //      binningFunction: <bin-func-name>  Function to find bin index
  //      initArguments: [<arg1>, ...]      List of argument strings (e.g.
  //                                           input file names)
  //      bins:  <number>                   Number of bins in the table
  //      variables: [<var1>, <var2>, ... ] Variables used for binning the
  //                                           table "X" coordinate by the
  //                                           binning function.
  //
  //
  // If initFunction is provided it will be called with the signature:
  //
  //    extern "C"
  //    int initFunc(char* name, int argc, char* argv[], int bins)
  //
  //        name -- The name of the table
  //        argc -- number of arguments
  //        argv -- argument strings (0 is table name)
  //                arguments 1+ are defined by the library
  //        bins -- The size of the table.
  //
  // The function should return 0 for success, and any other value for failure
  //
  // The updateFunction signature is:
  //
  //    extern "C"
  //    int updateFunc(char* name,
  //                 double table[], int bins,
  //                 double par[], int npar)
  //
  //        name  -- table name
  //        table -- address of the table to update
  //        bins  -- The size of the table
  //        par   -- The parameters.  Must match parameters
  //                   define in the dial definition.
  //        npar  -- number of parameters
  //
  // The function should return 0 for success, and any other value for failure
  //
  // The table will be filled with "bins" values calculated with uniform
  // spacing between "low" and "high".  If bins is one, there must be one
  // value calculated for "low", if bins is two or more, then the first point
  // is located at "low", and the last point is located at "high".  The step
  // between the bins is (high-low)/(bins-1).  Examples:
  //
  //    bins = 2, low = 1.0, high = 6.0
  //       values calculated at 1.0 and 6.0
  //
  //    bins = 3, low = 1.0, high = 6.0
  //       values calculated at 1.0, 3.5, and 6.0
  //
  //    extern "C"
  //    double weightFunc(char* name, int bins,
  //                      int nvar, double varv[],
  //                      int maxEntries,
  //                      int indices[], double weights[]);
  //        name -- table name
  //        bins -- The number of bins in the table.
  //        nvar -- number of (truth) variables used to find bin
  //        varv -- array of (truth) variables used to find bin
  //        maxEntries -- size of the indices and weights arrays
  //        indices -- array of returned indices
  //        weights -- array of returned weights.
  //
  // The function should fill the indices and weights arrays with the
  // kriging weights.  The indices should always be inside the table since
  // the dial won't do bounds checking (for efficiency).
  //
  // The code should be compiled with
  // gcc -fPIC -rdynamic --shared -o libLibraryName.so

  // Create a unique copy of this dial data so that it gets deleted if
  // there is a problem during initialization.
  std::unique_ptr<KrigedDialFactory> kriged
      = std::make_unique<KrigedDialFactory>(dialsDefinition_);

  // Save the new object (the move releases the pointer).
  _dialCollectionData_.emplace_back(std::move(kriged));

  // Get the index of the new dial collection data entry.  This is "back()",
  // but the index will be needed for the update closure, so use that instead.
  int index = _dialCollectionData_.size()-1;

  for (const std::string& var :
       getCollectionData<KrigedDialFactory>(index)->getWeightVariables()) {
    addExtraLeafName(var);
  }

  addUpdate(
      [index](DialCollection* dc){
        dc->getCollectionData<KrigedDialFactory>(index)
        ->updateTable(dc->getDialInputBufferList().front());
      });

  LogDebug << "Initialize dial with kriging" << std::endl;

  return true;
}

/// @par Implementation
/// Creates the ExternalWeightDialFactory from the `options` block and stores
/// it as collection data, registers its input event variables as extra
/// leaves and, for binned weights, creates one dial per bin of the factory's
/// binning. The update callback passes the first input buffer to
/// ExternalWeightDialFactory::updateWeights() before each reweight.
bool DialCollection::initializeDialsWithExternalWeight(const ConfigReader& dialsDefinition_){
  LogThrowIf(not dialsDefinition_.hasField("options"),
             "ExternalWeight dial requires an options configuration block.");


  auto externalWeightFactory = std::make_unique<ExternalWeightDialFactory>();
  externalWeightFactory->configure(dialsDefinition_.fetchValue<ConfigReader>("options"));
  _dialCollectionData_.emplace_back(std::move(externalWeightFactory));

  int index = int(_dialCollectionData_.size()) - 1;

  for( const auto& var : getCollectionData<ExternalWeightDialFactory>(index)->getInputEventVarNameList() ){
    addExtraLeafName(var);
  }

  auto* externalWeight = getCollectionData<ExternalWeightDialFactory>(index);
  if( externalWeight->useBinnedWeights() ){
    _dialBinSet_ = externalWeight->getBinning();
    _dialInterfaceList_.reserve(_dialBinSet_.getBinList().size());
    for( std::size_t iBin = 0 ; iBin < _dialBinSet_.getBinList().size() ; ++iBin ){
      _dialInterfaceList_.emplace_back();
      _dialInterfaceList_.back().setDial(
          DialBaseObject(externalWeight->makeBinnedDial(iBin))
      );
    }
  }

  addUpdate(
      [index](DialCollection* dc){
        dc->getCollectionData<ExternalWeightDialFactory>(index)
          ->updateWeights(dc->getDialInputBufferList().front());
      });

  return true;
}

bool DialCollection::initializeBinnedDials(const ConfigReader& dialsDefinition) {
  const bool treeBinning = dialsDefinition.hasField("dialsBinningDefinition");
  LogThrowIf(treeBinning and dialsDefinition.hasField("binning"),
             "Specify either binning or dialsBinningDefinition, not both");
  if( not treeBinning and not dialsDefinition.hasField("binning") ){ return false; }

  LogInfo << "Defining binned dials for " << getTitle() << std::endl;
  _dialBinSet_ = BinSet();
  _dialBinSet_.setName("dial binning");
  if( not treeBinning ){
    _dialBinSet_.configure(dialsDefinition.fetchValue<ConfigReader>("binning"));
  }
  // Keep bins and dials in source order.

  // Get the filename for a file with the object array of dials (graphs)
  // that will be applied based on the binning.
  auto filePath = dialsDefinition.fetchValue<std::string>("dialsFilePath", "");
  auto objPath = dialsDefinition.fetchValue<std::string>("dialsTreePath", "");
  auto branchName = dialsDefinition.fetchValue<std::string>("dialsBranch", "");
  LogThrowIf(dialsDefinition.hasField("dialsBranch") and branchName.empty(), "dialsBranch must not be empty");

  // Merge redundant declarations only when they identify the same source.
  auto mergePath = [](std::string& target_, const std::string& value_, const char* field_){
    LogThrowIf(value_.empty(), "Empty " << field_ << " in binned dial path");
    LogThrowIf(not target_.empty() and target_ != value_, "Conflicting " << field_ << " in binned dial definition");
    target_ = value_;
  };
  const auto branchSeparator = objPath.rfind(':');
  if( branchSeparator != std::string::npos ){
    mergePath(branchName, objPath.substr(branchSeparator + 1), "dialsBranch");
    objPath.resize(branchSeparator);
    LogThrowIf(objPath.empty(), "Empty dialsTreePath in binned dial path");
  }
  if( dialsDefinition.hasField("binnedDialsPath") ){
    const auto path = dialsDefinition.fetchValue<std::string>("binnedDialsPath");
    const auto source = GenericToolbox::parseRootObjectPath(path);
    const auto branchSeparator = source.objectPath.rfind(':');
    LogThrowIf(not source.hasObjectPath or branchSeparator == std::string::npos,
               "binnedDialsPath must use file.root:tree:branch syntax: " << path);
    mergePath(filePath, source.filePath, "dialsFilePath");
    mergePath(objPath, source.objectPath.substr(0, branchSeparator), "dialsTreePath");
    mergePath(branchName, source.objectPath.substr(branchSeparator + 1), "dialsBranch");
  }
  LogThrowIf(not branchName.empty() and (objPath.empty() or dialsDefinition.hasField("dialsList")),
             "An explicit binned dial branch requires a tree and cannot be combined with dialsList");
  LogThrowIf(treeBinning and (objPath.empty() or dialsDefinition.hasField("dialsList")),
             "dialsBinningDefinition requires a TTree source and cannot be combined with dialsList");
  LogThrowIf(filePath.empty(), "Missing dialsFilePath or binnedDialsPath for binned dials");
  filePath = GenericToolbox::expandEnvironmentVariables(filePath);

  LogThrowIf(not GenericToolbox::doesTFileIsValid(filePath), "Could not open: " << filePath);
  std::unique_ptr<TFile> dialsTFile{TFile::Open(filePath.c_str())};
  LogThrowIf(dialsTFile==nullptr, "Could not open: " << filePath);

  if      ( dialsDefinition.hasField("dialsList") ) {
    auto* dialsList = dialsTFile->Get<TObjArray>(dialsDefinition.fetchValue<std::string>("dialsList").c_str());

    LogThrowIf(
        dialsList==nullptr,
        "Could not find dialsList: " << dialsDefinition.fetchValue<std::string>("dialsList")
    );
    LogThrowIf(
        dialsList->GetEntries() != _dialBinSet_.getBinList().size(),
        this->getTitle() << ": Number of dials (" << dialsList->GetEntries()
                         << ") don't match the number of bins " << _dialBinSet_.getBinList().size()
    );

    std::vector<int> excludedBins{}; // bins without a dial; erased below
    int nBins(static_cast<int>(_dialBinSet_.getBinList().size()));
    _dialInterfaceList_.reserve( nBins ); // at most
    for( int iBin = 0 ; iBin < nBins ; iBin++ ){
      TObject* binnedInitializer = dialsList->At(iBin); // unused (makeDial() refetches it)

      _verboseShortCircuit_ = true;
      auto dial = DialBaseObject( this->makeDial( dialsList->At(iBin) ) );
      _verboseShortCircuit_ = false;

      if( dial.get() == nullptr ) {
        LogDebugIf(GundamGlobals::isDebug()) << getTitle() << " -> "
                 << _dialBinSet_.getBinList()[iBin].getSummary()
                 << " -> " << _verboseShortCircuitStr_ << std::endl;
        excludedBins.emplace_back(iBin);
        continue;
      }

      _dialInterfaceList_.emplace_back();
      _dialInterfaceList_.back().setDial( dial );
    }

      if( not excludedBins.empty() ){
      LogWarning << "Removing " << excludedBins.size() << " null dials out of " << nBins << " / " << 100*double(excludedBins.size())/double(nBins) << "%... (--debug for more info)" << std::endl;
      for( int iBin = nBins ; iBin >= 0 ; iBin-- ){
        if( GenericToolbox::doesElementIsInVector(iBin, excludedBins) ){
          _dialBinSet_.getBinList().erase(_dialBinSet_.getBinList().begin() + iBin);
        }
      }
    }

  }

    ///////////////////////////////////////////////////////////////////////
  else if ( not objPath.empty() ) {
    auto* dialsTTree = dialsTFile->Get<TTree>(objPath.c_str());
    LogThrowIf(dialsTTree== nullptr, objPath << " within " << filePath << " could not be opened.")

    const bool explicitBranch = not branchName.empty();
    if( not explicitBranch ){
      // Preserve the historical choice based on dialType.
      if( _dialType_ == DialType::Spline ){ branchName = "Spline"; }
      if( _dialType_ == DialType::Graph ){ branchName = "Graph"; }
      LogWarning << "Implicit binned dial branch selection is deprecated for " << getTitle()
                 << ". Set dialsBranch explicitly (legacy branch: " << branchName
                 << "), or use file:tree:branch in binnedDialsPath." << std::endl;
    }
    auto* branch = dialsTTree->GetBranch(branchName.c_str());
    LogThrowIf(branch == nullptr, "Could not find binned dial branch " << branchName << " in " << objPath);
    auto* branchClass = TClass::GetClass(branch->GetClassName());
    const bool graphBranch = explicitBranch
                             ? branchClass != nullptr and branchClass->InheritsFrom(TGraph::Class())
                             : _dialType_ == DialType::Graph;
    const bool splineBranch = explicitBranch
                              ? branchClass != nullptr and branchClass->InheritsFrom(TSpline3::Class())
                              : _dialType_ == DialType::Spline;
    LogThrowIf(not graphBranch and not splineBranch,
               "Binned dial branch " << branchName << " must contain TGraph or TSpline3 objects");

    TSpline3* splinePtr = nullptr; // branch buffer, used for TSpline3 branches
    TGraph* graphPtr = nullptr; // branch buffer, used for TGraph branches
    // Read only the curve and the explicitly declared binning branches.
    dialsTTree->SetBranchStatus("*", false);
    dialsTTree->SetBranchStatus(branchName.c_str(), true);
    if( branch->GetListOfBranches()->GetEntries() != 0 ){
      dialsTTree->SetBranchStatus((branchName + ".*").c_str(), true);
    }
    const int branchStatus = graphBranch ? dialsTTree->SetBranchAddress(branchName.c_str(), &graphPtr)
                                         : dialsTTree->SetBranchAddress(branchName.c_str(), &splinePtr);
    LogThrowIf(branchStatus < 0, "Could not bind binned dial branch " << branchName << " in " << objPath);

    struct BinningDimension {
      std::string name;
      TLeaf* minLeaf{nullptr};
      TLeaf* maxLeaf{nullptr};
      bool isValue{false};
    };
    std::vector<BinningDimension> dimensions;
    if( treeBinning ){
      auto definition = dialsDefinition.fetchValue<ConfigReader>("dialsBinningDefinition");
      LogThrowIf(not definition.getConfig().is_array() or definition.getConfig().empty(),
                 "dialsBinningDefinition must be a non-empty list");
      auto getScalarLeaf = [&](const std::string& name_) -> TLeaf* {
        auto* scalarBranch = dialsTTree->GetBranch(name_.c_str());
        LogThrowIf(name_.empty() or scalarBranch == nullptr,
                   "Could not find binning branch " << name_ << " in " << objPath);
        LogThrowIf(scalarBranch->GetListOfLeaves()->GetEntries() != 1,
                   "Binning branch " << name_ << " must contain one numeric scalar");
        auto* leaf = static_cast<TLeaf*>(scalarBranch->GetListOfLeaves()->At(0));
        const std::vector<std::string> numericTypes{
            "Bool_t", "Char_t", "UChar_t", "Short_t", "UShort_t", "Int_t", "UInt_t",
            "Long_t", "ULong_t", "Long64_t", "ULong64_t", "Float_t", "Double_t", "Float16_t", "Double32_t"};
        LogThrowIf(leaf->GetLeafCount() != nullptr or leaf->GetLenStatic() != 1
                   or leaf->InheritsFrom("TLeafC")
                   or std::find(numericTypes.begin(), numericTypes.end(), leaf->GetTypeName()) == numericTypes.end(),
                   "Binning branch " << name_ << " must contain one numeric scalar");
        dialsTTree->SetBranchStatus(name_.c_str(), true);
        return leaf;
      };
      for( auto entry : definition.loop() ){
        entry.clearFields();
        entry.defineFields({{"name"}, {"valueBranch"}, {"minValueBranch"}, {"maxBranchValue"}});
        entry.checkConfiguration();
        BinningDimension dimension;
        entry.fillValue(dimension.name, "name");
        LogThrowIf(dimension.name.empty(), "dialsBinningDefinition requires a non-empty name");
        LogThrowIf(std::any_of(dimensions.begin(), dimensions.end(), [&](const BinningDimension& other_){
          return other_.name == dimension.name;
        }), "Duplicate binning variable " << dimension.name);
        dimension.isValue = entry.hasField("valueBranch");
        LogThrowIf(dimension.isValue ? (entry.hasField("minValueBranch") or entry.hasField("maxBranchValue"))
                                     : (not entry.hasField("minValueBranch") or not entry.hasField("maxBranchValue")),
                   "Binning variable " << dimension.name
                   << " requires either valueBranch or both minValueBranch and maxBranchValue");
        dimension.minLeaf = getScalarLeaf(entry.fetchValue<std::string>(dimension.isValue ? "valueBranch" : "minValueBranch"));
        dimension.maxLeaf = dimension.isValue ? dimension.minLeaf
                                             : getScalarLeaf(entry.fetchValue<std::string>("maxBranchValue"));
        dimensions.emplace_back(dimension);
      }
    }

    const Long64_t nEntries = dialsTTree->GetEntries();
    LogThrowIf(not treeBinning and nEntries != _dialBinSet_.getBinList().size(),
               getTitle() << ": Number of tree entries (" << nEntries
               << ") does not match the number of bins " << _dialBinSet_.getBinList().size());
    std::vector<Bin> activeBins; // bins of the entries that gave a dial, in entry order
    activeBins.reserve(nEntries);
    _dialInterfaceList_.reserve(nEntries);
    LogInfo << "Reading binned dials from " << filePath << ":" << objPath << ":" << branchName << std::endl;
    for( Long64_t iEntry = 0 ; iEntry < nEntries ; ++iEntry ){
      LogThrowIf(dialsTTree->GetEntry(iEntry) <= 0, "Could not read binned dial entry " << iEntry);
      Bin bin;
      if( treeBinning ){
        bin.setIndex(int(iEntry));
        bin.setIsZeroWideRangesTolerated(true);
        for( const auto& dimension : dimensions ){
          const double min = dimension.minLeaf->GetValue();
          const double max = dimension.maxLeaf->GetValue();
          LogThrowIf(dimension.isValue ? not std::isfinite(min)
                                      : (std::isnan(min) or std::isnan(max) or min >= max),
                     "Invalid bin bounds for " << dimension.name << " at tree entry " << iEntry
                     << ": [" << min << ", " << max << "]");
          bin.addBinEdge(dimension.name, min, max);
        }
      }
      else{ bin = _dialBinSet_.getBinList().at(iEntry); }

      const TObject* initializer = graphBranch ? static_cast<TObject*>(graphPtr) : static_cast<TObject*>(splinePtr);
      auto dial = makeDial(initializer);
      if( dial == nullptr ){ continue; }
      activeBins.emplace_back(std::move(bin));
      _dialInterfaceList_.emplace_back();
      _dialInterfaceList_.back().setDial(DialBaseObject(std::move(dial)));
    }
    _dialBinSet_.getBinList() = std::move(activeBins);
    if( treeBinning ){ _dialBinSet_.checkBinning(); }
  } // Splines in TTree
  else{
    LogError << "Neither binnedDialsPath, dialsTreePath nor dialsList are provided..." << std::endl;
    return false;
  }
  return true;
}

/// @par Implementation
/// The cases are tested in a fixed order (Composite, Norm, Formula,
/// CompiledLibDial, Tabulated, Kriged, ExternalWeight, then `binning` or
/// `dialsBinningDefinition`, then branch data), so the dial type wins over
/// the other keys: a Norm definition with a `binning` key still gets one
/// unbinned Norm dial. A composite gets a single CompositeDial, built from
/// its `options` block and the parameter names of its `dialInputList`.
/// Formula and CompiledLibDial dials are built by
/// makeDial(const ConfigReader&, const Bin*); for a binned formula, each copy
/// gets its named TFormula parameters set to the centre of the bin edge
/// with the same variable name.
/// @note The response supervisor takes its minimum from the definition that
///       was read (the selected `dialsDefinitions` entry, if any), but reads
///       `maxDialResponse` from the top-level `_config_` first, using the
///       entry's value only when the top level has none.
/// @note The existing Formula comment ("a single dial for all events") does
///       not cover the binned case, which builds one dial per bin.
bool DialCollection::initializeDialsWithDefinition() {

  auto dialsDefinition = _config_; // _config_ or its dialsDefinitions entry
  if( _config_.hasField("dialsDefinitions") ) {
    // Fetch the dialSet corresponding to the selected parameter
    dialsDefinition = this->fetchDialsDefinition(_config_.fetchValue<ConfigReader>("dialsDefinitions"));
    prepareConfig(dialsDefinition);
  }

  if( dialsDefinition.getConfig().empty() ) {return false;}

  if( not dialsDefinition.fetchValue("isEnabled", true) ){
    LogDebug << "DialSet is disabled." << std::endl;
    return true;
  }

  this->readParametersFromConfig( ConfigReader(dialsDefinition) );

  LogThrowIf(dialsDefinition.hasField("binning") and dialsDefinition.hasField("dialsBinningDefinition"),
             "Specify either binning or dialsBinningDefinition, not both");
  LogThrowIf(dialsDefinition.hasField("dialsBinningDefinition")
             and _dialType_ != DialType::Graph and _dialType_ != DialType::Spline,
             "dialsBinningDefinition requires Graph or Spline dials");

  if( _dialType_ == DialType::Composite ){
    LogThrowIf(dialsDefinition.hasField("binning") or hasDialBranchData(),
               "Composite: put binning and dialBranchData on individual nodes");
    std::vector<std::string> parameterNames;
    for( const auto& input : _config_.fetchValue<JsonType>("dialInputList") ){
      parameterNames.emplace_back(input.at("parameterName").get<std::string>());
    }
    _isEventByEvent_ = false;
    _dialInterfaceList_.emplace_back();
    _dialInterfaceList_.back().setDial(DialBaseObject(std::make_unique<CompositeDial>(
        dialsDefinition.fetchValue<ConfigReader>("options"), parameterNames)));
  }
  else if( _dialType_ == DialType::Norm ) {
    // This dial collection is a normalization, so there is a single dial.
    // Create it here.
    _isEventByEvent_ = false;
    _dialInterfaceList_.emplace_back();
    _dialInterfaceList_.back().setDial( DialBaseObject(makeDial()) );
  }
  else if( _dialType_ == DialType::Formula ){
    // This dial collection calculates a function of the parameter values, so it
    // is a single dial for all events.  Create the dial here.
    _isEventByEvent_ = false;

    if( dialsDefinition.hasField("binning") ){
      auto binning = dialsDefinition.fetchValue("binning", ConfigReader());

      LogInfo << "Defining binned dials for " << getTitle() << std::endl;
      _dialBinSet_ = BinSet();
      _dialBinSet_.setName( "formula binning" );
      _dialBinSet_.configure( ConfigReader(binning) );

      _dialInterfaceList_.reserve( _dialBinSet_.getBinList().size() );
      for( auto& bin : _dialBinSet_.getBinList() ){

        _dialInterfaceList_.emplace_back();
        _dialInterfaceList_.back().setDial( DialBaseObject(makeDial( dialsDefinition, &bin )) );

        for( auto& var : bin.getEdgesList() ){
          auto& formula = ((RootFormula*) _dialInterfaceList_.back().getDialBaseRef())->getFormula();
          if( formula.GetParNumber(var.varName.c_str()) >= 0 ){
            formula.SetParameter(var.varName.c_str(), var.getCenterValue());
          }
        }
      }

    }
    else{
      _dialInterfaceList_.emplace_back();
      _dialInterfaceList_.back().setDial( DialBaseObject(makeDial( dialsDefinition )) );
    }

  }
  else if( _dialType_ == DialType::CompiledLibDial ){
    // This dial collection calculates a function of the parameter values so it
    // is a single dial for all events.  Create the dial here.
    _isEventByEvent_ = false;

    _dialInterfaceList_.emplace_back();
    _dialInterfaceList_.back().setDial( DialBaseObject(makeDial( dialsDefinition )) );
  }
  else if( _dialType_ == DialType::Tabulated ) {
    // This dial uses a precalculated table to apply weight to each event
    // (e.g. it might be used to implement neutrino osillations).  It has a
    // different weight for each event.
    _isEventByEvent_ = true;
    LogThrowIf(not initializeDialsWithTabulation(dialsDefinition),
               "Error initializing dials with tabulation");
  }
  else if( _dialType_ == DialType::Kriged ) {
    // This dial uses a precalculated table to apply weight to each event
    // (e.g. it might be used to implement neutrino osillations).  It has a
    // different weight for each event.
    _isEventByEvent_ = true;
    LogThrowIf(not initializeDialsWithKriging(dialsDefinition),
               "Error initializing dials with kriging");
  }
  else if( _dialType_ == DialType::ExternalWeight ) {
    LogThrowIf(not initializeDialsWithExternalWeight(dialsDefinition),
               "Error initializing dials with external weights");
    auto* externalWeight = getCollectionData<ExternalWeightDialFactory>(_dialCollectionData_.size() - 1);
    _isEventByEvent_ = not externalWeight->useBinnedWeights();
  }
  else if( dialsDefinition.hasField("binning") or dialsDefinition.hasField("dialsBinningDefinition") ) {
    // This dial collection is binned with different weights for each bin.
    // Create the dials here.
    _isEventByEvent_ = false;

    LogThrowIf(not initializeBinnedDials(dialsDefinition), "Error initializing binned dials");
  }
  else if (hasDialBranchData()) {
    // Branch data defines an event-by-event dial from a ROOT object
    // or a pair of TArray branches.  The generation of the dials will be handled in
    // DataDispenser.
    _isEventByEvent_ = true;
  }
  else{
    LogError << "The dial does not match a known dial type." << std::endl;
    LogError << "  dialType:     " << _dialType_ << std::endl;
    LogError << "  dialOptions:  " << _dialOptions_ << std::endl;
    LogError << "  dialLeafName: " << _dialLeafName_ << std::endl;
    LogThrow("Invalid dial type");
  }

  _dialResponseSupervisorList_.emplace_back();
  _dialResponseSupervisorList_.back().setMinResponse(
      _minDialResponse_
  );
  _dialResponseSupervisorList_.back().setMaxResponse(
      _config_.fetchValue("maxDialResponse", _maxDialResponse_)
  );

  return true;
}

/// @par Implementation
/// Walks the list in order and returns the first match. A named parameter
/// is matched by name only; an unnamed one only by position. The entry is
/// returned as a new ConfigReader with no key declared, which is why
/// initializeDialsWithDefinition() calls prepareConfig() on it.
ConfigReader DialCollection::fetchDialsDefinition(const ConfigReader& definitionsList_) const {
  auto* parSetPtr = this->getSupervisedParameterSet();
  LogThrowIf(parSetPtr == nullptr, "Can't fetch dial definition of parameter: par ref not set.");
  auto* par = &parSetPtr->getParameterList()[_supervisedParameterIndex_];
  for(size_t iDial = 0 ; iDial < definitionsList_.getConfig().size() ; iDial++ ){
    if( par->getName().empty() ){
      if( par->getParameterIndex() == iDial ){
        return ConfigReader(definitionsList_.getConfig().at(iDial));
      }
    }
    else if( par->getName() == GenericToolbox::Json::fetchValue<std::string>(definitionsList_.getConfig().at(iDial), {{"name"}, {"parameterName"}}, "") ){
      return ConfigReader(definitionsList_.getConfig().at(iDial));
    }
  }
  return {};
}

/// @par Implementation
/// Passes `this` to each callback, so callbacks need no captured pointer to
/// the collection. Runs on the calling thread.
void DialCollection::update() {
  for( auto& func : _dialCollectionCallbacks_ ){
    func(this);
  }
}

/// @par Implementation
/// Appends a copy of @p callback to `_dialCollectionCallbacks_`.
void DialCollection::addUpdate(const std::function<void(DialCollection* dc)>& callback) {
  _dialCollectionCallbacks_.emplace_back(callback);
}

/// @par Implementation
/// One LogInfo line with getTitle().
void DialCollection::printConfiguration() const {

  LogInfo << "DialCollection: " << this->getTitle() << std::endl;

}



/// @par Implementation
/// Returns a default-constructed Norm; the collection's options are not
/// applied to it.
std::unique_ptr<DialBase> DialCollection::makeDial() const{ return std::make_unique<Norm>(); }
/// @par Implementation
/// Dispatches on `_dialType_` to makeGraphDial(), makeSplineDial() or
/// makeSurfaceDial(), then sets the extrapolation flag on the result. These
/// helpers only read the collection's members (the verbose short-circuit
/// string is written only while `_verboseShortCircuit_` is on, during
/// configuration), which is what lets DataDispenser call this from several
/// threads. The one exception is the unsynchronised static flag behind the
/// one-time `pixar` message in makeSplineDial().
std::unique_ptr<DialBase> DialCollection::makeDial(const TObject* src_) const {
  std::unique_ptr<DialBase> out{nullptr};

  // always returns an invalid ptr if
  if( src_ == nullptr ){ return out; }

  if     ( _dialType_ == DialType::Graph   ){ out = makeGraphDial(src_); }
  else if( _dialType_ == DialType::Spline  ){ out = makeSplineDial(src_); }
  else if( _dialType_ == DialType::Surface ){ out = makeSurfaceDial(src_); }
  else{
    LogError << this->getSummary() << std::endl;
    LogThrow(this->getTitle() << ": invalid dial type to init with TObject: " << _dialType_);
  }

  if( out != nullptr ){
    out->setAllowExtrapolation( _allowDialExtrapolation_ );
  }

  return out;
}
/// @par Implementation
/// Reads `dialType` as a plain string (not as a DialType), so only the
/// three spellings tested below are accepted. A formula takes `formulaStr`
/// from the `options` block (an empty block if `options` is absent, so the
/// missing key then makes GenericToolbox::Json::fetchValue() throw), and
/// binds its inputs to the `name` of each `dialInputList` entry, taken from
/// the definition or else from this collection's configuration. For a
/// compiled library, `options` is required.
std::unique_ptr<DialBase> DialCollection::makeDial(const ConfigReader& config_, const Bin* bin_) const{

  std::unique_ptr<DialBase> dialBase{nullptr};
  std::string dialType{}; // the dialType key as written (string, not DialType)

  config_.fillValue(dialType, "dialType");

  if( dialType == "Formula" or dialType == "RootFormula" ){
    dialBase = std::make_unique<RootFormula>();
    auto* rootFormulaPtr{(RootFormula*) dialBase.get()};
    auto formulaConfig = config_.fetchValue("options", JsonType());
    rootFormulaPtr->setFormulaStr( GenericToolbox::Json::fetchValue<std::string>(formulaConfig, "formulaStr") );
    LogThrowIf(not config_.hasField("dialInputList") and not _config_.hasField("dialInputList"),
               "Formula requires an explicit named dialInputList");
    const auto inputs = config_.hasField("dialInputList") ? config_.fetchValue<JsonType>("dialInputList")
                                                        : _config_.fetchValue<JsonType>("dialInputList");
    LogThrowIf(not inputs.is_array(), "Formula dialInputList must be a list");
    std::vector<std::string> inputNames;
    for( const auto& input : inputs ){
      LogThrowIf(not input.is_object() or not input.contains("name"), "Formula input requires a name");
      inputNames.emplace_back(GenericToolbox::Json::fetchValue<std::string>(input, "name"));
    }
    std::vector<std::string> constantNames;
    if( bin_ != nullptr ){
      for( const auto& edge : bin_->getEdgesList() ){ constantNames.emplace_back(edge.varName); }
    }
    rootFormulaPtr->setInputNames(inputNames, constantNames);
  }
  else if( dialType == "CompiledLibDial" ){
    dialBase = std::make_unique<CompiledLibDial>();
    auto* compiledLibDialPtr{(CompiledLibDial*) dialBase.get()};

    auto formulaConfig(config_.fetchValue<JsonType>("options"));

    bool success = compiledLibDialPtr->loadLibrary( GenericToolbox::Json::fetchValue<std::string>(formulaConfig, "libraryFile") );
    if( not success ){
      LogThrow("Could not load CompiledLibDial. " << GenericToolbox::Json::fetchValue(formulaConfig, "messageOnError", std::string("")) );
    }
  }
  else{ LogThrow("Unknown dial type: " << dialType); }

  return dialBase;
}

/// @par Implementation
/// A TGraph is read without slopes (DialUtils::getPointListNoSlope()), since
/// a linear graph does not use them; any other object goes through
/// DialUtils::getPointList(), which handles TSpline3 and returns an empty
/// list for other classes. The points are then cleaned and sorted
/// (checkDialPointList()) before the short-circuit test.
std::unique_ptr<DialBase> DialCollection::makeGraphDial(const TObject* src_) const {
  // always returns an invalid ptr if
  if( src_ == nullptr ){ return nullptr; }

  std::vector<DialUtils::DialPoint> splinePointList{};
  if( src_->InheritsFrom(TGraph::Class()) ){
    splinePointList = DialUtils::getPointListNoSlope((TGraph *)src_);
  }
  else{
    // try something else (from TSpline3 for instance):
    splinePointList = DialUtils::getPointList(src_);
  }

  checkDialPointList(splinePointList);

  std::unique_ptr<DialBase> out{nullptr};
  if( makeDialShortCircuit(splinePointList, out) ){ return out; }

  return makeGraphDial(splinePointList);
}
/// @par Implementation
/// The knot slopes come from DialUtils::getPointList(), which takes them
/// from TSpline3::Derivative() (building a TSpline3 from a TGraph first).
/// Catmull-Rom and Akima overwrite them with their own slopes; the other
/// kinds keep them. checkDialPointList() runs after the slopes are taken,
/// so the knots kept by `definitionRange` keep the slopes of the full
/// curve. The uniformity tolerance is relative to the mean knot spacing
/// (DialUtils::isUniform()). With the default tolerance, non-uniform
/// Catmull-Rom knots only produce an error log of each interval, and a
/// Catmull-Rom spline is built anyway.
/// @note `natural` is selected but never tested afterwards, so it builds the
///       same dial as `not-a-knot`. The existing comment saying that the
///       not-a-knot and natural slopes are calculated by "FillFromGraph and
///       FillFromSpoline" is outdated: both come from
///       DialUtils::getPointList().
/// @note The branch for two points or fewer cannot run:
///       makeDialShortCircuit() already handles those lists, because
///       SHORT_CIRCUIT_SMALL_SPLINES is defined.
///
/// Once the dial is built, a non-ROOT spline whose knots and slopes all fit
/// one straight line is handed to
/// makeGraphDial(const std::vector<DialUtils::DialPoint>&) instead, which
/// gives a two-point Graph or a Norm; this requires a GeneralSpline or knots
/// uniform to within 64 double epsilons.
/// @note The `pixar` option prints a message to std::cout once per process,
///       outside the logger, through an unsynchronised static flag.
std::unique_ptr<DialBase> DialCollection::makeSplineDial(const TObject* src_) const{
  // always returns an invalid ptr if
  if( src_ == nullptr ){ return nullptr; }

  // The types of cubic splines are "not-a-knot", "natural", "catmull-rom",
  // and "ROOT".  The "not-a-knot" spline will give the same curve as ROOT
  // (and might be implemented with at TSpline3).  The "ROOT" spline will use
  // an actual TSpline3 (and is slow).  The "natural" and "catmull-rom"
  // splines are just as expected (you can see the underlying math on
  // Wikipedia or another source).  Be careful about the order since later
  // conditionals can override earlier ones.
  std::string splType = "not-a-knot";  // The default.
  if( _dialOptions_.find("akima") != std::string::npos ) splType = "akima";
  if( _dialOptions_.find("catmull") != std::string::npos) splType = "catmull-rom";
  if( _dialOptions_.find("natural") != std::string::npos) splType = "natural";
  if( _dialOptions_.find("not-a-knot") != std::string::npos) splType = "not-a-knot";
  if( _dialOptions_.find("pixar") != std::string::npos) {
    splType = "catmull-rom";
    // sneaky output... logger would tattle on me.
    static bool woody=true; // true until the message has been printed once
    if (woody) std::cout << std::endl << std::endl << "You got a friend in me!" << std::endl;
    woody=false;
  }
  if( _dialOptions_.find("ROOT") != std::string::npos) splType = "ROOT";


  // Get the numeric tolerance for when a uniform spline can be used.  We
  // should be able to set this in the DialSubType.
  // (It can now be set: the "uniformity(TOL)" option is parsed below.)
  const double defUniformityTolerance{16*std::numeric_limits<float>::epsilon()};
  double uniformityTolerance{defUniformityTolerance};
  if (_dialOptions_.find("uniformity(") != std::string::npos) {
    std::size_t bg = _dialOptions_.find("uniformity("); // index of '('
    bg = _dialOptions_.find('(',bg);
    std::size_t en = _dialOptions_.find(')',bg); // index of ')', then length
    LogThrowIf(en == std::string::npos,
               "Invalid spline uniformity with dialSubType: " << _dialOptions_
               << " dial: " << getTitle());
    en = en - bg;
    std::string uniformityString = _dialOptions_.substr(bg+1,en-1);
    std::istringstream unif(uniformityString); // parses the number between the parentheses
    unif >> uniformityTolerance;
  }

  auto splinePointList = DialUtils::getPointList(src_);
  checkDialPointList(splinePointList);

  std::unique_ptr<DialBase> out;

  if( makeDialShortCircuit(splinePointList, out) ) {
    // dial short circuit has been processed
    return out;
  }

  ////////////////////////////////////////////////////////////////
  // Check if the spline slope calculation should be updated.  The slopes for
  // not-a-knot and natural splines are calculated by FillFromGraph and
  // FillFromSpoline using ROOT code.  That means we need to fill in the
  // slopes for the other types ("catmull-rom", "akima")
  if     ( splType == "catmull-rom" ){ DialUtils::fillCatmullRomSlopes(splinePointList); }
  else if( splType == "akima" ){ DialUtils::fillAkimaSlopes(splinePointList); }

  ///////////////////////////////////////////////////////////
  // Create the right kind low level spline class base on all of the previous
  // queries.  This is pushing the if-elseif-elseif limit so watch for when it
  // should change to the do-while-false idiom.  Make sure the individual
  // conditionals are less than 10 lines.
  ///////////////////////////////////////////////////////////
  if      ( splType == "ROOT" ){
    // The ROOT implementation of the spline has been explicitly requested, so
    // use it.
    auto rootSpline = std::make_unique<RootSpline>();
    rootSpline->buildDial(splinePointList);
    out = std::move(rootSpline);
  }
  else {

    bool isMonotonic = ( _dialOptions_.find("monotonic") != std::string::npos );

    // If there are only two points, then force a catmull-rom.  This could be
    // handled using a graph, but Catmull-Rom is fast, and works better with the
    // GPU.  The isMonotonic is forced to false so that this uses CompactSpline
    // instead of MonotonicSpline.
    // NOTE: THIS SHOULD NOT HAPPEN -> <= 2 points graphs should be intercepted before
    if( splinePointList.size() <= 2 ){ splType = "catmull-rom"; isMonotonic = false; }

    if( splType == "catmull-rom" ){
      // Catmull-Rom is handled as a special case because it ignores the slopes,
      // and has an explicit monotonic implementatino.  It also must have
      // uniformly spaced knots.
      if( not DialUtils::isUniform(splinePointList, uniformityTolerance) ){
        LogError << "Catmull-rom splines need a uniformly spaced points"
                 << " Dial: " << getTitle()
                 << std::endl;
        double step = (splinePointList.back().x-splinePointList.front().x)/(static_cast<double>(splinePointList.size())-1.); // mean knot spacing
        for (int i = 0; i<splinePointList.size()-1; ++i) {
          LogError << i << " --  X: " << splinePointList[i].x
                   << " X+1: " << splinePointList[i+1].x
                   << " step: " << step
                   << " error: " << splinePointList[i+1].x - splinePointList[i].x - step
                   << std::endl;
        }
        // If the user specified a tolerance then crash, otherwise trust the
        // user knows that it's not uniform and continue.
        LogThrowIf(uniformityTolerance != defUniformityTolerance, "Invalid catmull-rom inputs -- Nonuniform spacing");
      }

      if( isMonotonic ) {
        // MonotonicSpline class is CATMULL-ROM with MONOTONIC option -> TODO: make that explicit in the class name
        auto monotonicSpline = std::make_unique<MonotonicSpline>();
        monotonicSpline->buildDial(splinePointList);
        out = std::move(monotonicSpline);
      }
      else {
        auto compactSpline = std::make_unique<CompactSpline>();
        compactSpline->buildDial(splinePointList);
        out = std::move(compactSpline);
      }
    }
    else{
      ////////////////////////////////////////////////////////////////
      // Check if the spline is supposed to be monotonic and condition the slopes
      // if necessary.  The Catmull-Rom
      // splines have a special implementation for monotonic splines, so save a
      // flag that can be checked later.
      ////////////////////////////////////////////////////////////////
      if( isMonotonic ) {
        DialUtils::applyMonotonicCondition(splinePointList);
      }

      if( DialUtils::isUniform(splinePointList, uniformityTolerance) ){
        // Haven't matched a specific special case, but we have uniformly spaced
        // knots so we can use the faster UniformSpline implementation.
        auto uniformSpline = std::make_unique<UniformSpline>();
        uniformSpline->buildDial(splinePointList);
        out = std::move(uniformSpline);
      }
      else {
        // Haven't matched a specific special case, and the knots are not
        // uniformly spaced, so we have to use the GeneralSpline implemenatation
        // which can handle any kind of cubic spline.
        auto generalSpline = std::make_unique<GeneralSpline>();
        generalSpline->buildDial(splinePointList);
        out = std::move(generalSpline);
      }
    }

  }

  // A two-point Graph is an affine response and retains the original domain.
  // Check the effective spline slopes as well as the knot coordinates.
  if( splType != "ROOT" and isAffine(splinePointList, true)
      and (dynamic_cast<const GeneralSpline*>(out.get()) != nullptr
           or DialUtils::isUniform(splinePointList, 64 * std::numeric_limits<double>::epsilon())) ){
    return makeGraphDial(splinePointList);
  }

  // Pass the ownership without any constraints!
  return out;

}
/// @par Implementation
/// The options are compared as whole strings, case included, unlike the
/// substring search of makeSplineDial(). The histogram is used as it is:
/// no definition range, sorting or short circuit applies to surfaces.
/// @note The existing comment in the Bilinear branch about a "cached
///       version" does not match the code: there is no cached variant
///       here, only Bilinear and Bicubic.
std::unique_ptr<DialBase> DialCollection::makeSurfaceDial(const TObject* src_) const{
  // always returns an invalid ptr if
  if( src_ == nullptr ){ return nullptr; }

  auto* srcObject = dynamic_cast<const TH2*>(src_);

  LogThrowIf(srcObject == nullptr, "Surface dial initializers must be a TH2");

  // Stuff the created dial into a unique_ptr, so it will be properly deleted
  // in the event of an exception.
  std::unique_ptr<DialBase> dialBase;

  if (_dialOptions_ == "Bilinear") {
    // Basic coding: Give a hint to the reader and put likely branch "first".
    // Do we really expect the cached version more than the uncached?
    auto bilinear = std::make_unique<Bilinear>();
    bilinear->buildDial(*srcObject);
    dialBase = std::move(bilinear);
  }
  else if (_dialOptions_ == "Bicubic") {
    auto bicubic = std::make_unique<Bicubic>();
    bicubic->buildDial(*srcObject);
    dialBase = std::move(bicubic);
  }

  if (not dialBase) {
    LogError << "Invalid dialSubType value: " << _dialOptions_ << std::endl;
    LogError << "Valid dialSubType values are: Bilinear, Bicubic" << std::endl;
    LogThrow("Invalid Surface dialSubType");
  }

  // Pass the ownership without any constraints!
  return dialBase;
}

/// @par Implementation
/// The list is only copied when at least one point is out of range. Both
/// range bounds are inclusive (GenericToolbox::Range::isInBounds()). After
/// sorting, knot values outside `knotResponseLimits` are clamped to it; for
/// a Spline collection with more than one point, the slopes are then
/// recomputed from a TSpline3 through the clamped knots. The
/// `mirrorDefinitionRange` test comes last, so it stops the job even after
/// the points were filtered and sorted.
void DialCollection::checkDialPointList(std::vector<DialUtils::DialPoint>& pointList_) const{

  if( _definitionRange_.hasBound() and not std::all_of(pointList_.begin(), pointList_.end(), [&](DialUtils::DialPoint dialPoint){ return _definitionRange_.isInBounds(dialPoint.x); }) ){
    auto temp{pointList_}; temp.clear(); // kept points
    for( auto& point : pointList_ ) {
      if( not _definitionRange_.isInBounds(point.x) ){ continue; }
      temp.emplace_back(point);
    }
    pointList_ = std::move(temp);
  }

  std::sort(pointList_.begin(), pointList_.end(),
    [](const DialUtils::DialPoint& a_, const DialUtils::DialPoint& b_){
      return a_.x < b_.x; // a goes first?
    });

  if( _knotResponseLimits_.hasBound() ){
    bool knotsChanged{false};
    for( auto& point : pointList_ ){
      if( _knotResponseLimits_.isBelowMin(point.y) ){
        point.y = _knotResponseLimits_.min;
        knotsChanged = true;
      }
      if( _knotResponseLimits_.isAboveMax(point.y) ){
        point.y = _knotResponseLimits_.max;
        knotsChanged = true;
      }
    }
    if( knotsChanged and _dialType_ == DialType::Spline and pointList_.size() > 1 ){
      // Rebuild the slopes from the bounded knots before choosing a spline type.
      auto spline = DialUtils::buildTSpline3(pointList_);
      for( auto& point : pointList_ ){ point.slope = spline.Derivative(point.x); }
    }
  }

  if( _mirrorDefinitionRange_.hasBound() ){
    // TODO: process mirroring options
    LogExit("_mirrorDefinitionRange_ not implemented yet.");
  }

}
/// @par Implementation
/// Flatness is tested with DialUtils::isFlat() (every y within two float
/// epsilons of the first), and "equal to 1" uses the same float-based
/// tolerance, except in a composite node, which keeps a Shift dial even for
/// a response of 1 so that every node has a dial. Only the first point's
/// value is used for the result. The
/// two-point case is guarded by SHORT_CIRCUIT_SMALL_SPLINES, which is
/// defined just before the test, so it is always active: a Spline
/// collection gets a linear Graph dial for two points.
/// @note The empty-list case does not update `_verboseShortCircuitStr_`,
///       so the debug log of initializeBinnedDials() then shows the
///       reason recorded for an earlier bin.
bool DialCollection::makeDialShortCircuit(const std::vector<DialUtils::DialPoint>& pointList_, std::unique_ptr<DialBase>& dial_) const{
  dial_ = nullptr;

  if( pointList_.empty() ){ return true; }

  if( pointList_.size() == 1 or DialUtils::isFlat(pointList_) ){
    // it's flat, let's shortcut

    if( not _isCompositeNode_ and std::abs( pointList_[0].y - 1.0 ) < 2 * std::numeric_limits<float>::epsilon() ){
      // one! no need for a dial
      if(_verboseShortCircuit_){ _verboseShortCircuitStr_ = "No dial: response is always 1."; }
      return true;
    }

    // Do the unique_ptr dance in case there are exceptions.
    auto dialBase = std::make_unique<Shift>();
    dialBase->setShiftValue(pointList_[0].y);
    dial_ = std::move(dialBase);
    if(_verboseShortCircuit_){ _verboseShortCircuitStr_ = "Shift dial: response is flat."; }
    return true;
  }

  // Defined right here, so the block below is always compiled in.
#define SHORT_CIRCUIT_SMALL_SPLINES
#ifdef  SHORT_CIRCUIT_SMALL_SPLINES
  if( pointList_.size() <= 2 ) {
    if(_verboseShortCircuit_){ _verboseShortCircuitStr_ = "Graph dial: 2 points dial."; }
    dial_ = makeGraphDial(pointList_);
    return true;
  }
#endif

  return false;
}
/// @par Implementation
/// This function and makeDialShortCircuit() call each other, but never in
/// a loop: zero or one point is resolved by makeDialShortCircuit() without
/// coming back here, and two points or more build the Graph directly.
/// Points on one straight line (isAffine(), without the slope test) are
/// reduced to the first and last point; if the line is exactly y = x and
/// extrapolation is allowed, a Norm dial, whose response is the input value,
/// replaces the graph.
std::unique_ptr<DialBase> DialCollection::makeGraphDial(const std::vector<DialUtils::DialPoint>& pointList_) const{

  if( pointList_.empty() or pointList_.size() == 1 ){
    // for sure, it's a short circuit
    std::unique_ptr<DialBase> out{nullptr};
    makeDialShortCircuit(pointList_, out);
    return out;
  }

  auto dial = std::make_unique<Graph>();
  if( isAffine(pointList_, false) ){
    if( _allowDialExtrapolation_ and std::all_of(pointList_.begin(), pointList_.end(),
          [](const DialUtils::DialPoint& point_){ return point_.x == point_.y; }) ){
      return std::make_unique<Norm>();
    }
    dial->buildDial(std::vector<DialUtils::DialPoint>{pointList_.front(), pointList_.back()});
  }
  else{ dial->buildDial(pointList_); }
  return dial;
}

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// End:
