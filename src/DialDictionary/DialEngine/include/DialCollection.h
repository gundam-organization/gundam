//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALCOLLECTION_H
#define GUNDAM_DIALCOLLECTION_H

#include "DialBase.h"
#include "DialInterface.h"
#include "DialInputBuffer.h"
#include "DialResponseSupervisor.h"
#include "SampleSet.h"

#include "GenericToolbox.Wrappers.h"

#include <vector>
#include <string>
#include <memory>

class DialCollection : public JsonBaseClass {
public:
  DialCollection() = delete;
  explicit DialCollection(std::vector<ParameterSet> *targetParameterSetListPtr): _parameterSetListPtr_(targetParameterSetListPtr) {}

  // A base class for any specialized DialCollection data needed by a
  // particular implementation.  For example, the Tabulated dials use this to
  // keep track of the external libraries needed to fill the tables.
  struct CollectionData {
    CollectionData() = default;
    virtual ~CollectionData() = default;
  };

  //  The PolymorphicObjectWrapper doesn't have the correct semantics since it
  // clones the payload when it's copied.  We want to leave the pointee alone
  // and just move the pointers around.
  //
  // Temporarily replace specialty class with shared_ptr.  The shared_ptr
  // class has the correct semantics (copyable, and deletes the object), but
  // we don't need the reference counting since we can only have one of each
  // object.  Also shared_ptr is a bit to memory hungry.
  typedef std::shared_ptr<DialBase> DialBaseObject;

  // setters
  void setIndex(int index){ _index_ = index; }
  void setSupervisedParameterIndex(int supervisedParameterIndex){ _supervisedParameterIndex_ = supervisedParameterIndex; }
  void setSupervisedParameterSetIndex(int supervisedParameterSetIndex){ _supervisedParameterSetIndex_ = supervisedParameterSetIndex; }

  // DEPRECATED: Replace with 'not isEventByEvent()' -- When GUNDAM
  // started, there were three types of dials that could be applied to
  // events (normalization, event-by-event splines, and binned
  // splines).  Since normalization didn't have a "real" dial, we only
  // needed to distinguish between binned and unbinned splines.  Now
  // we have event-by-event splines, and "everything else"
  // (Normalization, Formula, "binned", &c).  This may be
  // "undeprecated" as soon as we start adding different queries for
  // dial types.
  [[deprecated("Replace with 'not isEventByEvent()'")]] bool isBinned() const {return not isEventByEvent();}

  // const getters

  // Query if this has one DialBaseObject per event, or if DialBaseObjects are
  // shared between events.  Used to decide how to attach to events in
  // DataDispenser.
  [[nodiscard]] bool isEventByEvent() const {return _isEventByEvent_;}

  // Check if the dial is enabled during this run
  [[nodiscard]] bool isEnabled() const{ return _isEnabled_; }

  // Check if the DialBase should be allowed to calculate an extrapolation out
  // outside of the defined parameter bounds.
  [[nodiscard]] bool isAllowDialExtrapolation() const{ return _allowDialExtrapolation_; }

  // The location in the cache for this dialCollection that can be used to
  // indentify the collection.
  [[nodiscard]] int getIndex() const{ return _index_; }

  // The value for dialType: in the yaml file.
  [[nodiscard]] const std::string &getGlobalDialType() const{return _globalDialType_; }

  // The value for dialSubType: in the yaml file
  [[nodiscard]] const std::string &getGlobalDialSubType() const{ return _globalDialSubType_; }

  // If it exists, then this is the name of a leaf in the input file
  // containing data to weight the event in the entry.  This is empty if the
  // dial is not event-by-event, or if the dial is not based on a "spline"
  // (e.g. it might be an tabulated event-by-event dial).
  [[nodiscard]] const std::string &getGlobalDialLeafName() const{ return _globalDialLeafName_; }

  [[nodiscard]] const BinSet &getDialBinSet() const{ return _dialBinSet_; }
  [[nodiscard]] const std::vector<std::string> &getDataSetNameList() const{ return _dataSetNameList_; }

  // A formula to decide if the dial should be applied to an event.  The dial
  // should be applied if this returns a non-zero value.
  [[nodiscard]] const std::shared_ptr<TFormula> &getApplyConditionFormula() const{ return _applyConditionFormula_; }

  // non-const getters
  BinSet &getDialBinSet(){ return _dialBinSet_; }

  // One interface per DialBase.  The interface groups the input buffer,
  // response supervisor, dialBase (what actually calculates the weight) into
  // a single object and provide the methods to calculate the dial weight.
  std::vector<DialInterface> &getDialInterfaceList(){ return _dialInterfaceList_; }

  // The different sets of parameters to use for the interfaces.  There is a
  // single set of inputs for each DialBase in the collection, or a different
  // set of parameters for each DialBase (i.e. the number of DialInputBuffer
  // objects is 1, or _dialBaseList_.size()).  set, but for
  std::vector<DialInputBuffer> &getDialInputBufferList(){ return _dialInputBufferList_; }

  // non-trivial getters
  [[nodiscard]] bool isDatasetValid(const std::string& datasetName_) const;
  std::string getTitle() const;
  std::string getSummary(bool shallow_ = true);
  Parameter* getSupervisedParameter() const;
  ParameterSet* getSupervisedParameterSet() const;

  // core
  void clear();

  // Called after the DialCollection has been fully initialized, and reclaims
  // any unused space.
  void resizeContainers();

  // After the DialCollection is fully initialized, setup all of the pointers
  // in the DialInterface objects.  This is used after the size of the
  // DialCollection has changed to fix any pointer issues.
  void setupDialInterfaceReferences();

  // Update the dial collection.  Most dial collections don't need an update,
  // but this provides a handle for collections that precalculate values prior
  // to doing the event reweighting.  An example of that would be the
  // Tabulated dials (which can be used in implement things like oscillation
  // weights).  Update callbacks are std::function<void(DialCollection*)>, and
  // can be added with addUpdate();
  void update();

  // Add a dial collection update callback.  These are called in the order
  // that they are added.  They are activated by the "update()" method.
  void addUpdate(std::function<void(DialCollection* dc)> callback);

  // Check if the dial will need to be recalculated.  A recalculation
  // happens when a parameter value has changed since the last calculation.
  // This also checks if a dial has been masked, and flags that the
  // Dial calculation should be ignored.  That can only happen before the
  // reweight calculation has been "frozen".
  void updateInputBuffers();

  // Return the next slot in the DialBaseList that can b filled. Its
  // thread safe, so multiple threads can fill the list.
  size_t getNextDialFreeSlot(){ return _dialFreeSlot_++; }
  size_t getDialFreeSlotIndex() const { return _dialFreeSlot_.getValue(); }

  // Provide access to a collection data.  The ownership is retained by
  // the collection.
  template <typename T>
  T* getCollectionData(int i=0) const {return dynamic_cast<T*>(_dialCollectionData_[i].get());}

  // Add an extra leaf name needed for this dial
  void addExtraLeafName(const std::string& leaf) {_globalDialExtraLeafNames_.emplace_back(leaf);}

  std::vector<std::string>& getExtraLeafNames() {return _globalDialExtraLeafNames_;}

  void invalidateCachedInputBuffers(){ for( auto& inputBuffer : _dialInputBufferList_ ){ inputBuffer.invalidateBuffers(); }}

  void printConfiguration() const;

protected:
  void configureImpl() override;
  void initializeImpl() override;

  bool initializeNormDialsWithParBinning();
  bool initializeDialsWithDefinition();
  bool initializeDialsWithBinningFile(const JsonType& dialsDefinition);
  bool initializeDialsWithTabulation(const JsonType& dialsDefinition);

  void readGlobals(const JsonType &config_);
  JsonType fetchDialsDefinition(const JsonType &definitionsList_);

private:
  // parameters
  bool _isEventByEvent_{false};
  bool _isEnabled_{true};
  bool _useMirrorDial_{false};
  bool _enableDialsSummary_{false};
  bool _allowDialExtrapolation_{true};
  int _index_{-1};
  double _minDialResponse_{std::nan("unset")};
  double _maxDialResponse_{std::nan("unset")};
  double _mirrorLowEdge_{std::nan("unset")};
  double _mirrorHighEdge_{std::nan("unset")};
  double _mirrorRange_{std::nan("unset")};
  std::string _applyConditionStr_{};
  std::string _globalDialLeafName_{};
  std::string _globalDialType_{};
  std::string _globalDialSubType_{};
  std::vector<std::string> _dataSetNameList_{};
  std::vector<std::string> _globalDialExtraLeafNames_{};

  // internal
  int _supervisedParameterIndex_{-1};
  int _supervisedParameterSetIndex_{-1};
  BinSet _dialBinSet_{};

  // One interface per DialBase.  The interface groups the input buffer,
  // response supervisor, dialBase (what actually calculates the weight) into
  // a single object and provide the methods to calculate the dial weight.
  std::vector<DialInterface> _dialInterfaceList_{};

  // The different sets of parameters to use for the interfaces.  There is a
  // single set of inputs for each DialBase in the collection, or a different
  // set of parameters for each DialBase (i.e. the number of DialInputBuffer
  // objects is 1, or _dialBaseList_.size()).  set, but for
  std::vector<DialInputBuffer> _dialInputBufferList_{};

  // The response supervisors (which apply an upper and lower clamp).  There
  // is a single supervisor for all the interfaces, or one for each
  // DialBase object in the collection (just like for the
  // _dialInputBufferList_.
  std::vector<DialResponseSupervisor> _dialResponseSupervisorList_{};

  // A formula to decide if the dial should be applied to an event.
  std::shared_ptr<TFormula> _applyConditionFormula_{nullptr};
  GenericToolbox::Atomic<size_t> _dialFreeSlot_{0};

  // A pointer to dial specific data
  std::vector<std::shared_ptr<DialCollection::CollectionData>>  _dialCollectionData_;

  // The callbacks for this dial collection
  std::vector<std::function<void(DialCollection*)>> _dialCollectionCallbacks_;

  // external refs
  std::vector<ParameterSet>* _parameterSetListPtr_{nullptr};


  // dev
public:
  // methods to generate dials with factory
  std::unique_ptr<DialBase> makeDial() const;
  std::unique_ptr<DialBase> makeDial(const TObject* src_) const;
  std::unique_ptr<DialBase> makeDial(const JsonType& config_) const;

protected:
  std::unique_ptr<DialBase> makeGraphDial(const TObject* src_) const;
  std::unique_ptr<DialBase> makeSplineDial(const TObject* src_) const;
  std::unique_ptr<DialBase> makeSurfaceDial(const TObject* src_) const;


};
#endif //GUNDAM_DIALCOLLECTION_H

// Local Variables:
// mode:c++
// c-basic-offset:2
// End:
