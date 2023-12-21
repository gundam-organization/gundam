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

#include "nlohmann/json.hpp"

#include <vector>
#include <string>
#include <memory>

class DialCollection : public JsonBaseClass {

public:
  DialCollection() = delete;
  explicit DialCollection(std::vector<ParameterSet> *targetParameterSetListPtr): _parameterSetListPtr_(targetParameterSetListPtr) {}

  //  The PolymorphicObjectWrapper doesn't have the correct semantics since it
  // clones the payload when it's copied.  We want to leave the pointee alone
  // and just move the pointers around.
  //
  // Temporarily replace specialty class with shared_ptr.  The shared_ptr
  // class has the correct semantics (copyable, and deletes the object), but
  // we don't need the reference counting since we can only have one of each
  // object.  Also shared_ptr is a bit to memory hungry.
  typedef std::shared_ptr<DialBase> DialBaseObject;
  // typedef GenericToolbox::PolymorphicObjectWrapper<DialBase> DialBaseObject;

  // setters
  void setIndex(int index){ _index_ = index; }
  void setSupervisedParameterIndex(int supervisedParameterIndex){ _supervisedParameterIndex_ = supervisedParameterIndex; }
  void setSupervisedParameterSetIndex(int supervisedParameterSetIndex){ _supervisedParameterSetIndex_ = supervisedParameterSetIndex; }

  // const getters
  [[nodiscard]] bool isBinned() const{ return _isBinned_; }
  [[nodiscard]] bool isEnabled() const{ return _isEnabled_; }
  [[nodiscard]] bool isAllowDialExtrapolation() const{ return _allowDialExtrapolation_; }
  [[nodiscard]] int getIndex() const{ return _index_; }
  [[nodiscard]] const std::string &getGlobalDialType() const{return _globalDialType_; }
  [[nodiscard]] const std::string &getGlobalDialSubType() const{ return _globalDialSubType_; }
  [[nodiscard]] const std::string &getGlobalDialLeafName() const{ return _globalDialLeafName_; }
  [[nodiscard]] const DataBinSet &getDialBinSet() const{ return _dialBinSet_; }
  [[nodiscard]] const std::vector<std::string> &getDataSetNameList() const{ return _dataSetNameList_; }
  [[nodiscard]] const std::shared_ptr<TFormula> &getApplyConditionFormula() const{ return _applyConditionFormula_; }

  // non-const getters
  DataBinSet &getDialBinSet(){ return _dialBinSet_; }
  std::vector<DialBaseObject> &getDialBaseList(){ return _dialBaseList_; }
  std::vector<DialInterface> &getDialInterfaceList(){ return _dialInterfaceList_; }

  // non-trivial getters
  [[nodiscard]] bool isDatasetValid(const std::string& datasetName_) const;
  std::string getTitle() const;
  std::string getSummary(bool shallow_ = true);
  Parameter* getSupervisedParameter() const;
  ParameterSet* getSupervisedParameterSet() const;

  // core
  void clear();
  void resizeContainers();
  void setupDialInterfaceReferences();
  void updateInputBuffers();
  size_t getNextDialFreeSlot();


protected:
  void readConfigImpl() override;
  void initializeImpl() override;

  bool initializeNormDialsWithParBinning();
  bool initializeDialsWithDefinition();
  void readGlobals(const JsonType &config_);
  JsonType fetchDialsDefinition(const JsonType &definitionsList_);

private:
  // parameters
  bool _isBinned_{true};
  bool _isEnabled_{true};
  bool _useMirrorDial_{false};
  bool _disableDialCache_{false};
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

  // internal
  int _supervisedParameterIndex_{-1};
  int _supervisedParameterSetIndex_{-1};
  DataBinSet _dialBinSet_{};
  std::vector<DialInterface> _dialInterfaceList_{};
  std::vector<DialInputBuffer> _dialInputBufferList_{};
  std::vector<DialResponseSupervisor> _dialResponseSupervisorList_{};
  std::vector<DialBaseObject> _dialBaseList_{};
  std::shared_ptr<TFormula> _applyConditionFormula_{nullptr};
  GenericToolbox::Atomic<size_t> _dialFreeSlot_{0};

  // external refs
  std::vector<ParameterSet>* _parameterSetListPtr_{nullptr};

};


#endif //GUNDAM_DIALCOLLECTION_H
