#ifndef __AnaEvent_hh__
#define __AnaEvent_hh__

#include <iostream>
#include <mutex>
#include <vector>
#include <map>

#include <TMath.h>
#include "TLeaf.h"
#include "TTree.h"
#include "TROOT.h"
#include "TTreeFormula.h"
#include <TChain.h>

#include "Logger.h"
#include "GenericToolbox.h"

#include "DataBin.h"
#include "DataBinSet.h"


namespace AnaEventType{
  ENUM_EXPANDER(
    AnaEventType, -1,
    Undefined,
    MC,
    DATA
  )
};

class FitParameterSet;
class Dial;

class AnaEvent
{

public:

  AnaEvent();
  AnaEvent(AnaEventType::AnaEventType anaEventType_);
  AnaEvent(long int eventId_);

  void reset();

  // Setters
  void SetAnaEventType(AnaEventType::AnaEventType anaEventType_);
  void SetEventId(long int evid);
  void SetTrueBinIndex(int trueBinIndex_) { _trueBinIndex_ = trueBinIndex_; }
  void SetRecoBinIndex(int recoBinIndex_) { _recoBinIndex_ = recoBinIndex_; }
  void SetSignalEvent(const bool flag = true){ m_signal = flag; }
  void SetTrueEvent(const bool flag = true){ m_true_evt = flag; }
  void SetEvWght(double val)  { _eventWeight_  = val; }
  void SetIntVarNameListPtr(std::vector<std::string>*   intNameListPtr_);
  void SetFloatVarNameListPtr(std::vector<std::string>*   floatNameListPtr_);
  void SetIsBeingEdited(bool isBeingEdited_) { _isBeingEdited_ = isBeingEdited_; }

  // Initializer
  void DumpTreeEntryContent(TTree* tree_);
  void HookIntMembers();
  void HookFloatMembers();

  // Getters
  long int& GetEvId()    { return m_evid; }
  int& GetTrueBinIndex() { return _trueBinIndex_; }
  int& GetRecoBinIndex() { return _recoBinIndex_; }
  bool isTrueEvent()               const { return m_true_evt; }
  bool isSignalEvent()             const { return m_signal; }
  bool GetIsBeingEdited()          const { return _isBeingEdited_; }
  bool GetTreeEventHasBeenDumped() const { return _treeEventHasBeenDumped_; }
  std::vector<std::string>* GetIntVarNameListPtr(){ return _intNameListPtr_; }
  std::vector<std::string>* GetFloatVarNameListPtr(){ return _floatNameListPtr_; }
  double& GetEvWght()     { return _eventWeight_; }
  double GetEventWeight() const { return _eventWeight_; }

  // Core
  int GetGlobalIndex(const std::string& varName_, bool throwIfNotFound_ = true) const;
  int GetIntIndex(const std::string& intName_, bool throwIfNotFound_ = true) const;
  int GetFloatIndex(const std::string& floatName_, bool throwIfNotFound_ = true) const;
  Int_t GetEventVarInt(int varIndex_) const;
  Int_t GetEventVarInt(const std::string& varName_) const;
  Float_t GetEventVarFloat(int varIndex_) const;
  Float_t GetEventVarFloat(const std::string& varName_) const;
  double GetEventVarAsDouble(int varGlobalIndex_) const; // varGlobalIndex_ = intIndexes, then floatIndexes
  double GetEventVarAsDouble(const std::string& varName_) const;
  void AddEvWght(double val);
  void ResetEvWght();

  // Misc
  void Print() const;

  // Old: Direct Access Setters
  void SetTopology(Int_t val)   { *_topologyPtr_ = val; }
  void SetReaction(Int_t val)   { *_reactionPtr_ = val; }
  void SetTarget(Int_t val)     { *_targetPtr_ = val; }
  void SetSampleType(Int_t val) { *_samplePtr_ = val; }
  void SetFlavor(Int_t flavor)  { *_flavorPtr_ = flavor; }
  void SetBeamMode(Int_t val)   { *_beamModePtr_ = val; }
  void SetSignalType(Int_t val) { *_sigTypePtr_ = val; }

  void SetTrueEnu(Float_t val) { *_enuTruePtr_ = val; }
  void SetRecoEnu(Float_t val) { *_enuRecoPtr_ = val; }
  void SetTrueD1(Float_t val)  { *_d1TruePtr_ = val; }
  void SetRecoD1(Float_t val)  { *_d1RecoPtr_ = val; }
  void SetTrueD2(Float_t val)  { *_d2TruePtr_ = val; }
  void SetRecoD2(Float_t val)  { *_d2RecoPtr_ = val; }
  void SetQ2Reco(Float_t val)  { *_q2RecoPtr_ = val; }
  void SetQ2True(Float_t val)  { *_q2TruePtr_ = val; }
  void SetEvWghtMC(Float_t val){ *_weightMCPtr_ = val; }

  // Old: Direct Access Getters
  Int_t&   GetTopology()   { return *_topologyPtr_; }
  Int_t&   GetReaction()   { return *_reactionPtr_; }
  Int_t&   GetTarget()     { return *_targetPtr_; }
  Int_t&   GetSampleType() { return *_samplePtr_; }
  Int_t&   GetFlavor()     { return *_flavorPtr_; }
  Int_t&   GetBeamMode()   { return *_beamModePtr_; }
  Int_t&   GetSignalType() { return *_sigTypePtr_; }

  Float_t& GetTrueEnu()    { return *_enuTruePtr_; }
  Float_t& GetRecoEnu()    { return *_enuRecoPtr_; }
  Float_t& GetTrueD1()     { return *_d1TruePtr_; }
  Float_t& GetRecoD1()     { return *_d1RecoPtr_; }
  Float_t& GetTrueD2()     { return *_d2TruePtr_; }
  Float_t& GetRecoD2()     { return *_d2RecoPtr_; }
  Float_t& GetQ2True()     { return *_q2TruePtr_; }
  Float_t& GetQ2Reco()     { return *_q2RecoPtr_; }
  Float_t& GetEvWghtMC()   { return *_weightMCPtr_; }

  // Interfaces
  bool isInBin( const DataBin& dataBin_) const;
  std::map<FitParameterSet *, std::vector<Dial *>> *getDialCachePtr();
  double evalFormula(TFormula* formulaPtr_) const;

  // Deprecated
  Int_t GetEventVar(const std::string& var);

protected:
  void ResetIntContainer();
  void ResetFloatContainer();

private:

  AnaEventType::AnaEventType _anaEventType_;

  // Multi-thread security
  bool _isBeingEdited_{};
  bool _treeEventHasBeenDumped_{};

//  TChain* _chainPtr_{nullptr};
  TTree* _singleEntryTree_{nullptr};
//  std::vector<TLeaf*> _leafList_;

  // Int_t containers
  std::vector<std::string>* _intNameListPtr_;     // same ptr can be used for every event -> save memory
  std::vector<std::string>  _defaultIntNameList_; // here for memory management if the ptr is not manually assigned
  std::vector<Int_t> _intValuesList_;

  // Float_t containers
  std::vector<std::string>* _floatNameListPtr_;       // same ptr can be used for every event -> save memory
  std::vector<std::string>  _defaultFloatNameList_;   // here for memory management if the ptr is not manually assigned
  std::vector<Float_t> _floatValuesList_;

  bool m_signal{};     //flag if signal event
  bool m_true_evt{};   //flag if true event
  long int m_evid{};   //unique event id
  int _trueBinIndex_{};
  int _recoBinIndex_{};

  double _eventWeight_{};

  Int_t* _flavorPtr_{};     //flavor of neutrino (numu, etc.)
  Int_t* _beamModePtr_{};   //Forward horn current (+1) or reverse horn current (-1)
  Int_t* _topologyPtr_{};   //final state topology type
  Int_t* _reactionPtr_{};   //event interaction mode
  Int_t* _targetPtr_{};     //target nuclei
  Int_t* _samplePtr_{};     //sample type (aka cutBranch)
  Int_t* _sigTypePtr_{};    //signal definition

  Float_t* _enuTruePtr_{};
  Float_t* _enuRecoPtr_{};
  Float_t* _d1TruePtr_{};
  Float_t* _d1RecoPtr_{};
  Float_t* _d2TruePtr_{};
  Float_t* _d2RecoPtr_{};
  Float_t* _q2TruePtr_{};
  Float_t* _q2RecoPtr_{};
  Float_t* _weightMCPtr_{};

  // Cache
  std::map<FitParameterSet*, std::vector<Dial*>> _dialCache_; // _dialCache_[fitParSetPtr][ParIndex] = correspondingDialPtr;

};

#endif
