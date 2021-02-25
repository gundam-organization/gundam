#ifndef __AnaEvent_hh__
#define __AnaEvent_hh__

#include <iostream>
#include <mutex>
#include <vector>
#include <map>

#include <TMath.h>
#include "TLeaf.h"

#include "Logger.h"
#include "GenericToolbox.h"

#include <FitStructs.hh>

class AnaEvent
{

public:

    typedef enum{
        MC = 0,
        DATA
    } AnaEventType;

    AnaEvent(){
        _anaEventType_     = MC;
        _intNameListPtr_ = nullptr;
        _floatNameListPtr_ = nullptr;
        Reset();
    }
    AnaEvent(AnaEventType anaEventType_){
        _anaEventType_ = anaEventType_;
        _intNameListPtr_ = nullptr;
        _floatNameListPtr_ = nullptr;
        Reset();
    }
    explicit AnaEvent(long int evid){
        _anaEventType_     = MC;
        _intNameListPtr_ = nullptr;
        _floatNameListPtr_ = nullptr;
        Reset();
        m_evid = evid;
    }

    void Reset(){

        _isBeingEdited_ = false;
        _treeEventHasBeenDumped_ = false;

        // long int
        m_evid = -1;

        _trueBinIndex_ = -1;
        _recoBinIndex_ = -1;

        // bool
        m_signal   = false;
        m_true_evt = false;

        ResetIntContainer();
        ResetFloatContainer();

    }

    void ResetIntContainer(){
        _defaultIntNameList_.clear();

        _defaultIntNameList_.emplace_back("beammode");
        _defaultIntNameList_.emplace_back("topology");
        _defaultIntNameList_.emplace_back("cut_branch");

        if(_anaEventType_ == MC){
            _defaultIntNameList_.emplace_back("nutype");
            _defaultIntNameList_.emplace_back("reaction");
            _defaultIntNameList_.emplace_back("target");
            _defaultIntNameList_.emplace_back("signal");
        }

        _intNameListPtr_ = &_defaultIntNameList_;
        _intValuesList_.resize(_intNameListPtr_->size());

        HookIntMembers();
    }
    void ResetFloatContainer(){
        _defaultFloatNameList_.clear();

        _defaultFloatNameList_.emplace_back("enu_reco");
        _defaultFloatNameList_.emplace_back("D1Reco");
        _defaultFloatNameList_.emplace_back("D2Reco");
        _defaultFloatNameList_.emplace_back("q2_reco");
        _defaultFloatNameList_.emplace_back("weight"); // asimov

        if(_anaEventType_ == MC){
            _defaultFloatNameList_.emplace_back("enu_true");
            _defaultFloatNameList_.emplace_back("D1True");
            _defaultFloatNameList_.emplace_back("D2True");
            _defaultFloatNameList_.emplace_back("q2_true");
            _defaultFloatNameList_.emplace_back("weightMC");
        }

        _floatNameListPtr_ = &_defaultFloatNameList_;
        _floatValuesList_.resize(_floatNameListPtr_->size());

        HookFloatMembers();
    }

    void HookIntMembers(){
        // Hooking ptr members for faster access
        // CAVEAT: all these vars have to be in _intNameListPtr_
        _flavorPtr_ = &_intValuesList_[GetIntIndex("nutype")];
        _beamModePtr_ = &_intValuesList_[GetIntIndex("beammode")];
        _topologyPtr_ = &_intValuesList_[GetIntIndex("topology")];
        _reactionPtr_ = &_intValuesList_[GetIntIndex("reaction")];
        _targetPtr_ = &_intValuesList_[GetIntIndex("target")];
        _samplePtr_ = &_intValuesList_[GetIntIndex("cut_branch")];
        _sigTypePtr_ = &_intValuesList_[GetIntIndex("signal")];

    }
    void HookFloatMembers(){
        // Hooking ptr members for faster access
        // CAVEAT: all these vars have to be in _floatNameListPtr_
        _enuTruePtr_ = &_floatValuesList_[GetFloatIndex("enu_true")];
        _enuRecoPtr_ = &_floatValuesList_[GetFloatIndex("enu_reco")];
        _d1TruePtr_ = &_floatValuesList_[GetFloatIndex("D1True")];
        _d1RecoPtr_ = &_floatValuesList_[GetFloatIndex("D1Reco")];
        _d2TruePtr_ = &_floatValuesList_[GetFloatIndex("D2True")];
        _d2RecoPtr_ = &_floatValuesList_[GetFloatIndex("D2Reco")];
        _q2TruePtr_ = &_floatValuesList_[GetFloatIndex("q2_true")];
        _q2RecoPtr_ = &_floatValuesList_[GetFloatIndex("q2_reco")];
        _weightPtr_ = &_floatValuesList_[GetFloatIndex("weight")];
        _weightMCPtr_ = &_floatValuesList_[GetFloatIndex("weightMC")];
    }

    int GetIntIndex(const std::string& intName_, bool throwIfNotFound_ = true) const {
        int index = GenericToolbox::findElementIndex(intName_, *_intNameListPtr_);
        if(throwIfNotFound_ and index == -1){
            LogFatal << "Could not find int \"" << intName_ << "\" in _intNameListPtr_." << std::endl;
            LogFatal << "Available ints are: ";
            GenericToolbox::printVector(*_intNameListPtr_);
            LogFatal << std::endl;
            throw std::logic_error("Could not find int");
        }
        return index;
    }
    int GetFloatIndex(const std::string& floatName_, bool throwIfNotFound_ = true) const {
        int index = GenericToolbox::findElementIndex(floatName_, *_floatNameListPtr_);
        if(throwIfNotFound_ and index == -1){
            LogFatal << "Could not find float \"" << floatName_ << "\" in _floatNameListPtr_." << std::endl;
            LogFatal << "Available floats are: ";
            GenericToolbox::printVector(*_floatNameListPtr_);
            LogFatal << std::endl;
            throw std::logic_error("Could not find float");
        }
        return index;
    }

    void DumpTreeEntryContent(TTree* tree_){

        if(_treeEventHasBeenDumped_) return;

        TLeaf* leafBuffer = nullptr;
        int index;
        for( int iKey = 0 ; iKey < tree_->GetListOfLeaves()->GetEntries() ; iKey++ ){
            leafBuffer = (TLeaf*) tree_->GetListOfLeaves()->At(iKey);

            if( std::string(leafBuffer->GetTypeName()) == "Int_t" ){
                index = GetIntIndex(leafBuffer->GetName(), false);
                if(index != -1){
                    _intValuesList_[index] = leafBuffer->GetValue(0);
                }
            }
            else if( std::string(leafBuffer->GetTypeName()) == "Float_t" ){
                index = GetFloatIndex(leafBuffer->GetName(), false);
                if(index != -1){
                    _floatValuesList_[index] = leafBuffer->GetValue(0);
                }
            }

        }

        *_weightMCPtr_ = *_weightPtr_;

        _treeEventHasBeenDumped_ = true;

    }

    // Setters
    void SetAnaEventType(AnaEventType anaEventType_){
        _anaEventType_ = anaEventType_;
    }
    void SetEventId(long int evid){ m_evid = evid; }
    void SetTrueBinIndex(int trueBinIndex_) { _trueBinIndex_ = trueBinIndex_; }
    void SetRecoBinIndex(int recoBinIndex_) { _recoBinIndex_ = recoBinIndex_; }
    void SetSignalEvent(const bool flag = true){ m_signal = flag; }
    void SetTrueEvent(const bool flag = true){ m_true_evt = flag; }

    void SetIntVarNameListPtr(std::vector<std::string>*   intNameListPtr_){
        // Copy old values
        std::vector<Int_t> newIntValuesList(intNameListPtr_->size());
        for( size_t iName = 0 ; iName < _intNameListPtr_->size() ; iName++ ){
            for( size_t jName = 0 ; jName < intNameListPtr_->size() ; jName++ ){
                if(_intNameListPtr_->at(iName) == intNameListPtr_->at(jName)){
                    newIntValuesList.at(jName) = _intValuesList_.at(iName);
                }
            }
        }

        _intNameListPtr_ = intNameListPtr_;
        _defaultIntNameList_.clear(); // save memory since it's not used anymore

        _intValuesList_ = newIntValuesList;
        HookIntMembers();
    }
    void SetFloatVarNameListPtr(std::vector<std::string>*   floatNameListPtr_){
        // Copy old values
        std::vector<Float_t> newFloatValuesList(floatNameListPtr_->size());
        for( size_t iName = 0 ; iName < _floatNameListPtr_->size() ; iName++ ){
            for( size_t jName = 0 ; jName < floatNameListPtr_->size() ; jName++ ){
                if(_floatNameListPtr_->at(iName) == floatNameListPtr_->at(jName)){
                    newFloatValuesList.at(jName) = _floatValuesList_.at(iName);
                }
            }
        }

        _floatNameListPtr_ = floatNameListPtr_;
        _defaultFloatNameList_.clear(); // save memory since it's not used anymore
        _floatValuesList_ = newFloatValuesList;
        HookFloatMembers();
    }

    void SetTopology(int val)   { *_topologyPtr_ = val; }
    void SetReaction(int val)   { *_reactionPtr_ = val; }
    void SetTarget(int val)     { *_targetPtr_ = val; }
    void SetSampleType(int val) { *_samplePtr_ = val; }
    void SetFlavor(int flavor)  { *_flavorPtr_ = flavor; }
    void SetBeamMode(int val)   { *_beamModePtr_ = val; }
    void SetSignalType(int val) { *_sigTypePtr_ = val; }

    void SetTrueEnu(double val) { *_enuTruePtr_ = val; }
    void SetRecoEnu(double val) { *_enuRecoPtr_ = val; }
    void SetTrueD1(double val)  { *_d1TruePtr_ = val; }
    void SetRecoD1(double val)  { *_d1RecoPtr_ = val; }
    void SetTrueD2(double val)  { *_d2TruePtr_ = val; }
    void SetRecoD2(double val)  { *_d2RecoPtr_ = val; }
    void SetQ2Reco(double val)  { *_q2RecoPtr_ = val; }
    void SetQ2True(double val)  { *_q2TruePtr_ = val; }
    void SetEvWght(double val)  { *_weightPtr_  = val; }
    void SetEvWghtMC(double val){ *_weightMCPtr_ = val; }

    void SetIsBeingEdited(bool isBeingEdited_) { _isBeingEdited_ = isBeingEdited_; }

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
    Float_t& GetEvWght()     { return *_weightPtr_; }
    Float_t& GetEvWghtMC()   { return *_weightMCPtr_; }


    // Multiplies the current event weight with the input argument:
    void AddEvWght(double val){ *_weightPtr_ *= val; }
    void ResetEvWght(){ *_weightPtr_ = *_weightMCPtr_; }

    void Print() {

        LogInfo << "Event ID: " << m_evid << std::endl;

        LogInfo << "List of Int_t: {" << std::endl;
        for(size_t iInt = 0 ; iInt < _intNameListPtr_->size() ; iInt++){
            if(iInt != 0) LogInfo << ", " << std::endl;
            LogInfo << "  \"" << (*_intNameListPtr_)[iInt] << "\": " << _intValuesList_[GetIntIndex((*_intNameListPtr_)[iInt])];
        }
        LogInfo << std::endl << "}" << std::endl;

        LogInfo << "List of Float_t: {" << std::endl;
        for(size_t iFloat = 0 ; iFloat < _floatNameListPtr_->size() ; iFloat++){
            if(iFloat != 0) LogInfo << ", " << std::endl;
            LogInfo << "  \"" << (*_floatNameListPtr_)[iFloat] << "\": " << _floatValuesList_[GetFloatIndex((*_floatNameListPtr_)[iFloat])];
        }
        LogInfo << std::endl << "}" << std::endl;
    }

    Int_t& GetEventVarInt(const std::string& varName_) {
        return _intValuesList_.at(GetIntIndex(varName_));
    }
    Float_t& GetEventVarFloat(const std::string& varName_) {
        return _floatValuesList_.at(GetFloatIndex(varName_));
    }

    // Deprecated
    Int_t GetEventVar(const std::string& var) {
        return GetEventVarInt(var);
    }

private:

    AnaEventType _anaEventType_;

    // Multi-thread security
    bool _isBeingEdited_;
    bool _treeEventHasBeenDumped_;

    // Int_t containers
    std::vector<std::string>* _intNameListPtr_;     // same ptr can be used for every event -> save memory
    std::vector<std::string>  _defaultIntNameList_; // here for memory management if the ptr is not manually assigned
    std::vector<Int_t> _intValuesList_;

    // Float_t containers
    std::vector<std::string>* _floatNameListPtr_;       // same ptr can be used for every event -> save memory
    std::vector<std::string>  _defaultFloatNameList_;   // here for memory management if the ptr is not manually assigned
    std::vector<Float_t> _floatValuesList_;

    bool m_signal;     //flag if signal event
    bool m_true_evt;   //flag if true event
    long int m_evid;   //unique event id
    int _trueBinIndex_;
    int _recoBinIndex_;

    Int_t* _flavorPtr_;     //flavor of neutrino (numu, etc.)
    Int_t* _beamModePtr_;   //Forward horn current (+1) or reverse horn current (-1)
    Int_t* _topologyPtr_;   //final state topology type
    Int_t* _reactionPtr_;   //event interaction mode
    Int_t* _targetPtr_;     //target nuclei
    Int_t* _samplePtr_;     //sample type (aka cutBranch)
    Int_t* _sigTypePtr_;    //signal definition

    Float_t* _enuTruePtr_;
    Float_t* _enuRecoPtr_;
    Float_t* _d1TruePtr_;
    Float_t* _d1RecoPtr_;
    Float_t* _d2TruePtr_;
    Float_t* _d2RecoPtr_;
    Float_t* _q2TruePtr_;
    Float_t* _q2RecoPtr_;
    Float_t* _weightPtr_;
    Float_t* _weightMCPtr_;

};

#endif
