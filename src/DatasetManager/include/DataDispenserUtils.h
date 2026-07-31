//
// Created by Adrien Blanchet on 29/09/2023.
//

#ifndef GUNDAM_DATA_DISPENSER_UTILS_H
#define GUNDAM_DATA_DISPENSER_UTILS_H

#include "Propagator.h"
#include "EventVarTransformLib.h"

#include "GenericToolbox.Wrappers.h"
#include "GenericToolbox.Root.h"

#include "TChain.h"
#include "TTreeFormula.h"
#include "TFormula.h"


#include "string"
#include "map"


struct DataDispenserParameters{

  // should be load dials and request the associate variables?
  bool useReweightEngine{false};
  bool isData{false}; // shall fetch slpit vars?
  bool allowMultipleSamplesPerEntry{false};
  size_t debugNbMaxEventsToLoad{0};
  double fractionOfEntries{1.};

  std::string name{};
  std::string globalTreePath{};
  std::string dialIndexFormula{};
  std::string nominalWeightFormulaStr{};
  std::string selectionCutFormulaStr{};
  std::string eventVariableAsWeight{};
  std::vector<std::string> activeLeafNameList{};
  struct FilePathEntry{
    struct FriendTree{
      std::string name{}; // ROOT friend alias
      std::string path{}; // ROOT file path, optionally followed by :tree/path
    };

    std::string name{}; // config-list identifier, used by the override mechanism
    std::string path{}; // ROOT file path, optionally followed by :tree/path
    std::vector<FriendTree> friendList{};
  };
  std::vector<FilePathEntry> filePathList{};
  std::map<std::string, std::string> variableDict{};
  std::map<std::string, EventVarTransformLib> variableDictTransform{};
  std::vector<std::string> additionalVarsStorage{};
  std::vector<std::string> dummyVariablesList;

  struct FromHistContent{
    bool isEnabled{false};
    std::string rootFilePath{};

    struct SampleHist{
      std::string name{};
      std::string hist{};
      std::vector<std::string> axisList{};
    };
    std::vector<SampleHist> sampleHistList{};

    SampleHist& addSampleHist(const std::string& name_){
      for( auto& sampleHist : sampleHistList ){
        LogThrowIf(sampleHist.name == name_, "Duplicate sample hist with name: " << name_);
      }
      sampleHistList.emplace_back();
      sampleHistList.back().name = name_;
      return sampleHistList.back();
    }
    SampleHist* getSampleHistPtr(const std::string& name_){
      for( auto& sampleHist : sampleHistList ){
        if( sampleHist.name == name_ ){ return &sampleHist; }
      }
      return nullptr;
    }
  };
  FromHistContent fromHistContent;

//  JsonType fromHistContent;
  JsonType overridePropagatorConfig;
  JsonType evalModelAt;

  [[nodiscard]] std::string getSummary() const;
};

struct DataDispenserCache{
  Propagator* propagatorPtr{nullptr};

  size_t totalNbEvents{0};

  std::vector<Sample*> samplesToFillList{};
  std::vector<size_t> sampleNbOfEvents;
  GenericToolbox::CSRVector<int> entrySampleIndexList{};
  std::vector<size_t> sampleIndexOffsetList;
  std::vector< std::vector<Event>* > sampleEventListPtrToFill;
  std::vector<DialCollection*> dialCollectionsRefList{};

  std::vector<std::string> varsRequestedForIndexing{};
  std::map<std::string, std::pair<std::string, bool>> varToLeafDict; // varToLeafDict[EVENT_VAR_NAME] = {LEAF_NAME, IS_DUMMY}
  std::map<std::string, std::string> eventFormulaTreeExpressionAliases{};

  struct VariableDictEntry{
    enum EvalBackend{
      TreeBufferExpression,
      EventBufferFormula,
      LibraryTransform
    };

    std::string name{};
    std::string expr{};
    EvalBackend backend{TreeBufferExpression};
    const EventVarTransformLib* transformPtr{nullptr};
  };
  std::vector<VariableDictEntry> variableDictEvalList{};

  struct ThreadSelectionResult{
    std::vector<size_t> sampleNbOfEvents;
    GenericToolbox::CSRVector<int> entrySampleIndexList;
  };
  std::vector<ThreadSelectionResult> threadSelectionResults;

  void clear();
  void addVarRequestedForIndexing(const std::string& varName_);

};

struct ThreadSharedData{
  // I/O
  Long64_t nbEntries{0};
  std::shared_ptr<TChain> treeChain{nullptr};
  GenericToolbox::TreeBuffer treeBuffer{};

  // buffer
  struct VariableBuffer{
    struct EventFormula{
      std::string expr{};
      TFormula formula{};
      std::vector<int> varIndexList{};

      double eval(const Event& event_) const{
        std::vector<double> parArray(formula.GetNpar());
        for( int iPar = 0 ; iPar < formula.GetNpar() ; iPar++ ){
          parArray[iPar] = event_.getVariables().getVarList()[varIndexList[iPar]].getVarAsDouble();
        }
        return formula.EvalPar(nullptr, parArray.empty() ? nullptr : parArray.data());
      }
    };

    struct RuntimeFormula{
      enum EvalBackend{
        Disabled,
        TreeBufferExpression,
        EventBufferFormula
      };

      EvalBackend backend{Disabled};
      const GenericToolbox::TreeBuffer::ExpressionBuffer* treeExpression{nullptr};
      EventFormula eventFormula{};

      bool isEnabled() const{ return backend != Disabled; }
      double eval(const Event& event_) const{
        if( backend == TreeBufferExpression ){ return treeExpression->getBuffer().getValueAsDouble(); }
        if( backend == EventBufferFormula ){ return eventFormula.eval(event_); }
        return 0;
      }
    };

    struct VariableDictBuffer{
      std::string name{};
      int outputVarIndex{-1};
      RuntimeFormula formula{};
      EventVarTransformLib transform{};
      bool isLibraryTransform{false};
    };

    int eventVarAsWeightIndex{-1};
    std::vector<const GenericToolbox::TreeBuffer::ExpressionBuffer*> varIndexingList{};
    std::vector<const GenericToolbox::TreeBuffer::ExpressionBuffer*> varStorageList{};

    RuntimeFormula nominalWeightFormula{};
    RuntimeFormula dialIndexFormula{};
    std::vector<RuntimeFormula> dialApplyConditionFormulaList{};
    std::vector<RuntimeFormula> sampleWeightFormulaList{};
    std::vector<VariableDictBuffer> variableDictEvalList{};

    static void storeTempIndex(const GenericToolbox::TreeBuffer::ExpressionBuffer*& var_, int idx_){
      if( idx_ == -1 ){
        var_ = reinterpret_cast<const GenericToolbox::TreeBuffer::ExpressionBuffer *>(static_cast<size_t>(-1));
        return;
      }
      var_ = reinterpret_cast<const GenericToolbox::TreeBuffer::ExpressionBuffer *>(static_cast<size_t>(idx_ + 1));
    }
    static void unfoldTempIndex(const GenericToolbox::TreeBuffer::ExpressionBuffer*& var_, const std::vector<std::shared_ptr<GenericToolbox::TreeBuffer::ExpressionBuffer>>& list_){
      auto encodedIndex = reinterpret_cast<size_t>(var_);
      if( encodedIndex == 0 or encodedIndex == static_cast<size_t>(-1) ){
        var_ = nullptr;
        return;
      }
      int idx = static_cast<int>(encodedIndex - 1);
      var_ = list_[idx].get();
    }
  };
  VariableBuffer buffer{};

  // thread communication
  GenericToolbox::Atomic<bool> requestReadNextEntry{false};
  GenericToolbox::Atomic<bool> isEntryBufferReady{false};
  GenericToolbox::Atomic<bool> isDoneReading{false};
  GenericToolbox::Atomic<bool> isEventFillerReady{false};
};



#endif //GUNDAM_DATA_DISPENSER_UTILS_H
