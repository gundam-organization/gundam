//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_EVENT_H
#define GUNDAM_EVENT_H

#include "ParameterSet.h"
#include "DataBinSet.h"
#include "DataBin.h"

#include "GenericToolbox.Root.h"
#include "GenericToolbox.Utils.h"

#include "TTree.h"
#include "TFormula.h"

#include <map>
#include <mutex>
#include <vector>
#include <string>
#include <sstream>

class Event{

public:
  struct Indices{
    int dataset{-1}; // which DatasetDefinition?
    Long64_t entry{-1}; // which entry of the TChain?
    int sample{-1}; // this information is lost in the EventDialCache manager
    int bin{-1}; // which bin of the sample?

    [[nodiscard]] std::string getSummary() const{
      std::stringstream ss;
      ss << "dataset(" << dataset << ")";
      ss << ", " << "entry(" << entry << ")";
      ss << ", " << "sample(" << sample << ")";
      ss << ", " << "bin(" << bin << ")";
      return ss.str();
    }
    friend std::ostream& operator <<( std::ostream& o, const Indices& this_ ){ o << this_.getSummary(); return o; }
  };

public:
  Event() = default;

  // setters
  void setBaseWeight(double baseWeight_){ _baseWeight_ = baseWeight_; }
  void setEventWeight(double eventWeight){ _eventWeight_ = eventWeight; }
  void setNominalWeight(double nominalWeight){ _nominalWeight_ = nominalWeight; }
  void setCommonVarNameListPtr(const std::shared_ptr<std::vector<std::string>>& commonVarNameListPtr_);
  template<typename T> void setVariable(const T& value_, const std::string& leafName_, size_t arrayIndex_ = 0);

  // const getters
  double getBaseWeight() const { return _baseWeight_; }
  double getNominalWeight() const { return _nominalWeight_; }
  double getEventWeight() const;
  const Indices& getIndices() const{ return _indices_; }
  const GenericToolbox::AnyType& getVar(int varIndex_, size_t arrayIndex_ = 0) const { return _varHolderList_[varIndex_][arrayIndex_]; }
  const std::vector<GenericToolbox::AnyType>& getVarHolder(int index_) const { return _varHolderList_[index_]; }
  const std::vector<GenericToolbox::AnyType>& getVarHolder(const std::string &leafName_) const;
  const std::vector<std::vector<GenericToolbox::AnyType>> &getVarHolderList() const { return _varHolderList_; }
  const std::shared_ptr<std::vector<std::string>>& getCommonVarNameListPtr() const { return _commonVarNameListPtr_; }
  const GenericToolbox::AnyType& getVariableAsAnyType(const std::string& leafName_, size_t arrayIndex_ = 0) const;
  template<typename T> auto getVarValue(const std::string& leafName_, size_t arrayIndex_ = 0) const -> T;
  template<typename T> auto getVariable(const std::string& leafName_, size_t arrayIndex_ = 0) const -> const T&;

  // mutable getters
  double& getEventWeightRef(){ return _eventWeight_; }
  void* getVariableAddress(const std::string& leafName_, size_t arrayIndex_ = 0);
  Indices& getIndices(){ return _indices_; }
  std::vector<std::vector<GenericToolbox::AnyType>> &getVarHolderList(){ return _varHolderList_; }
  GenericToolbox::AnyType& getVariableAsAnyType(const std::string& leafName_, size_t arrayIndex_ = 0);

  // core
  void resetEventWeight(){ _eventWeight_ = _baseWeight_; }
  void resizeVarToDoubleCache();
  void invalidateVarToDoubleCache();
  void copyData(const std::vector<const GenericToolbox::LeafForm*>& leafFormList_);
  void allocateMemory(const std::vector<const GenericToolbox::LeafForm*>& leafFormList_);
  bool isInBin(const DataBin& bin_) const;
  int findBinIndex(const DataBinSet& binSet_) const;
  int findBinIndex(const std::vector<DataBin>& binSet_) const;
  int findVarIndex(const std::string& leafName_, bool throwIfNotFound_ = true) const;
  double getVarAsDouble(int varIndex_, size_t arrayIndex_ = 0) const;
  double getVarAsDouble(const std::string& leafName_, size_t arrayIndex_ = 0) const;
  double evalFormula(const TFormula* formulaPtr_, std::vector<int>* indexDict_ = nullptr) const;

  // misc
  void print() const;
  std::string getSummary() const;
  void copyVarHolderList(const Event& ref_);
  void copyOnlyExistingVarHolders(const Event& other_);
  void fillBuffer(const std::vector<int>& indexList_, std::vector<double>& buffer_) const;
  void fillBinIndex(const DataBinSet& binSet_){ _indices_.bin = findBinIndex(binSet_); }

  // operators
  friend std::ostream& operator <<( std::ostream& o, const Event& p );

private:
  // internals
  Indices _indices_;
  double _baseWeight_{1};
  double _nominalWeight_{1};
  double _eventWeight_{1};

  // Data storage variables
  std::shared_ptr<std::vector<std::string>> _commonVarNameListPtr_{nullptr};
  std::vector<std::vector<GenericToolbox::AnyType>> _varHolderList_{};
  mutable std::vector<std::vector<double>> _varToDoubleCache_{};

#ifdef GUNDAM_USING_CACHE_MANAGER
public:
  void setCacheManagerIndex(int i) { _cacheManagerIndex_ = i;}
  void setCacheManagerValuePointer(const double* v) { _cacheManagerValue_ = v;}
  void setCacheManagerValidPointer(const bool* v) { _cacheManagerValid_ = v;}
  void setCacheManagerUpdatePointer(void (*p)()) { _cacheManagerUpdate_ = p;}

  int getCacheManagerIndex() const {return _cacheManagerIndex_;}
private:
  // An "opaque" index into the cache that is used to simplify bookkeeping.
  int _cacheManagerIndex_{-1};
  // A pointer to the cached result.
  const double* _cacheManagerValue_{nullptr};
  // A pointer to the cache validity flag.
  const bool* _cacheManagerValid_{nullptr};
  // A pointer to a callback to force the cache to be updated.
  void (*_cacheManagerUpdate_)(){nullptr};
#endif

};


// TEMPLATES IMPLEMENTATION
template<typename T> auto Event::getVarValue( const std::string &leafName_, size_t arrayIndex_) const -> T {
  return this->getVariable<T>(leafName_, arrayIndex_);
}
template<typename T> auto Event::getVariable( const std::string& leafName_, size_t arrayIndex_) const -> const T&{
  return this->getVariableAsAnyType(leafName_, arrayIndex_).template getValue<T>();
}
template<typename T> void Event::setVariable( const T& value_, const std::string& leafName_, size_t arrayIndex_){
  int index = this->findVarIndex(leafName_, true);
  _varHolderList_[index][arrayIndex_].template getValue<T>() = value_;
  if( not _varToDoubleCache_.empty() ){ _varToDoubleCache_[index][arrayIndex_] = std::nan("unset"); }
}


#endif //GUNDAM_EVENT_H
