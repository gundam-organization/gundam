//
// Created by Nadrino on 06/03/2024.
//

#ifndef GUNDAM_EVENT_UTILS_H
#define GUNDAM_EVENT_UTILS_H

#include "DataBin.h"
#include "DataBinSet.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Root.h"

#include <iostream>
#include <string>


namespace EventUtils{

  struct Indices{

    // source
    int dataset{-1}; // which DatasetDefinition?
    Long64_t entry{-1}; // which entry of the TChain?

    // destination
    int sample{-1}; // this information is lost in the EventDialCache manager
    int bin{-1}; // which bin of the sample?

    [[nodiscard]] std::string getSummary() const;
    friend std::ostream& operator <<( std::ostream& o, const Indices& this_ ){ o << this_.getSummary(); return o; }
  };

  struct Weights{
    double base{1};
    double current{1};

    void resetCurrentWeight(){ current = base; }
    [[nodiscard]] std::string getSummary() const;
    friend std::ostream& operator <<( std::ostream& o, const Weights& this_ ){ o << this_.getSummary(); return o; }
  };

  class Variables{

  public:
    class Variable{

    public:
      Variable() = default;

      template<typename T>void set(const T& value_){ var = value_; updateCache(); }
      void set(const GenericToolbox::LeafForm& leafForm_);
      [[nodiscard]] const GenericToolbox::AnyType& get() const { return var; }
      [[nodiscard]] double getVarAsDouble() const { return cache; }

      GenericToolbox::AnyType& get(){ return var; }

    protected:
      // user should not have to worry about the cache
      void updateCache(){ cache = var.getValueAsDouble(); }

    private:
      GenericToolbox::AnyType var{};
      double cache{std::nan("unset")};

    };

  public:
    Variables() = default;

    // setters
    void setVarNameList(const std::shared_ptr<std::vector<std::string>>& nameListPtr_);

    // const-getters
    [[nodiscard]] const std::shared_ptr<std::vector<std::string>>& getNameListPtr() const{ return _nameListPtr_; }
    [[nodiscard]] const std::vector<Variable>& getVarList() const{ return _varList_; }

    // mutable-getters
    std::vector<Variable>& getVarList(){ return _varList_; }

    // memory
    void allocateMemory( const std::vector<const GenericToolbox::LeafForm*>& leafFormList_);
    void copyData( const std::vector<const GenericToolbox::LeafForm*>& leafFormList_);

    // fetch
    [[nodiscard]] int findVarIndex( const std::string& leafName_, bool throwIfNotFound_ = true) const;
    [[nodiscard]] const Variable& fetchVariable(const std::string& name_) const;
    Variable& fetchVariable(const std::string& name_);

    // bin tools
    [[nodiscard]] bool isInBin(const DataBin& bin_) const;
    [[nodiscard]] int findBinIndex(const std::vector<DataBin>& binList_) const;
    [[nodiscard]] int findBinIndex(const DataBinSet& binSet_) const;

    // formula
    [[nodiscard]] double evalFormula(const TFormula* formulaPtr_, std::vector<int>* indexDict_ = nullptr) const;

    // printouts
    [[nodiscard]] std::string getSummary() const;
    friend std::ostream& operator <<( std::ostream& o, const Variables& this_ ){ o << this_.getSummary(); return o; }

  private:
    std::vector<Variable> _varList_{};
    // keep only one list of name in memory -> shared_ptr is used to make sure it gets properly deleted
    std::shared_ptr<std::vector<std::string>> _nameListPtr_{nullptr};

  };

#ifdef GUNDAM_USING_CACHE_MANAGER
  struct Cache{
    // An "opaque" index into the cache that is used to simplify bookkeeping.
    // This is actually the index of the result in the buffer filled by the
    // Cache::Manager (i.e. the index on the GPU).
    int index{-1};
    // A pointer to the cached result.  Will not be a nullptr if the cache is
    // initialized and should be checked before calling getWeight.  If it is a
    // nullptr, then that means the cache must not be used.
    const double* valuePtr{nullptr};
    // A pointer to the cache validity flag.  When not nullptr it will point
    // to the flag for if the cache needs to be updated.  This is only
    // valid when valuePtr is not null.
    const bool* isValidPtr{nullptr};
    // A pointer to a callback to force the cache to be updated.  This will
    // force the value to be copied from the GPU to the host (if necessary).
    void (*updateCallbackPtr)(){nullptr};
    // Safely update the value.  The value may not be valid after
    // this call.
    void update() const;
    // Check if there is a valid value.  The update might not provide a valid
    // value for the cache, so THIS. MUST. BE. CHECKED.
    [[nodiscard]] bool valid() const;
    // Get the current value of the weight.  Only valid if valid() returned
    // true.
    [[nodiscard]] double getWeight() const;
  };
#endif

}


#endif //GUNDAM_EVENT_UTILS_H
