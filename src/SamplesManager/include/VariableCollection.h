//
// Created by Nadrino on 27/09/2024.
//

#ifndef GUNDAM_VARIABLE_COLLECTION_H
#define GUNDAM_VARIABLE_COLLECTION_H

#include "VariableHolder.h"
#include "DataBinSet.h"

#include "GenericToolbox.Utils.h" // Any objects

#include <memory>
#include <vector>
#include <string>


class VariableCollection{
  /*
   * This class aims at keeping together a list of variables and their associated name.
   * Although on paper this could be trivially achieved by keeping together a variable with
   * its associated name, we don't want to keep duplicates of name lists which
   * takes a lot of space in RAM.
   *
   * */

public:
  VariableCollection() = default;

  // setters
  void setVarNameList(const std::shared_ptr<std::vector<std::string>>& nameListPtr_);

  // const-getters
  [[nodiscard]] const std::shared_ptr<std::vector<std::string>>& getNameListPtr() const{ return _nameListPtr_; }
  [[nodiscard]] const std::vector<VariableHolder>& getVarList() const{ return _varList_; }

  // mutable-getters
  std::vector<VariableHolder>& getVarList(){ return _varList_; }

  // memory
  void allocateMemory(const std::vector<const GenericToolbox::LeafForm*>& leafFormList_);
  void copyData(const std::vector<const GenericToolbox::LeafForm*>& leafFormList_);

  // fetch
  [[nodiscard]] int findVarIndex( const std::string& leafName_, bool throwIfNotFound_ = true) const;
  [[nodiscard]] const VariableHolder& fetchVariable( const std::string& name_) const;
  VariableHolder& fetchVariable( const std::string& name_);

  // bin tools
  [[nodiscard]] bool isInBin(const DataBin& bin_) const;
  [[nodiscard]] int findBinIndex(const std::vector<DataBin>& binList_) const;
  [[nodiscard]] int findBinIndex(const DataBinSet& binSet_) const;

  // formula
  [[nodiscard]] double evalFormula(const TFormula* formulaPtr_, std::vector<int>* indexDict_ = nullptr) const;

  // printouts
  [[nodiscard]] std::string getSummary() const;
  friend std::ostream& operator <<( std::ostream& o, const VariableCollection& this_ ){ o << this_.getSummary(); return o; }

private:
  std::vector<VariableHolder> _varList_{};
  // keep only one list of name in memory -> shared_ptr is used to make sure it gets properly deleted
  std::shared_ptr<std::vector<std::string>> _nameListPtr_{nullptr};

};


#endif //GUNDAM_VARIABLE_COLLECTION_H
