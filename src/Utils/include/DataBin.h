//
// Created by Nadrino on 19/05/2021.
//

#ifndef GUNDAM_DATABIN_H
#define GUNDAM_DATABIN_H


#include "ConfigUtils.h"

#include "GenericToolbox.Json.h"

#include <TFormula.h>
#include <TTreeFormula.h>

#include <utility>
#include <vector>
#include <string>

class DataBin {

public:
  struct Edges{
    Edges() = delete;
    explicit Edges(int index_) : index(index_) {}

    // utils
    void readConfig(const JsonType& config_);
    [[nodiscard]] bool isOverlapping(const Edges& other_) const;
    [[nodiscard]] std::string getSummary() const;


    bool isConditionVar{false};
    int index{-1};
    int varIndexCache{-1};
    std::string varName{};
    double min{std::nan("unset")};
    double max{std::nan("unset")};
  };

public:
  DataBin() = delete;
  explicit DataBin(int index_) : _index_(index_) {}
  virtual ~DataBin() = default;

  // setters
  void setIndex(int index_){ _index_ = index_; }
  void setIsLowMemoryUsageMode(bool isLowMemoryUsageMode_){ _isLowMemoryUsageMode_ = isLowMemoryUsageMode_; }
  void setIsZeroWideRangesTolerated(bool isZeroWideRangesTolerated_){ _isZeroWideRangesTolerated_ = isZeroWideRangesTolerated_; } // make it explicit! -> double precision might not be enough if you play with long int
  void addBinEdge(const std::string& variableName_, double lowEdge_, double highEdge_);

  // const getters
  [[nodiscard]] bool isLowMemoryUsageMode() const { return _isLowMemoryUsageMode_; }
  [[nodiscard]] bool isZeroWideRangesTolerated() const { return _isZeroWideRangesTolerated_; }
  [[nodiscard]] int getIndex() const { return _index_; }
  [[nodiscard]] const std::string &getFormulaStr() const { return _formulaStr_; }
  [[nodiscard]] const std::string &getTreeFormulaStr() const{ return _treeFormulaStr_; }
  [[nodiscard]] TFormula *getFormula() const{ return _formula_.get(); }
  [[nodiscard]] TTreeFormula *getTreeFormula() const{ return _treeFormula_.get(); }
  [[nodiscard]] const std::vector<Edges> &getEdgesList() const { return _binEdgesList_; }
  [[nodiscard]] const Edges& getVarEdges( const std::string& varName_ ) const;
  [[nodiscard]] const Edges* getVarEdgesPtr( const std::string& varName_ ) const;

  // non-const getters
  std::vector<Edges> &getEdgesList() { return _binEdgesList_; }

  // Non-native Getters
  [[nodiscard]] double getVolume() const;

  // Management
  [[nodiscard]] bool isOverlapping(const DataBin& other_) const;
  [[nodiscard]] bool isInBin(const std::vector<double>& valuesList_) const;
  [[nodiscard]] bool isVariableSet(const std::string& variableName_) const;
  [[nodiscard]] bool isBetweenEdges(const std::string& variableName_, double value_) const;
  [[nodiscard]] bool isBetweenEdges(size_t varIndex_, double value_) const;
  [[nodiscard]] bool isBetweenEdges(const Edges& edges_, double value_) const;
  [[nodiscard]] std::vector<std::string> buildVariableNameList() const;

  // Misc
  void generateFormula();
  void generateTreeFormula();
  void readConfig( const JsonType& config_);
  [[nodiscard]] std::string getSummary() const;
  [[nodiscard]] std::vector<double> generateBinTarget(const std::vector<std::string>& varNameList_ = {}) const;

protected:
  std::string generateFormulaStr(bool varNamesAsTreeFormula_);

private:
  bool _isLowMemoryUsageMode_{false};
  bool _isZeroWideRangesTolerated_{false};
  bool _includeLowerBoundVal_{true}; // by default it's [a,b[
  bool _includeHigherBoundVal_{false};
  int _index_{-1};
  std::vector<Edges> _binEdgesList_{};

  std::string _formulaStr_{};
  std::string _treeFormulaStr_{};
  std::shared_ptr<TFormula> _formula_{nullptr};
  std::shared_ptr<TTreeFormula> _treeFormula_{nullptr};

};

#endif //GUNDAM_DATABIN_H
