//
// Created by Nadrino on 19/05/2021.
//

#ifndef GUNDAM_DATABIN_H
#define GUNDAM_DATABIN_H

#include <utility>
#include <vector>
#include <string>

#include <TFormula.h>
#include <TTreeFormula.h>


class DataBin {

public:
  DataBin() = default;
  virtual ~DataBin() = default;

  // Setters
  void setIsLowMemoryUsageMode(bool isLowMemoryUsageMode_){ _isLowMemoryUsageMode_ = isLowMemoryUsageMode_; }
  void setIsZeroWideRangesTolerated(bool isZeroWideRangesTolerated_){ _isZeroWideRangesTolerated_ = isZeroWideRangesTolerated_; } // make it explicit! -> double precision might not be enough if you play with long int
  void setEventVarIndexCache(const std::vector<int> &eventVarIndexCache){ _eventVarIndexCache_ = eventVarIndexCache; }
  void addBinEdge(double lowEdge_, double highEdge_);
  void addBinEdge(const std::string& variableName_, double lowEdge_, double highEdge_);


  // Init

  // Getters
  [[nodiscard]] bool isLowMemoryUsageMode() const { return _isLowMemoryUsageMode_; }
  [[nodiscard]] bool isZeroWideRangesTolerated() const { return _isZeroWideRangesTolerated_; }
  [[nodiscard]] const std::string &getFormulaStr() const { return _formulaStr_; }
  [[nodiscard]] const std::string &getTreeFormulaStr() const{ return _treeFormulaStr_; }
  [[nodiscard]] TFormula *getFormula() const{ return _formula_.get(); }
  [[nodiscard]] TTreeFormula *getTreeFormula() const{ return _treeFormula_.get(); }
  [[nodiscard]] const std::vector<int> &getEventVarIndexCache() const{ return _eventVarIndexCache_; }
  [[nodiscard]] const std::vector<std::string> &getVariableNameList() const { return _variableNameList_; }
  [[nodiscard]] const std::vector<std::pair<double, double>> &getEdgesList() const { return _edgesList_; }
  [[nodiscard]] const std::pair<double, double>& getVarEdges( const std::string& varName_ ) const;

  // Non-native Getters
  [[nodiscard]] size_t getNbEdges() const{ return _edgesList_.size(); }
  [[nodiscard]] double getVolume() const;

  // Management
  [[nodiscard]] bool isInBin(const std::vector<double>& valuesList_) const;
  [[nodiscard]] bool isVariableSet(const std::string& variableName_) const;
  [[nodiscard]] bool isBetweenEdges(const std::string& variableName_, double value_) const;
  [[nodiscard]] bool isBetweenEdges(size_t varIndex_, double value_) const;
  [[nodiscard]] bool isBetweenEdges(const std::pair<double,double>& edges_, double value_) const;

  // Misc
  void generateFormula();
  void generateTreeFormula();
  [[nodiscard]] std::string getSummary() const;
  [[nodiscard]] std::vector<double> generateBinTarget(const std::vector<std::string>& varNameList_ = {}) const;

protected:
  std::string generateFormulaStr(bool varNamesAsTreeFormula_);

private:
  bool _isLowMemoryUsageMode_{false};
  bool _isZeroWideRangesTolerated_{false};
  bool _includeLowerBoundVal_{true}; // by default it's [a,b[
  bool _includeHigherBoundVal_{false};
  std::vector<std::string> _variableNameList_{};
  std::vector<std::pair<double, double>> _edgesList_{};

  std::string _formulaStr_{};
  std::string _treeFormulaStr_{};
  std::shared_ptr<TFormula> _formula_{nullptr};
  std::shared_ptr<TTreeFormula> _treeFormula_{nullptr};

  std::vector<int> _eventVarIndexCache_{};

};

#endif //GUNDAM_DATABIN_H
