//
// Created by Nadrino on 19/05/2021.
//

#ifndef GUNDAM_DATABIN_H
#define GUNDAM_DATABIN_H

#include <utility>
#include "vector"
#include "string"

#include <TFormula.h>
#include <TTreeFormula.h>


class DataBin {

public:
  DataBin();
  virtual ~DataBin();

  void reset();

  // Setters
  void setIsLowMemoryUsageMode(bool isLowMemoryUsageMode_);
  void setIsZeroWideRangesTolerated(bool isZeroWideRangesTolerated_); // make it explicit! -> double precision might not be enough if you play with long int
  void addBinEdge(double lowEdge_, double highEdge_);
  void addBinEdge(const std::string& variableName_, double lowEdge_, double highEdge_);

  void setEventVarIndexCache(const std::vector<int> &eventVarIndexCache);

  // Init

  // Getters
  bool isLowMemoryUsageMode() const;
  bool isZeroWideRangesTolerated() const;
  const std::vector<std::string> &getVariableNameList() const;
  const std::vector<std::pair<double, double>> &getEdgesList() const;
  const std::pair<double, double>& getVarEdges( const std::string& varName_ ) const;
  const std::string &getFormulaStr() const;
  const std::string &getTreeFormulaStr() const;
  TFormula *getFormula() const;
  TTreeFormula *getTreeFormula() const;
  const std::vector<int> &getEventVarIndexCache() const;

  // Non-native Getters
  size_t getNbEdges() const;

  // Management
  bool isInBin(const std::vector<double>& valuesList_) const;
  bool isBetweenEdges(const std::string& variableName_, double value_) const;
  bool isBetweenEdges(size_t varIndex_, double value_) const;

  // Misc
  bool isVariableSet(const std::string& variableName_) const;
  std::string getSummary() const;
  void generateFormula();
  void generateTreeFormula();

  // Static
  static bool isBetweenEdges(const std::pair<double,double>& edges_, double value_);

protected:
  std::string generateFormulaStr(bool varNamesAsTreeFormula_);

private:
  bool _isLowMemoryUsageMode_{false};
  bool _isZeroWideRangesTolerated_{false};
  std::vector<std::string> _variableNameList_{};
  std::vector<std::pair<double, double>> _edgesList_{};

  std::string _formulaStr_;
  std::shared_ptr<TFormula> _formula_;
  std::string _treeFormulaStr_;
  std::shared_ptr<TTreeFormula> _treeFormula_;

  std::vector<int> _eventVarIndexCache_{};

};

#endif //GUNDAM_DATABIN_H
