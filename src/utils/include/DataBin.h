//
// Created by Adrien BLANCHET on 19/05/2021.
//

#ifndef XSLLHFITTER_DATABIN_H
#define XSLLHFITTER_DATABIN_H

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

  // Init

  // Getters
  bool isLowMemoryUsageMode() const;
  bool isZeroWideRangesTolerated() const;
  const std::vector<std::string> &getVariableNameList() const;
  const std::vector<std::pair<double, double>> &getEdgesList() const;
  const std::string &getFormulaStr() const;
  const std::string &getTreeFormulaStr() const;
  TFormula *getFormula() const;
  TTreeFormula *getTreeFormula() const;

  // Management
  bool isInBin(const std::vector<double>& valuesList_) const;
  bool isBetweenEdges(const std::string& variableName_, double value_) const;

  // Misc
  bool isVariableSet(const std::string& variableName_) const;
  std::string getSummary() const;
  void generateFormula();
  void generateTreeFormula();

protected:
  std::string generateFormulaStr(bool varNamesAsTreeFormula_);
  bool isBetweenEdges(size_t varIndex_, double value_) const;

private:
  bool _isLowMemoryUsageMode_{false};
  bool _isZeroWideRangesTolerated_{false};
  std::vector<std::string> _variableNameList_{};
  std::vector<std::pair<double, double>> _edgesList_{};

  std::string _formulaStr_;
  std::shared_ptr<TFormula> _formula_;
  std::string _treeFormulaStr_;
  std::shared_ptr<TTreeFormula> _treeFormula_;

};

#endif //XSLLHFITTER_DATABIN_H
