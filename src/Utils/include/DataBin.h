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
  void setIsLowMemoryUsageMode(bool isLowMemoryUsageMode_);
  void setIsZeroWideRangesTolerated(bool isZeroWideRangesTolerated_); // make it explicit! -> double precision might not be enough if you play with long int
  void addBinEdge(double lowEdge_, double highEdge_);
  void addBinEdge(const std::string& variableName_, double lowEdge_, double highEdge_);

  void setEventVarIndexCache(const std::vector<int> &eventVarIndexCache);

  // Init

  // Getters
  [[nodiscard]] bool isLowMemoryUsageMode() const;
  [[nodiscard]] bool isZeroWideRangesTolerated() const;
  [[nodiscard]] const std::vector<std::string> &getVariableNameList() const;
  [[nodiscard]] const std::vector<std::pair<double, double>> &getEdgesList() const;
  [[nodiscard]] const std::pair<double, double>& getVarEdges( const std::string& varName_ ) const;
  [[nodiscard]] const std::string &getFormulaStr() const;
  [[nodiscard]] const std::string &getTreeFormulaStr() const;
  [[nodiscard]] TFormula *getFormula() const;
  [[nodiscard]] TTreeFormula *getTreeFormula() const;
  [[nodiscard]] const std::vector<int> &getEventVarIndexCache() const;

  // Non-native Getters
  [[nodiscard]] size_t getNbEdges() const;
  [[nodiscard]] double getVolume() const;

  // Management
  [[nodiscard]] bool isInBin(const std::vector<double>& valuesList_) const;
  [[nodiscard]] bool isBetweenEdges(const std::string& variableName_, double value_) const;
  [[nodiscard]] bool isBetweenEdges(size_t varIndex_, double value_) const;
  [[nodiscard]] bool isVariableSet(const std::string& variableName_) const;

  // Misc
  void generateFormula();
  void generateTreeFormula();
  [[nodiscard]] std::string getSummary() const;
  [[nodiscard]] std::vector<double> generateBinTarget(const std::vector<std::string>& varNameList_ = {}) const;

  // Static
  static bool isBetweenEdges(const std::pair<double,double>& edges_, double value_);

protected:
  std::string generateFormulaStr(bool varNamesAsTreeFormula_);

private:
  bool _isLowMemoryUsageMode_{false};
  bool _isZeroWideRangesTolerated_{false};
  std::vector<std::string> _variableNameList_{};
  std::vector<std::pair<double, double>> _edgesList_{};

  std::string _formulaStr_{};
  std::string _treeFormulaStr_{};
  std::shared_ptr<TFormula> _formula_{nullptr};
  std::shared_ptr<TTreeFormula> _treeFormula_{nullptr};

  std::vector<int> _eventVarIndexCache_{};

};

#endif //GUNDAM_DATABIN_H
