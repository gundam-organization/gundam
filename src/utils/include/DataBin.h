//
// Created by Adrien BLANCHET on 19/05/2021.
//

#ifndef XSLLHFITTER_DATABIN_H
#define XSLLHFITTER_DATABIN_H

#include <utility>
#include "vector"
#include "string"

class DataBin {

public:
  DataBin();
  virtual ~DataBin();

  void reset();

  // Setters
  void setIsLowMemoryUsageMode(bool isLowMemoryUsageMode_);
  void setIsZeroWideRangesTolerated(bool isZeroWideRangesTolerated_); // make it explicit! -> double precision might not be enough if you play with long int

  // Getters
  bool isLowMemoryUsageMode() const;
  bool isZeroWideRangesTolerated() const;
  const std::vector<std::string> &getVariableNameList() const;
  const std::vector<std::pair<double, double>> &getEdgesList() const;

  // Management
  void addBinEdge(double lowEdge_, double highEdge_);
  void addBinEdge(const std::string& variableName_, double lowEdge_, double highEdge_);
  bool isInBin(const std::vector<double>& valuesList_);
  bool isBetweenEdges(const std::string& variableName_, double value_);

  // Misc
  bool isVariableSet(const std::string& variableName_);
  std::string generateSummary() const;

protected:
  bool isBetweenEdges(size_t varIndex_, double value_);

private:
  bool _isLowMemoryUsageMode_{false};
  bool _isZeroWideRangesTolerated_{false};
  std::vector<std::string> _variableNameList_{};
  std::vector<std::pair<double, double>> _edgesList_{};

};


#endif //XSLLHFITTER_DATABIN_H
