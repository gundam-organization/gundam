//
// Created by Adrien BLANCHET on 19/05/2021.
//

#ifndef XSLLHFITTER_DATABINSET_H
#define XSLLHFITTER_DATABINSET_H

#include "vector"
#include "string"

#include "DataBin.h"

class DataBinSet {

public:
  DataBinSet();
  virtual ~DataBinSet();

  void reset();

  // Setter
  void setName(const std::string &name);
  void readBinningDefinition(const std::string& filePath_);

  // Management
  void addBinContent(int binIndex_, double weight_);

  // Misc
  std::string generateSummary() const;

private:
  std::string _name_;
  std::vector<DataBin> _binsList_{};
  std::vector<double> _binContent_{};

};


#endif //XSLLHFITTER_DATABINSET_H
