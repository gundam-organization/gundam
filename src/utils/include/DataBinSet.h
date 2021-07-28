//
// Created by Nadrino on 19/05/2021.
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

  const std::vector<DataBin> &getBinsList() const;
  const std::string &getFilePath() const;

  // Misc
  std::string getSummary() const;

  // Globals
  static void setVerbosity(int maxLogLevel_);

private:
  std::string _name_;
  std::string _filePath_;
  std::vector<DataBin> _binsList_{};
  std::vector<double> _binContent_{};

};


#endif //XSLLHFITTER_DATABINSET_H
