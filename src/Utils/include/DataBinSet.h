//
// Created by Nadrino on 19/05/2021.
//

#ifndef GUNDAM_DATABINSET_H
#define GUNDAM_DATABINSET_H

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
  void addBin(const DataBin& bin_);

  // Management
  void addBinContent(int binIndex_, double weight_);

  // Getters
  const std::vector<DataBin> &getBinsList() const;
  const std::string &getFilePath() const;
  const std::vector<std::string> &getBinVariables() const;

  std::vector<DataBin> &getBinsList();

  // Misc
  std::string getSummary() const;

  // Globals
  static void setVerbosity(int maxLogLevel_);

private:
  std::string _name_;
  std::string _filePath_;
  std::vector<DataBin> _binsList_{};
  std::vector<double> _binContent_{};
  std::vector<std::string> _binVariables_{};

};


#endif //GUNDAM_DATABINSET_H
