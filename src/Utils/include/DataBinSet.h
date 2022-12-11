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

  // Getters
  [[nodiscard]] const std::vector<DataBin> &getBinsList() const;
  [[nodiscard]] const std::string &getFilePath() const;
  [[nodiscard]] const std::vector<std::string> &getBinVariables() const;
  std::vector<DataBin> &getBinsList();

  // Misc
  bool isEmpty() const;
  [[nodiscard]] std::string getSummary() const;
  void addBinContent(int binIndex_, double weight_);

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
