//
// Created by Nadrino on 19/05/2021.
//

#ifndef GUNDAM_DATABINSET_H
#define GUNDAM_DATABINSET_H


#include "DataBin.h"

#include <vector>
#include <string>


class DataBinSet {

public:
  // static
  static void setVerbosity(int maxLogLevel_);

public:
  DataBinSet() = default;

  // setter
  void setName(const std::string &name){ _name_ = name; }

  // const getters
  [[nodiscard]] const std::string &getFilePath() const { return _filePath_; }
  [[nodiscard]] const std::vector<DataBin> &getBinList() const { return _binList_; }

  // getters
  std::vector<DataBin> &getBinList() { return _binList_; }

  // core
  void readBinningDefinition(const JsonType& binning_);
  void checkBinning();
  [[nodiscard]] std::string getSummary() const;

  // utils
  void sortBinEdges();
  void sortBins();
  [[nodiscard]] std::vector<std::string> buildVariableNameList() const;

protected:
  void readTxtBinningDefinition();    // original txt
  void readBinningConfig(const JsonType& binning_); // yaml/json

private:
  std::string _name_;
  std::string _filePath_;
  std::vector<DataBin> _binList_{};

};


#endif //GUNDAM_DATABINSET_H
