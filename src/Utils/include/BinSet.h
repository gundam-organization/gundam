//
// Created by Nadrino on 19/05/2021.
//

#ifndef GUNDAM_BINSET_H
#define GUNDAM_BINSET_H

#include "Bin.h"

#include <vector>
#include <string>


class BinSet {

public:
  // static
  static void setVerbosity(int maxLogLevel_);

public:
  BinSet() = default;

  // setter
  void setName(const std::string &name){ _name_ = name; }

  // const getters
  [[nodiscard]] const std::string &getFilePath() const { return _filePath_; }
  [[nodiscard]] const std::vector<Bin> &getBinList() const { return _binList_; }

  // getters
  std::vector<Bin> &getBinList() { return _binList_; }

  // core
  void readBinningDefinition(const JsonType& binning_);
  void checkBinning();
  [[nodiscard]] std::string getSummary() const;

  // utils
  void sortBinEdges();
  [[nodiscard]] std::vector<std::string> buildVariableNameList() const;

  // caused a lot of problem
  [[deprecated("don't sort bins. let the user ordering.")]] void sortBins();

protected:
  void readTxtBinningDefinition();    // original txt
  void readBinningConfig(const JsonType& binning_); // yaml/json

private:
  std::string _name_;
  std::string _filePath_;
  std::vector<Bin> _binList_{};

};


#endif //GUNDAM_BINSET_H
