//
// Created by Nadrino on 19/05/2021.
//

#ifndef GUNDAM_BINSET_H
#define GUNDAM_BINSET_H

#include "Bin.h"
#include "ConfigUtils.h"

#include <vector>
#include <string>


class BinSet : public JsonBaseClass {

public:
  // static
  static void setVerbosity(int maxLogLevel_);

protected:
  void configureImpl() override;
  void initializeImpl() override;

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
  void checkBinning() const;
  [[nodiscard]] std::string getSummary() const;

  // utils
  void sortBinEdges();
  [[nodiscard]] std::vector<std::string> buildVariableNameList() const;

  // careful with this one: the ordering might refer to a dial list... So don't sort in that case
  void sortBins();

protected:
  void readTxtBinningDefinition();    // original txt
  void readBinningConfig(const ConfigReader& binning_); // yaml/json

private:
  bool _sortBins_{false};
  std::string _name_;
  std::string _filePath_;
  std::vector<Bin> _binList_{};

};


#endif //GUNDAM_BINSET_H
