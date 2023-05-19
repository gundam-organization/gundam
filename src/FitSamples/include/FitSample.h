//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_FITSAMPLE_H
#define GUNDAM_FITSAMPLE_H


#include "SampleElement.h"
#include "DataBinSet.h"
#include "JsonBaseClass.h"

#include "nlohmann/json.hpp"
#include <TH1D.h>
#include <TTreeFormula.h>

#include "vector"
#include "string"
#include "memory"


class FitSample : public JsonBaseClass {

public:
  // SETTERS
  void setName(const std::string &name);
  void setIndex(int index);
  void setLlhStatBuffer(double llhStatBuffer_) { _llhStatBuffer_ = llhStatBuffer_; }
  void setBinningFilePath(const std::string &binningFilePath_);
  void setSelectionCutStr(const std::string &selectionCutStr_);
  void setVarSelectionFormulaStr(const std::string &varSelectionFormulaStr_);
  void setEnabledDatasetList(const std::vector<std::string>& enabledDatasetList_);

  // GETTERS
  [[nodiscard]] bool isEnabled() const;
  [[nodiscard]] int getIndex() const;
  [[nodiscard]] double getLlhStatBuffer() const { return _llhStatBuffer_; }
  [[nodiscard]] const std::string &getName() const;
  [[nodiscard]] const std::string &getSelectionCutsStr() const;
  [[nodiscard]] const std::string &getVarSelectionFormulaStr() const;
  [[nodiscard]] const DataBinSet &getBinning() const;
  [[nodiscard]] const SampleElement &getMcContainer() const;
  [[nodiscard]] const SampleElement &getDataContainer() const;
  SampleElement &getMcContainer();
  SampleElement &getDataContainer();

  // Misc
  bool isDatasetValid(const std::string& datasetName_);

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

private:
  // Yaml
  bool _isEnabled_{false};
  int _index_{-1};
  double _mcNorm_{1};
  double _dataNorm_{1};
  std::string _name_;
  std::string _selectionCutStr_;
  std::string _varSelectionFormulaStr_;
  std::string _binningFilePath_;
  std::vector<std::string> _enabledDatasetList_;

  // Internals
  double _llhStatBuffer_{std::nan("unset")}; // set by FitSampleSet which hold the joinProbability obj
  DataBinSet _binning_;
  SampleElement _mcContainer_;
  SampleElement _dataContainer_;
  std::vector<size_t> _dataSetIndexList_;

};


#endif //GUNDAM_FITSAMPLE_H
