//
// Created by Nadrino on 25/06/2025.
//

#ifndef CROSSSECTIONCALCULATOR_H
#define CROSSSECTIONCALCULATOR_H

#include "CrossSectionHistogramData.h"

#include "FitterEngine.h"

#include "ConfigUtils.h"

#include <string>


class CrossSectionCalculator : public JsonBaseClass {

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  void setFitterRootFilePath(const std::string& fitterFilePath_){ _fitterFilePath_ = fitterFilePath_; }
  void setUsePrefit(bool usePrefit_){ _usePrefit_ = usePrefit_; }
  void setUseBestFitAsCentralValue(bool useBestFitAsCentralValue_){ _useBestFitAsCentralValue_ = useBestFitAsCentralValue_; }

  const auto& getOutputFolder(){ return _outputFolder_; }

  void throwToys(int nToys_);

private:
  bool _useBestFitAsCentralValue_{false};
  bool _enableStatThrowInToys_{false};
  bool _enableEventMcThrow_{false};
  bool _usePrefit_{false};
  std::string _outputFolder_{};
  std::string _fitterFilePath_{};

  std::unique_ptr<TFile> fitterRootFile{nullptr};
  JsonType _fitterEngineConfig_;
  FitterEngine _fitterEngine_{nullptr};
  GenericToolbox::TFilePath _savePath_{nullptr};

  std::vector<CrossSectionHistogramData> crossSectionDataList{};

};



#endif //CROSSSECTIONCALCULATOR_H
