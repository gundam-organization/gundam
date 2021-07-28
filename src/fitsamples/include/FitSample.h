//
// Created by Nadrino on 22/07/2021.
//

#ifndef XSLLHFITTER_FITSAMPLE_H
#define XSLLHFITTER_FITSAMPLE_H

#include "vector"
#include "string"
#include "memory"
#include <TH1D.h>
#include <TTreeFormula.h>
#include "json.hpp"

#include "DataSet.h"
#include "PhysicsEvent.h"
#include "DataBinSet.h"

class FitSample {

public:
  FitSample();
  virtual ~FitSample();

  void reset();

  void setConfig(const nlohmann::json &config_);

  void initialize();

  bool isEnabled() const;

  const std::string &getName() const;
  const std::string &getSelectionCutsStr() const;
  std::vector<PhysicsEvent> &getMcEventList();
  std::vector<PhysicsEvent> &getDataEventList();

  bool isDataSetValid(const std::string& dataSetName_);


private:
  nlohmann::json _config_;

  // internals
  bool _isEnabled_{false};
  std::string _name_;

  DataBinSet _binning_;

  std::string _selectionCuts_;
  std::vector<std::string> _selectedDataSets_;

  double _mcNorm_{1};
  std::vector<PhysicsEvent> _mcEventList_;
  std::shared_ptr<TH1D> _mcHistogram_{nullptr};

  double _dataNorm_{1};
  std::vector<PhysicsEvent> _dataEventList_;
  std::shared_ptr<TH1D> _dataHistogram_{nullptr};

};


#endif //XSLLHFITTER_FITSAMPLE_H
