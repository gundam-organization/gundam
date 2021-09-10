//
// Created by Nadrino on 11/06/2021.
//

#ifndef XSLLHFITTER_PROPAGATOR_H
#define XSLLHFITTER_PROPAGATOR_H

#include "vector"
#include "future"

#include "json.hpp"

#include "FitParameterSet.h"
#include "AnaSample.hh"
#include "PlotGenerator.h"
#include "FitSampleSet.h"

class Propagator {

public:
  Propagator();
  virtual ~Propagator();

  // Initialize
  void reset();

  // Setters
  void setShowTimeStats(bool showTimeStats);
  void setSaveDir(TDirectory *saveDir);
  void setConfig(const json &config);

  //! TODO: GET RID OF THOSE TEST METHOD
  void setDataTree(TTree *dataTree_);
  void setMcFilePath(const std::string &mcFilePath);

  // Init
  void initialize();

  // Getters
  bool isUseResponseFunctions() const;
  FitSampleSet &getFitSampleSet();
  std::vector<AnaSample> &getSamplesList();
  std::vector<FitParameterSet> &getParameterSetsList();
  PlotGenerator &getPlotGenerator();
  const json &getConfig() const;

  // Core
  void propagateParametersOnSamples();
  void reweightSampleEvents();
  void refillSampleHistograms();
  void applyResponseFunctions();

  // Switches
  void preventRfPropagation();
  void allowRfPropagation();

  // Monitor

protected:
  void initializeThreads();
  void initializeCaches();

  void fillEventDialCaches();
  void makeResponseFunctions();

  // multi-threaded
  void reweightSampleEvents(int iThread_);
  void fillEventDialCaches(int iThread_);
  void applyResponseFunctions(int iThread_);

private:
  // Parameters
  bool _showTimeStats_{false};
  TDirectory* _saveDir_{nullptr};
  nlohmann::json _config_;

  // Internals
  bool _isInitialized_{false};
  bool _useResponseFunctions_{false};
  bool _isRfPropagationEnabled_{false};
  std::vector<FitParameterSet> _parameterSetsList_;
  std::vector<AnaSample> _samplesList_;
  FitSampleSet _fitSampleSet_;
  PlotGenerator _plotGenerator_;

  // Response functions
  std::map<FitSample*, std::shared_ptr<TH1D>> _nominalSamplesMcHistogram_;
  std::map<FitSample*, std::vector<std::shared_ptr<TH1D>>> _responseFunctionsSamplesMcHistogram_;

  // TEST
  TTree* dataTree{nullptr};
  std::string mc_file_path;

public:
  struct CycleTimer{
    long long counts{0};
    long long cumulated{0};
    friend std::ostream& operator<< (std::ostream& stream, const CycleTimer& timer_) {
      stream << GenericToolbox::parseTimeUnit(timer_.cumulated / timer_.counts);
      return stream;
    }
  };

  CycleTimer weightProp;
  CycleTimer fillProp;
  CycleTimer applyRf;

  long long nbWeightProp = 0;
  long long cumulatedWeightPropTime = 0;

};


#endif //XSLLHFITTER_PROPAGATOR_H
