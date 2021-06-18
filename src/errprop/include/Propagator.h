//
// Created by Adrien BLANCHET on 11/06/2021.
//

#ifndef XSLLHFITTER_PROPAGATOR_H
#define XSLLHFITTER_PROPAGATOR_H

#include "vector"
#include "future"

#include "json.hpp"

#include "FitParameterSet.h"
#include "AnaSample.hh"
#include "PlotGenerator.h"

class Propagator {

public:
  Propagator();
  virtual ~Propagator();

  // Initialize
  void reset();

  // Setters
  void setParameterSetConfig(const json &parameterSetConfig);
  void setSamplesConfig(const json &samplesConfig);

  void setSamplePlotGeneratorConfig(const json &samplePlotGeneratorConfig);

  // test
  void setDataTree(TTree *dataTree_);

  void setMcFilePath(const std::string &mcFilePath);

  // Init
  void initialize();

  // Getters
  const std::vector<FitParameterSet> &getParameterSetsList() const;

  // Core
  void propagateParametersOnSamples();
  void fillSampleHistograms();

protected:
  void initializeThreads();
  void initializeCaches();

  // multi-threaded
  void fillEventDialCaches();

  void propagateParametersOnSamples(int iThread_);
  void fillEventDialCaches(int iThread_);

private:
  nlohmann::json _parameterSetsConfig_;
  nlohmann::json _samplesConfig_;
  nlohmann::json _samplePlotGeneratorConfig_;

  // Internals
  bool _isInitialized_{false};
  std::vector<FitParameterSet> _parameterSetsList_;
  std::vector<AnaSample> _samplesList_;
  PlotGenerator _plotGenerator_;

  // Threads
  std::vector<std::future<void>> _threadsList_;
  int _nbThreads_{1};
  std::mutex _propagatorMutex_;
  bool _stopThreads_{false};

  struct ThreadTriggers{
    bool fillSampleHistograms{false};
    bool propagateOnSampleEvents{false};
    bool fillDialCaches{false};
  };
  std::vector<ThreadTriggers> _threadTriggersList_;



  // TEST
  TTree* dataTree{nullptr};
  std::string mc_file_path;

};


#endif //XSLLHFITTER_PROPAGATOR_H
