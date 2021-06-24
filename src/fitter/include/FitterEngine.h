//
// Created by Adrien BLANCHET on 11/06/2021.
//

#ifndef XSLLHFITTER_FITTERENGINE_H
#define XSLLHFITTER_FITTERENGINE_H

#include "string"
#include "vector"
#include "memory"

#include "TDirectory.h"
#include "Math/Functor.h"
#include "Math/Minimizer.h"
#include "json.hpp"

#include "Propagator.h"

class FitterEngine {

public:
  FitterEngine();
  virtual ~FitterEngine();

  void reset();

  void setSaveDir(TDirectory *saveDir);
  void setConfig(const json &config);

  void initialize();

  void generateSamplePlots(const std::string& saveSubDirPath_ = "");
  void generateOneSigmaPlots(const std::string& saveSubDirPath_ = "");

  double evalFit(const double* par);

protected:
  void initializePropagator();
  void initializeMinimizer();

private:
  // Parameters
  TDirectory* _saveDir_{nullptr};
  nlohmann::json _config_;

  // Internals
  Propagator _propagator_;
  int _nb_fit_parameters_;
  std::shared_ptr<ROOT::Math::Minimizer> _minimizer_{nullptr};
  std::shared_ptr<ROOT::Math::Functor> _functor_{nullptr};

};


#endif //XSLLHFITTER_FITTERENGINE_H
