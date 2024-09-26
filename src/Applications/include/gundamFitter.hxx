//
// Created by Nadrino on 26/09/2024.
//

#ifndef GUNDAM_GUNDAMFITTER_H
#define GUNDAM_GUNDAMFITTER_H

#include "GundamAppTemplate.h"
#include "FitterEngine.h"

class GundamFitterApp : public GundamAppTemplate {

public:
  GundamFitterApp(const std::string& name_, int argc_, char** argv_){ _argc_ = argc_; _argv_ = argv_; _name_ = name_; }

  void run() override { _fitter_.fit(); }

protected:
  void defineCommandLineOptions() override;

  void configureImpl() override;

private:
  FitterEngine _fitter_{nullptr};

};

#endif //GUNDAM_GUNDAMFITTER_H
