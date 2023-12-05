//
// Created by Adrien Blanchet on 05/12/2023.
//

#ifndef GUNDAM_PYTHONBINDER_H
#define GUNDAM_PYTHONBINDER_H


#include "FitterEngine.h"
#include "DataBin.h"
#include "ConfigUtils.h"
#include "GundamApp.h"

#include "GenericToolbox.h"

#include <pybind11/stl.h> // support for vectors

#include <cmath>
#include <utility>
#include <vector>
#include <iostream>
#include <string>


class PythonBinder{

  std::string filePath{};

  GundamApp app{"test fitter"};
  FitterEngine* fitter{nullptr};

public:

  std::vector<double> v_data{};
  std::vector<double> v_gamma{};

  PythonBinder() = default;
  explicit PythonBinder( std::string  filePath_ ) : filePath(std::move(filePath_)) {
    ConfigUtils::ConfigHandler configHandler(filePath);
    configHandler.override( std::vector<std::string>{{"./override/onlyRun4and5.yaml"}} );

    app.openOutputFile("test.root");
    app.writeAppInfo();

    fitter = new FitterEngine{GenericToolbox::mkdirTFile(app.getOutfilePtr(), "FitterEngine")};

    fitter->readConfig(GenericToolbox::Json::fetchSubEntry(configHandler.getConfig(), {"fitterEngineConfig"}));
    fitter->getPropagator().setLoadAsimovData( true );
    fitter->initialize();
  }

  void run() {
    fitter->fit();
  }

};


#endif //GUNDAM_PYTHONBINDER_H
