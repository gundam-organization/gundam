//
// Created by Adrien Blanchet on 05/12/2023.
//

#ifndef GUNDAM_PYTHON_INTERFACE_H
#define GUNDAM_PYTHON_INTERFACE_H


#include "FitterEngine.h"
#include "ConfigUtils.h"
#include "GundamApp.h"

#include <vector>
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

    fitter->setConfig( GenericToolbox::Json::fetchValue<JsonType>(configHandler.getConfig(), "fitterEngineConfig") );
    fitter->configure();

    fitter->getLikelihoodInterface().setForceAsimovData( true );
    fitter->initialize();
  }

  void run() {
    fitter->fit();
  }

};


#endif //GUNDAM_PYTHON_INTERFACE_H
