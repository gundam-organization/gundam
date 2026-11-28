//
// Created by Adrien Blanchet on 30/05/2023.
//

#ifndef GUNDAM_GUNDAMAPP_H
#define GUNDAM_GUNDAMAPP_H


#include "GundamGreetings.h"
#include "GundamOutputManager.h"

#include "CmdLineParser.h"

#include "TFile.h"

#include <string>


class GundamApp {

public:
  explicit GundamApp(std::string  appName_);
  virtual ~GundamApp();

  void setCmdLinePtr(const CmdLineParser *cmdLinePtr);
  void setConfigString(const std::string &configString);
  GundamOutputManager& getOutputManager(){ return _outputManager_; }
  TFile* getOutfilePtr(){ return _outputManager_.getFile(); }

  void openOutputFile(const std::string& filePath_);
  void writeAppInfo();

private:
  std::string _configString_{};
  const CmdLineParser* _cmdLinePtr_{nullptr};
  GundamOutputManager _outputManager_;
  GundamGreetings _greeting_{};

};


#endif //GUNDAM_GUNDAMAPP_H
