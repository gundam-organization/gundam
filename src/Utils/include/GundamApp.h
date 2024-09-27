//
// Created by Adrien Blanchet on 30/05/2023.
//

#ifndef GUNDAM_GUNDAMAPP_H
#define GUNDAM_GUNDAMAPP_H


#include "GundamUtils.h"
#include "GundamGreetings.h"

#include "CmdLineParser.h"

#include "TFile.h"

#include <string>
#include <memory>
#include <utility>


class GundamApp {

public:
  explicit GundamApp(std::string  appName_);
  virtual ~GundamApp();

  void setCmdLinePtr(const CmdLineParser *cmdLinePtr);
  void setConfigString(const std::string &configString);

  TFile* getOutfilePtr(){ return _outFile_.get(); }

  void openOutputFile(const std::string& filePath_);
  void writeAppInfo();

private:
  // parameters
  std::string _appName_{};
  std::string _configString_{};
  const CmdLineParser* _cmdLinePtr_{nullptr};

  // internals
  std::unique_ptr<TFile> _outFile_{nullptr};
  GundamGreetings _greeting_{};

};


#endif //GUNDAM_GUNDAMAPP_H
