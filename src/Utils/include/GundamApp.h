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
#include <vector>

namespace LoggerUtils { class StreamCapture; }
class TTree;


class GundamApp {

public:
  explicit GundamApp(std::string  appName_);
  virtual ~GundamApp();

  void setCmdLinePtr(const CmdLineParser *cmdLinePtr);
  void setConfigString(const std::string &configString);
  void startLogCapture();
  void flushRecordedLogs();

  TFile* getOutfilePtr(){ return _outFile_.get(); }

  void openOutputFile(const std::string& filePath_);
  void writeAppInfo();

private:
  void writeLogLines(std::vector<std::string> lines_);

  // parameters
  std::string _appName_{};
  std::string _configString_{};
  const CmdLineParser* _cmdLinePtr_{nullptr};

  // internals
  std::unique_ptr<LoggerUtils::StreamCapture> _logCapture_;
  std::string _logLine_;
  std::unique_ptr<TFile> _outFile_{nullptr};
  std::unique_ptr<TTree> _logTree_;
  GundamGreetings _greeting_{};

};


#endif //GUNDAM_GUNDAMAPP_H
