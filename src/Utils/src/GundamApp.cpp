//
// Created by Adrien Blanchet on 30/05/2023.
//

#include "GundamApp.h"
#include "ShowLogsSource.h"

#include "GenericToolbox.Root.h"
#include "GenericToolbox.Time.h"
#include "Logger.h"

#include <TFile.h>
#include <TMacro.h>
#include <TTree.h>

#include <iostream>
#include <memory>
#include <sstream>


GundamApp::GundamApp(std::string  appName_)
  : _appName_(std::move(appName_)), _logCapture_(std::make_unique<LoggerUtils::StreamCapture>()) {
  _greeting_.setAppName( _appName_ );
  _greeting_.hello();
}
GundamApp::~GundamApp() {
  _greeting_.goodbye();
  if( _outFile_ != nullptr and _outFile_->IsOpen() and not _outFile_->IsZombie() ){
    LogWarning << "Closing output file \"" << _outFile_->GetName() << "\"..." << std::endl;
    closeOutputFile();
    LogInfo << "Output file closed." << std::endl;
  }
}

void GundamApp::closeOutputFile(){
  if( _outFile_ == nullptr or not _outFile_->IsOpen() or _outFile_->IsZombie() ) return;
  if( _logTree_ != nullptr ){
    if( _logCapture_ != nullptr ) writeLogLines(_logCapture_->finishLines());
    if( _logTree_->AutoSave("FlushBaskets") <= 0 ){
      std::cerr << "Failed to save the recorded log tree." << std::endl;
    }
    _logTree_.reset();
  }
  _outFile_->Close();
}

void GundamApp::setCmdLinePtr(const CmdLineParser *cmdLinePtr) { _cmdLinePtr_ = cmdLinePtr; }
void GundamApp::setConfigString(const std::string &configString) {
  _configString_ = configString;
}
void GundamApp::writeLogLines(std::vector<std::string> lines_){
  for( auto& line : lines_ ){
    _logLine_ = std::move(line);
    if( _logTree_->Fill() < 0 ){
      _logCapture_.reset();
      std::cerr << "Failed to write recorded logs to the ROOT output file." << std::endl;
      break;
    }
  }
  _logLine_.clear();
}

void GundamApp::flushRecordedLogs(){
  if( _logTree_ == nullptr or _logCapture_ == nullptr ) return;
  constexpr std::size_t retainedLines{64};
  constexpr std::size_t minBatchSize{64};
  writeLogLines(_logCapture_->drain(retainedLines, minBatchSize));
}

void GundamApp::openOutputFile(const std::string& filePath_){
  LogWarning << "Creating output file: \"" << filePath_ << "\"..." << std::endl;
  GenericToolbox::mkdir( GenericToolbox::getFolderPath( filePath_ ) );
  _logTree_.reset();
  _outFile_ = std::make_unique<TFile>( filePath_.c_str(), "RECREATE" );
  if( _logCapture_ != nullptr and _outFile_->IsOpen() and not _outFile_->IsZombie() and _outFile_->IsWritable() ){
    auto* gundamDir = GenericToolbox::mkdirTFile(_outFile_.get(), "gundam");
    auto* runtimeDir = GenericToolbox::mkdirTFile(gundamDir, "runtime");
    if( runtimeDir != nullptr ){
      TDirectory::TContext context(runtimeDir);
      _logTree_ = std::make_unique<TTree>("logger", "Recorded console output");
      // Keep the file backing the baskets, but let only AutoSave write the tree header.
      runtimeDir->Remove(_logTree_.get());
      _logTree_->Branch("text", &_logLine_, 32000, 0);
      _logTree_->SetAutoSave(-256*1024);

      TMacro showLogs("showLogs", "Display the GUNDAM run log");
      std::istringstream source(showLogsSource);
      for( std::string line; std::getline(source, line); ){
        showLogs.AddLine(line.c_str());
      }
      runtimeDir->WriteObject(&showLogs, "showLogs", "Overwrite");
      _logExitHandler_ = std::make_unique<LoggerDetail::ScopedExitHandler>([this]{ closeOutputFile(); });
    }
    flushRecordedLogs();
  }
}
void GundamApp::writeAppInfo(){
  if( _outFile_ == nullptr ){
    LogWarning << "Skipping " << __METHOD_NAME__ << " as no output file is opened." << std::endl;
  }
  LogInfo << "Writing general app info in output file..." << std::endl;

  auto* dir = GenericToolbox::mkdirTFile(_outFile_.get(), "gundam");

  auto* runtimeDir = GenericToolbox::mkdirTFile(dir, "runtime");
  GenericToolbox::writeInTFileWithObjTypeExt( runtimeDir, TNamed("date", GenericToolbox::getNowDateString("%Y.%m.%d %H:%M:%S").c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( runtimeDir, TNamed("user", GenericToolbox::getUserName().c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( runtimeDir, TNamed("host", GenericToolbox::getHostName().c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( runtimeDir, TNamed("pwd", GenericToolbox::getCurrentWorkingDirectory().c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( runtimeDir, TNamed("os", GenericToolbox::getOsName().c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( runtimeDir, TNamed("dist", GenericToolbox::getOsVersion().c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( runtimeDir, TNamed("arch", GenericToolbox::getOsArchitecture().c_str()) );
  if( _cmdLinePtr_ != nullptr ){
    GenericToolbox::writeInTFileWithObjTypeExt( runtimeDir, TNamed("commandLine", _cmdLinePtr_->getCommandLineString().c_str()) );
  }

  auto* buildDir = GenericToolbox::mkdirTFile(dir, "build");
  GenericToolbox::writeInTFileWithObjTypeExt( buildDir, TNamed("version", GundamUtils::getVersionFullStr().c_str()) );

  auto* rootDir = GenericToolbox::mkdirTFile(buildDir, "root");
  GenericToolbox::writeInTFileWithObjTypeExt( rootDir, TNamed("version", std::string(gROOT->GetVersion()).c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( rootDir, TNamed("date", std::to_string(gROOT->GetVersionDate()).c_str()) );
  GenericToolbox::writeInTFileWithObjTypeExt( rootDir, TNamed("install", std::string(gROOT->GetDataDir()).c_str()) );

  if( not _configString_.empty() ){
    GenericToolbox::writeInTFileWithObjTypeExt( GenericToolbox::mkdirTFile(dir, "config"), TNamed("unfoldedJson", _configString_.c_str()) );
  }

  GenericToolbox::triggerTFileWrite( dir );
}


