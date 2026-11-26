//
// Created by Adrien Blanchet on 30/05/2023.
//

#include "GundamApp.h"
#include "GundamUtils.h"

#include "GenericToolbox.Root.h"
#include "GenericToolbox.Time.h"
#include "Logger.h"

#include <utility>

GundamApp::GundamApp(std::string appName_){
  _greeting_.setAppName(std::move(appName_));
  _greeting_.hello();
}

GundamApp::~GundamApp(){
  _greeting_.goodbye();
}

void GundamApp::setCmdLinePtr(const CmdLineParser* cmdLinePtr_){ _cmdLinePtr_ = cmdLinePtr_; }
void GundamApp::setConfigString(const std::string& configString_){ _configString_ = configString_; }
void GundamApp::openOutputFile(const std::string& filePath_){ _outputManager_.open(filePath_); }

void GundamApp::writeAppInfo(){
  LogInfo << "Writing general app info in output file..." << std::endl;

  _outputManager_.directory("gundam").write([this](TDirectory* dir){
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
  });
}
