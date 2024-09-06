//
// Created by Adrien Blanchet on 30/05/2023.
//

#include "GundamApp.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"

#include <TFile.h>

#include <memory>

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::getUserHeader() << "[GundamApp]";
});
#endif


GundamApp::GundamApp(std::string  appName_) : _appName_(std::move(appName_)) {
  _greeting_.setAppName( _appName_ );
  _greeting_.hello();
}
GundamApp::~GundamApp() {
  _greeting_.goodbye();
  if( _outFile_ != nullptr ){
    LogWarning << "Closing output file \"" << _outFile_->GetName() << "\"..." << std::endl;
    _outFile_->Close();
    LogInfo << "Output file closed." << std::endl;
  }
}

void GundamApp::setCmdLinePtr(const CmdLineParser *cmdLinePtr) { _cmdLinePtr_ = cmdLinePtr; }
void GundamApp::setConfigString(const std::string &configString) {
  _configString_ = configString;
}

void GundamApp::openOutputFile(const std::string& filePath_){
  LogWarning << "Creating output file: \"" << filePath_ << "\"..." << std::endl;
  GenericToolbox::mkdir( GenericToolbox::getFolderPath( filePath_ ) );
  _outFile_ = std::make_unique<TFile>( filePath_.c_str(), "RECREATE" );
}
void GundamApp::writeAppInfo(){
  if( _outFile_ == nullptr ){
    LogWarning << "Skipping " << __METHOD_NAME__ << " as no output file is opened." << std::endl;
  }
  LogInfo << "Writing general app info in output file..." << std::endl;

  auto* dir = GenericToolbox::mkdirTFile(_outFile_.get(), "gundam");

  // Gundam version?
  GenericToolbox::writeInTFile( dir, TNamed("version", GundamUtils::getVersionFullStr().c_str()) );

  // Command line?
  if( _cmdLinePtr_ != nullptr ){
    GenericToolbox::writeInTFile( dir, TNamed("commandLine", _cmdLinePtr_->getCommandLineString().c_str()) );
  }

  if( not _configString_.empty() ){
    GenericToolbox::writeInTFile( dir, TNamed("config", _configString_.c_str()) );
  }

  GenericToolbox::triggerTFileWrite( dir );
}


