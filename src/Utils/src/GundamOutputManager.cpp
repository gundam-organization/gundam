//
// Created by Adrien Blanchet on 30/05/2023.
//

#include "GundamOutputManager.h"
#include "ShowLogsSource.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"

#include <TFile.h>
#include <TMacro.h>
#include <TTree.h>

#include <iostream>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

GundamOutputTDirectory::GundamOutputTDirectory(GundamOutputManager& manager_, std::string path_)
  : _manager_(&manager_), _path_(std::move(path_)) {}

void GundamOutputTDirectory::write(const std::function<void(TDirectory*)>& write_) const{
  if( _manager_ != nullptr ) _manager_->writeTo(_path_, write_);
  else if( _borrowedDirectory_ != nullptr ) write_(_borrowedDirectory_);
}

void GundamOutputTDirectory::runOnOutputThread(const std::function<void()>& task_) const{
  if( _manager_ != nullptr ) _manager_->writeOutput(task_);
  else task_();
}

TDirectory* GundamOutputTDirectory::getDirectDirectory() const{
  return _manager_ != nullptr ? _manager_->getDirectDirectory(_path_) : _borrowedDirectory_;
}

class GundamOutputManager::OutputWorker {
public:
  explicit OutputWorker(GundamOutputManager& manager_) : _manager_(manager_), _thread_([this]{ run(); }) {}
  ~OutputWorker(){
    {
      std::lock_guard<std::mutex> lock(_mutex_);
      _stopping_ = true;
    }
    _ready_.notify_one();
    _thread_.join();
  }
  void submit(const std::function<void()>& write_){
    if( std::this_thread::get_id() == _thread_.get_id() ){
      write_();
      return;
    }
    std::packaged_task<void()> task(write_);
    auto result = task.get_future();
    {
      std::lock_guard<std::mutex> lock(_mutex_);
      _pending_.push_back(std::move(task));
    }
    _ready_.notify_one();
    result.get();
  }

private:
  void run(){
    LoggerDetail::ScopedExitHandler exitHandler([this]{ _manager_.closeOutputFile(); });
    std::unique_lock<std::mutex> lock(_mutex_);
    while( not _stopping_ ){
      _ready_.wait_for(lock, std::chrono::milliseconds(50), [this]{ return _stopping_ or not _pending_.empty(); });
      if( _stopping_ ) break;
      if( _manager_._isInterruptRequested_ and _manager_._isInterruptRequested_() ){
        lock.unlock();
        _manager_.closeOutputFile(true);
        _manager_._onInterrupted_();
        lock.lock();
      }
      if( not _pending_.empty() ){
        auto task = std::move(_pending_.front());
        _pending_.pop_front();
        lock.unlock();
        _manager_.flushRecordedLogs();
        task();
        lock.lock();
      }
      else{
        lock.unlock();
        _manager_.flushRecordedLogs();
        lock.lock();
      }
    }
    // Destroy the file and its directories on the owning thread as well.
    lock.unlock();
    _manager_.closeOutputFile();
    _manager_._outFile_.reset();
  }

  GundamOutputManager& _manager_;
  std::mutex _mutex_;
  std::condition_variable _ready_;
  std::deque<std::packaged_task<void()>> _pending_;
  bool _stopping_{false};
  std::thread _thread_;
};

GundamOutputManager::GundamOutputManager()
  : _logCapture_(std::make_unique<LoggerUtils::StreamCapture>()) {}

GundamOutputManager::GundamOutputManager(const std::string& filePath_, Mode mode_)
  : GundamOutputManager() {
  open(filePath_, mode_);
}

GundamOutputManager::~GundamOutputManager(){
  close();
}

void GundamOutputManager::close(){
  writeOutput([this]{
    if( _outFile_ != nullptr and _outFile_->IsOpen() and not _outFile_->IsZombie() ){
      LogWarning << "Closing output file \"" << _outFile_->GetName() << "\"..." << std::endl;
      closeOutputFile();
      LogInfo << "Output file closed." << std::endl;
    }
    _outFile_.reset();
    _logCapture_.reset();
  });
  _outputWorker_.reset();
  _logExitHandler_.reset();
}

void GundamOutputManager::closeOutputFile(bool interrupted_){
  if( _outFile_ == nullptr or not _outFile_->IsOpen() or _outFile_->IsZombie() ) return;
  if( _logTree_ != nullptr ){
    if( _logCapture_ != nullptr ){
      if( interrupted_ ){
        auto lines = _logCapture_->snapshotLines();
        lines.emplace_back("Interrupt requested (SIGINT). Stopping the fitter.");
        writeLogLines(std::move(lines));
      }
      else{
        writeLogLines(_logCapture_->finishLines());
      }
    }
    if( _logTree_->AutoSave("FlushBaskets") <= 0 ){
      _logWriteFailed_ = true;
      std::cerr << "Failed to save the recorded log tree." << std::endl;
    }
    _logTree_.reset();
  }
  _outFile_->Close();
  _isOpen_.store(false);
  if( not interrupted_ ) _logCapture_.reset();
  if( interrupted_ ){
    const bool writeFailed = _logWriteFailed_ or _outFile_->TestBit(TFile::kWriteError);
    const std::string message = writeFailed
        ? "\nSIGINT: Failed to save the complete log to ROOT output file \""
        : "\nSIGINT: Log saved and ROOT output file closed: \"";
    std::cerr << (message + _outFile_->GetName() + "\".\n");
  }
}

void GundamOutputManager::writeLogLines(std::vector<std::string> lines_){
  if( _logWriteFailed_ ) return;
  for( auto& line : lines_ ){
    _logLine_ = std::move(line);
    if( _logTree_->Fill() < 0 ){
      _logWriteFailed_ = true;
      std::cerr << "Failed to write recorded logs to the ROOT output file." << std::endl;
      break;
    }
  }
  _logLine_.clear();
}

void GundamOutputManager::flushRecordedLogs(){
  if( _logTree_ == nullptr or _logCapture_ == nullptr ) return;
  constexpr std::size_t retainedLines{64};
  constexpr std::size_t minBatchSize{64};
  writeLogLines(_logCapture_->drain(retainedLines, minBatchSize));
}

void GundamOutputManager::writeOutput(const std::function<void()>& write_){
  if( _outputWorker_ == nullptr ){ write_(); return; }
  _outputWorker_->submit(write_);
}

GundamOutputTDirectory GundamOutputManager::directory(std::string path_){
  return GundamOutputTDirectory(*this, std::move(path_));
}

void GundamOutputManager::writeTo(const std::string& directory_, const std::function<void(TDirectory*)>& write_){
  writeOutput([&]{
    auto* destination = getDirectory(directory_);
    if( destination == nullptr ) throw std::runtime_error("No writable ROOT output directory: " + directory_);
    TDirectory::TContext context(destination);
    write_(destination);
  });
}

TDirectory* GundamOutputManager::getDirectory(const std::string& path_){
  if( _outFile_ == nullptr or not _outFile_->IsOpen() or not _outFile_->IsWritable() ) return nullptr;
  return path_.empty() ? _outFile_.get() : GenericToolbox::mkdirTFile(_outFile_.get(), path_);
}

TDirectory* GundamOutputManager::getDirectDirectory(const std::string& path_){
  if( isThreaded() ){
    throw std::logic_error("A threaded ROOT output directory can only be accessed inside a write request.");
  }
  return getDirectory(path_);
}

void GundamOutputManager::setInterruptHandler(std::function<bool()> isRequested_, std::function<void()> onInterrupted_){
  if( _outputWorker_ != nullptr ) throw std::logic_error("Set the interrupt handler before opening threaded ROOT output.");
  if( bool(isRequested_) != bool(onInterrupted_) ) throw std::invalid_argument("An interrupt handler requires both callbacks.");
  _isInterruptRequested_ = std::move(isRequested_);
  _onInterrupted_ = std::move(onInterrupted_);
}

void GundamOutputManager::open(const std::string& filePath_, Mode mode_){
  if( _isOpen_.load() ) close();
  if( _logCapture_ == nullptr ) _logCapture_ = std::make_unique<LoggerUtils::StreamCapture>();
  if( mode_ == Mode::Threaded ){
    ROOT::EnableThreadSafety();
    _outputWorker_ = std::make_unique<OutputWorker>(*this);
  }
  try{
    writeOutput([&]{ openOutputFile(filePath_); });
  }
  catch(...){
    close();
    throw;
  }
  _logExitHandler_ = std::make_unique<LoggerDetail::ScopedExitHandler>([this]{
    writeOutput([this]{ closeOutputFile(); });
  });
}

void GundamOutputManager::openOutputFile(const std::string& filePath_){
  LogWarning << "Creating output file: \"" << filePath_ << "\"..." << std::endl;
  GenericToolbox::mkdir( GenericToolbox::getFolderPath( filePath_ ) );
  _logTree_.reset();
  _logWriteFailed_ = false;
  _outFile_ = std::make_unique<TFile>( filePath_.c_str(), "RECREATE" );
  if( not _outFile_->IsOpen() or _outFile_->IsZombie() or not _outFile_->IsWritable() ){
    throw std::runtime_error("Could not open ROOT output file: " + filePath_);
  }
  _isOpen_.store(true);
  if( _logCapture_ != nullptr ){
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
    }
    flushRecordedLogs();
  }
}
