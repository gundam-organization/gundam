#ifndef GUNDAM_GUNDAM_OUTPUT_MANAGER_H
#define GUNDAM_GUNDAM_OUTPUT_MANAGER_H

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class TDirectory;
class TFile;
class TTree;
class GundamOutputManager;
namespace LoggerUtils { class StreamCapture; }
namespace LoggerDetail { class ScopedExitHandler; }

class GundamOutputTDirectory {
public:
  GundamOutputTDirectory() = default;
  explicit GundamOutputTDirectory(TDirectory* directory_) : _borrowedDirectory_(directory_) {}

  // The referenced manager or borrowed ROOT directory must outlive the handle.
  void write(const std::function<void(TDirectory*)>& write_) const;
  void runOnOutputThread(const std::function<void()>& task_) const;
  TDirectory* getDirectDirectory() const;

private:
  friend class GundamOutputManager;
  GundamOutputTDirectory(GundamOutputManager& manager_, std::string path_);

  GundamOutputManager* _manager_{nullptr};
  std::string _path_;
  TDirectory* _borrowedDirectory_{nullptr};
};

class GundamOutputManager {
public:
  enum class Mode { Direct, Threaded };

  GundamOutputManager();
  explicit GundamOutputManager(const std::string& filePath_, Mode mode_ = Mode::Direct);
  ~GundamOutputManager();

  void open(const std::string& filePath_, Mode mode_ = Mode::Direct);
  void close();
  bool isOpen() const { return _isOpen_.load(); }
  bool isThreaded() const { return _outputWorker_ != nullptr; }

  GundamOutputTDirectory directory(std::string path_ = "");
  TFile* getFile(){ return isThreaded() ? nullptr : _outFile_.get(); }
  void setInterruptHandler(std::function<bool()> isRequested_, std::function<void()> onInterrupted_);

private:
  friend class GundamOutputTDirectory;
  class OutputWorker;
  void writeOutput(const std::function<void()>& write_);
  void writeTo(const std::string& directory_, const std::function<void(TDirectory*)>& write_);
  TDirectory* getDirectDirectory(const std::string& path_);
  TDirectory* getDirectory(const std::string& path_);
  void openOutputFile(const std::string& filePath_);
  void closeOutputFile(bool interrupted_ = false);
  void flushRecordedLogs();
  void writeLogLines(std::vector<std::string> lines_);

  std::unique_ptr<LoggerUtils::StreamCapture> _logCapture_;
  std::string _logLine_;
  std::unique_ptr<TFile> _outFile_;
  std::unique_ptr<TTree> _logTree_;
  bool _logWriteFailed_{false};
  std::atomic<bool> _isOpen_{false};
  std::function<bool()> _isInterruptRequested_;
  std::function<void()> _onInterrupted_;
  std::unique_ptr<LoggerDetail::ScopedExitHandler> _logExitHandler_;
  std::unique_ptr<OutputWorker> _outputWorker_;
};

#endif
