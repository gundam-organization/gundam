//
// Created by Adrien BLANCHET on 24/06/2021.
//

#ifndef XSLLHFITTER_THREADPOOL_H
#define XSLLHFITTER_THREADPOOL_H

#include "future"
#include "vector"
#include "functional"
#include "map"

class ThreadPool {

public:
  ThreadPool();
  virtual ~ThreadPool();

  void reset();

  void setIsVerbose(bool isVerbose);
  void setNThreads(int nThreads);

  void initialize();

  const std::vector<std::string> &getJobNameList() const;

  void addJob(const std::string& jobName_, const std::function<void(int)>& function_); // int arg is supposed to be the thread id
  void setPostParallelJob(const std::string& jobName_, const std::function<void()>& function_);
  void runJob(const std::string& jobName_);
  void removeJob(const std::string& jobName_);

  void pauseParallelThreads();
  void unPauseParallelThreads();

protected:
  void reStartThreads();
  void stopThreads();

private:
  // Parameters
  bool _isVerbose_{true};
  int _nThreads_{-1};

  // Internals
  bool _isInitialized_{false};
  bool _stopThreads_{false};
  bool _pauseThreads_{false};
  std::mutex* _threadMutexPtr_{nullptr};
  std::vector<std::future<void>> _threadsList_;
  std::vector<std::function<void(int)>> _jobFunctionList_;
  std::vector<std::function<void()>> _jobFunctionPostParallelList_;
  std::vector<std::string> _jobNameList_;
  std::vector<std::vector<bool>> _jobTriggerList_;

};


#endif //XSLLHFITTER_THREADPOOL_H
