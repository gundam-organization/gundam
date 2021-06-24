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

  void setNThreads(int nThreads);

  void initialize();

  // int arg is the thread id
  void addJob(const std::string& jobName_, const std::function<void(int)>& function_);
  void runJob(const std::string& jobName_);
  void removeJob(const std::string& jobName_);


private:
  // Parameters
  bool _stopThreads_{false};
  bool _pauseThreads_{false};
  int _nThreads_{-1};

  // Internals
  bool _isInitialized_{false};
  std::vector<std::future<void>> _threadsList_;
  std::vector<std::function<void(int)>> _jobFunctionList_;
  std::vector<std::string> _jobNameList_;
  std::vector<std::vector<bool>> _jobTriggerList_;

};


#endif //XSLLHFITTER_THREADPOOL_H
