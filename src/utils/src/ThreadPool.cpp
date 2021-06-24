//
// Created by Adrien BLANCHET on 24/06/2021.
//

#include "functional"

#include "GenericToolbox.h"

#include "ThreadPool.h"

ThreadPool::ThreadPool(){
  this->reset();
}
ThreadPool::~ThreadPool() {
  this->reset();
}

void ThreadPool::reset() {
  _isInitialized_ = false;
  _stopThreads_ = true;
  for( auto& thread : _threadsList_ ){
    thread.get();
  }
  _threadsList_.clear();

  _nThreads_ = -1;
  _stopThreads_ = false;
}

void ThreadPool::setNThreads(int nThreads) {
  if(_isInitialized_){
    throw std::logic_error("Can't set the number of threads while already initialized.");
  }
  _nThreads_ = nThreads;
}

void ThreadPool::initialize() {

  if( _nThreads_ < 1 ){
    throw std::logic_error("_nThreads_ should be >= 1");
  }

  for( int iThread = 0 ; iThread < _nThreads_-1 ; iThread++ ){
    std::function<void()> asyncLoop = [this, iThread](){
      size_t jobIndex = 0;
      while( not _stopThreads_ ){
        while( _pauseThreads_ ); // wait

        jobIndex = 0;
        for( jobIndex = 0 ; jobIndex < _jobTriggerList_.size() ; jobIndex++ ){
          if( _pauseThreads_ ) break; // jump out!
          if( _jobTriggerList_[jobIndex][iThread] ){ // is it triggered?
            _jobFunctionList_.at(jobIndex)(iThread); // run
            _jobTriggerList_[jobIndex][iThread] = false; // un-trigger this thread
          }
        } // jobIndex
      } // not stop
    };
    _threadsList_.emplace_back( std::async( std::launch::async, asyncLoop ) );
  }

  _isInitialized_ = true;
}

void ThreadPool::addJob(const std::string &jobName_, const std::function<void(int)> &function_) {
  if( not _isInitialized_ ){
    throw std::logic_error("Can't add job while not initialized");
  }
  if( GenericToolbox::doesElementIsInVector(jobName_, _jobNameList_) ){
    throw std::logic_error("A job with the same name has already been added: " + jobName_);
  }
  if( not function_ ){ // is it callable?
    throw std::logic_error("the provided function is not callable");
  }
  _pauseThreads_ = true; // prevent the threads to loop over the available jobs
  for( const auto& threadTriggers : _jobTriggerList_ ){
    for( int iThread = 0 ; iThread < _nThreads_-1 ; iThread++ ){
      while( threadTriggers.at(iThread) ); // wait
    }
  }
  _jobNameList_.emplace_back(jobName_);
  _jobFunctionList_.emplace_back(function_);
  _jobTriggerList_.emplace_back(std::vector<bool>(_nThreads_, false));
  _pauseThreads_ = false;
}
void ThreadPool::runJob(const std::string &jobName_) {
  if( not _isInitialized_ ){
    throw std::logic_error("Can't run job while not initialized");
  }
  int jobIndex = GenericToolbox::findElementIndex(jobName_, _jobNameList_);
  if( jobIndex == -1 ){
    throw std::logic_error(jobName_ + ": is not in the available jobsList");
  }

  for( int iThread = 0 ; iThread < _nThreads_-1 ; iThread++ ){
    _jobTriggerList_.at(jobIndex).at(iThread) = true;
  }

  _jobFunctionList_.back()(_nThreads_-1); // do the last job in the main thread

  for( int iThread = 0 ; iThread < _nThreads_-1 ; iThread++ ){
    while( _jobTriggerList_.at(jobIndex).at(iThread)); // wait
  }

}
void ThreadPool::removeJob(const std::string& jobName_){
  if( not _isInitialized_ ){
    throw std::logic_error("Can't run job while not initialized");
  }
  int jobIndex = GenericToolbox::findElementIndex(jobName_, _jobNameList_);
  if( jobIndex == -1 ){
    throw std::logic_error(jobName_ + ": is not in the available jobsList");
  }
  _pauseThreads_ = true; // prevent the threads to loop over the available jobs
  for( const auto& threadTriggers : _jobTriggerList_ ){
    for( int iThread = 0 ; iThread < _nThreads_-1 ; iThread++ ){
      while( threadTriggers.at(iThread) ); // wait
    }
  }
  _jobNameList_.erase(_jobNameList_.begin() + jobIndex);
  _jobFunctionList_.erase(_jobFunctionList_.begin() + jobIndex);
  _jobTriggerList_.erase(_jobTriggerList_.begin() + jobIndex);
  _pauseThreads_ = false;
}
