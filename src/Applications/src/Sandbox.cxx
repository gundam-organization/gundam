//
// Created by Nadrino on 17/06/2021.
//

#include "GenericToolbox.Thread.h"

#include "Logger.h"

#include <string>
#include <vector>
#include <memory>
#include <future>


#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::getUserHeader() << "[" << FILENAME << "]"; });
#endif


int main(int argc, char** argv){
  int nThreads = 4;

  GenericToolbox::ParallelWorker w;
  w.setNThreads(nThreads);
  w.initialize();

  w.addJob("test", [](int iThread_){ LogTrace << GET_VAR_NAME_VALUE(iThread_) << std::endl; });
  w.runJob("test");
  w.removeJob("test");

  LogWarning << "NEXT" << std::endl;

  int nRound{100};
  while(nRound-- > 0){
    LogWarning << GET_VAR_NAME_VALUE(nRound) << std::endl;
    w.addJob("test", [](int iThread_){ LogTrace << GET_VAR_NAME_VALUE(iThread_) << std::endl; });
    w.runJob("test");
    w.removeJob("test");
  }

//  bool signal;
//  std::mutex m;
//  std::condition_variable v;
//
//  std::vector<std::future<void>> workers;
//  for(int iThread=0 ; iThread < nThreads ; iThread++){
//    LogInfo << GET_VAR_NAME_VALUE(iThread) << std::endl;
//    std::function<void()> workerFct = [&, iThread](){
//      std::unique_lock<std::mutex> lock(m);
//      v.wait(lock, [&](){ return signal; });
//      lock.unlock();
//      LogWarning << "UNLOCKED " << GET_VAR_NAME_VALUE(iThread) << std::endl;
//      std::this_thread::sleep_for(std::chrono::milliseconds(500));
//    };
//    workers.emplace_back(std::async( std::launch::async, workerFct ));
//  }
//
//  LogInfo << "TAKING LOCK?" << std::endl;
//  std::unique_lock<std::mutex> lock(m);
//
//  LogInfo << "SIGNAL ON" << std::endl;
//  signal = true;
//
//  LogInfo << "NOTIFY" << std::endl;
//  v.notify_all();
//  lock.unlock();
//
//  LogInfo << "GET THREADS" << std::endl;
//  for( auto& worker : workers ){
//    worker.get();
//  }

  LogInfo << "END" << std::endl;
  return EXIT_SUCCESS;
}


