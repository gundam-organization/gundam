//
// Created by Adrien BLANCHET on 11/02/2021.
//

#include "../include/GlobalVariables.h"

// INIT
int GlobalVariables::_nbThreads_ = 1;
std::mutex GlobalVariables::_threadMutex_;
std::map<std::string, bool> GlobalVariables::_boolMap_;
std::vector<TChain*> GlobalVariables::_chainList_;


void GlobalVariables::setNbThreads(int nbThreads_){
    _nbThreads_ = nbThreads_;
}

const int& GlobalVariables::getNbThreads(){ return _nbThreads_; }
std::mutex& GlobalVariables::getThreadMutex() { return _threadMutex_; }
std::map<std::string, bool>& GlobalVariables::getBoolMap() { return _boolMap_; }
std::vector<TChain*>& GlobalVariables::getChainList() { return _chainList_; }
