//
// Created by Adrien BLANCHET on 11/02/2021.
//

#ifndef XSLLHFITTER_GLOBALVARIABLES_H
#define XSLLHFITTER_GLOBALVARIABLES_H

#include <map>
#include <mutex>

class GlobalVariables{

public:

    // Setters
    static void setNbThreads(int nbThreads_);

    // Getters
    static int& getNbThreads();
    static std::mutex& getThreadMutex();
    static std::map<std::string, bool>& getBoolMap();


private:

    static int _nbThreads_;
    static std::mutex _threadMutex_;
    static std::map<std::string, bool> _boolMap_;

};

#endif // XSLLHFITTER_GLOBALVARIABLES_H
