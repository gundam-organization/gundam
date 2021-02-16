//
// Created by Adrien BLANCHET on 18/11/2020.
//

#include "Logger.h"
#include <fstream>
#include <json.hpp>
using json = nlohmann::json;

// User parameters
std::string __jsonConfigPath__;
std::string __outFilePath__;

// Customs
struct InputFile{

    InputFile() = default;

    std::string path;
    std::string treeName;
    std::string detectorName;

};

// Internals
int __argc__;
char** __argv__;
std::string __commandLine__;
std::vector<InputFile> __inputFileList__;

// Setup
std::string remindUsage();
void resetParameters();
void getUserParameters();
void readJsonConfigFile();

int main(int argc, char** argv){

    Logger::setUserHeaderStr("[xsllhGetDetParCov.cxx]");

    resetParameters();
    getUserParameters();
    readJsonConfigFile();

}


std::string remindUsage(){

    std::stringstream remind_usage_ss;
    remind_usage_ss << "*********************************************************" << std::endl;
    remind_usage_ss << " > Command Line Arguments" << std::endl;
    remind_usage_ss << "  -j : JSON input (Current : " << __jsonConfigPath__ << ")" << std::endl;
    remind_usage_ss << "  -o : Override output file path (Current : " << __outFilePath__ << ")" << std::endl;
//    remind_usage_ss << "  -t : Override number of threads (Current : " << __nb_threads__ << ")" << std::endl;
//    remind_usage_ss << "  -d : Enable dry run (Current : " << __is_dry_run__ << ")" << std::endl;
    remind_usage_ss << "*********************************************************" << std::endl;

    LogInfo << remind_usage_ss.str();
    return remind_usage_ss.str();

}
void resetParameters(){
    __jsonConfigPath__   = "";
    __outFilePath__      = "";
}
void getUserParameters(){

    if(__argc__ == 1){
        remindUsage();
        exit(EXIT_FAILURE);
    }

    // Sanity check
    const std::string XSLLHFITTER = std::getenv("XSLLHFITTER");
    if(XSLLHFITTER.empty()){
        LogError << "Environment variable \"XSLLHFITTER\" not set." << std::endl
                 << "Cannot determine source tree location." << std::endl;
        remindUsage();
        exit(EXIT_FAILURE);
    }

    LogWarning << "Reading user parameters" << std::endl;

    for(int i_arg = 0; i_arg < __argc__; i_arg++){
        __commandLine__ += __argv__[i_arg];
        __commandLine__ += " ";
    }

    for(int i_arg = 0; i_arg < __argc__; i_arg++){

        if (std::string(__argv__[i_arg]) == "-j"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __jsonConfigPath__ = std::string(__argv__[j_arg]);
            }
            else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-o"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __outFilePath__ = std::string(__argv__[j_arg]);
            }
            else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }

    }

}
void readJsonConfigFile(){
    // Read in the .json config file:
    std::fstream configFileStream;
    configFileStream.open(__jsonConfigPath__, std::ios::in);
    LogInfo << "Opening " << __jsonConfigPath__ << std::endl;
    if(!configFileStream.is_open())
    {
        LogError << "Unable to open JSON configure file." << std::endl;
        exit(EXIT_FAILURE);
    }

    json configJson;
    configFileStream >> configJson;

    if(__outFilePath__.empty()){
        __outFilePath__ = configJson["defaultOutPath"];
    }

    for(auto& inputFileConfig : configJson["inputFiles"]){
        __inputFileList__.emplace_back(InputFile());

        __inputFileList__.back().path         = inputFileConfig["path"];
        __inputFileList__.back().treeName     = inputFileConfig["treeName"];
        __inputFileList__.back().detectorName = inputFileConfig["detectorName"];

    }

}

