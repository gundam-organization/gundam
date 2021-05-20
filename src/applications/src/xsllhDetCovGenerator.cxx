//
// Created by Adrien BLANCHET on 21/04/2021.
//

#include "string"

#include "Logger.h"
#include "json.hpp"

#include "../include/xsllhDetCovGenerator.h"

int main( int argc, char** argv ){

  xsllhDetCovGenerator::readParameters(argc, argv);

}


namespace xsllhDetCovGenerator{

  void readParameters(int argc, char** argv){

    for(int iArg = 0 ; iArg < argc ; iArg++){
      if( std::string(argv[iArg]) == "-j" ){
        int jArg = iArg + 1;
        if ( jArg < argc ) {
          xsllhDetCovGenerator::jsonFilePath = std::string(argv[jArg]);
        }
        else {
          LogError << "Give an argument after " << argv[iArg] << std::endl;
          throw std::logic_error(std::string(argv[iArg]) + " : no argument found");
        }
      }
    }

  }
  void readJsonFile(){

    if( xsllhDetCovGenerator::jsonFilePath.empty() ){
      LogError << "jsonFilePath not specified." << std::endl;
      throw std::logic_error("jsonFilePath not specified.");
    }



  }

}

