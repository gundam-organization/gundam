//
// Created by Adrien Blanchet on 21/05/2023.
//

#ifndef GUNDAM_GUNDAMUTILS_H
#define GUNDAM_GUNDAMUTILS_H

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "CmdLineParser.h"

#include "string"
#include "map"
#include "vector"
#include "utility"



namespace GundamUtils {

  std::string generateFileName(const CmdLineParser& clp_, const std::vector<std::pair<std::string, std::string>>& appendixDict_){
    std::vector<std::string> appendixList{};
    for( const auto& appendixDictEntry : appendixDict_ ){
      if( clp_.isOptionTriggered(appendixDictEntry.first) ){
        appendixList.emplace_back( appendixDictEntry.second );
        if( clp_.getNbValueSet(appendixDictEntry.first) > 0 ){

          auto args = clp_.getOptionValList<std::string>(appendixDictEntry.first);
          for( auto& arg : args ){
            // strip potential slashes and extensions
            arg = GenericToolbox::getFileNameFromFilePath(arg, false);
            arg = arg.substr(0, 24); // cap length
            if( arg.size() == 24 ){
              // print dotdot if too long
              arg[arg.size()-1] = '.';
              arg[arg.size()-2] = '.';
              arg[arg.size()-3] = '.';
            }

            // cleanup from special chars
            arg = GenericToolbox::generateCleanBranchName(arg);
          }

          appendixList.back() = Form(
              appendixList.back().c_str(),
              GenericToolbox::joinVectorString(args, "_").c_str()
          );
        }
        else{
          appendixList.back() = GenericToolbox::trimString(Form( appendixList.back().c_str(), "" ), "_");
        }
      }
    }

    return GenericToolbox::joinVectorString(appendixList, "_");
  }

}

#endif //GUNDAM_GUNDAMUTILS_H
