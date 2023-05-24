//
// Created by Adrien Blanchet on 21/05/2023.
//

#ifndef GUNDAM_GUNDAMUTILS_H
#define GUNDAM_GUNDAMUTILS_H

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "CmdLineParser.h"

#include <map>
#include <string>
#include <vector>
#include <utility>



namespace GundamUtils {

  inline static std::string generateFileName(const CmdLineParser& clp_, const std::vector<std::pair<std::string, std::string>>& appendixDict_){
    std::vector<std::string> appendixList{};

    int maxArgLength{64};

    for( const auto& appendixDictEntry : appendixDict_ ){
      if( clp_.isOptionTriggered(appendixDictEntry.first) ){
        appendixList.emplace_back( appendixDictEntry.second );
        if( clp_.getNbValueSet(appendixDictEntry.first) > 0 ){

          auto args = clp_.getOptionValList<std::string>(appendixDictEntry.first);
          for( auto& arg : args ){
            // strip potential slashes and extensions
            arg = GenericToolbox::getFileNameFromFilePath(arg, false);
            if( arg.size() > maxArgLength ){
              // print dotdot if too long
              arg = arg.substr(0, maxArgLength);
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

  // dict
  inline static const std::map<int, std::string> minuitStatusCodeStr{
      { 0 , "status = 0    : OK" },
      { 1 , "status = 1    : Covariance was mad! Thus made pos defined"},
      { 2 , "status = 2    : Hesse is invalid"},
      { 3 , "status = 3    : Edm is above max"},
      { 4 , "status = 4    : Reached call limit"},
      { 5 , "status = 5    : Any other failure"},
      { -1, "status = -1   : We don't even know what happened"}
  };
  inline static const std::map<int, std::string> hesseStatusCodeStr{
      { 0, "status = 0    : OK" },
      { 1, "status = 1    : Hesse failed"},
      { 2, "status = 2    : Matrix inversion failed"},
      { 3, "status = 3    : Matrix is not pos defined"},
      { -1, "status = -1    : Minimize wasn't called before"}
  };
  inline static const std::map<int, std::string> minosStatusCodeStr{
      { 0, "status = 0    : last MINOS run was succesfull" },
      { 1, "status = 1    : Maximum number of function calls exceeded when running for lower error"},
      { 2, "status = 2    : maximum number of function calls exceeded when running for upper error"},
      { 3, "status = 3    : new minimum found when running for lower error"},
      { 4, "status = 4    : new minimum found when running for upper error"},
      { 5, "status = 5    : any other failure"},
      { -1, "status = -1   : Minos is not run"}
  };

  // 0 not calculated 1 approximated 2 made pos def , 3 accurate
  inline static const std::map<int, std::string> covMatrixStatusCodeStr{
      { 0, "status = 0    : not calculated" },
      { 1, "status = 1    : approximated"},
      { 2, "status = 2    : made pos def"},
      { 3, "status = 3    : accurate"}
  };

}

#endif //GUNDAM_GUNDAMUTILS_H
