//
// Created by Nadrino on 19/05/2021.
//

#include "DataBinSet.h"

#include "GenericToolbox.h"
#include "Logger.h"

#include <string>
#include <sstream>
#include <stdexcept>


LoggerInit([]{
  Logger::setUserHeaderStr("[DataBinSet]");
} );

void DataBinSet::setVerbosity(int maxLogLevel_) { Logger::setMaxLogLevel(maxLogLevel_); }

// core
void DataBinSet::readBinningDefinition(const std::string &filePath_) {

  _filePath_ = GenericToolbox::expandEnvironmentVariables(filePath_);

  if( not GenericToolbox::doesPathIsFile(_filePath_) ){
    LogError << GET_VAR_NAME_VALUE(_filePath_) << ": file not found." << std::endl;
    throw std::runtime_error(GET_VAR_NAME_VALUE(_filePath_) + ": file not found.");
  }

  auto lines = GenericToolbox::dumpFileAsVectorString(_filePath_);
  std::vector<std::string> expectedVariableList;
  std::vector<std::string> expectedVariableTitleList;
  std::vector<bool> expectedVariableIsRangeList;
  int nbExpectedValues{0};

  for( const auto& line : lines ){

    if( line.empty() ){ continue; }

    int offSet = 0;
    char firstChar = line[offSet];
    while( firstChar == ' ' ){
      offSet++;
      if( offSet >= line.size() ){
        firstChar = '#'; // consider it as a comment
        break;
      }
      firstChar = line[offSet];
    }
    if( firstChar == '#' ) continue;

    std::vector<std::string> lineElements = GenericToolbox::splitString(line, " ", true);
    if( lineElements.empty() ) continue;

    if( lineElements.at(0) == "variables:" ){

      nbExpectedValues = 0;
      expectedVariableIsRangeList.clear();
      expectedVariableTitleList.clear();
      expectedVariableList.clear();

      for( size_t iElement = 1 ; iElement < lineElements.size() ; iElement++ ){

        if( not expectedVariableList.empty() and lineElements.at(iElement) == expectedVariableList.back() ){
          LogThrowIf(
              expectedVariableIsRangeList.back(),
              "Same variable appear more than 2 times: " << GenericToolbox::parseVectorAsString(lineElements)
          );
          expectedVariableIsRangeList.back() = true;
          nbExpectedValues += 1;
        }
        else{

          if( GenericToolbox::doesElementIsInVector(lineElements.at(iElement), expectedVariableList) ){
            LogError << lineElements.at(iElement) << " is already set: " << GenericToolbox::parseVectorAsString(lineElements) << std::endl;
            throw std::runtime_error("Invalid bin definition line.");
          }

          expectedVariableList.emplace_back( lineElements.at(iElement) );
          expectedVariableIsRangeList.push_back(false);
          nbExpectedValues += 1;
        }

      }

      for( size_t iVar = 0 ; iVar < expectedVariableList.size() ; iVar++ ){
        expectedVariableTitleList.emplace_back(expectedVariableList.at(iVar));
        if( expectedVariableIsRangeList.at(iVar) ){
          expectedVariableTitleList.back() += " (range)";
        }
        else{
          expectedVariableTitleList.back() += " (point)";
        }
      }
    }
    else if( expectedVariableList.empty() ){
      LogError << "Can't fill bin info while variables have not been set" << std::endl;
      throw std::runtime_error("Can't fill bin info while variables have not been set");
    }
    else{

      if( int(lineElements.size()) != nbExpectedValues ){
        LogError << "(" << GET_VAR_NAME_VALUE(lineElements.size()) << ") != (" << GET_VAR_NAME_VALUE(lineElements.size()) << ")" << std::endl;
        LogError << "Expected: " << GenericToolbox::parseVectorAsString(expectedVariableTitleList) << std::endl;
        LogError << "Got: " << GenericToolbox::parseVectorAsString(lineElements) << std::endl;
        throw std::runtime_error("lineElements.size() != nbExpectedValues");
      }

      size_t iElement = 0;
      _binList_.emplace_back(_binList_.size());
      for( size_t iVar = 0; iVar < expectedVariableList.size() ; iVar++ ){

        if( expectedVariableIsRangeList.at(iVar) ){
          _binList_.back().addBinEdge(
            expectedVariableList[iVar],
            std::stod(lineElements.at(iElement)),
            std::stod(lineElements.at(iElement+1))
          );
          iElement += 2;
        }
        else{
          _binList_.back().setIsZeroWideRangesTolerated(true);
          _binList_.back().addBinEdge(
            expectedVariableList[iVar],
            std::stod(lineElements.at(iElement)),
            std::stod(lineElements.at(iElement))
          );
          iElement += 1;
        }

      }

    }
  }

  // sort bins?
  this->sortBins();
}
void DataBinSet::sortBins(){

  std::vector<std::string> varNameList{};
  for( auto& bin : _binList_ ){
    for( auto& edges : bin.getEdgesList() ){
      GenericToolbox::addIfNotInVector(edges.varName, varNameList);
    }
  }

  GenericToolbox::sortVector(
      _binList_,
      [&](const DataBin& bin1_, const DataBin& bin2_){
        // returns: does bin1 goes first?
        for( auto& varName : varNameList ){
          auto* edges1 = bin1_.getVarEdgesPtr(varName);
          auto* edges2 = bin2_.getVarEdgesPtr(varName);
          if( edges1 == nullptr ){ return true; } // missing variable bins goes first
          if( edges2 == nullptr ){ return false; } // missing variable bins goes first

          if( edges1->min < edges2->min ){ return true; } // lowest bins first
        }

        return false; // default
      }
      );

  // update indices
  for( int iBin = 0 ; iBin < int(_binList_.size()) ; iBin++ ){
    _binList_[iBin].setIndex( iBin );
  }

}
std::string DataBinSet::getSummary() const{
  std::stringstream ss;
  ss << "DataBinSet";
  if( not _name_.empty() ) ss << "(" << _name_ << ")";
  ss << ": holding " << _binList_.size() << " bins.";

  if( not _binList_.empty() ){
    for(size_t iBin = 0 ; iBin < _binList_.size() ; iBin++ ){
      ss << std::endl << GenericToolbox::indentString("#" + std::to_string(iBin) + ": " + _binList_.at(iBin).getSummary(), 2);
    }
  }
  return ss.str();
}
std::vector<std::string> DataBinSet::buildVariableNameList() const{
  std::vector<std::string> out;
  for( auto& bin : _binList_ ){
    for( auto& edges : bin.getEdgesList() ){
      GenericToolbox::addIfNotInVector(edges.varName, out);
    }
  }
  return out;
}
