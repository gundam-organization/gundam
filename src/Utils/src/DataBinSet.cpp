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
      _binsList_.emplace_back(_binsList_.size());
      for( size_t iVar = 0; iVar < expectedVariableList.size() ; iVar++ ){

        if( expectedVariableIsRangeList.at(iVar) ){
          _binsList_.back().addBinEdge(
            expectedVariableList[iVar],
            std::stod(lineElements.at(iElement)),
            std::stod(lineElements.at(iElement+1))
          );
          iElement += 2;
        }
        else{
          _binsList_.back().setIsZeroWideRangesTolerated(true);
          _binsList_.back().addBinEdge(
            expectedVariableList[iVar],
            std::stod(lineElements.at(iElement)),
            std::stod(lineElements.at(iElement))
          );
          iElement += 1;
        }

      }

    }
  }
}
std::string DataBinSet::getSummary() const{
  std::stringstream ss;
  ss << "DataBinSet";
  if( not _name_.empty() ) ss << "(" << _name_ << ")";
  ss << ": holding " << _binsList_.size() << " bins.";

  if( not _binsList_.empty() ){
    for( size_t iBin = 0 ; iBin < _binsList_.size() ; iBin++ ){
      ss << std::endl << GenericToolbox::indentString( "#" + std::to_string(iBin) + ": " + _binsList_.at(iBin).getSummary(), 2);
    }
  }
  return ss.str();
}
std::vector<std::string> DataBinSet::buildVariableNameList() const{
  std::vector<std::string> out;
  for( auto& bin : _binsList_ ){
    for( auto& edges : bin.getEdgesList() ){
      GenericToolbox::addIfNotInVector(edges.varName, out);
    }
  }
  return out;
}
