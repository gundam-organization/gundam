//
// Created by Nadrino on 19/05/2021.
//

#include "../include/DataBinSet.h"

#include "stdexcept"
#include "string"
#include "sstream"

#include "GenericToolbox.h"
#include "Logger.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[DataBinSet]");
} )

DataBinSet::DataBinSet() {
  this->reset();
}
DataBinSet::~DataBinSet() { this->reset(); }

void DataBinSet::reset() {
  _binsList_.clear();
}

// Setters
void DataBinSet::setName(const std::string &name) {
  _name_ = name;
}
void DataBinSet::readBinningDefinition(const std::string &filePath_) {

  LogInfo << "Reading binning definition from: \"" << filePath_ << "\"" << std::endl;
  _filePath_ = filePath_;

  if( not GenericToolbox::doesPathIsFile(filePath_) ){
    LogError << GET_VAR_NAME_VALUE(filePath_) << ": file not found." << std::endl;
    throw std::runtime_error(GET_VAR_NAME_VALUE(filePath_) + ": file not found.");
  }

  auto lines = GenericToolbox::dumpFileAsVectorString(filePath_);
  std::vector<std::string> expectedVariableList;
  std::vector<std::string> expectedVariableTitleList;
  std::vector<bool> expectedVariableIsRangeList;
  int nbExpectedValues{0};

  for( const auto& line : lines ){

    if( line.empty() ) continue;

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
          if( expectedVariableIsRangeList.back() == true ){
            LogError << "Same variable appear more than 2 times: " << GenericToolbox::parseVectorAsString(lineElements) << std::endl;
            throw std::runtime_error("Invalid bin definition line.");
          }
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

      LogDebug << "Defined edges for variables: " << GenericToolbox::parseVectorAsString(expectedVariableTitleList) << std::endl;

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
      _binsList_.emplace_back();
      _binContent_.emplace_back(0);
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

  LogInfo << _binsList_.size() << " bins have been defined." << std::endl;

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


void DataBinSet::addBinContent(int binIndex_, double weight_) {
  if( binIndex_ < 0 or binIndex_ >= _binsList_.size() ){
    LogError << GET_VAR_NAME_VALUE(binIndex_) << " is out of range: " << GET_VAR_NAME_VALUE(_binsList_.size()) << std::endl;
    throw std::logic_error("Invalid binIndex");
  }
  _binContent_.at(binIndex_) += weight_;
}

const std::vector<DataBin> &DataBinSet::getBinsList() const {
  return _binsList_;
}

const std::string &DataBinSet::getFilePath() const {
  return _filePath_;
}

void DataBinSet::setVerbosity(int maxLogLevel_) {
  Logger::setMaxLogLevel(maxLogLevel_);
}
