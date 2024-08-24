//
// Created by Nadrino on 19/05/2021.
//

#include "DataBinSet.h"
#include "ConfigUtils.h"

#include "GenericToolbox.Json.h"
#include "Logger.h"

#include <string>
#include <sstream>
#include <stdexcept>


LoggerInit([]{
  Logger::setUserHeaderStr("[DataBinSet]");
} );

void DataBinSet::setVerbosity(int maxLogLevel_){ Logger::setMaxLogLevel(maxLogLevel_); }

// core
void DataBinSet::readBinningDefinition(const JsonType& binning_) {

  if( binning_.is_structured() ){
    // config like -> should already be unfolded
    this->readBinningConfig( binning_ );
  }
  else if( binning_.is_string() ){
    _filePath_ = GenericToolbox::expandEnvironmentVariables( binning_.get<std::string>() );
    if( not GenericToolbox::isFile(_filePath_) ){
      LogError << GET_VAR_NAME_VALUE(_filePath_) << ": file not found." << std::endl;
      throw std::runtime_error(GET_VAR_NAME_VALUE(_filePath_) + ": file not found.");
    }

    if( GenericToolbox::hasExtension(_filePath_, "txt") ){ this->readTxtBinningDefinition(); }
  }
  else{
    LogThrow("Unknown binning config entry: " << GenericToolbox::Json::toReadableString(binning_));
  }

  this->sortBinEdges();
  this->checkBinning();
}

void DataBinSet::checkBinning(){

  bool hasErrors{false};

  for( auto& bin : _binList_ ){
    for( auto& otherBin : _binList_ ) {
      if( &otherBin == &bin ){ continue; } // skip if it's the same bin
      if( bin.isOverlapping( otherBin ) ){
        LogError << "BIN OVERLAP DETECTED" << std::endl;
        LogError << bin.getSummary() << std::endl;
        LogError << otherBin.getSummary() << std::endl;
        hasErrors = true;
      }
    }
  }

  LogThrowIf(hasErrors);

}
void DataBinSet::sortBins(){

  /// DON'T SORT THE BINS FOR DIALS!!! THE ORDER MIGHT REFER TO THE COV MATRIX DEFINITION

//  // weird things going on if uncommented...
//  std::vector<std::string> varNameList{this->buildVariableNameList()};
//  std::sort(
//      _binList_.begin(), _binList_.end(),
//      [&](const DataBin& bin1_, const DataBin& bin2_){
//        // returns: does bin1 goes first?
//        for( auto& varName : varNameList ){
//          auto* edges1 = bin1_.getVarEdgesPtr(varName);
//          if( edges1 == nullptr ){ return true; } // missing variable bins goes first
//
//          auto* edges2 = bin2_.getVarEdgesPtr(varName);
//          if( edges2 == nullptr ){ return false; } // missing variable bins goes first
//
//          if( edges1->min < edges2->min ){ return true; } // lowest bins first
//        }
//
//        return false; // default
//      }
//  );
//
//  // update indices
//  for( int iBin = 0 ; iBin < int(_binList_.size()) ; iBin++ ){
//    _binList_[iBin].setIndex( iBin );
//  }

}
std::string DataBinSet::getSummary() const{
  std::stringstream ss;
  ss << "DataBinSet";
  if( not _name_.empty() ) ss << "(" << _name_ << ")";
  ss << ": holding " << _binList_.size() << " bins.";


  for( auto& bin : _binList_ ){
    ss << std::endl << GenericToolbox::indentString("#" + std::to_string(bin.getIndex()) + ": " + bin.getSummary(), 2);
  }

  return ss.str();
}


void DataBinSet::sortBinEdges(){
  for( auto& bin : _binList_ ){
    std::sort(
        bin.getEdgesList().begin(), bin.getEdgesList().end(),
        [](const DataBin::Edges& edges1_, const DataBin::Edges& edges2_){
          if( edges1_.isConditionVar and not edges2_.isConditionVar ){ return true; }
          if( not edges1_.isConditionVar and edges2_.isConditionVar ){ return false; }
          return GenericToolbox::toLowerCase(edges1_.varName) < GenericToolbox::toLowerCase(edges2_.varName);
        }
    );
    // update the indices
    for( int iEdges = 0 ; iEdges < int(bin.getEdgesList().size()) ; iEdges++ ){ bin.getEdgesList()[iEdges].index = iEdges; }
  }
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


void DataBinSet::readTxtBinningDefinition(){

  auto lines = GenericToolbox::dumpFileAsVectorString(_filePath_);

  std::vector<std::string> expectedVariableList;
  std::vector<std::string> expectedVariableTitleList;
  std::vector<bool> expectedVariableIsRangeList;
  int nbExpectedValues{0};

  for( auto& line : lines ){

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

    // stripping comments
    line = GenericToolbox::splitString(line, "#")[0];
    GenericToolbox::trimInputString(line, " "); // removing trailing spaces
    GenericToolbox::removeEscapeCodes(line);

    std::vector<std::string> lineElements = GenericToolbox::splitString(line, " ", true);
    if( lineElements.empty() ){ continue; }

    if( lineElements.at(0) == "variables:" ){

      nbExpectedValues = 0;
      expectedVariableIsRangeList.clear();
      expectedVariableTitleList.clear();
      expectedVariableList.clear();

      for( size_t iElement = 1 ; iElement < lineElements.size() ; iElement++ ){

        if( not expectedVariableList.empty() and lineElements.at(iElement) == expectedVariableList.back() ){
          LogThrowIf(
              expectedVariableIsRangeList.back(),
              "Same variable appear more than 2 times: " << GenericToolbox::toString(lineElements)
          );
          expectedVariableIsRangeList.back() = true;
          nbExpectedValues += 1;
        }
        else{

          if( GenericToolbox::doesElementIsInVector(lineElements.at(iElement), expectedVariableList) ){
            LogError << lineElements.at(iElement) << " is already set: " << GenericToolbox::toString(lineElements) << std::endl;
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
        LogError << "(" << GET_VAR_NAME_VALUE(lineElements.size()) << ") != (" << GET_VAR_NAME_VALUE(nbExpectedValues) << ")" << std::endl;
        LogError << "Expected: " << GenericToolbox::toString(expectedVariableTitleList) << std::endl;
        LogError << "Got: " << GenericToolbox::toString(lineElements) << std::endl;
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

}

void DataBinSet::readBinningConfig(const JsonType& binning_){

  if( GenericToolbox::Json::doKeyExist(binning_, {"binningDefinition"}) ){

    auto binningDefinition = GenericToolbox::Json::fetchValue<JsonType>(binning_, "binningDefinition");
    struct Dimension{
      int nBins{0};
      int nModulo{1};
      std::string var{};
      bool isEdgesDiscreteValues{false};
      std::vector<double> edgesList{};
    };
    std::vector<Dimension> dimensionList{};
    dimensionList.reserve( binningDefinition.size() );

    for( auto& binDefEntry : binningDefinition ){
      dimensionList.emplace_back();
      auto& dim = dimensionList.back();

      dim.var = GenericToolbox::Json::fetchValue<std::string>(binDefEntry, "name");

      if( GenericToolbox::Json::doKeyExist(binDefEntry, "edges") ){
        dim.edgesList = GenericToolbox::Json::fetchValue(binDefEntry, "edges", dim.edgesList);
      }
      else if( GenericToolbox::Json::doKeyExist(binDefEntry, "values") ){
        dim.edgesList = GenericToolbox::Json::fetchValue(binDefEntry, "values", dim.edgesList);
        dim.isEdgesDiscreteValues = true;
      }
      else if( GenericToolbox::Json::doKeyExist(binDefEntry, "nBins") ){
        // TH1D-like definition
        auto nBins( GenericToolbox::Json::fetchValue<int>(binDefEntry, "nBins") );
        auto minVal( GenericToolbox::Json::fetchValue<double>(binDefEntry, "min") );
        auto maxVal( GenericToolbox::Json::fetchValue<double>(binDefEntry, "max") );

        double step{(maxVal - minVal)/nBins};
        LogThrowIf( step <= 0, "Invalid binning: " << GenericToolbox::Json::toReadableString(binDefEntry) );

        dim.edgesList.reserve( nBins + 1 );
        double edgeValue{minVal};
        for( int iBin = 0 ; iBin < nBins ; iBin++ ){
          dim.edgesList.emplace_back( edgeValue );
          edgeValue += step;
        }
        dim.edgesList.emplace_back( edgeValue );
      }
      else{
        LogThrow("Unrecognised binning definition: " << binningDefinition);
      }

      dim.nBins = int( dim.edgesList.size() );
      if( not dim.isEdgesDiscreteValues ){ dim.nBins--; }

      LogThrowIf(dim.nBins == 0, "Invalid edgesList for binEdgeEntry: " << GenericToolbox::Json::toReadableString(binDefEntry));
    }

    int nBinsTotal{1};
    for( int iDim = int( dimensionList.size() )-1 ; iDim >= 0 ; iDim-- ){
      dimensionList[iDim].nModulo = nBinsTotal;
      nBinsTotal *= dimensionList[iDim].nBins;
    }

    _binList_.reserve( nBinsTotal );
    for( int iBin = 0 ; iBin < nBinsTotal ; iBin++ ){
      JsonType binDefConfig{};
      auto& edgesListConfig = binDefConfig["edgesList"];

      for( auto& dim : dimensionList ){
        edgesListConfig.emplace_back(); auto& edge = edgesListConfig.back();
        edge["name"] = dim.var;

        int edgeIndex = ( iBin/dim.nModulo ) % dim.nBins;

        if( dim.isEdgesDiscreteValues ){
          edge["value"] = dim.edgesList[edgeIndex];
        }
        else{
          edge["bounds"].emplace_back( dim.edgesList[edgeIndex] );
          edge["bounds"].emplace_back( dim.edgesList[edgeIndex+1] );
        }
      }

      _binList_.emplace_back( _binList_.size() );
      _binList_.back().readConfig( binDefConfig );
    }

  }

  for( auto& binDef : GenericToolbox::Json::fetchValue(binning_, "binList", JsonType()) ){
    _binList_.emplace_back( _binList_.size() );
    _binList_.back().readConfig( binDef );
  }


}
