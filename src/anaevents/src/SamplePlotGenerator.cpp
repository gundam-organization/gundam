//
// Created by Adrien BLANCHET on 16/06/2021.
//

#include "Logger.h"

#include "SamplePlotGenerator.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[SamplePlotGenerator]");
})


SamplePlotGenerator::SamplePlotGenerator() { this->reset(); }
SamplePlotGenerator::~SamplePlotGenerator() { this->reset(); }

void SamplePlotGenerator::reset() {
  _saveTDirectory_ = nullptr;
}

void SamplePlotGenerator::setSaveTDirectory(TDirectory *saveTDirectory_) {
  _saveTDirectory_ = saveTDirectory_;
}
void SamplePlotGenerator::setConfig(const nlohmann::json &config) {
  _config_ = config;
}

void SamplePlotGenerator::initialize() {

  if(_saveTDirectory_ == nullptr ){
    LogError << "_saveTDirectory_ not set." << std::endl;
    throw std::logic_error("_saveTDirectory_ not set.");
  }



}


