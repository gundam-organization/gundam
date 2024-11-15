//
// Created by Adrien Blanchet on 28/02/2023.
//

#include "ConfigUtils.h"
#include "GundamUtils.h"

#include "GenericToolbox.Root.h"
#include "GenericToolbox.Yaml.h"
#include "Logger.h"

#include <string>
#include <vector>
#include <sstream>
#include <iostream>


namespace ConfigUtils {

  // open file
  JsonType readConfigFile(const std::string& configFilePath_){
    if( not GenericToolbox::isFile(configFilePath_) ){
      LogError << "\"" << configFilePath_ << "\" could not be found." << std::endl;
      throw std::runtime_error("file not found.");
    }

    JsonType output;

    try{
      if( GenericToolbox::hasExtension(configFilePath_, {{"yaml"}, {"yml"}}) ){
        output = ConfigUtils::convertYamlToJson( configFilePath_ );
      }
      else{
        output = GenericToolbox::Json::readConfigFile(configFilePath_);
      }
    }
    catch(...){ LogThrow("Error while reading config file: " << configFilePath_); }

    // resolve sub-references to other config files
    ConfigUtils::unfoldConfig( output );

    return output;
  }

  // YAML to JSON converting
  JsonType convertYamlToJson(const std::string& configFilePath_){
    return ConfigUtils::convertYamlToJson(GenericToolbox::Yaml::readConfigFile(configFilePath_));
  }
  JsonType convertYamlToJson(const YAML::Node& yaml){
    JsonType output = JsonType::parse(GenericToolbox::Yaml::toJsonString(yaml));

    auto is_number = [](const std::string& s){
      return !s.empty() && std::find_if(s.begin(),
                                        s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
    };
    auto is_numeric = [](std::string const & str){
      auto result = double();
      auto i = std::istringstream(str);
      i >> result;
      return !i.fail() && i.eof();
    };

    std::function<void(JsonType&)> recursiveFix;
    recursiveFix = [&recursiveFix, is_number, is_numeric](JsonType& jsonEntry_){

      if( jsonEntry_.is_null() ){
        return;
      }
      else if(jsonEntry_.is_array() or jsonEntry_.is_structured()){
        for( auto &jsonSubEntry : jsonEntry_ ){
          recursiveFix(jsonSubEntry);
        }
      }
      else if(jsonEntry_.is_string()){

        auto value = jsonEntry_.get<std::string>();
        if( value == "true" ){
          jsonEntry_ = true;
        }
        else if( value == "false" ) {
          jsonEntry_ = false;
        }
        else if( is_number(value) ){
          jsonEntry_ = std::stoi(value);
        }
        else if( is_numeric(value) ){
          jsonEntry_ = std::stod(value);
        }

      }
    };

    recursiveFix(output);
    return output;
  }

  // unfolding
  JsonType getForwardedConfig(const JsonType& config_){
    JsonType out = config_;
    while( out.is_string() ){
      out = ConfigUtils::readConfigFile(out.get<std::string>());
    }
    return out;
  }
  void forwardConfig(JsonType& config_){
    while( config_.is_string() and
         ( GenericToolbox::endsWith(config_.get<std::string>(), ".yaml", true)
        or GenericToolbox::endsWith(config_.get<std::string>(), ".json", true) )
        ){
      auto name = config_.get<std::string>();
      std::string expand = GenericToolbox::expandEnvironmentVariables(name);
      config_ = ConfigUtils::readConfigFile(expand);
    }
  }
  void unfoldConfig(JsonType& config_){

    std::function<void(JsonType&)> unfoldRecursive = [&](JsonType& outEntry_){
      for( auto& entry : config_ ){
        if( entry.is_string() and (
               GenericToolbox::endsWith(entry.get<std::string>(), ".yaml", true)
            or GenericToolbox::endsWith(entry.get<std::string>(), ".json", true)
        ) ){
          ConfigUtils::forwardConfig( entry );
          ConfigUtils::unfoldConfig( config_ ); // remake the loop on the unfolder config
          break; // don't touch anymore
        }

        if( entry.is_structured() ){ ConfigUtils::unfoldConfig( entry ); }
      }
    };
    unfoldRecursive(config_);

  }

  // class impl
  ConfigHandler::ConfigHandler(const std::string& filePath_){
    if( GenericToolbox::hasExtension( filePath_, "root" ) ){
      LogInfo << "Extracting config file for fitter file: " << filePath_ << std::endl;
      LogThrowIf( not GenericToolbox::doesTFileIsValid(filePath_), "Invalid root file: " << filePath_ );
      auto fitFile = std::shared_ptr<TFile>( GenericToolbox::openExistingTFile( filePath_ ) );

      auto* conf = fitFile->Get<TNamed>("gundam/config_TNamed");
      if( conf == nullptr ){
        // legacy
        conf = fitFile->Get<TNamed>("gundamFitter/unfoldedConfig_TNamed");
      }
      LogThrowIf(conf==nullptr, "no config in ROOT file " << filePath_);
      config = GenericToolbox::Json::readConfigJsonStr( conf->GetTitle() );
      fitFile->Close();
    }
    else{
      LogInfo << "Reading config file: " << filePath_ << std::endl;
      config = ConfigUtils::readConfigFile(filePath_ ); // works with yaml
    }
  }

  void ConfigHandler::override( const JsonType& overrideConfig_ ){
    LogWarning << GenericToolbox::Json::applyOverrides(config, overrideConfig_);
  }
  void ConfigHandler::override( const std::string& filePath_ ){
    LogInfo << "Overriding config with \"" << filePath_ << "\"" << std::endl;
    LogThrowIf(not GenericToolbox::isFile(filePath_), "Could not find " << filePath_);
    this->override( ConfigUtils::readConfigFile(filePath_) );
  }
  void ConfigHandler::override( const std::vector<std::string>& filesList_ ){
    for( auto& file : filesList_ ){ this->override( file ); }
  }
  void ConfigHandler::flatOverride( const std::string& flattenEntry_ ){
    // Override the configuration values.  If the old value was a string then
    // replace with the new string. Otherwise, the input value is parsed.  The
    // configuration value are references like path names
    // (e.g. /fitterEngineConfig/mcmcConfig/steps to change the MCMC interface
    // "steps" value.)  This is intended to make minor changes to the behavior,
    // so for sanity's sake, the key must already exist in the configuration
    // files (if the key does not exist an exception will be thrown).  The
    // command line syntax to change the number of mcmc steps to 1000 per cycle
    // would be
    //
    // gundamFitter.exe -O /fitterEngineConfig/mcmcConfig/steps=1000 ...
    //

    std::vector<std::string> split = GenericToolbox::splitString( flattenEntry_,"=" );
    LogWarning << "Override " << split[0] << " with " << split[1]
               << std::endl;
    JsonType flat = config.flatten();
    LogWarning << "    Original value: " << flat.at(split[0])
               << std::endl;
    if (flat.at(split[0]).is_string()) flat.at(split[0]) = split[1];
    else flat.at(split[0]) = JsonType::parse(split[1]);
    LogWarning << "         New value: " << flat.at(split[0])
               << std::endl;
    config = flat.unflatten();
  }
  void ConfigHandler::flatOverride( const std::vector<std::string>& flattenEntryList_ ){
    for( auto& flattenEntry : flattenEntryList_ ){ this->flatOverride( flattenEntry ); }
  }
  void ConfigHandler::exportToJsonFile(const std::string &filePath_) const {
    auto outPath{filePath_};

    if( not GenericToolbox::endsWith(outPath, ".json") ){
      // add extension if missing
      outPath += ".json";
    }

    LogInfo << "Writing as: " << outPath << std::endl;
    GenericToolbox::dumpStringInFile(outPath, this->toString());
    LogInfo << "Unfolded config written as: " << outPath << std::endl;
  }

}
