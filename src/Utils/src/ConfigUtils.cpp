//
// Created by Adrien Blanchet on 28/02/2023.
//

#include "ConfigUtils.h"

#include "GenericToolbox.Json.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Yaml.h"
#include "Logger.h"

#include "nlohmann/json.hpp"

#include <string>
#include <vector>
#include <utility>
#include <sstream>
#include <iostream>


LoggerInit([]{
  Logger::setUserHeaderStr("[ConfigUtils]");
} );


namespace ConfigUtils {

  nlohmann::json readConfigFile(const std::string& configFilePath_){
    if( not GenericToolbox::doesPathIsFile(configFilePath_) ){
      LogError << "\"" << configFilePath_ << "\" could not be found." << std::endl;
      throw std::runtime_error("file not found.");
    }

    nlohmann::json output;

    if( GenericToolbox::doesFilePathHasExtension(configFilePath_, "yml")
        or GenericToolbox::doesFilePathHasExtension(configFilePath_,"yaml")
        ){
      output = ConfigUtils::convertYamlToJson(configFilePath_);
    }
    else{
      output = GenericToolbox::Json::readConfigFile(configFilePath_);
    }

    return output;
  }

  nlohmann::json convertYamlToJson(const std::string& configFilePath_){
    return ConfigUtils::convertYamlToJson(GenericToolbox::Yaml::readConfigFile(configFilePath_));
  }
  nlohmann::json convertYamlToJson(const YAML::Node& yaml){
    nlohmann::json output = nlohmann::json::parse(GenericToolbox::Yaml::toJsonString(yaml));

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

    std::function<void(nlohmann::json&)> recursiveFix;
    recursiveFix = [&recursiveFix, is_number, is_numeric](nlohmann::json& jsonEntry_){

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

  nlohmann::json getForwardedConfig(const nlohmann::json& config_){
    nlohmann::json out = config_;
    while( out.is_string() ){
      out = ConfigUtils::readConfigFile(out.get<std::string>());
    }
    return out;
  }
  nlohmann::json getForwardedConfig(const nlohmann::json& config_, const std::string& keyName_){
    return ConfigUtils::getForwardedConfig(GenericToolbox::Json::fetchValue<nlohmann::json>(config_, keyName_));
  }
  void forwardConfig(nlohmann::json& config_, const std::string& className_){
    while( config_.is_string() ){
//      LogDebug << "Forwarding " << (className_.empty()? "": className_ + " ") << "config: \"" << config_.get<std::string>() << "\"" << std::endl;
      auto name = config_.get<std::string>();
      std::string expand = GenericToolbox::expandEnvironmentVariables(name);
      config_ = ConfigUtils::readConfigFile(expand);
    }
  }
  void unfoldConfig(nlohmann::json& config_){

    std::function<void(nlohmann::json&)> unfoldRecursive = [&](nlohmann::json& outEntry_){
      for( auto& entry : config_ ){
        if( entry.is_string() and (
               GenericToolbox::doesStringEndsWithSubstring(entry.get<std::string>(), ".yaml", true)
            or GenericToolbox::doesStringEndsWithSubstring(entry.get<std::string>(), ".json", true)
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

  void applyOverrides(nlohmann::json& outConfig_, const nlohmann::json& overrideConfig_){

    // dev options
    bool debug{false};
    bool allowAddMissingKey{true};

    // specific keys like "name" might help reference the lists
    std::vector<std::string> listOfIdentifiers{{"name"}, {"__INDEX__"}};

    std::vector<std::string> jsonPath{};
    std::function<void(nlohmann::json&, const nlohmann::json&)> overrideRecursive =
        [&](nlohmann::json& outEntry_, const nlohmann::json& overrideEntry_){
      LogDebug(debug) << GET_VAR_NAME_VALUE(GenericToolbox::joinPath( jsonPath )) << std::endl;

      if( overrideEntry_.is_array() ){
        // entry is list
        LogThrowIf( not outEntry_.is_array(), GenericToolbox::joinPath( jsonPath ) << " is not an array: " << std::endl << outEntry_ << std::endl << std::endl << overrideEntry_ );

        // is it empty? -> erase
        if( overrideEntry_.empty() ){
          LogWarning << "Overriding list: " << GenericToolbox::joinPath(jsonPath) << std::endl;
          outEntry_ = overrideEntry_;
          return;
        }

        // is it an array of primitive type? like std::vector<std::string>?
        bool isStructured{false};
        for( auto& outListEntry : outEntry_.items() ){ if( outListEntry.value().is_structured() ){ isStructured = true; break; } }
        if( not isStructured ){
          LogWarning << "Overriding list: " << GenericToolbox::joinPath(jsonPath) << std::endl;
          outEntry_ = overrideEntry_;
          return;
        }

        // loop over to find the right entry
        for( auto& overrideListEntry: overrideEntry_.items() ){

          // fetch identifier if available using override
          std::string identifier{};
          for( auto& identifierCandidate : listOfIdentifiers ){
            if( GenericToolbox::Json::doKeyExist( overrideListEntry.value(), identifierCandidate ) ){
              identifier = identifierCandidate;
            }
          }

          if( not identifier.empty() ){
            // will i
            LogDebug(debug) << "Will identify override list item with key \"" << identifier << "\" = " << overrideListEntry.value()[identifier] << std::endl;

            nlohmann::json* outListEntryMatch{nullptr};

            if( identifier == "__INDEX__" ){
              if( overrideListEntry.value()[identifier].get<int>() == -1 ){
                // add entry
                if( allowAddMissingKey ){
                  LogAlert << "Adding: " << GenericToolbox::joinPath(jsonPath, outEntry_.size());
                  if( overrideListEntry.value().is_primitive() ){ LogAlert << " -> " << overrideListEntry.value(); }
                  LogAlert << std::endl;
                  outEntry_.emplace_back(overrideListEntry.value());
                }
              }
              else if( overrideListEntry.value()[identifier].get<size_t>() < outEntry_.size() ){
                jsonPath.emplace_back(overrideListEntry.key());
                overrideRecursive(outEntry_[overrideListEntry.value()[identifier].get<size_t>()], overrideListEntry.value());
                jsonPath.pop_back();
              }
              else{
                LogThrow("Invalid __INDEX__: " << overrideListEntry.value()[identifier].get<int>());
              }
            }
            else{
              for( auto& outListEntry : outEntry_ ){
                if( GenericToolbox::Json::doKeyExist( outListEntry, identifier )
                and outListEntry[identifier] == overrideListEntry.value()[identifier] ){
                  outListEntryMatch = &outListEntry;
                  break;
                }
              }

              if( outListEntryMatch == nullptr ){
                if( allowAddMissingKey ) {
                  LogAlert << "Adding: " << GenericToolbox::joinPath(jsonPath, outEntry_.size()) << "(" << identifier
                           << ":" << overrideListEntry.value()[identifier] << ")" << std::endl;
                  outEntry_.emplace_back(overrideListEntry.value());
                  continue;
                }
              }
              jsonPath.emplace_back(GenericToolbox::joinAsString("",overrideListEntry.key(),"(",identifier,":",overrideListEntry.value()[identifier],")"));
              overrideRecursive(*outListEntryMatch, overrideListEntry.value());
              jsonPath.pop_back();
            }
          }
          else{
            LogAlert << "No identifier found for list def in " << GenericToolbox::joinPath(jsonPath) << std::endl;
            continue;
          }
        }
      }
      else{

        if( overrideEntry_.empty() ){
          LogWarning << "Removing entry: " << GenericToolbox::joinPath(jsonPath) << std::endl;
          outEntry_ = overrideEntry_;
          return;
        }

        // entry is dictionary
        for( auto& overrideEntry : overrideEntry_.items() ){

          // addition mode:
          if( not GenericToolbox::Json::doKeyExist(outEntry_, overrideEntry.key()) ){
            if( overrideEntry.key() != "__INDEX__" ){
              if( allowAddMissingKey ){
                LogAlert << "Adding: " << GenericToolbox::joinPath(jsonPath, overrideEntry.key());
                if( overrideEntry.value().is_primitive() ){ LogAlert << " -> " << overrideEntry.value(); }
                LogAlert << std::endl;
                outEntry_[overrideEntry.key()] = overrideEntry.value();
              }
              else{
                LogThrow("Could not edit missing key \"" << GenericToolbox::joinPath(jsonPath, overrideEntry.key()) << "\" ("
                << GET_VAR_NAME_VALUE(allowAddMissingKey) << ")"
                );
              }
            }
            continue;
          }

          // override
          auto& outSubEntry = outEntry_[overrideEntry.key()];

          if( overrideEntry.value().is_structured() ){
            // recursive candidate
            jsonPath.emplace_back(overrideEntry.key());
            overrideRecursive(outSubEntry, overrideEntry.value());
            jsonPath.pop_back();
          }
          else{
            // override
            if( outSubEntry != overrideEntry.value() ){
              LogWarning << "Overriding: " << GenericToolbox::joinPath(jsonPath, overrideEntry.key()) << ": "
                         << outSubEntry << " -> " << overrideEntry.value() << std::endl;
              outSubEntry = overrideEntry.value();
            }
          }
        }
      }

    };

    // recursive
    if( overrideConfig_.is_array() ){
      // old nlohmann json version -> can be defined as array
      overrideRecursive(outConfig_, overrideConfig_[0]);
    }
    else{
      overrideRecursive(outConfig_, overrideConfig_);
    }

  }

  // class impl
  ConfigHandler::ConfigHandler(const std::string& filePath_){
    if( GenericToolbox::doesFilePathHasExtension( filePath_, "root" ) ){
      LogInfo << "Extracting config file for fitter file: " << filePath_ << std::endl;
      LogThrowIf( not GenericToolbox::doesTFileIsValid(filePath_), "Invalid root file: " << filePath_ );
      auto fitFile = std::shared_ptr<TFile>( GenericToolbox::openExistingTFile( filePath_ ) );
      auto* conf = fitFile->Get<TNamed>("gundamFitter/unfoldedConfig_TNamed");
      LogThrowIf(conf==nullptr, "no config in ROOT file " << filePath_);
      config = GenericToolbox::Json::readConfigJsonStr( conf->GetTitle() );
      fitFile->Close();
    }
    else{
      LogInfo << "Reading config file: " << filePath_ << std::endl;
      config = ConfigUtils::readConfigFile(filePath_ ); // works with yaml
      ConfigUtils::unfoldConfig( config );
    }

  }
  ConfigHandler::ConfigHandler(const nlohmann::json& config_) : config(config_) {}

  std::string ConfigHandler::toString() const{
    return GenericToolbox::Json::toReadableString( config );
  }
  const nlohmann::json &ConfigHandler::getConfig() const {
    return config;
  }

  nlohmann::json &ConfigHandler::getConfig(){
    return config;
  }


  void ConfigHandler::override( const std::string& filePath_ ){
    LogInfo << "Overriding config with \"" << filePath_ << "\"" << std::endl;
    LogThrowIf(not GenericToolbox::doesPathIsFile(filePath_), "Could not find " << filePath_);

    auto override{ConfigUtils::readConfigFile(filePath_)};
    ConfigUtils::unfoldConfig(override);
    ConfigUtils::applyOverrides(config, override);
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
    nlohmann::json flat = config.flatten();
    LogWarning << "    Original value: " << flat.at(split[0])
               << std::endl;
    if (flat.at(split[0]).is_string()) flat.at(split[0]) = split[1];
    else flat.at(split[0]) = nlohmann::json::parse(split[1]);
    LogWarning << "         New value: " << flat.at(split[0])
               << std::endl;
    config = flat.unflatten();
  }
  void ConfigHandler::flatOverride( const std::vector<std::string>& flattenEntryList_ ){
    for( auto& flattenEntry : flattenEntryList_ ){ this->override( flattenEntry ); }
  }


  void ConfigHandler::exportToJsonFile(const std::string &filePath_) const {
    auto outPath{filePath_};

    if( not GenericToolbox::doesStringEndsWithSubstring(outPath, ".json") ){
      // add extension if missing
      outPath += ".json";
    }

    LogInfo << "Writing as: " << outPath << std::endl;
    GenericToolbox::dumpStringInFile(outPath, this->toString());
    LogInfo << "Unfolded config written as: " << outPath << std::endl;
  }



}
