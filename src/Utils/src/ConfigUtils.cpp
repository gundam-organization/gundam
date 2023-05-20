//
// Created by Adrien Blanchet on 28/02/2023.
//

#include "ConfigUtils.h"

#include "GenericToolbox.Json.h"
#include "GenericToolbox.Yaml.h"
#include "Logger.h"

#include "nlohmann/json.hpp"


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
    std::vector<std::string> listOfIdentifiers{{"name"}, {"__INDEX__"}}; // specific keys like "name" might help reference the lists

    std::vector<std::string> jsonPath{};
    std::function<void(nlohmann::json&, const nlohmann::json&)> overrideRecursive = [&](nlohmann::json& outEntry_, const nlohmann::json& overrideEntry_){
      LogDebug(debug) << GET_VAR_NAME_VALUE(GenericToolbox::joinPath( jsonPath )) << std::endl;

      if( overrideEntry_.is_array() ){
        // entry is list
        LogReturnIf(not outEntry_.is_array(), GenericToolbox::joinPath( jsonPath ) << " is not an array.");

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
                LogAlert << "Adding: " << GenericToolbox::joinPath(jsonPath, outEntry_.size()) << std::endl;
                outEntry_.emplace_back(overrideListEntry.value());
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
                LogError << "Could not find key" << std::endl;
                continue;
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
        // entry is dictionary
        for( auto& overrideEntry : overrideEntry_.items() ){

          // addition mode:
          if( not GenericToolbox::Json::doKeyExist(outEntry_, overrideEntry.key()) ){
            if( overrideEntry.key() != "__INDEX__" ){
              if( allowAddMissingKey ){
                LogAlert << "Adding: " << GenericToolbox::joinPath(jsonPath, overrideEntry.key()) << std::endl;
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
    overrideRecursive(outConfig_, overrideConfig_);

  }

}
