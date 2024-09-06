//
// Created by Adrien BLANCHET on 16/02/2022.
//

#include "GundamGreetings.h"
#include "ConfigUtils.h"
#include "GundamUtils.h"

#include "CmdLineParser.h"
#include "Logger.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Json.h"

#include "nlohmann/json.hpp"

#include <string>
#include <vector>
#include <cstdlib>


LoggerInit([]{ Logger::setPrefixFormat("{TIME} {SEVERITY}"); });


CmdLineParser clp{};

void compareConfigStage(const JsonType& config1_, const JsonType& config2_);


int main( int argc, char** argv ){
  GundamGreetings g;
  g.setAppName("config compare tool");
  g.hello();

  clp.addOption("config-1", {"-c1"}, "Path to first config file.", 1);
  clp.addOption("config-2", {"-c2"}, "Path to second config file.", 1);
  clp.addOption("overrides-1", {"-of1", "--override-files-1"}, "Provide config files that will override keys in config file 1", -1);
  clp.addOption("overrides-2", {"-of2", "--override-files-2"}, "Provide config files that will override keys in config file 2", -1);

  clp.addTriggerOption("show-all-keys", {"-a"}, "Show all keys. (debug option)");

  LogInfo << clp.getDescription().str() << std::endl;

  LogInfo << "Usage: " << std::endl;
  LogInfo << clp.getConfigSummary() << std::endl << std::endl;

  clp.parseCmdLine(argc, argv);

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clp.getValueSummary() << std::endl << std::endl;
  LogInfo << clp.dumpConfigAsJsonStr() << std::endl;

  if(      clp.isNoOptionTriggered()
    or not clp.isOptionTriggered("config-1")
    or not clp.isOptionTriggered("config-2")
  ){
    LogError << "Missing options. Reminding usage..." << std::endl;
    LogInfo << clp.getConfigSummary() << std::endl;
    exit(EXIT_FAILURE);
  }

  LogInfo << "Reading config..." << std::endl;
  ConfigUtils::ConfigHandler c1{clp.getOptionVal<std::string>("config-1")};
  ConfigUtils::ConfigHandler c2{clp.getOptionVal<std::string>("config-2")};

  c1.override( clp.getOptionValList<std::string>("overrides-1") );
  c2.override( clp.getOptionValList<std::string>("overrides-2") );

  if( GenericToolbox::doesTFileIsValid( clp.getOptionVal<std::string>("config-1") ) ){
    auto f = std::unique_ptr<TFile>( GenericToolbox::openExistingTFile( clp.getOptionVal<std::string>("config-1") ) );
    TNamed* version{nullptr};
    GundamUtils::ObjectReader::readObject<TNamed>( f.get(), {{"gundam/version_TNamed"}, {"gundamFitter/gundamVersion_TNamed"}}, [&](TNamed* obj_){ version = obj_; });
    TNamed* cmdLine{nullptr};
    GundamUtils::ObjectReader::readObject<TNamed>( f.get(), {{"gundam/commandLine_TNamed"}, {"gundamFitter/commandLine_TNamed"}}, [&](TNamed* obj_){ version = cmdLine; });
    if( version != nullptr and cmdLine != nullptr ){
      LogInfo << "config-1 is within .root file. Ran under GUNDAM v" << version->GetTitle() << " with cmdLine: "<< cmdLine->GetTitle() << std::endl;
    }
  }
  if( GenericToolbox::doesTFileIsValid( clp.getOptionVal<std::string>("config-2") ) ){
    auto f = std::unique_ptr<TFile>( GenericToolbox::openExistingTFile( clp.getOptionVal<std::string>("config-2") ) );
    TNamed* version{nullptr};
    GundamUtils::ObjectReader::readObject<TNamed>( f.get(), {{"gundam/version_TNamed"}, {"gundamFitter/gundamVersion_TNamed"}}, [&](TNamed* obj_){ version = obj_; });
    TNamed* cmdLine{nullptr};
    GundamUtils::ObjectReader::readObject<TNamed>( f.get(), {{"gundam/commandLine_TNamed"}, {"gundamFitter/gundamVersion_TNamed"}}, [&](TNamed* obj_){ version = cmdLine; });
    if( version != nullptr and cmdLine != nullptr ){
      LogInfo << "config-2 is within .root file. Ran under GUNDAM v" << version->GetTitle() << " with cmdLine: "<< cmdLine->GetTitle() << std::endl;
    }
  }

  compareConfigStage( c1.getConfig(), c2.getConfig() );


  g.goodbye();

  return EXIT_SUCCESS;
}


void compareConfigStage(const JsonType& config1_, const JsonType& config2_){


  std::vector<std::string> pathBuffer;

  std::function<void(const JsonType&, const JsonType&)> recursiveFct =
      [&](const JsonType& entry1_, const JsonType& entry2_){
        std::string path = GenericToolbox::joinVectorString(pathBuffer, "/");

        if( entry1_.is_array() and entry2_.is_array() ){

          if( entry1_.size() != entry2_.size() ){
            LogAlert << path << "Array size mismatch: " << entry1_.size() << " <-> " << entry2_.size() << std::endl;
          }

          if( entry1_.empty() or entry2_.empty() ){ return; }

          if( GenericToolbox::Json::doKeyExist(entry1_[0], "name") ){
            // trying to fetch by key "name"
            for( int iEntry1 = 0 ; iEntry1 < entry1_.size() ; iEntry1++ ){
              auto name1 = GenericToolbox::Json::fetchValue(entry1_[iEntry1], "name", "");
              bool found1{false};

              for( int iEntry2 = 0 ; iEntry2 < entry2_.size() ; iEntry2++){
                auto name2 = GenericToolbox::Json::fetchValue(entry2_[iEntry2], "name", "");
                if( name1 == name2 ){
                  found1 = true;
                  pathBuffer.emplace_back( "#"+std::to_string(iEntry1) + "(name:" + name1 + ")" );
                  if( iEntry1 != iEntry2 ) pathBuffer.back() += "<->" + std::to_string(iEntry2);
                  recursiveFct(entry1_[iEntry1], entry2_[iEntry2]);
                  pathBuffer.pop_back();
                  break;
                }
              }
              if( not found1 ){
                LogError << path << ": could not find key with name \"" << name1 << "\" in config2." << std::endl;
              }
            }

            // looking for missing keys in 2
            for( const auto & entry2 : entry2_ ){
              auto name2 = GenericToolbox::Json::fetchValue(entry2, "name", "");
              bool found2{false};
              for(const auto & entry1 : entry1_){
                auto name1 = GenericToolbox::Json::fetchValue(entry1, "name", "");
                if( name1 == name2 ){
                  found2 = true;
                  break;
                }
              }
              if( not found2 ){
                LogError << path << ": could not find key with name \"" << name2 << "\" in config1." << std::endl;
              }
            }
          }
          else{
            for( int iEntry = 0 ; iEntry < std::min( entry1_.size(), entry2_.size() ) ; iEntry++ ){
              pathBuffer.emplace_back("#"+std::to_string(iEntry));
              recursiveFct(entry1_[iEntry], entry2_[iEntry]);
              pathBuffer.pop_back();
            }
          }

        }
        else if( entry1_.is_structured() and entry2_.is_structured() ){
          std::vector<std::string> keysToFetch(GenericToolbox::Json::ls(entry1_));
          for( auto& key2 : GenericToolbox::Json::ls(entry2_) ){
            if( not GenericToolbox::doesElementIsInVector(key2, keysToFetch) ){ keysToFetch.emplace_back(key2); }
          }

          for( auto& key : keysToFetch ){
            if     ( not GenericToolbox::Json::doKeyExist(entry1_, key) ){
              LogError << path <<  " -> missing key \"" << key << "\" in c1. Value in c2 is: " << entry2_[key] << std::endl;
              continue;
            }
            else if( not GenericToolbox::Json::doKeyExist(entry2_, key ) ){
              LogError << path << " -> missing key \"" << key << "\" in c2. Value in c1 is: " << entry1_[key] << std::endl;
              continue;
            }

            // both have the key:
            auto content1 = GenericToolbox::Json::fetchValue<JsonType>(entry1_, key);
            auto content2 = GenericToolbox::Json::fetchValue<JsonType>(entry2_, key);

            pathBuffer.emplace_back(key);
            recursiveFct(content1, content2);
            pathBuffer.pop_back();
          }
        }
        else{
          if( entry1_ != entry2_ ){
            LogWarning << path << ": " << entry1_ << " <-> " << entry2_ << std::endl;
          }
          else if( clp.isOptionTriggered("show-all-keys") ){
            LogInfo << path << ": " << entry1_ << " <-> " << entry2_ << std::endl;
          }
        }

        return;
  };

  LogInfo << "Recursive function call..." << std::endl;
  recursiveFct( config1_, config2_ );

}
