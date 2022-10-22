//
// Created by Adrien BLANCHET on 23/10/2022.
//

#ifndef GUNDAM_CONFIGBASEDCLASS_H
#define GUNDAM_CONFIGBASEDCLASS_H

#include "nlohmann/json.hpp"

class ConfigBasedClass {

public:
  ConfigBasedClass() = default;
  ConfigBasedClass(const nlohmann::json& config_);

  virtual ~ConfigBasedClass() = default;

  void setConfig(const nlohmann::json& config_);

  void readConfig();
  void readConfig(const nlohmann::json& config_);

  void initialize();

  bool isConfigReadDone() const;
  bool isInitialized() const;
  const nlohmann::json &getConfig() const;

protected:
  virtual void readConfigImpl() = 0;
  virtual void initializeImpl() = 0;

  nlohmann::json _config_{};

private:
  bool _isConfigReadDone_{false};
  bool _isInitialized_{false};

};


#endif //GUNDAM_CONFIGBASEDCLASS_H
