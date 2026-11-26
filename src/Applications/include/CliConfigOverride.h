#ifndef GUNDAM_CLI_CONFIG_OVERRIDE_H
#define GUNDAM_CLI_CONFIG_OVERRIDE_H

#include "ConfigUtils.h"

class CmdLineParser;

namespace CliConfigOverride {
  void applyConfigOverride(const CmdLineParser& parser_, JsonType& config_);
}

#endif // GUNDAM_CLI_CONFIG_OVERRIDE_H
