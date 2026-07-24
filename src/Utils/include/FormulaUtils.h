#ifndef GUNDAM_FORMULA_UTILS_H
#define GUNDAM_FORMULA_UTILS_H

#include <string>
#include <vector>
#include <map>

namespace ConfigUtils { class ConfigReader; }

namespace FormulaUtils {

  struct FormulaComponent{
    std::string name{};
    std::string expr{};
  };

  enum class FormulaResolutionMode{
    StrictVariableDictOnly,
    AllowTreeLeafFallback
  };

  std::vector<FormulaComponent> readFormulaComponents(
      const ConfigUtils::ConfigReader& config_,
      const std::string& fieldName_
  );

  std::string joinFormulaComponents(
      const std::vector<FormulaComponent>& components_,
      const std::string& joinStr_,
      bool skipEmptyExpr_ = true
  );

  std::string buildFormulaString(
      const ConfigUtils::ConfigReader& config_,
      const std::string& fieldName_,
      const std::string& joinStr_,
      bool skipEmptyExpr_ = true
  );

  std::string resolveFormulaReferences(
      const std::string& formula_,
      const std::map<std::string, std::string>& variableDict_,
      FormulaResolutionMode resolutionMode_ = FormulaResolutionMode::StrictVariableDictOnly
  );

  std::string buildAndResolveFormulaString(
      const ConfigUtils::ConfigReader& config_,
      const std::string& fieldName_,
      const std::string& joinStr_,
      const std::map<std::string, std::string>& variableDict_,
      FormulaResolutionMode resolutionMode_ = FormulaResolutionMode::StrictVariableDictOnly,
      bool skipEmptyExpr_ = true
  );

}

#endif
