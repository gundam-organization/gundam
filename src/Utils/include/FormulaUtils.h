#ifndef GUNDAM_FORMULA_UTILS_H
#define GUNDAM_FORMULA_UTILS_H

#include <string>
#include <vector>
#include <map>

namespace FormulaUtils {

  struct FormulaComponent{
    std::string name{};
    std::string expr{};
  };

  enum class FormulaResolutionMode{
    StrictVariableDictOnly,
    AllowTreeLeafFallback
  };

  std::string joinFormulaComponents(
      const std::vector<FormulaComponent>& components_,
      const std::string& joinStr_,
      bool skipEmptyExpr_ = true
  );

  std::string resolveFormulaReferences(
      const std::string& formula_,
      const std::map<std::string, std::string>& variableDict_,
      FormulaResolutionMode resolutionMode_ = FormulaResolutionMode::StrictVariableDictOnly
  );

  std::vector<std::string> extractFormulaReferenceNames(const std::string& formula_);

  std::vector<std::string> extractBareVariableNames(const std::string& formula_);

  std::vector<std::string> extractEventVariableNames(const std::string& formula_);

  std::string convertBareVariablesToFormulaParameters(const std::string& formula_);

}

#endif
