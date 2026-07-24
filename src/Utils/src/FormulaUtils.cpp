#include "FormulaUtils.h"

#include "ConfigUtils.h"

#include "GenericToolbox.Json.h"
#include "Logger.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace {

  std::string readScalarFormulaExpr(
      const JsonType& entry_,
      const std::string& location_
  ){
    if( entry_.is_string() ){
      return GenericToolbox::Json::get<std::string>(entry_);
    }
    if( entry_.is_number() or entry_.is_boolean() ){
      return entry_.dump();
    }

    LogExit(location_ << ": formula expression must be a scalar value.");
    return {};
  }

  bool isFormulaReferenceName(const std::string& name_){
    if( name_.empty() ){ return false; }
    if( not (std::isalpha(static_cast<unsigned char>(name_[0])) or name_[0] == '_') ){ return false; }

    return std::all_of(name_.begin() + 1, name_.end(), [](char c_){
      return std::isalnum(static_cast<unsigned char>(c_)) or c_ == '_';
    });
  }

  bool isStandaloneFormulaReference(const std::string& formula_, size_t openingBracketPos_){
    if( openingBracketPos_ == 0 ){ return true; }

    auto previousChar = formula_[openingBracketPos_ - 1];
    return not (std::isalnum(static_cast<unsigned char>(previousChar)) or previousChar == '_');
  }

  std::string joinReferenceStack(const std::vector<std::string>& stack_, const std::string& extra_){
    std::stringstream ss;
    for( const auto& entry : stack_ ){
      if( ss.tellp() != std::streampos(0) ){ ss << " -> "; }
      ss << entry;
    }
    if( not extra_.empty() ){
      if( ss.tellp() != std::streampos(0) ){ ss << " -> "; }
      ss << extra_;
    }
    return ss.str();
  }

  std::string buildComponentLocation(
      const ConfigUtils::ConfigReader& config_,
      const std::string& fieldName_,
      size_t componentIndex_
  ){
    return GenericToolbox::joinPath(config_.getParentPath(), fieldName_, componentIndex_);
  }

  std::string resolveFormulaReferencesImpl(
      const std::string& formula_,
      const std::map<std::string, std::string>& variableDict_,
      std::vector<std::string>& referenceStack_,
      FormulaUtils::FormulaResolutionMode resolutionMode_
  ){
    std::string output;
    output.reserve(formula_.size());

    size_t cursor{0};
    while( cursor < formula_.size() ){
      auto openingBracketPos = formula_.find('[', cursor);
      if( openingBracketPos == std::string::npos ){
        output.append(formula_, cursor, std::string::npos);
        break;
      }

      output.append(formula_, cursor, openingBracketPos - cursor);

      auto closingBracketPos = formula_.find(']', openingBracketPos + 1);
      LogExitIf(
          closingBracketPos == std::string::npos,
          "Invalid formula reference in \"" << formula_ << "\": missing closing ']'."
      );

      auto referenceName = formula_.substr(openingBracketPos + 1, closingBracketPos - openingBracketPos - 1);
      if(
          not isFormulaReferenceName(referenceName)
          or not isStandaloneFormulaReference(formula_, openingBracketPos)
      ){
        output.append(formula_, openingBracketPos, closingBracketPos - openingBracketPos + 1);
        cursor = closingBracketPos + 1;
        continue;
      }

      auto dictEntry = variableDict_.find(referenceName);
      if( dictEntry == variableDict_.end() ){
        if( resolutionMode_ == FormulaUtils::FormulaResolutionMode::AllowTreeLeafFallback ){
          output += referenceName;
          cursor = closingBracketPos + 1;
          continue;
        }
        LogExit("Unknown variableDict reference [" << referenceName << "] in formula \"" << formula_ << "\".");
      }

      LogExitIf(
          std::find(referenceStack_.begin(), referenceStack_.end(), referenceName) != referenceStack_.end(),
          "Cyclic variableDict reference detected: " << joinReferenceStack(referenceStack_, referenceName)
      );

      referenceStack_.emplace_back(referenceName);
      auto resolvedExpression = resolveFormulaReferencesImpl(
          dictEntry->second,
          variableDict_,
          referenceStack_,
          resolutionMode_
      );
      referenceStack_.pop_back();

      output += "(" + resolvedExpression + ")";
      cursor = closingBracketPos + 1;
    }

    return output;
  }

}

namespace FormulaUtils {

std::vector<FormulaComponent> readFormulaComponents(
    const ConfigUtils::ConfigReader& config_,
    const std::string& fieldName_
){
  std::vector<FormulaComponent> out;

  auto keyValuePair = config_.getConfigEntry(fieldName_);
  if( keyValuePair.second == nullptr ){ return out; }

  const auto& formulaConfig = *keyValuePair.second;

  if( formulaConfig.is_string() or formulaConfig.is_number() or formulaConfig.is_boolean() ){
    out.emplace_back();
    out.back().expr = readScalarFormulaExpr(formulaConfig, GenericToolbox::joinPath(config_.getParentPath(), keyValuePair.first));
    return out;
  }

  LogExitIf(
      not formulaConfig.is_array(),
      config_.getParentPath() << "/" << keyValuePair.first << ": formula entry must be a string or an array."
  );

  if( formulaConfig.empty() ){ return out; }

  bool hasStringEntries{false};
  bool hasObjectEntries{false};
  for( const auto& entry : formulaConfig ){
    hasStringEntries = hasStringEntries or entry.is_string() or entry.is_number() or entry.is_boolean();
    hasObjectEntries = hasObjectEntries or entry.is_structured();
    LogExitIf(
        not entry.is_string() and not entry.is_number() and not entry.is_boolean() and not entry.is_structured(),
        config_.getParentPath() << "/" << keyValuePair.first << ": formula list entries must be scalars or dictionaries."
    );
  }

  LogExitIf(
      hasStringEntries and hasObjectEntries,
      config_.getParentPath() << "/" << keyValuePair.first << ": mixed string/dictionary formula lists are not supported."
  );

  out.reserve(formulaConfig.size());
  for( size_t iEntry = 0 ; iEntry < formulaConfig.size() ; iEntry++ ){
    const auto& entry = formulaConfig[iEntry];
    out.emplace_back();

    if( entry.is_string() or entry.is_number() or entry.is_boolean() ){
      out.back().expr = readScalarFormulaExpr(entry, buildComponentLocation(config_, keyValuePair.first, iEntry));
      continue;
    }

    ConfigUtils::ConfigReader componentConfig(entry);
    componentConfig.setParentPath(buildComponentLocation(config_, keyValuePair.first, iEntry));
    componentConfig.defineFields({
        {ConfigUtils::ConfigReader::FieldDefinition::MANDATORY, "name"},
        {ConfigUtils::ConfigReader::FieldDefinition::MANDATORY, "expr"}
    });
    componentConfig.checkConfiguration();
    componentConfig.fillValue(out.back().name, "name");
    auto exprEntry = componentConfig.getConfigEntry("expr").second;
    LogExitIf(exprEntry == nullptr, componentConfig.getParentPath() << ": missing mandatory field \"expr\".");
    out.back().expr = readScalarFormulaExpr(*exprEntry, GenericToolbox::joinPath(componentConfig.getParentPath(), "expr"));
    componentConfig.printUnusedKeys();

    LogExitIf(
        out.back().name.empty(),
        componentConfig.getParentPath() << ": formula component \"name\" cannot be empty."
    );
  }

  return out;
}

std::string joinFormulaComponents(
    const std::vector<FormulaComponent>& components_,
    const std::string& joinStr_,
    bool skipEmptyExpr_
){
  std::vector<std::string> formulaList;
  formulaList.reserve(components_.size());

  for( const auto& component : components_ ){
    if( skipEmptyExpr_ and component.expr.empty() ){ continue; }
    formulaList.emplace_back("(" + component.expr + ")");
  }

  if( formulaList.empty() ){ return ""; }
  return GenericToolbox::joinVectorString(formulaList, joinStr_);
}

std::string buildFormulaString(
    const ConfigUtils::ConfigReader& config_,
    const std::string& fieldName_,
    const std::string& joinStr_,
    bool skipEmptyExpr_
){
  return joinFormulaComponents(readFormulaComponents(config_, fieldName_), joinStr_, skipEmptyExpr_);
}

std::string resolveFormulaReferences(
    const std::string& formula_,
    const std::map<std::string, std::string>& variableDict_,
    FormulaResolutionMode resolutionMode_
){
  std::vector<std::string> referenceStack;
  return resolveFormulaReferencesImpl(formula_, variableDict_, referenceStack, resolutionMode_);
}

std::string buildAndResolveFormulaString(
    const ConfigUtils::ConfigReader& config_,
    const std::string& fieldName_,
    const std::string& joinStr_,
    const std::map<std::string, std::string>& variableDict_,
    FormulaResolutionMode resolutionMode_,
    bool skipEmptyExpr_
){
  return resolveFormulaReferences(
      buildFormulaString(config_, fieldName_, joinStr_, skipEmptyExpr_),
      variableDict_,
      resolutionMode_
  );
}

}
