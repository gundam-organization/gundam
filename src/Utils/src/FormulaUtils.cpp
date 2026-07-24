#include "FormulaUtils.h"

#include "GenericToolbox.Utils.h"
#include "Logger.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace {

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

std::string resolveFormulaReferences(
    const std::string& formula_,
    const std::map<std::string, std::string>& variableDict_,
    FormulaResolutionMode resolutionMode_
){
  std::vector<std::string> referenceStack;
  return resolveFormulaReferencesImpl(formula_, variableDict_, referenceStack, resolutionMode_);
}

}
