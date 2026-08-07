#include "FormulaUtils.h"

#include "GenericToolbox.Utils.h"
#include "Logger.h"

#include <algorithm>
#include <cctype>
#include <set>
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

  size_t findNextNonSpace(const std::string& str_, size_t pos_){
    while( pos_ < str_.size() and std::isspace(static_cast<unsigned char>(str_[pos_])) ){ pos_++; }
    return pos_;
  }

  bool isIdentifierChar(char c_){
    return std::isalnum(static_cast<unsigned char>(c_)) or c_ == '_';
  }

  bool isIdentifierStart(char c_){
    return std::isalpha(static_cast<unsigned char>(c_)) or c_ == '_';
  }

  // ROOT qualifies friend-tree leaves with an alias, e.g. friend.branch.
  // Treat the complete qualified path as one leaf expression rather than as
  // two independent bare variables.
  size_t consumeQualifiedIdentifier(const std::string& formula_, size_t cursor_){
    while( cursor_ < formula_.size() and isIdentifierChar(formula_[cursor_]) ){ cursor_++; }
    while(
        cursor_ + 1 < formula_.size()
        and formula_[cursor_] == '.'
        and isIdentifierStart(formula_[cursor_ + 1])
    ){
      cursor_ += 2;
      while( cursor_ < formula_.size() and isIdentifierChar(formula_[cursor_]) ){ cursor_++; }
    }
    return cursor_;
  }

  bool isBareIdentifierFormulaToken(
      const std::string& formula_,
      size_t begin_,
      size_t end_
  ){
    if( begin_ >= end_ ){ return false; }

    // Namespace components and function names are not event variables.
    if( begin_ >= 2 and formula_[begin_ - 1] == ':' and formula_[begin_ - 2] == ':' ){ return false; }
    if( end_ + 1 < formula_.size() and formula_[end_] == ':' and formula_[end_ + 1] == ':' ){ return false; }

    auto nextNonSpace = findNextNonSpace(formula_, end_);
    if( nextNonSpace < formula_.size() and formula_[nextNonSpace] == '(' ){ return false; }

    auto token = formula_.substr(begin_, end_ - begin_);
    static const std::set<std::string> reservedTokens{
        "true", "false",
        "and", "or", "not",
        "pi", "e"
    };
    return reservedTokens.find(token) == reservedTokens.end();
  }

  void addUnique(std::vector<std::string>& list_, const std::string& value_){
    if( std::find(list_.begin(), list_.end(), value_) == list_.end() ){ list_.emplace_back(value_); }
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

std::vector<std::string> extractFormulaReferenceNames(const std::string& formula_){
  std::vector<std::string> output;

  size_t cursor{0};
  while( cursor < formula_.size() ){
    auto openingBracketPos = formula_.find('[', cursor);
    if( openingBracketPos == std::string::npos ){ break; }

    auto closingBracketPos = formula_.find(']', openingBracketPos + 1);
    LogExitIf(
        closingBracketPos == std::string::npos,
        "Invalid formula reference in \"" << formula_ << "\": missing closing ']'."
    );

    auto referenceName = formula_.substr(openingBracketPos + 1, closingBracketPos - openingBracketPos - 1);
    if(
        isFormulaReferenceName(referenceName)
        and isStandaloneFormulaReference(formula_, openingBracketPos)
    ){
      addUnique(output, referenceName);
    }

    cursor = closingBracketPos + 1;
  }

  return output;
}

std::vector<std::string> extractBareVariableNames(const std::string& formula_){
  std::vector<std::string> output;

  size_t cursor{0};
  while( cursor < formula_.size() ){
    if( formula_[cursor] == '[' ){
      auto closingBracketPos = formula_.find(']', cursor + 1);
      LogExitIf(
          closingBracketPos == std::string::npos,
          "Invalid formula reference in \"" << formula_ << "\": missing closing ']'."
      );
      cursor = closingBracketPos + 1;
      continue;
    }

    if( not isIdentifierStart(formula_[cursor]) ){
      cursor++;
      continue;
    }

    auto begin = cursor;
    cursor = consumeQualifiedIdentifier(formula_, cursor + 1);

    if( isBareIdentifierFormulaToken(formula_, begin, cursor) ){
      addUnique(output, formula_.substr(begin, cursor - begin));
    }
  }

  return output;
}

std::vector<std::string> extractEventVariableNames(const std::string& formula_){
  auto output = extractFormulaReferenceNames(formula_);
  for( const auto& varName : extractBareVariableNames(formula_) ){ addUnique(output, varName); }
  return output;
}

std::string convertBareVariablesToFormulaParameters(const std::string& formula_){
  std::string output;
  output.reserve(formula_.size() + 16);

  size_t cursor{0};
  while( cursor < formula_.size() ){
    if( formula_[cursor] == '[' ){
      auto closingBracketPos = formula_.find(']', cursor + 1);
      LogExitIf(
          closingBracketPos == std::string::npos,
          "Invalid formula reference in \"" << formula_ << "\": missing closing ']'."
      );
      output.append(formula_, cursor, closingBracketPos - cursor + 1);
      cursor = closingBracketPos + 1;
      continue;
    }

    if( not isIdentifierStart(formula_[cursor]) ){
      output += formula_[cursor];
      cursor++;
      continue;
    }

    auto begin = cursor;
    cursor = consumeQualifiedIdentifier(formula_, cursor + 1);

    if( isBareIdentifierFormulaToken(formula_, begin, cursor) ){
      output += "[";
      output.append(formula_, begin, cursor - begin);
      output += "]";
    }
    else{
      output.append(formula_, begin, cursor - begin);
    }
  }

  return output;
}

}
