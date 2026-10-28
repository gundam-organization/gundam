#include "FormulaCompiler.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
  using namespace Backends;

  struct Function {
    const char* name;
    FormulaOpcode opcode;
    int arity;
  };

  const Function functions[]{
      {"exp", FormulaOpcode::Exp, 1}, {"log", FormulaOpcode::Log, 1},
      {"log10", FormulaOpcode::Log10, 1}, {"sqrt", FormulaOpcode::Sqrt, 1},
      {"abs", FormulaOpcode::Abs, 1}, {"fabs", FormulaOpcode::Abs, 1},
      {"sin", FormulaOpcode::Sin, 1}, {"cos", FormulaOpcode::Cos, 1},
      {"tan", FormulaOpcode::Tan, 1}, {"asin", FormulaOpcode::Asin, 1},
      {"acos", FormulaOpcode::Acos, 1}, {"atan", FormulaOpcode::Atan, 1},
      {"atan2", FormulaOpcode::Atan2, 2}, {"sinh", FormulaOpcode::Sinh, 1},
      {"cosh", FormulaOpcode::Cosh, 1}, {"tanh", FormulaOpcode::Tanh, 1},
      {"erf", FormulaOpcode::Erf, 1}, {"erfc", FormulaOpcode::Erfc, 1},
      {"floor", FormulaOpcode::Floor, 1}, {"ceil", FormulaOpcode::Ceil, 1},
      {"pow", FormulaOpcode::Pow, 2}, {"power", FormulaOpcode::Pow, 2},
      {"min", FormulaOpcode::Min, 2}, {"max", FormulaOpcode::Max, 2}};

  int arity(FormulaOpcode opcode_) {
    switch( opcode_ ){
      case FormulaOpcode::LoadInput:
      case FormulaOpcode::LoadConstant: return 0;
      case FormulaOpcode::Add:
      case FormulaOpcode::Subtract:
      case FormulaOpcode::Multiply:
      case FormulaOpcode::Divide: return 2;
      case FormulaOpcode::Negate: return 1;
      default:
        for( const auto& function : functions ){
          if( function.opcode == opcode_ ){ return function.arity; }
        }
        throw std::runtime_error("Formula bytecode: unknown opcode");
    }
  }

  class Parser {
  public:
    Parser(const std::string& expression_, const std::vector<std::string>& inputNames_,
           const std::map<std::string, double>& constants_)
        : _expression_(expression_), _constants_(constants_) {
      _program_.inputCount = inputNames_.size();
      for( std::size_t i = 0; i < inputNames_.size(); ++i ){
        // Empty slots are unused inputs when importing ROOT's binding layout.
        if( inputNames_[i].empty() ){ continue; }
        if( not _inputs_.emplace(inputNames_[i], i).second ){ fail("duplicate input " + inputNames_[i]); }
      }
    }

    FormulaProgram parse() {
      expression();
      whitespace();
      if( _position_ != _expression_.size() ){ fail("unsupported token"); }
      std::size_t depth{0};
      for( const auto& instruction : _program_.instructions ){
        const auto count = arity(instruction.opcode);
        depth = depth + 1 - count;
        _program_.stackSize = std::max(_program_.stackSize, depth);
      }
      _program_.validate();
      return std::move(_program_);
    }

  private:
    [[noreturn]] void fail(const std::string& message_) const {
      throw std::runtime_error("Formula bytecode at position " + std::to_string(_position_)
                               + ": " + message_ + " in \"" + _expression_ + "\"");
    }
    void whitespace() {
      while( _position_ < _expression_.size()
             and std::isspace(static_cast<unsigned char>(_expression_[_position_])) ){ ++_position_; }
    }
    bool take(char token_) {
      whitespace();
      if( _position_ == _expression_.size() or _expression_[_position_] != token_ ){ return false; }
      ++_position_;
      return true;
    }
    void require(char token_) {
      if( not take(token_) ){ fail(std::string("expected '") + token_ + "'"); }
    }
    void emit(FormulaOpcode opcode_, std::size_t operand_ = 0) {
      if( _program_.instructions.size() >= 4096
          or operand_ > std::numeric_limits<std::uint32_t>::max() ){ fail("expression is too large"); }
      _program_.instructions.push_back({opcode_, std::uint32_t(operand_)});
    }
    void constant(double value_) {
      emit(FormulaOpcode::LoadConstant, _program_.constants.size());
      _program_.constants.push_back(value_);
    }
    void expression() {
      product();
      while( true ){
        if( take('+') ){ product(); emit(FormulaOpcode::Add); }
        else if( take('-') ){ product(); emit(FormulaOpcode::Subtract); }
        else{ return; }
      }
    }
    void product() {
      unary();
      while( true ){
        if( take('*') ){ unary(); emit(FormulaOpcode::Multiply); }
        else if( take('/') ){ unary(); emit(FormulaOpcode::Divide); }
        else{ return; }
      }
    }
    void unary() {
      if( ++_depth_ > 128 ){ fail("expression nesting exceeds 128"); }
      if( take('+') ){ unary(); }
      else if( take('-') ){ unary(); emit(FormulaOpcode::Negate); }
      else{ power(); }
      --_depth_;
    }
    void power() {
      primary();
      whitespace();
      bool exponent = take('^');
      if( not exponent and _expression_.compare(_position_, 2, "**") == 0 ){
        _position_ += 2;
        exponent = true;
      }
      if( exponent ){ unary(); emit(FormulaOpcode::Pow); }
    }
    void primary() {
      whitespace();
      if( take('(') ){ expression(); require(')'); return; }
      if( take('[') ){
        const auto end = _expression_.find(']', _position_);
        if( end == std::string::npos ){ fail("unterminated input name"); }
        const auto name = _expression_.substr(_position_, end - _position_);
        _position_ = end + 1;
        const auto input = _inputs_.find(name);
        if( input != _inputs_.end() ){ emit(FormulaOpcode::LoadInput, input->second); return; }
        const auto value = _constants_.find(name);
        if( value != _constants_.end() ){ constant(value->second); return; }
        fail("undeclared input [" + name + "]");
      }
      if( _position_ == _expression_.size() ){ fail("expected an operand"); }
      const auto first = static_cast<unsigned char>(_expression_[_position_]);
      if( std::isdigit(first) or first == '.' ){
        const char* begin = _expression_.c_str() + _position_;
        char* end{nullptr};
        const double value = std::strtod(begin, &end);
        if( end == begin ){ fail("invalid number"); }
        _position_ += std::size_t(end - begin);
        constant(value);
        return;
      }
      const auto begin = _position_;
      while( _position_ < _expression_.size() ){
        const auto c = static_cast<unsigned char>(_expression_[_position_]);
        if( not std::isalnum(c) and c != '_' and c != ':' ){ break; }
        ++_position_;
      }
      std::string name = _expression_.substr(begin, _position_ - begin);
      // Explicitly support common spellings, without accepting arbitrary C++.
      if( name.compare(0, 7, "TMath::") == 0 ){ name.erase(0, 7); }
      else if( name.compare(0, 5, "std::") == 0 ){ name.erase(0, 5); }
      std::transform(name.begin(), name.end(), name.begin(),
                     [](unsigned char c_){ return char(std::tolower(c_)); });
      if( name == "pi" or name == "twopi" or name == "e" ){
        if( take('(') ){ require(')'); }
        constant(name == "e" ? std::exp(1.) : (name == "twopi" ? 2. : 1.) * std::acos(-1.));
        return;
      }
      for( const auto& function : functions ){
        if( name != function.name ){ continue; }
        require('(');
        expression();
        if( function.arity == 2 ){ require(','); expression(); }
        require(')');
        emit(function.opcode);
        return;
      }
      fail("unsupported function or token '" + name + "'");
    }

    const std::string& _expression_;
    const std::map<std::string, double>& _constants_;
    std::map<std::string, std::size_t> _inputs_;
    FormulaProgram _program_;
    std::size_t _position_{0};
    int _depth_{0};
  };
}

void Backends::FormulaProgram::validate() const {
  std::size_t depth{0}, maximum{0};
  for( const auto& instruction : instructions ){
    const auto count = std::size_t(arity(instruction.opcode));
    if( depth < count
        or (instruction.opcode == FormulaOpcode::LoadInput and instruction.operand >= inputCount)
        or (instruction.opcode == FormulaOpcode::LoadConstant and instruction.operand >= constants.size()) ){
      throw std::runtime_error("Invalid formula bytecode operand or stack underflow");
    }
    depth = depth - count + 1;
    maximum = std::max(maximum, depth);
  }
  if( depth != 1 or maximum != stackSize ){
    throw std::runtime_error("Invalid formula bytecode result or stack size");
  }
}

Backends::FormulaProgram Backends::FormulaCompiler::compile(const std::string& expression_,
    const std::vector<std::string>& inputNames_, const std::map<std::string, double>& constants_) {
  return Parser(expression_, inputNames_, constants_).parse();
}
