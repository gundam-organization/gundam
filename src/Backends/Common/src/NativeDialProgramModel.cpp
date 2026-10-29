#include "NativeDialProgramModel.h"
#include "DialProgramValidation.h"
#include "Semantics/NativeFormulaProgram.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace {
  using namespace Backends;
  template<typename T> void key(std::string& out_, const T& value_) {
    out_.append(reinterpret_cast<const char*>(&value_), sizeof(value_));
  }
  unsigned int arity(FormulaOpcode op_) {
    switch( op_ ){
      case FormulaOpcode::LoadInput:
      case FormulaOpcode::LoadConstant: return 0;
#define GUNDAM_FORMULA_UNARY(opcode, operation) case FormulaOpcode::opcode: return 1;
#include "DeviceMath/SharedFormulaOperations.h"
#undef GUNDAM_FORMULA_UNARY
#define GUNDAM_FORMULA_BINARY(opcode, operation) case FormulaOpcode::opcode: return 2;
#include "DeviceMath/SharedFormulaOperations.h"
#undef GUNDAM_FORMULA_BINARY
    }
    throw std::runtime_error("Invalid native formula opcode");
  }
  bool fold(FormulaOpcode op_, double a_, double b_, double& result_) {
    switch( op_ ){
      case FormulaOpcode::Negate: result_ = -a_; return true;
      case FormulaOpcode::Add: result_ = a_ + b_; return true;
      case FormulaOpcode::Subtract: result_ = a_ - b_; return true;
      case FormulaOpcode::Multiply: { volatile double product = a_ * b_; result_ = product; return true; }
      case FormulaOpcode::Divide: result_ = a_ / b_; return true;
      case FormulaOpcode::Min: result_ = a_ <= b_ ? a_ : b_; return true;
      case FormulaOpcode::Max: result_ = a_ >= b_ ? a_ : b_; return true;
      default: return false; // Keep transcendental evaluation on its native hardware.
    }
  }
  void recognize(FormulaProgram& program_) {
    auto& plan = program_.nativePlan;
    const auto& code = program_.instructions;
    auto load = [&](std::size_t instruction_, unsigned int slot_) {
      const auto& in = code[instruction_];
      if( in.opcode == FormulaOpcode::LoadInput ){ plan.inputs[slot_] = in.operand; return true; }
      if( in.opcode == FormulaOpcode::LoadConstant ){
        plan.literalMask |= 1u << slot_;
        plan.literals[slot_] = program_.constants[in.operand];
        return true;
      }
      return false;
    };
    if( code.size() == 1 and load(0, 0) ){
      plan.kernel = plan.literalMask ? NativeFormulaKernel::Constant : NativeFormulaKernel::Input;
    }
    else if( code.size() == 3 and load(0, 0) and load(1, 1) ){
      const auto op = code[2].opcode;
      if( op == FormulaOpcode::Add or op == FormulaOpcode::Subtract or op == FormulaOpcode::Multiply
          or op == FormulaOpcode::Divide or op == FormulaOpcode::Min or op == FormulaOpcode::Max ){
        plan.kernel = NativeFormulaKernel::Binary; plan.operation = op;
      }
    }
    else if( code.size() == 5 and load(0, 0) and load(1, 1) and load(2, 2)
             and code[3].opcode == FormulaOpcode::Multiply and code[4].opcode == FormulaOpcode::Add ){
      plan.kernel = NativeFormulaKernel::MultiplyThenAdd;
    }
  }
  struct Node {
    FormulaInstruction instruction{};
    int left{-1}, right{-1};
    bool constant{false}, bound{false};
    double value{0.};
  };
  struct Builder {
    NativeDialProgramModel& out;
    const PropagationView& source;
    std::unordered_map<std::string, std::size_t> formulas, composites;

    std::pair<std::size_t, std::size_t> formula(const BackendDialDescriptor& dial_) {
      const auto& original = source.formulaPrograms.at(dial_.formulaProgramIndex);
      const auto* bindings = original.inputCount ? source.formulaInputs.data() + dial_.firstFormulaInput : nullptr;
      std::vector<Node> nodes;
      std::vector<int> stack;
      nodes.reserve(original.instructions.size()); stack.reserve(original.stackSize);
      for( const auto& instruction : original.instructions ){
        Node node; node.instruction = instruction;
        const auto count = arity(instruction.opcode);
        if( count == 0 ){
          if( instruction.opcode == FormulaOpcode::LoadConstant ){
            node.constant = true; node.value = original.constants[instruction.operand];
          }
          else if( bindings[instruction.operand].parameterIndex < 0 ){
            node.constant = node.bound = true; node.value = bindings[instruction.operand].constant;
          }
        }
        else{
          node.right = count == 2 ? stack.back() : -1;
          if( count == 2 ){ stack.pop_back(); }
          node.left = stack.back(); stack.pop_back();
          const auto& a = nodes[node.left];
          const auto& b = count == 2 ? nodes[node.right] : a;
          node.bound = a.bound or b.bound;
          if( a.constant and b.constant ){ node.constant = fold(instruction.opcode, a.value, b.value, node.value); }
        }
        stack.push_back(static_cast<int>(nodes.size())); nodes.push_back(node);
      }
      FormulaProgram optimized;
      std::vector<FormulaInput> inputs;
      std::unordered_map<unsigned int, unsigned int> inputSlots;
      auto emit = [&](auto&& self_, int index_) -> void {
        const auto& node = nodes[index_];
        if( node.constant ){
          if( node.bound ){
            optimized.instructions.push_back({FormulaOpcode::LoadInput, static_cast<FormulaIndex>(inputs.size())});
            inputs.push_back({-1, node.value});
          }
          else{
            optimized.instructions.push_back({FormulaOpcode::LoadConstant, static_cast<FormulaIndex>(optimized.constants.size())});
            optimized.constants.push_back(node.value);
          }
        }
        else if( node.instruction.opcode == FormulaOpcode::LoadInput ){
          auto found = inputSlots.emplace(node.instruction.operand, static_cast<unsigned int>(inputs.size()));
          if( found.second ){ inputs.push_back(bindings[node.instruction.operand]); }
          optimized.instructions.push_back({FormulaOpcode::LoadInput, found.first->second});
        }
        else{
          self_(self_, node.left);
          if( node.right >= 0 ){ self_(self_, node.right); }
          optimized.instructions.push_back(node.instruction);
        }
      };
      emit(emit, stack.back());
      std::size_t depth = 0;
      for( const auto& instruction : optimized.instructions ){
        depth = depth + 1 - arity(instruction.opcode);
        optimized.stackSize = std::max(optimized.stackSize, depth);
      }
      optimized.inputCount = inputs.size();
      optimized.validate(); recognize(optimized);
      std::string signature;
      key(signature, optimized.inputCount);
      key(signature, optimized.instructions.size()); key(signature, optimized.constants.size());
      for( const auto& instruction : optimized.instructions ){ key(signature, instruction.opcode); key(signature, instruction.operand); }
      for( double value : optimized.constants ){ key(signature, value); }
      const auto inserted = formulas.emplace(std::move(signature), out.formulaPrograms.size());
      if( inserted.second ){ out.formulaPrograms.push_back(std::move(optimized)); }
      const auto offset = out.formulaInputs.size();
      out.formulaInputs.insert(out.formulaInputs.end(), inputs.begin(), inputs.end());
      return {inserted.first->second, offset};
    }
    void composite(const CompositeProgram& original_) {
      std::vector<CompositeInstruction> instructions;
      std::vector<BackendDialInputDescriptor> inputs;
      const auto firstData = out.compositeNodeData.size();
      std::string signature;
      key(signature, original_.inputCount); key(signature, original_.outputSlot); key(signature, original_.instructionCount);
      for( std::size_t i = 0; i < original_.instructionCount; ++i ){
        auto instruction = source.compositeInstructions[original_.firstInstruction + i];
        auto& dial = instruction.dial;
        std::size_t data = dial.payloadOffset;
        if( dial.type == BackendDialType::Formula ){
          const auto compiled = formula(dial);
          dial.formulaProgramIndex = compiled.first; data = compiled.second;
        }
        out.compositeNodeData.push_back(data);
        const auto inputOffset = inputs.size();
        for( std::size_t j = 0; j < dial.inputCount; ++j ){
          const auto input = source.compositeInputs[dial.firstInput + j];
          inputs.push_back(input);
          key(signature, input.parameterIndex); key(signature, input.useMirror);
          key(signature, input.mirrorMin); key(signature, input.mirrorRange);
        }
        dial.firstInput = inputOffset; dial.payloadOffset = dial.firstFormulaInput = 0;
        key(signature, dial.type); key(signature, dial.inputCount); key(signature, dial.payloadSize);
        key(signature, dial.formulaProgramIndex); key(signature, dial.allowExtrapolation);
        key(signature, dial.hasMinResponse); key(signature, dial.hasMaxResponse);
        key(signature, dial.minResponse); key(signature, dial.maxResponse);
        const auto& smoothing = dial.smoothing;
        key(signature, smoothing.minResponse); key(signature, smoothing.maxResponse);
        key(signature, smoothing.minWidth); key(signature, smoothing.maxWidth);
        key(signature, smoothing.minInverseWidth); key(signature, smoothing.maxInverseWidth);
        key(signature, instruction.inputMin); key(signature, instruction.inputMax);
        instructions.push_back(instruction);
      }
      const auto inserted = composites.emplace(std::move(signature), out.compositePrograms.size());
      if( inserted.second ){
        NativeCompositeTemplate program{out.compositeInstructions.size(), original_.instructionCount,
                                       original_.inputCount, original_.outputSlot, false};
        if( program.inputCount == 2 and instructions.size() == 3 and program.outputSlot == 4 ){
          const auto& a = instructions[0].dial; const auto& b = instructions[1].dial; const auto& c = instructions[2].dial;
          const bool leaf = b.type == BackendDialType::CompactSpline or b.type == BackendDialType::UniformSpline
              or b.type == BackendDialType::MonotonicSpline or b.type == BackendDialType::GeneralSpline or b.type == BackendDialType::Graph;
          program.isChain = a.type == BackendDialType::Formula and leaf and b.inputCount == 1
              and c.type == BackendDialType::Formula and c.inputCount == 1
              and inputs[b.firstInput].parameterIndex == 2 and inputs[c.firstInput].parameterIndex == 3
              and out.formulaPrograms[a.formulaProgramIndex].nativePlan.kernel != NativeFormulaKernel::Interpreter
              and out.formulaPrograms[c.formulaProgramIndex].nativePlan.kernel != NativeFormulaKernel::Interpreter;
        }
        const auto inputBase = out.compositeInputs.size();
        for( auto instruction : instructions ){
          instruction.dial.firstInput += inputBase;
          out.compositeInstructions.push_back(instruction);
        }
        out.compositeInputs.insert(out.compositeInputs.end(), inputs.begin(), inputs.end());
        out.compositePrograms.push_back(program);
      }
      out.compositeInstances.push_back({inserted.first->second, firstData});
    }
  };
}

void Backends::NativeDialProgramModel::build(const PropagationView& source_, const std::vector<BackendDialDescriptor>& roots_) {
  *this = NativeDialProgramModel{};
  dialInputs = source_.dialInputs.data(); payloads = source_.dialPayloads.data();
  if( source_.formulaPrograms.empty() and source_.compositePrograms.empty() ){ return; }
  dials = roots_;
  for( const auto& program : source_.formulaPrograms ){ program.validate(); }
  validateCompositePrograms(source_);
  Builder builder{*this, source_};
  for( auto& dial : dials ){
    validateFormulaDial(source_, dial);
    if( dial.type != BackendDialType::Formula ){ continue; }
    const auto compiled = builder.formula(dial);
    dial.formulaProgramIndex = compiled.first; dial.firstFormulaInput = compiled.second;
  }
  compositeInstances.reserve(source_.compositePrograms.size());
  compositeNodeData.reserve(source_.compositeInstructions.size());
  for( const auto& program : source_.compositePrograms ){ builder.composite(program); }
}
