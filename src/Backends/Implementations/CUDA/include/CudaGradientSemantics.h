#ifndef GUNDAM_CUDA_GRADIENT_SEMANTICS_H
#define GUNDAM_CUDA_GRADIENT_SEMANTICS_H
#include "CudaDialPrograms.h"
#include "Semantics/BackendDeviceGradient.h"

// Copyright (c) 2024 Clark McGrew
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


// Analytic forward differentiation; no finite-difference evaluations.
namespace Backends::CudaGradient {
using S = double;
using namespace Backends;
struct Dual {
  S value, slope;
  GUNDAM_BACKEND_HOST_DEVICE Dual(S value_ = S(0), S slope_ = S(0)) : value(value_), slope(slope_) {}
};
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE Dual operator+(Dual a, Dual b) { return {a.value+b.value, a.slope+b.slope}; }
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE Dual operator-(Dual a, Dual b) { return {a.value-b.value, a.slope-b.slope}; }
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE Dual operator-(Dual a) { return {-a.value, -a.slope}; }
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE Dual operator*(Dual a, Dual b) { return {a.value*b.value, a.slope*b.value+a.value*b.slope}; }
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE Dual operator/(Dual a, Dual b) { return {a.value/b.value, (a.slope*b.value-a.value*b.slope)/(b.value*b.value)}; }
// Zero tangents avoid propagating undefined partials of constant operands.
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE S chain(S tangent, S partial) { return tangent == S(0) ? S(0) : tangent*partial; }
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE Dual mirror(Dual x, unsigned enabled, S minimum, S range) {
  if( !enabled ) return x;
  S folded = ::fmod(x.value-minimum, S(2)*range);
  S slope = folded < S(0) ? S(-1) : S(1);
  if( NativeFormulaMath::abs(folded) > range ) slope = -slope;
  return {(NativeFormulaMath::abs(folded) > range ? S(2)*range-NativeFormulaMath::abs(folded) : NativeFormulaMath::abs(folded))+minimum, chain(x.slope, slope)};
}
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE Dual supervise(Dual x, BackendDialDescriptor dial) {
  S slope(1);
  auto smoothing = dial.smoothing;
  if( dial.hasMinResponse && x.value < dial.minResponse ){
    if( smoothing.minWidth == S(0) || x.value <= smoothing.minResponse-smoothing.minWidth ) slope = S(0);
    else{
      S t = S(.5f)+(x.value-smoothing.minResponse)*smoothing.minInverseWidth;
      slope = t*t*(S(3)-S(2)*t);
    }
  }
  else if( dial.hasMaxResponse && x.value > dial.maxResponse ){
    if( smoothing.maxWidth == S(0) || x.value >= smoothing.maxResponse+smoothing.maxWidth ) slope = S(0);
    else{
      S t = S(.5f)+(smoothing.maxResponse-x.value)*smoothing.maxInverseWidth;
      slope = t*t*(S(3)-S(2)*t);
    }
  }
  return {Semantics::clampDialResponse(dial, x.value), chain(x.slope, slope)};
}
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE Dual cubic(Dual x, S x0, Dual p0, S x1, Dual p1, S x2, Dual p2, S x3, Dual p3) {
  Dual step(x2-x1), fraction = (x-Dual(x1))/step;
  Dual m1 = step*(p2-p0)/Dual(x2-x0), m2 = step*(p3-p1)/Dual(x3-x1);
  if( fraction.value < S(0) ) m2 = m1;
  if( fraction.value > S(1) ) m1 = m2;
  return ((((-Dual(S(2))*p2+Dual(S(2))*p1+m2+m1)*fraction
    +Dual(S(3))*p2-Dual(S(3))*p1-m2-Dual(S(2))*m1)*fraction+m1)*fraction+p1);
}
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE Dual surface(Dual x, Dual y, const S* data, bool bicubic, bool extrapolate) {
  int nx = int(data[0]), ny = int(data[1]);
  const S* xx = data+2;
  const S* yy = xx+nx;
  const S* knots = yy+ny;
  if( !extrapolate ){
    if( x.value <= xx[0] ) x = Dual(xx[0]);
    if( x.value >= xx[nx-1] ) x = Dual(xx[nx-1]);
    if( y.value <= yy[0] ) y = Dual(yy[0]);
    if( y.value >= yy[ny-1] ) y = Dual(yy[ny-1]);
  }
  int ix = GundamDeviceMath::SurfaceSegment(x.value, xx, nx);
  int iy = GundamDeviceMath::SurfaceSegment(y.value, yy, ny);
  Dual value;
  if( !bicubic ){
    Dual f = (x-Dual(xx[ix]))/Dual(xx[ix+1]-xx[ix]);
    Dual first = Dual(knots[ix*ny+iy])+Dual(knots[(ix+1)*ny+iy]-knots[ix*ny+iy])*f;
    Dual second = Dual(knots[ix*ny+iy+1])+Dual(knots[(ix+1)*ny+iy+1]-knots[ix*ny+iy+1])*f;
    value = first+(second-first)*(y-Dual(yy[iy]))/Dual(yy[iy+1]-yy[iy]);
  }
  else{
    Dual rows[4];
    S x0 = ix > 0 ? xx[ix-1] : S(2)*xx[ix]-xx[ix+1];
    S x3 = ix+2 < nx ? xx[ix+2] : S(2)*xx[ix+1]-xx[ix];
    for( int row = 0; row < 4; ++row ){
      int j = iy+row-1;
      S p1 = GundamDeviceMath::SurfaceRowKnot(knots, ix, j, ny);
      S p2 = GundamDeviceMath::SurfaceRowKnot(knots, ix+1, j, ny);
      S p0 = ix > 0 ? GundamDeviceMath::SurfaceRowKnot(knots, ix-1, j, ny) : S(2)*p1-p2;
      S p3 = ix+2 < nx ? GundamDeviceMath::SurfaceRowKnot(knots, ix+2, j, ny) : S(2)*p2-p1;
      rows[row] = cubic(x, x0, Dual(p0), xx[ix], Dual(p1), xx[ix+1], Dual(p2), x3, Dual(p3));
    }
    S y0 = iy > 0 ? yy[iy-1] : S(2)*yy[iy]-yy[iy+1];
    S y3 = iy+2 < ny ? yy[iy+2] : S(2)*yy[iy+1]-yy[iy];
    value = cubic(y, y0, rows[0], yy[iy], rows[1], yy[iy+1], rows[2], y3, rows[3]);
  }
  if( value.value < S(-1E20) ) return Dual(S(-1E20));
  if( value.value > S(1E20) ) return Dual(S(1E20));
  return value;
}
template<typename Reader>
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE Dual formula(CudaFormulaProgram program, const FormulaInstruction* instructions, const S* constants,
             Reader readInput, Dual* stack) {
  unsigned top = 0;
  for( unsigned i = 0; i < program.instructionCount; ++i ){
    auto instruction = instructions[program.firstInstruction+i];
    auto op = instruction.opcode;
    if( op == FormulaOpcode::LoadInput ){
      stack[top++] = readInput(instruction.operand); continue;
    }
    if( op == FormulaOpcode::LoadConstant ){
      stack[top++] = Dual(constants[program.firstConstant+instruction.operand]); continue;
    }
    Dual a = stack[top-1];
    S x = a.value, partial(0), value(0);
    bool unary = true;
    switch( op ){
      case FormulaOpcode::Negate: partial = S(-1); break;
      case FormulaOpcode::Exp: partial = NativeFormulaMath::exp(x); break;
      case FormulaOpcode::Log: partial = S(1)/x; break;
      case FormulaOpcode::Log10: partial = S(1)/(x*NativeFormulaMath::log(S(10))); break;
      case FormulaOpcode::Sqrt: partial = S(.5f)/NativeFormulaMath::sqrt(x); break;
      case FormulaOpcode::Abs: partial = x < S(0) ? S(-1) : (x > S(0) ? S(1) : S(0)); break;
      case FormulaOpcode::Sin: partial = NativeFormulaMath::cos(x); break;
      case FormulaOpcode::Cos: partial = -NativeFormulaMath::sin(x); break;
      case FormulaOpcode::Tan: partial = S(1)/NativeFormulaMath::pow(NativeFormulaMath::cos(x), S(2)); break;
      case FormulaOpcode::Asin: partial = S(1)/NativeFormulaMath::sqrt(S(1)-x*x); break;
      case FormulaOpcode::Acos: partial = -S(1)/NativeFormulaMath::sqrt(S(1)-x*x); break;
      case FormulaOpcode::Atan: partial = S(1)/(S(1)+x*x); break;
      case FormulaOpcode::Sinh: partial = NativeFormulaMath::cosh(x); break;
      case FormulaOpcode::Cosh: partial = NativeFormulaMath::sinh(x); break;
      case FormulaOpcode::Tanh: partial = S(1)/NativeFormulaMath::pow(NativeFormulaMath::cosh(x), S(2)); break;
      case FormulaOpcode::Erf: partial = S(2)*S(.56418958354775628695)*NativeFormulaMath::exp(-x*x); break;
      case FormulaOpcode::Erfc: partial = -S(2)*S(.56418958354775628695)*NativeFormulaMath::exp(-x*x); break;
      case FormulaOpcode::Floor: case FormulaOpcode::Ceil: break;
      default: unary = false;
    }
    if( unary ){
      switch( op ){
#define GUNDAM_FORMULA_UNARY(opcode, operation) \
        case FormulaOpcode::opcode: value = NativeFormulaMath::operation(x); break;
#include "DeviceMath/SharedFormulaOperations.h"
#undef GUNDAM_FORMULA_UNARY
        default: value = S(NAN);
      }
      stack[top-1] = Dual(value, chain(a.slope, partial));
      continue;
    }
    Dual left = stack[top-2];
    S l = left.value, r = a.value, dl(0), dr(0);
    switch( op ){
      case FormulaOpcode::Add: dl = dr = S(1); break;
      case FormulaOpcode::Subtract: dl = S(1); dr = S(-1); break;
      case FormulaOpcode::Multiply: dl = r; dr = l; break;
      case FormulaOpcode::Divide: dl = S(1)/r; dr = -l/(r*r); break;
      case FormulaOpcode::Pow: dl = r == S(0) ? S(0) : r*NativeFormulaMath::pow(l, r-S(1)); dr = NativeFormulaMath::pow(l,r)*NativeFormulaMath::log(l); break;
      case FormulaOpcode::Atan2: dl = r/(l*l+r*r); dr = -l/(l*l+r*r); break;
      case FormulaOpcode::Min: dl = l <= r ? S(1) : S(0); dr = S(1)-dl; break;
      case FormulaOpcode::Max: dl = l >= r ? S(1) : S(0); dr = S(1)-dl; break;
      default: break;
    }
    switch( op ){
#define GUNDAM_FORMULA_BINARY(opcode, operation) \
      case FormulaOpcode::opcode: value = NativeFormulaMath::operation(l,r); break;
#include "DeviceMath/SharedFormulaOperations.h"
#undef GUNDAM_FORMULA_BINARY
      default: value = S(NAN);
    }
    stack[--top-1] = Dual(value, chain(left.slope,dl)+chain(a.slope,dr));
  }
  return stack[0];
}

struct View {
  const BackendDialDescriptor* dials;
  const BackendDialInputDescriptor* inputs;
  const double* payloads;
  const CudaFormulaProgram* formulas;
  const FormulaInstruction* instructions;
  const double* constants;
  const FormulaInput* formulaInputs;
  const CompositeProgram* composites;
  const CompositeInstruction* nodes;
  const BackendDialInputDescriptor* edges;
  const BackendTableTerm* terms;
  const ExternalWeightBlockDescriptor* blocks;
};
struct Math : LikelihoodSemantics::DoubleMath {
  GUNDAM_BACKEND_HOST_DEVICE static double remainder(double a, double b) { return ::fmod(a,b); }
  GUNDAM_BACKEND_HOST_DEVICE static int integer(double x) { return int(x); }
};
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
Dual input(BackendDialInputDescriptor binding, const Dual* values) {
  return mirror(values[binding.parameterIndex], binding.useMirror, binding.mirrorMin, binding.mirrorRange);
}
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
Dual node(View view, BackendDialDescriptor dial, const BackendDialInputDescriptor* bindings,
          const Dual* values, Dual* stack) {
  Dual result;
  if( dial.type == BackendDialType::Formula ){
    auto read = [&](std::size_t i){
      auto binding = view.formulaInputs[dial.firstFormulaInput+i];
      return binding.parameterIndex < 0 ? Dual(binding.constant) : input(bindings[binding.parameterIndex], values);
    };
    result = formula(view.formulas[dial.formulaProgramIndex], view.instructions, view.constants, read, stack);
  }
  else{
    Dual first = dial.inputCount ? input(bindings[0],values) : Dual();
    if( dial.type == BackendDialType::Bilinear || dial.type == BackendDialType::Bicubic ){
      result = surface(first, input(bindings[1],values), view.payloads+dial.payloadOffset,
                       dial.type == BackendDialType::Bicubic, dial.allowExtrapolation);
    }
    else{
      GradientDial<double> descriptor{};
      descriptor.type = unsigned(dial.type); descriptor.payload = dial.payloadOffset;
      descriptor.payloadSize = dial.payloadSize; descriptor.extrapolate = dial.allowExtrapolation;
      auto scalar = GradientSemantics::evaluateDial<Math>(descriptor, view.payloads, first.value);
      result = Dual(scalar.value, chain(first.slope, scalar.derivative));
    }
  }
  return supervise(result,dial);
}
GUNDAM_BACKEND_FORCE_INLINE GUNDAM_BACKEND_HOST_DEVICE
Dual evaluate(View view, unsigned dialIndex, unsigned direction, const double* parameters,
              const double* external, double producerSlope, Dual* scratch, Dual* stack) {
  auto dial = view.dials[dialIndex];
  if( dial.type == BackendDialType::ExternalWeight || dial.type == BackendDialType::Tabulated || dial.type == BackendDialType::Kriged ){
    auto block = view.blocks[dial.externalBlockIndex];
    double value = 0.;
    if( dial.type == BackendDialType::ExternalWeight ) value = external[block.offset+dial.externalWeightIndex];
    else for( std::size_t i = 0; i < dial.tableTermCount; ++i ){
      auto term = view.terms[dial.firstTableTerm+i]; value += term.coefficient*external[block.offset+term.index];
    }
    double slope = 0.;
    if( direction < dial.inputCount ){
      auto binding = view.inputs[dial.firstInput+direction];
      slope = chain(mirror(Dual(parameters[binding.parameterIndex],1.), binding.useMirror,
                            binding.mirrorMin,binding.mirrorRange).slope,producerSlope);
    }
    return supervise(Dual(value,slope),dial);
  }
  // Root inputs are local independent coordinates. Composite edges can mirror
  // them again; formula bindings can reuse them without losing multiplicity.
  for( std::size_t j = 0; j < dial.inputCount; ++j ){
    auto binding = view.inputs[dial.firstInput+j];
    scratch[j] = Dual(parameters[binding.parameterIndex],j == direction ? 1. : 0.);
  }
  if( dial.type != BackendDialType::Composite ){
    // Bind root inputs to local slots while retaining their individual mirrors.
    if( dial.type == BackendDialType::Formula ){
      auto read = [&](std::size_t i){
        auto f = view.formulaInputs[dial.firstFormulaInput+i];
        if( f.parameterIndex < 0 ) return Dual(f.constant);
        auto binding = view.inputs[dial.firstInput+f.parameterIndex];
        return mirror(scratch[f.parameterIndex], binding.useMirror,binding.mirrorMin,binding.mirrorRange);
      };
      return supervise(formula(view.formulas[dial.formulaProgramIndex],view.instructions,view.constants,read,stack),dial);
    }
    Dual first;
    if( dial.inputCount ){
      auto binding = view.inputs[dial.firstInput];
      first = mirror(scratch[0],binding.useMirror,binding.mirrorMin,binding.mirrorRange);
    }
    Dual result;
    if( dial.type == BackendDialType::Bilinear || dial.type == BackendDialType::Bicubic ){
      auto binding = view.inputs[dial.firstInput+1];
      result = surface(first,mirror(scratch[1],binding.useMirror,binding.mirrorMin,binding.mirrorRange),
        view.payloads+dial.payloadOffset,dial.type == BackendDialType::Bicubic,dial.allowExtrapolation);
    }
    else{
      GradientDial<double> descriptor{};
      descriptor.type = unsigned(dial.type); descriptor.payload = dial.payloadOffset;
      descriptor.payloadSize = dial.payloadSize; descriptor.extrapolate = dial.allowExtrapolation;
      auto scalar = GradientSemantics::evaluateDial<Math>(descriptor,view.payloads,first.value);
      result = Dual(scalar.value,chain(first.slope,scalar.derivative));
    }
    return supervise(result,dial);
  }
  for( std::size_t j = 0; j < dial.inputCount; ++j ){
    auto binding = view.inputs[dial.firstInput+j];
    scratch[j] = mirror(scratch[j],binding.useMirror,binding.mirrorMin,binding.mirrorRange);
  }
  auto program = view.composites[dial.compositeProgram];
  for( std::size_t j = 0; j < program.instructionCount; ++j ){
    auto leaf = view.nodes[program.firstInstruction+j].dial;
    scratch[program.inputCount+j] = node(view,leaf,leaf.inputCount ? view.edges+leaf.firstInput : nullptr,scratch,stack);
  }
  return supervise(scratch[program.outputSlot],dial);
}
}
#endif
