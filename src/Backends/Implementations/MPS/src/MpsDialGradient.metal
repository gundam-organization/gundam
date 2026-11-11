namespace MpsGradient {
using S = Backends::ObservableScalar;
using namespace Backends;
struct Dual {
  S value, slope;
  Dual(S value_ = S(0), S slope_ = S(0)) : value(value_), slope(slope_) {}
};
Dual operator+(Dual a, Dual b) { return {a.value+b.value, a.slope+b.slope}; }
Dual operator-(Dual a, Dual b) { return {a.value-b.value, a.slope-b.slope}; }
Dual operator-(Dual a) { return {-a.value, -a.slope}; }
Dual operator*(Dual a, Dual b) { return {a.value*b.value, a.slope*b.value+a.value*b.slope}; }
Dual operator/(Dual a, Dual b) { return {a.value/b.value, (a.slope*b.value-a.value*b.slope)/(b.value*b.value)}; }
// Zero tangents avoid propagating undefined partials of constant operands.
S chain(S tangent, S partial) { return tangent == S(0) ? S(0) : tangent*partial; }
Dual mirror(Dual x, uint enabled, S minimum, S range) {
  if( !enabled ) return x;
  S folded = GundamDeviceMath::remainder(x.value-minimum, S(2)*range);
  S slope = folded < S(0) ? S(-1) : S(1);
  if( FormulaMath::abs(folded) > range ) slope = -slope;
  return {mirror_program_input(x.value, enabled, minimum, range), chain(x.slope, slope)};
}
Dual supervise(Dual x, DevicePropagationDial dial) {
  S slope(1);
  auto smoothing = dial.smoothing;
  if( x.value < dial.minResponse ){
    if( smoothing.minWidth == S(0) || x.value <= smoothing.minResponse-smoothing.minWidth ) slope = S(0);
    else{
      S t = S(.5f)+(x.value-smoothing.minResponse)*smoothing.minInverseWidth;
      slope = t*t*(S(3)-S(2)*t);
    }
  }
  else if( x.value > dial.maxResponse ){
    if( smoothing.maxWidth == S(0) || x.value >= smoothing.maxResponse+smoothing.maxWidth ) slope = S(0);
    else{
      S t = S(.5f)+(smoothing.maxResponse-x.value)*smoothing.maxInverseWidth;
      slope = t*t*(S(3)-S(2)*t);
    }
  }
  return {Semantics::processResponse(x.value, dial), chain(x.slope, slope)};
}
Dual cubic(Dual x, S x0, Dual p0, S x1, Dual p1, S x2, Dual p2, S x3, Dual p3) {
  Dual step(x2-x1), fraction = (x-Dual(x1))/step;
  Dual m1 = step*(p2-p0)/Dual(x2-x0), m2 = step*(p3-p1)/Dual(x3-x1);
  if( fraction.value < S(0) ) m2 = m1;
  if( fraction.value > S(1) ) m1 = m2;
  return ((((-Dual(S(2))*p2+Dual(S(2))*p1+m2+m1)*fraction
    +Dual(S(3))*p2-Dual(S(3))*p1-m2-Dual(S(2))*m1)*fraction+m1)*fraction+p1);
}
Dual surface(Dual x, Dual y, device const S* data, bool bicubic, bool extrapolate) {
  int nx = int(data[0]), ny = int(data[1]);
  device const S* xx = data+2;
  device const S* yy = xx+nx;
  device const S* knots = yy+ny;
  if( !extrapolate ){
    if( x.value <= xx[0] ) x = Dual(xx[0]);
    if( x.value >= xx[nx-1] ) x = Dual(xx[nx-1]);
    if( y.value <= yy[0] ) y = Dual(yy[0]);
    if( y.value >= yy[ny-1] ) y = Dual(yy[ny-1]);
  }
  int ix = GundamObservableMath::SurfaceSegment(x.value, xx, nx);
  int iy = GundamObservableMath::SurfaceSegment(y.value, yy, ny);
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
      S p1 = GundamObservableMath::SurfaceRowKnot(knots, ix, j, ny);
      S p2 = GundamObservableMath::SurfaceRowKnot(knots, ix+1, j, ny);
      S p0 = ix > 0 ? GundamObservableMath::SurfaceRowKnot(knots, ix-1, j, ny) : S(2)*p1-p2;
      S p3 = ix+2 < nx ? GundamObservableMath::SurfaceRowKnot(knots, ix+2, j, ny) : S(2)*p2-p1;
      rows[row] = cubic(x, x0, Dual(p0), xx[ix], Dual(p1), xx[ix+1], Dual(p2), x3, Dual(p3));
    }
    S y0 = iy > 0 ? yy[iy-1] : S(2)*yy[iy]-yy[iy+1];
    S y3 = iy+2 < ny ? yy[iy+2] : S(2)*yy[iy+1]-yy[iy];
    value = cubic(y, y0, rows[0], yy[iy], rows[1], yy[iy+1], rows[2], y3, rows[3]);
  }
  if( value.value < S(-1E20f) ) return Dual(S(-1E20f));
  if( value.value > S(1E20f) ) return Dual(S(1E20f));
  return value;
}
Dual read(DeviceProgramInput input, device const S* parameters, device const Dual* frame, uint parameter) {
  if( input.source == ProgramInputSource::Constant ) return Dual(input.value);
  Dual value = input.source == ProgramInputSource::Parameter
      ? Dual(parameters[input.index], input.index == parameter ? S(1) : S(0)) : frame[input.index];
  return mirror(value, input.useMirror, input.mirrorMin, input.mirrorRange);
}
Dual formula(DeviceFormulaProgram program, uint firstInput,
    device const FormulaInstruction* instructions, device const S* constants,
    device const DeviceProgramInput* inputs, device const S* parameters,
    device const Dual* frame, device Dual* stack, uint parameter) {
#define GUNDAM_FORMULA_GRADIENT_MATH FormulaMath
#define GUNDAM_FORMULA_GRADIENT_INV_SQRT_PI FormulaMath::invSqrtPi()
  uint top = 0;
  for( uint i = 0; i < program.instructionCount; ++i ){
    auto instruction = instructions[program.firstInstruction+i];
    auto op = instruction.opcode;
    if( op == FormulaOpcode::LoadInput ){
      stack[top++] = read(inputs[firstInput+instruction.operand], parameters, frame, parameter); continue;
    }
    if( op == FormulaOpcode::LoadConstant ){
      stack[top++] = Dual(constants[program.firstConstant+instruction.operand]); continue;
    }
    Dual a = stack[top-1];
    S x = a.value, partial(0), value(0);
    bool unary = true;
    switch( op ){
#define GUNDAM_FORMULA_UNARY_GRADIENT(opcode, operation) \
      case FormulaOpcode::opcode: partial = operation; break;
#include "DeviceMath/SharedFormulaGradientOperations.h"
#undef GUNDAM_FORMULA_UNARY_GRADIENT
      default: unary = false;
    }
    if( unary ){
      switch( op ){
#define GUNDAM_FORMULA_UNARY(opcode, operation) \
        case FormulaOpcode::opcode: value = FormulaMath::operation(x); break;
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
#define GUNDAM_FORMULA_BINARY_GRADIENT(opcode, left, right) \
      case FormulaOpcode::opcode: dl = left; dr = right; break;
#include "DeviceMath/SharedFormulaGradientOperations.h"
#undef GUNDAM_FORMULA_BINARY_GRADIENT
      default: break;
    }
    switch( op ){
#define GUNDAM_FORMULA_BINARY(opcode, operation) \
      case FormulaOpcode::opcode: value = FormulaMath::operation(l,r); break;
#include "DeviceMath/SharedFormulaOperations.h"
#undef GUNDAM_FORMULA_BINARY
      default: value = S(NAN);
    }
    stack[--top-1] = Dual(value, chain(left.slope,dl)+chain(a.slope,dr));
  }
#undef GUNDAM_FORMULA_GRADIENT_MATH
#undef GUNDAM_FORMULA_GRADIENT_INV_SQRT_PI
  return stack[0];
}
Dual leaf(DevicePropagationDial dial, Dual first, Dual second, device const S* payloads,
          bool composite, S inputMin, S inputMax) {
  auto kind = BackendDialType(dial.type);
  if( kind == BackendDialType::Bilinear || kind == BackendDialType::Bicubic )
    return surface(first, second, payloads+dial.payloadOffset, kind == BackendDialType::Bicubic, dial.allowExtrapolation);
  // Inputs have already been mirrored. The shared scalar derivative uses the
  // same interpolation conventions; composite nodes additionally use their
  // stored input bounds for non-graph splines.
  if( composite && !dial.allowExtrapolation && kind != BackendDialType::Norm && kind != BackendDialType::Shift && kind != BackendDialType::Graph ){
    if( first.value <= inputMin ) first = Dual(inputMin);
    if( first.value >= inputMax ) first = Dual(inputMax);
  }
  GradientDial<S> descriptor{};
  descriptor.type = dial.type; descriptor.payload = dial.payloadOffset;
  descriptor.payloadSize = dial.payloadSize; descriptor.extrapolate = dial.allowExtrapolation;
  auto result = GradientSemantics::evaluateDial<LikelihoodSemantics::MetalMath>(descriptor, payloads, first.value);
  return {result.value, chain(first.slope, result.derivative)};
}
}
