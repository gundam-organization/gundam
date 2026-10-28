#ifndef GUNDAM_FORMULA_MATH_H
#define GUNDAM_FORMULA_MATH_H
#ifndef __METAL_VERSION__
#include "DeviceMath/DoubleFloat.h"
#include <cmath>
#define GUNDAM_FORMULA_MATH std
#else
#define GUNDAM_FORMULA_MATH metal
#endif

namespace Backends::FormulaMath {
  using Scalar = GundamDeviceMath::DoubleFloat;
  inline bool finite(Scalar x_) { return GUNDAM_FORMULA_MATH::isfinite(x_.high); }
  inline Scalar pi() { return {3.14159265358979323846f, -8.742277657347586e-8f}; }
  inline Scalar ln2() { return {.69314718055994530942f, -1.904654299957768e-9f}; }
  inline Scalar invSqrtPi() { return {.56418958354775628695f, -2.9317691714321197e-8f}; }
  inline Scalar abs(Scalar x_) {
    if( x_ == Scalar(0) ){ return Scalar(0); }
    return x_ < Scalar(0) ? -x_ : x_;
  }
  inline Scalar add(Scalar a_, Scalar b_) {
    float result = float(a_) + float(b_);
    return GUNDAM_FORMULA_MATH::isfinite(result) ? a_ + b_ : Scalar(result);
  }
  inline Scalar multiply(Scalar a_, Scalar b_) {
    float result = float(a_) * float(b_);
    return GUNDAM_FORMULA_MATH::isfinite(result) ? a_ * b_ : Scalar(result);
  }
  inline Scalar divide(Scalar a_, Scalar b_) {
    float result = float(a_) / float(b_);
    return GUNDAM_FORMULA_MATH::isfinite(result) ? a_ / b_ : Scalar(result);
  }
  inline Scalar sqrt(Scalar x_) {
    if( !finite(x_) || x_ <= Scalar(0) ){ return Scalar(GUNDAM_FORMULA_MATH::sqrt(float(x_))); }
    Scalar value(GUNDAM_FORMULA_MATH::sqrt(float(x_)));
    for( int i = 0; i < 2; ++i ){ value += (x_ - value * value) / (Scalar(2) * value); }
    return value;
  }
  inline Scalar exp(Scalar x_) {
    if( !finite(x_) || x_ > Scalar(88) || x_ < Scalar(-87) ){
      return Scalar(GUNDAM_FORMULA_MATH::exp(float(x_)));
    }
    int exponent = int(GUNDAM_FORMULA_MATH::round(float(x_ / ln2())));
    Scalar reduced = x_ - Scalar(exponent) * ln2();
    Scalar term(1), value(1);
    for( int i = 1; i <= 18; ++i ){ term = term * reduced / Scalar(i); value += term; }
    return {GUNDAM_FORMULA_MATH::ldexp(value.high, exponent), GUNDAM_FORMULA_MATH::ldexp(value.low, exponent)};
  }
  inline Scalar log(Scalar x_) {
    if( !finite(x_) || x_ <= Scalar(0) ){ return Scalar(GUNDAM_FORMULA_MATH::log(float(x_))); }
    int exponent{0};
#ifdef __METAL_VERSION__
    metal::frexp(x_.high, exponent);
#else
    GUNDAM_FORMULA_MATH::frexp(x_.high, &exponent);
#endif
    Scalar mantissa{GUNDAM_FORMULA_MATH::ldexp(x_.high, -exponent), GUNDAM_FORMULA_MATH::ldexp(x_.low, -exponent)};
    Scalar z = (mantissa - Scalar(1)) / (mantissa + Scalar(1));
    Scalar power = z, value = z, square = z * z;
    for( int i = 3; i <= 49; i += 2 ){ power *= square; value += power / Scalar(i); }
    return Scalar(2) * value + Scalar(exponent) * ln2();
  }
  inline Scalar sinh(Scalar x_) {
    if( !finite(x_) || abs(x_) > Scalar(80) ){ return Scalar(GUNDAM_FORMULA_MATH::sinh(float(x_))); }
    return divide(add(exp(x_), -exp(-x_)), Scalar(2));
  }
  inline Scalar cosh(Scalar x_) {
    if( !finite(x_) || abs(x_) > Scalar(80) ){ return Scalar(GUNDAM_FORMULA_MATH::cosh(float(x_))); }
    return divide(add(exp(x_), exp(-x_)), Scalar(2));
  }
  inline Scalar sin(Scalar x_) {
    if( !finite(x_) || abs(x_) > Scalar(1000000) ){ return Scalar(GUNDAM_FORMULA_MATH::sin(float(x_))); }
    Scalar period = Scalar(2) * pi();
    Scalar x = x_ - Scalar(GUNDAM_FORMULA_MATH::round(float(x_ / period))) * period;
    Scalar term = x, value = x, square = -(x * x);
    for( int i = 1; i <= 16; ++i ){ term = term * square / Scalar((2*i)*(2*i+1)); value += term; }
    return value;
  }
  inline Scalar cos(Scalar x_) {
    if( !finite(x_) || abs(x_) > Scalar(1000000) ){ return Scalar(GUNDAM_FORMULA_MATH::cos(float(x_))); }
    Scalar period = Scalar(2) * pi();
    Scalar x = x_ - Scalar(GUNDAM_FORMULA_MATH::round(float(x_ / period))) * period;
    Scalar term(1), value(1), square = -(x * x);
    for( int i = 1; i <= 16; ++i ){ term = term * square / Scalar((2*i-1)*(2*i)); value += term; }
    return value;
  }
  inline Scalar atan2(Scalar y_, Scalar x_) {
    float seed = GUNDAM_FORMULA_MATH::atan2(float(y_), float(x_));
    if( !finite(x_) || !finite(y_) ){ return Scalar(seed); }
    Scalar scale = abs(x_) > abs(y_) ? abs(x_) : abs(y_);
    if( scale == Scalar(0) ){
      return GUNDAM_FORMULA_MATH::abs(seed) > 1.f ? (seed < 0 ? -pi() : pi()) : Scalar(seed);
    }
    Scalar x = x_ / scale, y = y_ / scale, angle(seed);
    Scalar s = sin(angle), c = cos(angle);
    return angle + (y * c - x * s) / (x * c + y * s);
  }
  inline Scalar pow(Scalar x_, Scalar y_) {
    float exponent = float(y_);
    if( finite(y_) && GUNDAM_FORMULA_MATH::abs(exponent) <= 64.f
        && y_ == Scalar(GUNDAM_FORMULA_MATH::trunc(exponent)) ){
      int n = int(exponent);
      Scalar value(1), factor = x_;
      for( int power = n < 0 ? -n : n; power; power >>= 1 ){
        if( power & 1 ){ value = multiply(value, factor); }
        if( power > 1 ){ factor = multiply(factor, factor); }
      }
      return n < 0 ? divide(Scalar(1), value) : value;
    }
    if( x_ > Scalar(0) && finite(x_) && finite(y_) ){ return exp(y_ * log(x_)); }
    return Scalar(GUNDAM_FORMULA_MATH::pow(float(x_), exponent));
  }
  inline Scalar floor(Scalar x_) {
    if( !finite(x_) ){ return x_; }
    Scalar base(GUNDAM_FORMULA_MATH::floor(x_.high));
    Scalar residual = x_ - base;
    return base + Scalar(GUNDAM_FORMULA_MATH::floor(float(residual)));
  }
  inline Scalar ceil(Scalar x_) { return -floor(-x_); }
  inline Scalar erfcPositive(Scalar x_) {
    if( x_ > Scalar(10) ){ return Scalar(0); }
    if( x_ >= Scalar(2) ){
      // Laplace continued fraction, evaluated backwards without subtraction.
      Scalar tail(0);
      for( int i = 100; i >= 1; --i ){ tail = Scalar(i) / (Scalar(2) * (x_ + tail)); }
      return exp(-(x_ * x_)) * invSqrtPi() / (x_ + tail);
    }
    Scalar term = x_, sum = x_, square = Scalar(2) * x_ * x_;
    for( int i = 1; i <= 40; ++i ){ term = term * square / Scalar(2*i+1); sum += term; }
    return Scalar(1) - Scalar(2) * invSqrtPi() * exp(-(x_*x_)) * sum;
  }
  inline Scalar erfc(Scalar x_) {
    if( !finite(x_) ){ return Scalar(x_.high < 0 ? 2.f : x_.high > 0 ? 0.f : x_.high); }
    Scalar value = erfcPositive(abs(x_));
    return x_ < Scalar(0) ? Scalar(2) - value : value;
  }
  inline Scalar erf(Scalar x_) {
    if( !finite(x_) ){ return Scalar(x_.high < 0 ? -1.f : x_.high > 0 ? 1.f : x_.high); }
    Scalar value = Scalar(1) - erfcPositive(abs(x_));
    return x_ < Scalar(0) ? -value : value;
  }
}
#undef GUNDAM_FORMULA_MATH
#endif
