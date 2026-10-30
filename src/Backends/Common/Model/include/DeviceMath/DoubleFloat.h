#ifndef GUNDAM_DOUBLE_FLOAT_H
#define GUNDAM_DOUBLE_FLOAT_H
#ifndef __METAL_VERSION__
#include <cmath>
#endif

namespace GundamDeviceMath {
  // Two normalized floats retain about 48 significant bits. This is an
  // observable representation, independent of host objects and device APIs.
  // Explicit FMA residuals prevent bin decisions from depending on float32
  // rounding of the reference value or of intermediate variable transforms.
  struct DoubleFloat {
    float high{0}, low{0};
    DoubleFloat() = default;
    DoubleFloat(float value_) : high(value_) {}
    DoubleFloat(int value_) : high(float(value_)) {}
    DoubleFloat(float high_, float low_) : high(high_), low(low_) {}
#ifndef __METAL_VERSION__
    DoubleFloat(double value_) : high(float(value_)), low(std::isfinite(value_) ? float(value_ - double(high)) : 0.f) {}
#endif
    explicit operator float() const { return high + low; }
    explicit operator int() const {
      int value = int(high);
      if( high == float(value) ){
        if( value > 0 && low < 0 ){ --value; }
        else if( value < 0 && low > 0 ){ ++value; }
      }
      return value;
    }
  };
  inline DoubleFloat normalized(float high_, float low_) {
    float sum = high_ + low_;
    return {sum, low_ - (sum - high_)};
  }
  inline DoubleFloat operator+(DoubleFloat a_, DoubleFloat b_) {
    float sum = a_.high + b_.high;
    float part = sum - a_.high;
    float error = (a_.high - (sum - part)) + (b_.high - part);
    return normalized(sum, error + a_.low + b_.low);
  }
  inline DoubleFloat operator-(DoubleFloat value_) { return {-value_.high, -value_.low}; }
  inline DoubleFloat operator-(DoubleFloat a_, DoubleFloat b_) { return a_ + (-b_); }
  inline DoubleFloat operator*(DoubleFloat a_, DoubleFloat b_) {
    float product = a_.high * b_.high;
#ifdef __METAL_VERSION__
    float error = metal::fma(a_.high, b_.high, -product);
#else
    float error = std::fma(a_.high, b_.high, -product);
#endif
    error += a_.high * b_.low + a_.low * b_.high;
    error += a_.low * b_.low;
    return normalized(product, error);
  }
  inline DoubleFloat operator/(DoubleFloat a_, DoubleFloat b_) {
    float first = a_.high / b_.high;
    DoubleFloat residual = a_ - b_ * DoubleFloat(first);
    float second = (residual.high + residual.low) / b_.high;
    return normalized(first, second);
  }
  inline DoubleFloat& operator+=(DoubleFloat& a_, DoubleFloat b_) { a_ = a_ + b_; return a_; }
  inline DoubleFloat& operator-=(DoubleFloat& a_, DoubleFloat b_) { a_ = a_ - b_; return a_; }
  inline DoubleFloat& operator*=(DoubleFloat& a_, DoubleFloat b_) { a_ = a_ * b_; return a_; }
  inline DoubleFloat& operator/=(DoubleFloat& a_, DoubleFloat b_) { a_ = a_ / b_; return a_; }
  inline bool operator==(DoubleFloat a_, DoubleFloat b_) { return a_.high == b_.high && a_.low == b_.low; }
  inline bool operator!=(DoubleFloat a_, DoubleFloat b_) { return !(a_ == b_); }
  inline bool operator<(DoubleFloat a_, DoubleFloat b_) { return a_.high < b_.high || (a_.high == b_.high && a_.low < b_.low); }
  inline bool operator>(DoubleFloat a_, DoubleFloat b_) { return b_ < a_; }
  inline bool operator<=(DoubleFloat a_, DoubleFloat b_) { return a_ < b_ || a_ == b_; }
  inline bool operator>=(DoubleFloat a_, DoubleFloat b_) { return b_ <= a_; }
  inline DoubleFloat remainder(DoubleFloat a_, DoubleFloat b_) {
    DoubleFloat value = a_;
    // Successive quotients avoid losing the remainder when the period is
    // small compared with the input. Each pass removes about 24 bits.
    for( int i = 0; i < 8; ++i ){
#ifdef __METAL_VERSION__
      float quotient = metal::trunc(float(value / b_));
#else
      float quotient = std::trunc(float(value / b_));
#endif
      if( quotient == 0 ){ break; }
      value -= b_ * DoubleFloat(quotient);
    }
    if( a_ >= DoubleFloat(0.f) && value < DoubleFloat(0.f) ){ value += b_; }
    else if( a_ < DoubleFloat(0.f) && value > DoubleFloat(0.f) ){ value -= b_; }
    return value;
  }
}
#endif
