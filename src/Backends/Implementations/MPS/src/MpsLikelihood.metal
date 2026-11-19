namespace Backends::LikelihoodSemantics {
  struct MetalMath {
    using Scalar = ObservableScalar;
    static Scalar remainder(Scalar a_, Scalar b_) { return GundamDeviceMath::remainder(a_, b_); }
    static int integer(Scalar x_) { return int(x_); }
    static Scalar sqrt(Scalar x_) { return FormulaMath::sqrt(x_); }
    static Scalar log(Scalar x_) { return FormulaMath::log(x_); }
    static Scalar log1p(Scalar x_) {
      // The small-residual branch is evaluated as a series by evaluate().
      return FormulaMath::log(Scalar(1) + x_);
    }
    static Scalar exp(Scalar x_) { return FormulaMath::exp(x_); }
    static Scalar expm1(Scalar x_) {
      if( x_ < Scalar(-1)/Scalar(2) || x_ > Scalar(1)/Scalar(2) ) return exp(x_) - Scalar(1);
      Scalar term = x_, sum = x_;
      for( int order = 2; order <= 20; ++order ) { term = term * x_ / Scalar(order); sum += term; }
      return sum;
    }
    static Scalar hypot(Scalar a_, Scalar b_) {
      a_ = absolute(a_); b_ = absolute(b_);
      Scalar larger = maximum(a_, b_), smaller = LikelihoodSemantics::minimum(a_, b_);
      if( larger == Scalar(0) ) return Scalar(0);
      Scalar ratio = smaller / larger;
      return larger * sqrt(Scalar(1) + ratio * ratio);
    }
    static bool finite(Scalar x_) { return FormulaMath::finite(x_); }
    // Metal has no binary64 exponent range. Zero represents DBL_MIN for the
    // floor/branch; its logarithm is retained explicitly for empty predictions.
    static Scalar minimum() { return Scalar(0); }
    static Scalar logMinimum() { return Scalar(-708.3964233398438f, 4.8075796712510055e-06f); }
    static Scalar infinity() { return Scalar(INFINITY); }
    static Scalar invalid() { return Scalar(NAN); }
  };
  inline ObservableScalar addLikelihood(ObservableScalar a_, ObservableScalar b_) {
    if( !MetalMath::finite(a_) || !MetalMath::finite(b_) ){ return ObservableScalar(float(a_) + float(b_)); }
    return a_ + b_;
  }
}

kernel void evaluate_likelihood_partials(
    device const Backends::LikelihoodBin<Backends::ObservableScalar>* bins [[buffer(0)]],
    device const Backends::ObservableScalar* sums [[buffer(1)]],
    device const Backends::ObservableScalar* squares [[buffer(2)]],
    device Backends::ObservableScalar* partials [[buffer(3)]],
    constant Backends::LikelihoodWorkLayout& work [[buffer(4)]], uint index [[thread_position_in_grid]]) {
  using namespace Backends;
  using namespace LikelihoodSemantics;
  ObservableScalar total(0);
  for( uint i = index * work.binsPerThread; i < min(work.binCount, (index + 1) * work.binsPerThread); ++i ){
    auto bin = bins[i];
    if( bin.ignored ){ continue; }
    auto error = MetalMath::sqrt(squares[bin.globalBin]);
    auto variance = bin.options.useNominalVariance ? bin.nominalVariance : error * error;
    total = addLikelihood(total, evaluate<MetalMath>(bin.data, sums[bin.globalBin], variance, bin.options));
  }
  partials[index] = total;
}
kernel void finish_likelihood(
    device const Backends::ObservableScalar* partials [[buffer(0)]],
    device Backends::ObservableScalar* total [[buffer(1)]],
    constant Backends::LikelihoodWorkLayout& work [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]) {
  using namespace Backends;
  threadgroup ObservableScalar values[GUNDAM_LIKELIHOOD_REDUCTION_WIDTH];
  ObservableScalar value(0);
  for( uint i = lane; i < work.partialCount; i += GUNDAM_LIKELIHOOD_REDUCTION_WIDTH ){
    value = LikelihoodSemantics::addLikelihood(value, partials[i]);
  }
  values[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for( uint stride = GUNDAM_LIKELIHOOD_REDUCTION_WIDTH / 2; stride; stride /= 2 ){
    if( lane < stride ){
      values[lane] = LikelihoodSemantics::addLikelihood(values[lane], values[lane + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if( lane == 0 ){ total[0] = values[0]; }
}
