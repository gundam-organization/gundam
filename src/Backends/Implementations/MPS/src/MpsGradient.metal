namespace Backends::GradientSemantics {
  using MetalGradientMath = LikelihoodSemantics::MetalMath;
}
kernel void gradient_dials(
    device const Backends::GradientDial<Backends::ObservableScalar>* dials [[buffer(0)]],
    device const Backends::ObservableScalar* payloads [[buffer(1)]],
    device const Backends::ObservableScalar* parameters [[buffer(2)]],
    device Backends::GradientSemantics::DialEvaluation<Backends::ObservableScalar>* out [[buffer(3)]],
    constant Backends::GradientWork<Backends::ObservableScalar>& work [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  using namespace Backends;
  if( i >= work.dials ) return;
  auto dial = dials[i];
  out[i] = GradientSemantics::evaluateDial<GradientSemantics::MetalGradientMath>(dial, payloads,
      dial.type == uint(BackendDialType::Shift) ? ObservableScalar(0) : parameters[dial.parameter]);
}
kernel void gradient_bins(
    device const Backends::LikelihoodBin<Backends::ObservableScalar>* bins [[buffer(0)]],
    device const Backends::ObservableScalar* sums [[buffer(1)]],
    device const Backends::ObservableScalar* squares [[buffer(2)]],
    device Backends::GradientSemantics::DialEvaluation<Backends::ObservableScalar>* out [[buffer(3)]],
    constant Backends::GradientWork<Backends::ObservableScalar>& work [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if( i < work.bins ) out[i] = Backends::GradientSemantics::binScales<Backends::GradientSemantics::MetalGradientMath>(bins[i], sums[i], squares[i]);
}
kernel void gradient_events(
    device const Backends::GradientEvent<Backends::ObservableScalar>* events [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const Backends::GradientSemantics::DialEvaluation<Backends::ObservableScalar>* responses [[buffer(2)]],
    device const Backends::GradientSemantics::DialEvaluation<Backends::ObservableScalar>* scales [[buffer(3)]],
    device const Backends::ObservableScalar* weights [[buffer(4)]],
    device Backends::ObservableScalar* out [[buffer(5)]],
    constant Backends::GradientWork<Backends::ObservableScalar>& work [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if( i < work.events ) Backends::GradientSemantics::eventContributions(events[i], work, indices, responses, scales, weights, out);
}
kernel void gradient_reduce(
    device const uint* offsets [[buffer(0)]], device const uint* occurrences [[buffer(1)]],
    device const Backends::ObservableScalar* contributions [[buffer(2)]],
    device Backends::ObservableScalar* out [[buffer(3)]],
    uint parameter [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  using Backends::ObservableScalar;
  threadgroup ObservableScalar values[128];
  ObservableScalar value(0);
  for( uint i = offsets[parameter]+lane; i < offsets[parameter+1]; i += 128 ) value += contributions[occurrences[i]];
  values[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for( uint stride = 64; stride; stride /= 2 ){
    if( lane < stride ) values[lane] += values[lane+stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if( lane == 0 ) out[parameter] = values[0];
}
