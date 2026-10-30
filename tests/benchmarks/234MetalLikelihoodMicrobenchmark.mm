// Isolated likelihood-kernel experiment. Compile with the generated Metal source:
// clang++ -O3 -std=c++17 -I src/Backends/Common/include \
//   -I cmake-build-release/src/Backends/generated \
//   tests/benchmarks/234MetalLikelihoodMicrobenchmark.mm \
//   -framework Foundation -framework Metal -o /tmp/gundam-llh-micro
#include "DeviceMath/DoubleFloat.h"
#include "LikelihoodTypes.h"
#include "Semantics/BackendLikelihood.h"
#include "MpsBackendKernelSource.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using Scalar = GundamDeviceMath::DoubleFloat;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start_) { return std::chrono::duration<double>(Clock::now()-start_).count(); }
static double median(std::vector<double> values_) { std::sort(values_.begin(), values_.end()); return values_[values_.size()/2]; }

int main(int argc_, char** argv_) {
  @autoreleasepool {
    auto device = MTLCreateSystemDefaultDevice();
    auto queue = [device newCommandQueue];
    if( !device || !queue ){ throw std::runtime_error("Metal unavailable"); }
    std::cout << "[\n";
    bool first = true;
    std::vector<unsigned> chunks{64u, 8u, 1u};
    if( argc_ > 1 ){ std::reverse(chunks.begin(), chunks.end()); }
    for( unsigned chunk : chunks ){
      NSString* source = GUNDAM_MPS_BACKEND_KERNEL_SOURCE;
      auto options = [MTLCompileOptions new];
      options.mathMode = MTLMathModeSafe;
      NSError* error = nil;
      auto library = [device newLibraryWithSource:source options:options error:&error];
      [options release];
      if( !library ){ throw std::runtime_error([[error description] UTF8String]); }
      auto makePipeline = [&](NSString* name_){
        auto function = [library newFunctionWithName:name_];
        auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        if( !pipeline ){ throw std::runtime_error([[error description] UTF8String]); }
        return pipeline;
      };
      auto partialPipeline = makePipeline(@"evaluate_likelihood_partials");
      auto finalPipeline = makePipeline(@"finish_likelihood");
      for( unsigned count : {374u, 4140u, 65536u} ){
        for( auto kind : {Backends::LikelihoodKind::Poisson, Backends::LikelihoodKind::BarlowBeeston2022} ){
          std::vector<Backends::LikelihoodBin<Scalar>> bins(count);
          std::vector<Scalar> sums(count), squares(count);
          std::vector<double> data(count), predictions(count), variances(count);
          Backends::LikelihoodOptions<double> config{kind, 0, 0, 1, 1e150, -1.};
          for( unsigned i = 0; i < count; ++i ){
            data[i] = .1 + (i%1000)*.1;
            predictions[i] = data[i]*(1.+.03*(int(i%17)-8));
            variances[i] = predictions[i]*.2;
            sums[i] = Scalar(predictions[i]); squares[i] = Scalar(variances[i]);
            bins[i] = {i, 0, Scalar(data[i]), Scalar(0),
                       {kind, 0, 0, 1, Scalar(INFINITY), Scalar(-1)}};
          }
          auto upload = [&](const auto& values_){
            return [device newBufferWithBytes:values_.data() length:values_.size()*sizeof(values_[0]) options:MTLResourceStorageModeShared];
          };
          auto bBins = upload(bins), bSums = upload(sums), bSquares = upload(squares);
          auto partials = [device newBufferWithLength:((count+chunk-1)/chunk)*sizeof(Scalar) options:MTLResourceStorageModePrivate];
          auto total = [device newBufferWithLength:sizeof(Scalar) options:MTLResourceStorageModePrivate];
          auto readback = [device newBufferWithLength:sizeof(Scalar) options:MTLResourceStorageModeShared];
          const Backends::LikelihoodWorkLayout work{count, chunk, (count+chunk-1)/chunk};
          auto encode = [&](id<MTLComputeCommandEncoder> encoder_){
            [encoder_ setComputePipelineState:partialPipeline];
            [encoder_ setBuffer:bBins offset:0 atIndex:0];
            [encoder_ setBuffer:bSums offset:0 atIndex:1];
            [encoder_ setBuffer:bSquares offset:0 atIndex:2];
            [encoder_ setBuffer:partials offset:0 atIndex:3];
            [encoder_ setBytes:&work length:sizeof(work) atIndex:4];
            [encoder_ dispatchThreads:MTLSizeMake((count+chunk-1)/chunk,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder_ setComputePipelineState:finalPipeline];
            [encoder_ setBuffer:partials offset:0 atIndex:0];
            [encoder_ setBuffer:total offset:0 atIndex:1];
            [encoder_ setBytes:&work length:sizeof(work) atIndex:2];
            [encoder_ dispatchThreads:MTLSizeMake(GUNDAM_LIKELIHOOD_REDUCTION_WIDTH,1,1)
                      threadsPerThreadgroup:MTLSizeMake(GUNDAM_LIKELIHOOD_REDUCTION_WIDTH,1,1)];
          };
          std::vector<double> cpuTimes, latency, gpuTimes, throughput;
          volatile double sink = 0.;
          double expected = 0.;
          for( unsigned repeat = 0; repeat < 30; ++repeat ){
            auto start = Clock::now();
            double sum = 0.;
            for( unsigned i = 0; i < count; ++i ){
              const double e = std::sqrt(variances[i]);
              sum += Backends::LikelihoodSemantics::evaluate<Backends::LikelihoodSemantics::DoubleMath>(data[i],predictions[i],e*e,config);
            }
            sink = sum;
            cpuTimes.push_back(seconds(start));
            expected = sum;
          }
          for( unsigned repeat = 0; repeat < 120; ++repeat ){
            @autoreleasepool {
              auto start = Clock::now();
              auto command = [queue commandBuffer];
              auto encoder = [command computeCommandEncoder];
              encode(encoder);
              [encoder endEncoding];
              auto blit = [command blitCommandEncoder];
              [blit copyFromBuffer:total sourceOffset:0 toBuffer:readback destinationOffset:0 size:sizeof(Scalar)];
              [blit endEncoding];
              [command commit];
              [command waitUntilCompleted];
              if( command.status != MTLCommandBufferStatusCompleted ){ throw std::runtime_error("GPU execution failed"); }
              const auto pair = *static_cast<Scalar*>(readback.contents);
              double result = double(pair.high)+pair.low;
              if( std::abs(result-expected) > 1e-8*std::max(1.,std::abs(expected)) ){
                throw std::runtime_error("Likelihood mismatch");
              }
              if( repeat >= 20 ){
                latency.push_back(seconds(start));
                gpuTimes.push_back(command.GPUEndTime-command.GPUStartTime);
              }
            }
          }
          for( unsigned repeat = 0; repeat < 5; ++repeat ){
            auto command = [queue commandBuffer];
            auto encoder = [command computeCommandEncoder];
            for( unsigned i = 0; i < 50; ++i ){ encode(encoder); }
            [encoder endEncoding];
            [command commit]; [command waitUntilCompleted];
            if( command.status != MTLCommandBufferStatusCompleted ){ throw std::runtime_error("GPU batch failed"); }
            throughput.push_back((command.GPUEndTime-command.GPUStartTime)/50.);
          }
          if( !first ){ std::cout << ",\n"; } first = false;
          std::cout << "{\"bins\":" << count << ",\"chunk\":" << chunk << ",\"reduction\":\"parallel\""
                    << ",\"kind\":\"" << (kind == Backends::LikelihoodKind::Poisson ? "Poisson" : "Barlow2022")
                    << "\",\"cpuUs\":" << median(cpuTimes)*1e6 << ",\"latencyUs\":" << median(latency)*1e6
                    << ",\"gpuUs\":" << median(gpuTimes)*1e6 << ",\"batchedGpuUs\":" << median(throughput)*1e6 << "}" << std::flush;
          for( auto buffer : {bBins,bSums,bSquares,partials,total,readback} ){ [buffer release]; }
        }
      }
      [partialPipeline release]; [finalPipeline release]; [library release];
    }
    std::cout << "\n]\n";
    [queue release]; [device release];
  }
}
