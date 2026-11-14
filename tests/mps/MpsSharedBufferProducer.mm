#include "MpsSharedBuffer.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cerrno>
#include <csignal>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace {
  const char* kernels = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void produce(device float* parameters [[buffer(0)]],
                    device float* x [[buffer(1)]], device float* weights [[buffer(2)]],
                    device atomic_uint* errors [[buffer(3)]],
                    constant uint& count [[buffer(4)]], constant uint& generation [[buffer(5)]],
                    uint i [[thread_position_in_grid]]) {
  if( i >= count ){ return; }
  x[i] = float(i % 257 + generation % 3) / 16.f;
  if( i == 0 ){
    parameters[0] = 1.f + float(generation) * .125f;
    parameters[1] = .25f - float(generation) * .03125f;
    if( generation == 0 ){
      weights[0] = -7.f; weights[count + 1] = -7.f;
      atomic_store_explicit(errors, 0, memory_order_relaxed);
    }
  }
}
kernel void consume(device const float* weights [[buffer(0)]],
                    device atomic_uint* errors [[buffer(1)]],
                    constant uint& count [[buffer(2)]], constant uint& generation [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
  if( i >= count ){ return; }
  float x = float(i % 257 + generation % 3) / 16.f;
  float expected = (1.f + float(generation) * .125f) * x + .25f - float(generation) * .03125f;
  float actual = weights[i + 1];
  if( !isfinite(actual) || abs(actual - expected) > 1e-6f ){
    atomic_fetch_add_explicit(errors, 1, memory_order_relaxed);
  }
  if( i == 0 && (weights[0] != -7.f || weights[count + 1] != -7.f) ){
    atomic_fetch_add_explicit(errors, 1, memory_order_relaxed);
  }
}
)METAL";

  class WorkerProcess {
  public:
    explicit WorkerProcess(std::vector<std::string> arguments_) {
      int input[2], output[2];
      if( pipe(input) != 0 ){ throw std::runtime_error("Cannot create worker input pipe."); }
      if( pipe(output) != 0 ){ close(input[0]); close(input[1]); throw std::runtime_error("Cannot create worker output pipe."); }
      posix_spawn_file_actions_t actions;
      posix_spawn_file_actions_init(&actions);
      posix_spawn_file_actions_adddup2(&actions, input[0], STDIN_FILENO);
      posix_spawn_file_actions_adddup2(&actions, output[1], STDOUT_FILENO);
      for( int fd : {input[0], input[1], output[0], output[1]} ){ posix_spawn_file_actions_addclose(&actions, fd); }
      std::vector<char*> arguments;
      for( auto& value : arguments_ ){ arguments.push_back(value.data()); }
      arguments.push_back(nullptr);
      const auto status = posix_spawn(&_pid_, arguments[0], &actions, nullptr, arguments.data(), environ);
      posix_spawn_file_actions_destroy(&actions);
      close(input[0]); close(output[1]);
      if( status != 0 ){
        close(input[1]); close(output[0]);
        throw std::runtime_error(std::string("Cannot start selected Python: ") + std::strerror(status));
      }
      _input_ = input[1]; _output_ = output[0];
    }
    ~WorkerProcess() {
      if( _input_ >= 0 ){ close(_input_); }
      if( _output_ >= 0 ){ close(_output_); }
      if( _pid_ > 0 ){ kill(_pid_, SIGKILL); waitpid(_pid_, nullptr, 0); }
    }
    void send(const std::string& value_) {
      const auto line = value_ + "\n";
      std::size_t offset = 0;
      while( offset < line.size() ){
        const auto size = write(_input_, line.data() + offset, line.size() - offset);
        if( size < 0 and errno == EINTR ){ continue; }
        if( size <= 0 ){ throw std::runtime_error("Worker pipe write failed."); }
        offset += size;
      }
    }
    std::string receive() {
      std::string result;
      while( result.size() < 1024 ){
        pollfd fd{_output_, POLLIN, 0};
        const auto status = poll(&fd, 1, 30000);
        if( status < 0 and errno == EINTR ){ continue; }
        if( status <= 0 ){ throw std::runtime_error("Worker response timed out."); }
        char character;
        if( read(_output_, &character, 1) != 1 ){ throw std::runtime_error("Worker exited before responding."); }
        if( character == '\n' ){ return result; }
        result += character;
      }
      throw std::runtime_error("Worker response is too long.");
    }
    pid_t pid() const { return _pid_; }
    void finish() {
      int status = 0;
      close(_input_); _input_ = -1;
      if( waitpid(_pid_, &status, 0) != _pid_ ){ throw std::runtime_error("Cannot wait for worker."); }
      _pid_ = -1;
      if( not WIFEXITED(status) or WEXITSTATUS(status) != 0 ){ throw std::runtime_error("Worker failed."); }
    }
  private:
    pid_t _pid_{-1};
    int _input_{-1}, _output_{-1};
  };

  void complete(id<MTLCommandBuffer> command_) {
    [command_ commit]; [command_ waitUntilCompleted];
    if( [command_ status] == MTLCommandBufferStatusError ){
      throw std::runtime_error([[[command_ error] description] UTF8String]);
    }
  }
}

int main(int argc_, char** argv_) {
  @autoreleasepool {
    try {
      if( argc_ != 6 ){ throw std::invalid_argument("Expected python, worker script, extension directory, mapping path and count."); }
      signal(SIGPIPE, SIG_IGN);
      const auto parsedCount = std::stoul(argv_[5]);
      if( parsedCount == 0 or parsedCount > 1000000 ){ throw std::invalid_argument("Invalid event count."); }
      const auto count = std::uint32_t(parsedCount);
      auto device = MTLCreateSystemDefaultDevice();
      if( device == nil or not [device hasUnifiedMemory] ){
        throw std::runtime_error("This test requires a unified-memory Metal device.");
      }
      MpsSharedBuffer mapping(device, argv_[4], count, true);
      auto queue = [device newCommandQueue];
      NSError* error = nil;
      auto options = [MTLCompileOptions new]; options.mathMode = MTLMathModeSafe;
      auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:kernels] options:options error:&error];
      [options release];
      if( library == nil ){ throw std::runtime_error([[error description] UTF8String]); }
      auto makePipeline = [&](NSString* name_){
        auto function = [library newFunctionWithName:name_];
        auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        if( pipeline == nil ){ throw std::runtime_error([[error description] UTF8String]); }
        return pipeline;
      };
      auto producer = makePipeline(@"produce"), consumer = makePipeline(@"consume");
      WorkerProcess worker({argv_[1], "-B", argv_[2], "--worker", argv_[3], argv_[4], argv_[5]});
      if( worker.receive() != "ready " + std::to_string(worker.pid()) ){
        throw std::runtime_error("Worker did not confirm its separate process.");
      }
      for( std::uint32_t generation = 0; generation < 8; ++generation ){
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:producer];
        [encoder setBuffer:mapping.buffer() offset:0 atIndex:0];
        [encoder setBuffer:mapping.buffer() offset:mapping.inputOffset() atIndex:1];
        [encoder setBuffer:mapping.buffer() offset:mapping.weightOffset() - sizeof(float) atIndex:2];
        [encoder setBuffer:mapping.buffer() offset:mapping.errorOffset() atIndex:3];
        [encoder setBytes:&count length:sizeof(count) atIndex:4];
        [encoder setBytes:&generation length:sizeof(generation) atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding]; complete(command);

        worker.send("evaluate " + std::to_string(generation));
        if( worker.receive() != "ok " + std::to_string(generation) ){
          throw std::runtime_error("Worker did not acknowledge the current evaluation.");
        }
        command = [queue commandBuffer]; encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:consumer];
        [encoder setBuffer:mapping.buffer() offset:mapping.weightOffset() - sizeof(float) atIndex:0];
        [encoder setBuffer:mapping.buffer() offset:mapping.errorOffset() atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBytes:&generation length:sizeof(generation) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding]; complete(command);
      }
      worker.send("shutdown");
      if( worker.receive() != "bye" ){ throw std::runtime_error("Worker shutdown failed."); }
      worker.finish();
      const auto errors = mapping.errorCount();
      [producer release]; [consumer release]; [library release]; [queue release];
      if( errors != 0 ){ throw std::runtime_error("GPU validation errors: " + std::to_string(errors)); }
      std::cout << "PASS: " << count << " events, 8 generations, native producer and separate Python worker, "
                   "shared Metal/Torch buffers, GPU validation and guards; only final error counter read on CPU.\n";
      return 0;
    }
    catch( const std::exception& error ){
      std::cerr << "FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
