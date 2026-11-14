#ifndef GUNDAM_TEST_MPS_SHARED_BUFFER_H
#define GUNDAM_TEST_MPS_SHARED_BUFFER_H

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <string>

// Test transport, independent of CPython and Torch. Both processes map the same
// file and wrap its pages with Metal. Never pass a process-local pointer over IPC.
class MpsSharedBuffer {
public:
  MpsSharedBuffer(id<MTLDevice> device_, const std::string& path_,
                  std::uint32_t count_, bool create_);
  ~MpsSharedBuffer();
  MpsSharedBuffer(const MpsSharedBuffer&) = delete;
  MpsSharedBuffer& operator=(const MpsSharedBuffer&) = delete;
  id<MTLBuffer> buffer() const { return _buffer_; }
  std::size_t inputOffset() const { return 64; }
  std::size_t weightOffset() const { return inputOffset() + _count_ * sizeof(float) + sizeof(float); }
  std::size_t errorOffset() const { return weightOffset() + (_count_ + 1) * sizeof(float); }
  std::uint32_t errorCount() const;
  std::uint32_t count() const { return _count_; }

private:
  void release();
  std::string _path_;
  std::uint32_t _count_;
  bool _owner_;
  void* _mapping_{nullptr};
  std::size_t _bytes_{0};
  int _fd_{-1};
  id<MTLBuffer> _buffer_{nil};
};

#endif
