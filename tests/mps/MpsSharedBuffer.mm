#include "MpsSharedBuffer.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

MpsSharedBuffer::MpsSharedBuffer(id<MTLDevice> device_, const std::string& path_,
                                std::uint32_t count_, bool create_)
    : _path_(path_), _count_(count_), _owner_(create_) {
  if( count_ == 0 or count_ > 1000000 ){
    throw std::invalid_argument("Expected between 1 and 1000000 events.");
  }
  const auto page = std::size_t(sysconf(_SC_PAGESIZE));
  _bytes_ = ((errorOffset() + sizeof(std::uint32_t) + page - 1) / page) * page;
  try {
    _fd_ = open(path_.c_str(), O_RDWR | (create_ ? O_CREAT | O_EXCL : 0), 0600);
    if( _fd_ < 0 ){ throw std::runtime_error(std::strerror(errno)); }
    // Don't let a worker inherit the producer's file descriptor.
    if( fcntl(_fd_, F_SETFD, FD_CLOEXEC) < 0 ){ throw std::runtime_error(std::strerror(errno)); }
    if( create_ and ftruncate(_fd_, off_t(_bytes_)) != 0 ){
      throw std::runtime_error(std::strerror(errno));
    }
    struct stat info{};
    if( fstat(_fd_, &info) != 0 or info.st_size != off_t(_bytes_) ){
      throw std::runtime_error("Unexpected shared buffer size.");
    }
    _mapping_ = mmap(nullptr, _bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, _fd_, 0);
    if( _mapping_ == MAP_FAILED ){ _mapping_ = nullptr; throw std::runtime_error(std::strerror(errno)); }
    _buffer_ = [device_ newBufferWithBytesNoCopy:_mapping_ length:_bytes_
                                      options:MTLResourceStorageModeShared deallocator:nil];
    if( _buffer_ == nil ){ throw std::runtime_error("Metal rejected the shared no-copy mapping."); }
  }
  catch( ... ){ release(); throw; }
}

void MpsSharedBuffer::release() {
  [_buffer_ release];
  _buffer_ = nil;
  if( _mapping_ ){ munmap(_mapping_, _bytes_); _mapping_ = nullptr; }
  if( _fd_ >= 0 ){ close(_fd_); _fd_ = -1; if( _owner_ ){ unlink(_path_.c_str()); } }
}

MpsSharedBuffer::~MpsSharedBuffer() { release(); }

std::uint32_t MpsSharedBuffer::errorCount() const {
  return *reinterpret_cast<const std::uint32_t*>(static_cast<const char*>(_mapping_) + errorOffset());
}
