///////////////////////////////////////////////////////////////////////////////
//
// "Hemi" CUDA Portable C/C++ Utilities
//
// Copyright 2012-2015 NVIDIA Corporation
//
// License: BSD License, see LICENSE file in Hemi home directory
//
// The home for Hemi is https://github.com/harrism/hemi
//
///////////////////////////////////////////////////////////////////////////////
// Please see the file README.md (https://github.com/harrism/hemi/README.md)
// for full documentation and discussion.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "hemi/hemi.h"
#include <cstring>
#include <mutex>

#ifdef HEMI_ARRAY_DEBUG
#include <iostream>
#define HEMI_ARRAY_OUTPUT(arg) std::cout << arg << std::endl
#else
#define HEMI_ARRAY_OUTPUT(arg) /* nothing */
#endif

#ifndef HEMI_ARRAY_DEFAULT_LOCATION
#ifdef HEMI_CUDA_COMPILER
#define HEMI_ARRAY_DEFAULT_LOCATION hemi::device
#else
#define HEMI_ARRAY_DEFAULT_LOCATION hemi::host
#endif
#endif

#ifndef HEMI_CUDA_COMPILER
#ifndef HEMI_CUDA_DISABLE
#define HEMI_CUDA_DISABLE
#endif
#endif

namespace hemi {
#ifndef HEMI_DISABLE_THREADS
     inline std::mutex& deviceLock() {
          static std::mutex lock;
          return lock;
     }
#endif

     template <typename T> class Array; // forward decl

     enum Location {
          host   = 0,
          device = 1
     };

     template <typename T>
          class Array
     {
     public:
          // Construct the an array of "n" objects of class "T".object.
     Array(size_t n, bool usePinned=true) :
          nSize(n),
               hPtr(0),
               dPtr(0),
               isForeignHostPtr(false),
               isPinned(usePinned),
               isHostAlloced(false),
               isDeviceAlloced(false),
               isHostValid(false),
               isDeviceValid(false)
          {
          }

          // Use a pre-allocated host pointer (use carefully!)
     Array(T *hostMem, size_t n) :
          nSize(n),
               hPtr(hostMem),
               dPtr(0),
               isForeignHostPtr(true),
               isPinned(false),
               isHostAlloced(true),
               isDeviceAlloced(false),
               isHostValid(true),
               isDeviceValid(false)
          {
          }

          ~Array()
          {
               deallocateDevice();
               if (!isForeignHostPtr)
                    deallocateHost();
          }

          size_t size() const { return nSize; }

          // copy from/to raw external pointers (host or device)

          void copyFromHost(const T *other, size_t n)
          {
               if ((isHostAlloced || isDeviceAlloced) && nSize != n) {
                    deallocateHost();
                    deallocateDevice();
                    nSize = n;
               }
               memcpy(writeOnlyHostPtr(), other, nSize * sizeof(T));
          }

          void copyToHost(T *other, size_t n) const
          {
               assert(isHostAlloced);
               assert(n <= nSize);
               memcpy(other, readOnlyHostPtr(), n * sizeof(T));
          }

#ifndef HEMI_CUDA_DISABLE
          void copyFromDevice(const T *other, size_t n)
          {
               if ((isHostAlloced || isDeviceAlloced) && nSize != n) {
                    deallocateHost();
                    deallocateDevice();
                    nSize = n;
               }
               checkCuda( cudaMemcpy(writeOnlyDevicePtr(), other,
                                     nSize * sizeof(T),
                                     cudaMemcpyDeviceToDevice) );
          }

          void copyToDevice(T *other, size_t n)
          {
               assert(isDeviceAlloced);
               assert(n <= nSize);
               checkCuda( cudaMemcpy(other, readOnlyDevicePtr(),
                                     nSize * sizeof(T),
                                     cudaMemcpyDeviceToDevice) );
          }
#endif

          // read/write pointer access

          // R/W pointer access: Decide if the host or device pointer is needed
          // using hostPtr() or devicePtr().
          T* ptr(Location loc = HEMI_ARRAY_DEFAULT_LOCATION)
          {
               if (loc == host) return hostPtr();
               else return devicePtr();
          }

          // R/W pointer access: Reference the memory that resides on the host.
          // Copies data from device to host (if needed) and marks the device
          // memory as invalid.
          T* hostPtr()
          {
               if (isDeviceValid && !isHostValid) copyDeviceToHost();
               else if (!isHostAlloced) allocateHost();
               else assert(isHostValid);
               isDeviceValid = false;
               isHostValid   = true;
               return hPtr;
          }

          // R/W pointer access: Reference the memory that resides on the
          // device.  Copies data from host to device (if needed) and marks the
          // host memory as invalid.
          T* devicePtr()
          {
               if (!isDeviceValid && isHostValid) copyHostToDevice();
               else if (!isDeviceAlloced) allocateDevice();
               else assert(isDeviceValid);
               isDeviceValid = true;
               isHostValid = false;
               return dPtr;
          }

          // read-only pointer access

          // Read-only: Decide if a host or a device pointer is needed.
          const T* readOnlyPtr(Location loc = HEMI_ARRAY_DEFAULT_LOCATION) const
          {
               if (loc == host) return readOnlyHostPtr();
               else return readOnlyDevicePtr();
          }

          // Read-only: Reference the memory on the host.  Copies data from
          // the device to the host (if needed).
          const T* readOnlyHostPtr() const
          {
               if (isDeviceValid && !isHostValid) copyDeviceToHost();
               else assert(isHostValid);
               return hPtr;
          }

          // Read-only: Reference the memory on the device.  Copies data from
          // the host to the device (if needed).
          const T* readOnlyDevicePtr() const
          {
               if (!isDeviceValid && isHostValid) copyHostToDevice();
               else assert(isDeviceValid);
               return dPtr;
          }

          // Write-only: Decide if the host or the device is being accessed.
          // Ignore validity of existing data.
          T* writeOnlyPtr(Location loc = HEMI_ARRAY_DEFAULT_LOCATION)
          {
               if (loc == host) return writeOnlyHostPtr();
               else return writeOnlyDevicePtr();
          }

          // Write-only: Reference memory on the host.  Do not copy data from
          // the device and mark the device memory as invalid.
          T* writeOnlyHostPtr()
          {
               if (!isHostAlloced) allocateHost();
               isDeviceValid = false;
               isHostValid   = true;
               return hPtr;
          }

          // Write-only: Reference memory on the device.  Do not copy data from
          // the host and mark the host memory as invalid.
          T* writeOnlyDevicePtr()
          {
               if (!isDeviceAlloced) allocateDevice();
               isDeviceValid = true;
               isHostValid   = false;
               return dPtr;
          }

     private:
          size_t          nSize;

          mutable T       *hPtr;
          mutable T       *dPtr;

          bool            isForeignHostPtr;
          bool            isPinned;

          mutable bool    isHostAlloced;
          mutable bool    isDeviceAlloced;

          mutable bool    isHostValid;
          mutable bool    isDeviceValid;

     protected:
          void allocateHost() const
          {
               HEMI_ARRAY_OUTPUT("allocateHost");
               assert(!isHostAlloced);
#ifndef HEMI_CUDA_DISABLE
               if (isPinned) {
                    HEMI_ARRAY_OUTPUT("allocateHost is pinned");
                    checkCuda( cudaHostAlloc((void**)&hPtr, nSize * sizeof(T), 0));
               }
               else
#endif
                    hPtr = new T[nSize];

               isHostAlloced = true;
               isHostValid = false;

          }

          void allocateDevice() const
          {
#ifndef HEMI_CUDA_DISABLE
               HEMI_ARRAY_OUTPUT("allocateDevice");
               assert(!isDeviceAlloced);
               checkCuda( cudaMalloc((void**)&dPtr, nSize * sizeof(T)) );
               isDeviceAlloced = true;
               isDeviceValid = false;
#endif
          }

          void deallocateHost()
          {
               assert(!isForeignHostPtr);
               if (isHostAlloced) {
                    HEMI_ARRAY_OUTPUT("deallocateHost");
#ifndef HEMI_CUDA_DISABLE
                    if (isPinned) {
                         HEMI_ARRAY_OUTPUT("deallocateHost was pinned");
                         checkCuda( cudaFreeHost(hPtr) );
                    }
                    else
#endif
                         delete [] hPtr;
                    nSize = 0;
                    isHostAlloced = false;
                    isHostValid   = false;
               }
          }

          void deallocateDevice()
          {
#ifndef HEMI_CUDA_DISABLE
               if (isDeviceAlloced) {
                    HEMI_ARRAY_OUTPUT("deallocateDevice");
                    checkCuda( cudaFree(dPtr) );
                    isDeviceAlloced = false;
                    isDeviceValid   = false;
               }
#endif
          }

          void copyHostToDevice() const
          {
#ifndef HEMI_CUDA_DISABLE
               assert(isHostAlloced);
               if (!isDeviceAlloced) allocateDevice();
               HEMI_ARRAY_OUTPUT("copyHostToDevice");
               checkCuda( cudaMemcpy(dPtr,
                                     hPtr,
                                     nSize * sizeof(T),
                                     cudaMemcpyHostToDevice) );
               isDeviceValid = true;
#endif
          }

          void copyDeviceToHost() const
          {
#ifndef HEMI_CUDA_DISABLE
#ifndef HEMI_DISABLE_THREADS
               std::lock_guard<std::mutex> guard(deviceLock());
#endif
               assert(isDeviceAlloced);
               if (!isHostAlloced) allocateHost();
#ifndef HEMI_DISABLE_THREADS
               if (isHostValid) return; // done while waiting for lock
#endif
               HEMI_ARRAY_OUTPUT("copyDeviceToHost");
               checkCuda( cudaMemcpy(hPtr,
                                     dPtr,
                                     nSize * sizeof(T),
                                     cudaMemcpyDeviceToHost) );
               isHostValid = true;
#endif
          }

     };
}
