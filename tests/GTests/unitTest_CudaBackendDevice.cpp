#include "CudaBackend.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <stdexcept>

TEST(CudaBackendDevice, BuildsSharedModelAndRestoresDeviceSelection) {
  int deviceCount = 0;
  ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
  ASSERT_GT(deviceCount, 0) << "Run cuda-backend-skeleton inside a GPU allocation.";
  int originalDevice = 0;
  ASSERT_EQ(cudaGetDevice(&originalDevice), cudaSuccess);
  // With several visible GPUs, initialize device zero while another is current.
  ASSERT_EQ(cudaSetDevice(deviceCount - 1), cudaSuccess);
  {
    Backends::CudaBackend backend;
    ASSERT_TRUE(backend.isCompiled());
    backend.configure(ConfigReader(JsonType{{"deviceIndex", 0}}));
    ASSERT_NO_THROW(backend.initialize());
    auto model = std::make_shared<Backends::EngineView>();
    model->propagation.parameterCount = 3;
    const auto* address = model.get();
    ASSERT_NO_THROW(backend.buildShared(model));
    model.reset();
    EXPECT_EQ(&backend.getEngineView(), address);
    EXPECT_EQ(backend.getEngineView().propagation.parameterCount, 3);
    cudaDeviceProp properties{};
    ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
    EXPECT_EQ(backend.getCapabilities().deviceName, properties.name);
    EXPECT_TRUE(backend.getCapabilities().supportsEventWeights);
    const auto resources = backend.getResourceUsage();
    EXPECT_TRUE(resources.available);
    EXPECT_FALSE(resources.processMemory);
    EXPECT_GT(resources.allocatedBytes, 0);
    EXPECT_TRUE(resources.hasDeviceMemoryInfo);
    EXPECT_GT(resources.totalDeviceBytes, 0);
    EXPECT_LE(resources.freeDeviceBytes, resources.totalDeviceBytes);
    EXPECT_EQ(resources.deviceName, properties.name);
    EXPECT_NE(backend.getMonitorLine().find("Memory:"), std::string::npos);
    int currentDevice = -1;
    ASSERT_EQ(cudaGetDevice(&currentDevice), cudaSuccess);
    EXPECT_EQ(currentDevice, deviceCount - 1);
    EXPECT_THROW(backend.requestPropagation({}), std::runtime_error);
    EXPECT_FALSE(backend.isReady({1, true}));

    // A failed rebuild must not leave the preceding model accessible.
    EXPECT_THROW(backend.buildShared(nullptr), std::runtime_error);
    EXPECT_THROW((void)backend.getEngineView(), std::runtime_error);
    ASSERT_NO_THROW(backend.build(Backends::EngineView{}));
    backend.configure(ConfigReader(JsonType{{"deviceIndex", std::numeric_limits<int>::max()}}));
    EXPECT_THROW(backend.build(Backends::EngineView{}), std::runtime_error);
    EXPECT_THROW((void)backend.getEngineView(), std::runtime_error);
  }
  int currentDevice = -1;
  ASSERT_EQ(cudaGetDevice(&currentDevice), cudaSuccess);
  EXPECT_EQ(currentDevice, deviceCount - 1);
  EXPECT_EQ(cudaSetDevice(originalDevice), cudaSuccess);
}
