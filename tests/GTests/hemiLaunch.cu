#include "gtest/gtest.h"
#include "hemi/hemi.h"
#include "hemi/launch.h"

#define ASSERT_SUCCESS(res) ASSERT_EQ(cudaSuccess, (res));
#define ASSERT_FAILURE(res) ASSERT_NE(cudaSuccess, (res));

// for testing hemi::launch()
struct KernelClass {
	template <typename... Arguments>
	HEMI_DEV_CALLABLE_MEMBER void operator()(int *count, int *bdim, int *gdim, Arguments... args) {
		*count = sizeof...(args);
		*bdim = blockDim.x;
		*gdim = gridDim.x;
	}
};

// for testing hemi::cudaLaunch()
template <typename... Arguments>
HEMI_LAUNCHABLE void KernelFunc(int *count, int *bdim, int *gdim, Arguments... args) {
	KernelClass k;
	k(count, bdim, gdim, args...);
}

class hemiLaunchTestDevice : public ::testing::Test {
protected:
  virtual void SetUp() {
  	cudaMalloc(&dCount, sizeof(int));
    cudaMalloc(&dBdim, sizeof(int));
	cudaMalloc(&dGdim, sizeof(int));

	int devId;
	cudaGetDevice(&devId);
	cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, devId);
  }

  virtual void TearDown() {
  	cudaFree(dCount);
  	cudaFree(dBdim);
  	cudaFree(dGdim);
  }

  KernelClass kernel;
  int smCount;

  int *dCount;

  int *dBdim;
  int *dGdim;

  int count;

  int bdim;
  int gdim;
};


TEST_F(hemiLaunchTestDevice, CorrectVariadicParams) {
        hemi::launch(kernel, dCount, dBdim, dGdim, 1);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
	ASSERT_EQ(count, 1);

	hemi::launch(kernel, dCount, dBdim, dGdim, 1, 2);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
	ASSERT_EQ(count, 2);

	hemi::launch(kernel, dCount, dBdim, dGdim, 1, 2, 'a', 4.0, "hello");
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
	ASSERT_EQ(count, 5);
}

TEST_F(hemiLaunchTestDevice, AutoConfigMaximalLaunch) {
	hemi::launch(kernel, dCount, dBdim, dGdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));

	ASSERT_GE(gdim, smCount);
	ASSERT_EQ(gdim%smCount, 0);
	ASSERT_GE(bdim, 32);
}

TEST_F(hemiLaunchTestDevice, ExplicitBlockSize)
{
	hemi::ExecutionPolicy ep;
	ep.setBlockSize(128);
	hemi::launch(ep, kernel, dCount, dBdim, dGdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));

	ASSERT_GE(gdim, smCount);
	ASSERT_EQ(gdim%smCount, 0);
	ASSERT_EQ(bdim, 128);
}

TEST_F(hemiLaunchTestDevice, ExplicitGridSize)
{
	hemi::ExecutionPolicy ep;
	ep.setGridSize(100);
	hemi::launch(ep, kernel, dCount, dBdim, dGdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));

	ASSERT_EQ(gdim, 100);
	ASSERT_GE(bdim, 32);
}

#ifdef CONTINUE_WITHOUT_ABORT_ON_CUDA_ERROR
TEST_F(hemiLaunchTestDevice, InvalidConfigShouldFail)
{
	// Fail due to block size too large
	hemi::ExecutionPolicy ep;
	ep.setBlockSize(10000);
        try {
            hemi::launch(ep, kernel, dCount, dBdim, dGdim);
            ASSERT_FAILURE(checkCudaErrors());
        }
        catch (...) {
            std::cout << "caught" << std::endl;
            ASSERT_SUCCESS(checkCudaErrors());
        }


	// Fail due to excessive shared memory size
	ep.setBlockSize(0);
	ep.setGridSize(0);
	ep.setSharedMemBytes(1000000);
        try {
            hemi::launch(ep, kernel, dCount, dBdim, dGdim);
            ASSERT_FAILURE(checkCudaErrors());
        }
        catch (...) {
            ASSERT_SUCCESS(checkCudaErrors());
        }
}
#endif

TEST_F(hemiLaunchTestDevice, CorrectVariadicParams_cudaLaunch) {
	hemi::cudaLaunch(KernelFunc, dCount, dBdim, dGdim, 1);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
	ASSERT_EQ(count, 1);

	hemi::cudaLaunch(KernelFunc, dCount, dBdim, dGdim, 1, 2);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
	ASSERT_EQ(count, 2);

	hemi::cudaLaunch(KernelFunc, dCount, dBdim, dGdim, 1, 2, 'a', 4.0, "hello");
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&count, dCount, sizeof(int), cudaMemcpyDefault));
	ASSERT_EQ(count, 5);
}

TEST_F(hemiLaunchTestDevice, AutoConfigMaximalLaunch_cudaLaunch) {
	hemi::cudaLaunch(KernelFunc, dCount, dBdim, dGdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));

	ASSERT_GE(gdim, smCount);
	ASSERT_EQ(gdim%smCount, 0);
	ASSERT_GE(bdim, 32);
}

TEST_F(hemiLaunchTestDevice, ExplicitBlockSize_cudaLaunch)
{
	hemi::ExecutionPolicy ep;
	ep.setBlockSize(128);
	hemi::cudaLaunch(ep, KernelFunc, dCount, dBdim, dGdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));

	ASSERT_GE(gdim, smCount);
	ASSERT_EQ(gdim%smCount, 0);
	ASSERT_EQ(bdim, 128);
}

TEST_F(hemiLaunchTestDevice, ExplicitGridSize_cudaLaunch)
{
	hemi::ExecutionPolicy ep;
	ep.setGridSize(100);
	hemi::cudaLaunch(ep, KernelFunc, dCount, dBdim, dGdim);
	ASSERT_SUCCESS(cudaDeviceSynchronize());
	ASSERT_SUCCESS(cudaMemcpy(&bdim, dBdim, sizeof(int), cudaMemcpyDefault));
	ASSERT_SUCCESS(cudaMemcpy(&gdim, dGdim, sizeof(int), cudaMemcpyDefault));

	ASSERT_EQ(gdim, 100);
	ASSERT_GE(bdim, 32);
}

#ifdef CONTINUE_WITHOUT_ABORT_ON_CUDA_ERROR
TEST_F(hemiLaunchTestDevice, InvalidConfigShouldFail_cudaLaunch)
{
	// Fail due to block size too large
	hemi::ExecutionPolicy ep;
	ep.setBlockSize(10000);
        try {
            hemi::cudaLaunch(ep, KernelFunc, dCount, dBdim, dGdim);
            ASSERT_FAILURE(checkCudaErrors());
        }
        catch (...) {
            ASSERT_SUCCESS(checkCudaErrors());
        }

	// Fail due to excessive shared memory size
	ep.setBlockSize(0);
	ep.setGridSize(0);
	ep.setSharedMemBytes(1000000);
        try {
            hemi::cudaLaunch(ep, KernelFunc, dCount, dBdim, dGdim);
            ASSERT_FAILURE(checkCudaErrors());
        }
        catch (...) {
            ASSERT_SUCCESS(checkCudaErrors());
        }
}
#endif
