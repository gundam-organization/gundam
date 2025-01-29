#include "gtest/gtest.h"
#include "hemi/launch.h"

#ifdef HEMI_CUDA_COMPILER
#define ASSERT_SUCCESS(res) ASSERT_EQ(cudaSuccess, (res));
#define ASSERT_FAILURE(res) ASSERT_NE(cudaSuccess, (res));
#define hemiPortableLaunchTest hemiPortableLaunchTestDevice
#else
#define ASSERT_SUCCESS(res)
#define ASSERT_FAILURE(res)
#define hemiPortableLaunchTest hemiPortableLaunchTestHost
#endif

HEMI_MEM_DEVICE int result;
HEMI_MEM_DEVICE int rGDim;
HEMI_MEM_DEVICE int rBDim;

template <typename T, typename... Arguments>
HEMI_DEV_CALLABLE
T first(T f, Arguments...) {
	return f;
}

template <typename... Arguments>
struct k {
	HEMI_DEV_CALLABLE_MEMBER void operator()(Arguments... args) const {
		result = first(args...); //sizeof...(args);
#ifdef HEMI_DEV_CODE
		rGDim = 1;//gridDim.x;
		rBDim = 1;//blockDim.x;
#endif
	}
};

TEST(hemiPortableLaunchTest, KernelFunction_AutoConfig) {
	k<int> kernel;
	hemi::launch(kernel, 1);
}
