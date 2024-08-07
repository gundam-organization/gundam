Fixes relative to 1.8.4

Fixes: Guarrantee that all parameter value access is through the setter and getter so that the validity checks are correctly applied.

Fixes relative to 1.8.3

Issue #545 : Make the handling of ROOT config information portable between linux and macos.  This no longer assumes how ROOT is installed, and uses root-config portably.

Update (Related to #540): Strengthen the NaN trap in the JointProbability calculation.  An NaN there means the fit cannot continue, so throw an error.  Continue to allow INF.

Update (Prompted by #541): Make compilation less alarming by quieting warnings.  They were originally part of debugging.

Issue #536 : Add control over the size of a "kick" when starting a fit.  The starting point can be fluctuated around the prior, and the size of the step is controlled by an optional argument to --kickmc

Issue #534 : Backport CPU and GPU calculation backing from `main`, and add validation that the two calculations agree within machine precision.

Issue #530 : Fix an inefficiency in the summation of the event weights on the GPU and double the speed of the likelihood calculation using the GPU.

Issue # 524 : Apply the global event weight cap when using the Cache::Manager to calculate the likelihood.

Fixes relative to 1.8.2

Associated Fix: The CalculateGeneralSpline and CalculateGraph functions have been changed to use a brute-force binary search in place of the linear search for the correct index.  This results in a speed up of about 50% for those routines.

Issue #522 : Backport updates to extended-tests and slow-tests from the head of main.  This fixes some spurious test failures for the MCMC where the statistical fluctuations were not allowed for.

Issue #510 : Make the CacheManager calculation much more thread save when being run on the CPU.  This isn't the normal mode, but is an important cross check.  The CPU now uses (lock free) atomic addition, multiplication and value setting.

Issue #513 : Fix JointProbability so that an infinite log likelihood produces a warning, but does not terminate the program.  Infinities are valid (e.g. a zero probability).  This does produce a warning since they shouldn't occur often during a normal run.

Issue #506 : Fix output statement in ParameterSet.cpp to remove duplicated output.  

Issue #505 : Add the "isFixed" option for parameters in the configuration file.

Issue #508 : Improve error checking for Cache::Manager.  This makes sure that LogThrow is preferred to std::runtime_error.  There is an issue filed for simple-cpp-logger to make sure that the output is flushed before throwing, so that should make the error output much more readable.

Issue #502 : Update the validation code.  A validation failure caused by the recent corrections to the monotonic spline calculation (#486 and $494) has been fixed.  The expected result has been updated.  Unit testing with GoogleTest has been added and used for the HEMI GPU interface code.  Further tests are being implemented.  

Issue #492 : Fix compiler warnings from NVCC.  This is removing some unused variables.  It also fixes a "fix" that applied some correct C++ conventions to code that is aimed at the GPU (the fix produces inefficient code on a SIMD processor, and was primarily aesthetic.

Issue #490 : Don't set the value for disabled parameters in the output histograms.  This will prevent uninitialized values (i.e. NAN) from appearing in plots.

Fixes relative to 1.8.1

Issue #499 : Update to a new version of the cpp-generic-toolkit submodule. The cpp-generic-toolkit fix sets the branch status to enable branches that are used for the data.

Issue #494 : Rearrange the application of the Fritschle-Carlson criteria so that it can be tested.  This also applies a fix to Fritschle-Carlson so that the end of the spline are handled in a more reasonable fashion.

Issue #493 : Add tests for ComputeGeneralSplines to the validation suite.  As a special bonus it also adds tests for CalculateUniformSpline.h, and resolves unused variable warnings for both those functions.

Fixes relative to 1.8.0

Issue #485 : A job will correctly continue when the likelihood returns an infinite value. The job will stop if the likelihood returns a NaN.

Issue #486 : Fix the Catmull-Rom splines so that they have symmetric behavior. This changes the extrapolation behavior for both Catmull-Rom and Catmull-Rom,monotonic splines. The Catmull-Rom monotonic splines are updated to use the full Fritsche-Carlson criteria so that the interpolation is smoother.
