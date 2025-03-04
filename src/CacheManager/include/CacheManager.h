#ifndef CacheManager_h_seen
#define CacheManager_h_seen

#include "CacheSampleHistFiller.h"
#include "CacheEventWeightFiller.h"
#include "CacheParameters.h"
#include "CacheWeights.h"

#include "WeightNormalization.h"
#include "WeightCompactSpline.h"
#include "WeightMonotonicSpline.h"
#include "WeightUniformSpline.h"
#include "WeightGeneralSpline.h"
#include "WeightGraph.h"
#include "WeightBilinear.h"
#include "WeightBicubic.h"
#include "WeightTabulated.h"

#ifdef CACHE_MANAGER_USE_INDEXED_SUMS
// An older implementation of the histogram summing that may be faster for
// some (peculiar) data sets.
#include "CacheIndexedSums.h"
namespace Cache {
  using HistogramSum = Cache::IndexedSums;
}
#else
// A GPU optimized implementation of histogram summing that will be faster
// for most data sets.
#include "CacheRecursiveSums.h"
namespace Cache {
  using HistogramSum = Cache::RecursiveSums;
}
#endif

#include "SampleSet.h"
#include "EventDialCache.h"

#include "GenericToolbox.Time.h"

#include "hemi/array.h"

#include <map>
#include <vector>

namespace Cache {
  class Manager;
}

class Parameter;

/// Manage the cache calculations on the GPU.  This will work even when there
/// isn't a GPU, but it's really slow on the CPU.  This is a singleton.
class Cache::Manager {

private:
  // Holds static members of the CacheManager in one place
  struct Parameters{

    // You get one guess...
    Manager* fSingleton{nullptr};

    /// Control whether the Cache::Manager should be constructed.  This
    /// parameter should be set in the front-end applications.  This replaces a
    /// flag that was in GundamGlobals.h
    bool fIsEnabled{false};

    /// Enabled printouts when moving data between host and device
    bool fEnableDebugPrintouts{false};

    /// Control if a propagator should fill the histograms with the GPU (when
    /// avavailable), or force a parallel CPU calculation.  This really
    /// belongs in the propagator, but it's dynamic object, so put it here.
    /// The parallel calculation is used to check the accuracy of the CPU
    /// computation
    bool fForceCpuCalculation{false};

    /// A map between the fit parameter pointers and the parameter index used
    /// by the fitter.
    std::map<const Parameter*, int> ParameterMap{};

    /// True if the histogram content should be copied back from the GPU.
    /// This should almost always be true.
    bool fIsHistContentCopyEnabled{true};

    /// True if the individual event weights should be copied back from the
    /// Cache::Manager.  This should almost always be false.
    bool fIsEventWeightCopyEnabled{false};

    /// Declare all of the actual GPU caches here.  There is one GPU, so this
    /// is the ONE place that everything is collected together.
    std::vector<CacheSampleHistFiller> fSampleHistFillerList{};
    std::vector<CacheEventWeightFiller> fEventWeightFillerList{};

    // Time monitoring
    GenericToolbox::Time::AveragedTimer<10> cacheFillTimer;
    GenericToolbox::Time::AveragedTimer<10> pullFromDeviceTimer;
  };

  // instance defined in cpp file
  static Parameters fParameters;

public:

  /// Get the pointer to the cache manager.  This will be a nullptr if the
  /// cache is not being used or hasn't been constructed.
  static Manager* Get(){ return fParameters.fSingleton; }

  ///////////////////////////////////////////////////////////////
  // Static functions to set control parameters.  These are static since they
  // may be called during the program configuration before the Cache::Manager
  // is constructed.
  ///////////////////////////////////////////////////////////////

  /// Return true if CUDA was used during compilation.  CUDA is required to
  /// run on the GPU, but the Cache::Manager does not require a GPU to run the
  /// calculation (slowly).
  static bool HasCUDA();

  /// Return true if a GPU is available at runtime.  This checks if a device
  /// exists, is accessible, and has sufficient capabilities to run the
  /// Cache::Manager.
  static bool HasGPU(bool dump = false);

  /// Check if the caches for the manager have been built and the
  /// Cache::Manager will be used to fill the weights and histograms.  This
  /// will only be true when the Cache::Manager is enabled, and the
  /// Cache::Manager::Build() method has been called.
  static bool IsBuilt() {
    if (Cache::Manager::Get() == nullptr) return false;
    if (Cache::Manager::Get()->fUpdateRequired) return false;
    return true;
  }

  /// The Cache::Manager has been enabled (usually based on command line, or
  /// compilation options), and should be built.
  static void SetIsEnabled(bool enable_){ fParameters.fIsEnabled = enable_; }

  /// Provide a convenient place to save that the CPU event weight calculation
  /// should be run, even if the GPU has already calculated the event weights
  /// and filled the histograms.  This doesn't change the behavior of the
  /// Cache::Manager.
  static void SetIsForceCpuCalculation(bool enable_){ fParameters.fForceCpuCalculation = enable_; }

  /// Predicate to check if the Cache::Manager is enabled and should be built.
  /// This is only used in the Cache::Manager.  User code should use
  /// Cache::Manager::IsBuilt() which checks if the manager is ready to run.
  static bool IsCacheManagerEnabled(){ return fParameters.fIsEnabled; }

  /// Predicate to check if the Propagator should also calculate the weights
  /// using the CPU.
  static bool IsForceCpuCalculation(){ return fParameters.fForceCpuCalculation; }

  /// Used to set the flag that the histogram results should be copied from
  /// the Cache::Manager into the GUNDAM histogram object.
  static void SetIsHistContentCopyEnabled(bool fIsHistContentCopyEnabled_){ fParameters.fIsHistContentCopyEnabled = fIsHistContentCopyEnabled_; }

  /// Use to set the flag that the event weight results should be copied from
  /// the Cache::Manager into GUNDAM event weights.
  static void SetIsEventWeightCopyEnabled(bool fIsEventWeightCopyEnabled_){ fParameters.fIsEventWeightCopyEnabled = fIsEventWeightCopyEnabled_; }

  /// Set the debug output.
  static void SetEnableDebugPrintouts(bool fEnableDebugPrintouts_){ fParameters.fEnableDebugPrintouts = fEnableDebugPrintouts_; }

  /// Get the timer for the time that Cache::Manager spends passing
  /// information needed to fill the histograms, and triggering the
  /// propagation.
  static const GenericToolbox::Time::AveragedTimer<10>& GetCacheFillTimer() { return fParameters.cacheFillTimer; }

  /// Get the timer for the time spent waiting for the Cache::Manager to
  /// finish the calculation.
  static const GenericToolbox::Time::AveragedTimer<10>& GetPullFromDeviceTimer() { return fParameters.pullFromDeviceTimer; }

  /////////////////////////////////////////////////////////////////////
  // Methods used to run the Cache::Manager during the calculation.
  /////////////////////////////////////////////////////////////////////

  /// Build the cache and load it into the device.  This is used to construct
  /// the Cache::Manager singleton, and copy the event information to the
  /// device.  This will use Update to fill the event and spline information.
  static bool Build(SampleSet& sampleSet, EventDialCache& eventDialCache);

  /// Same as Propagator::propagateParameters().  This is a convenient way to
  /// fill the event weights and the histograms, but it will block until the
  /// Cache::Manager calculation is completed.
  static bool PropagateParameters();

  /// Copy the event weights from the Cache::Manager into the GUNDAM event
  /// weight classes.  This will block if the Cache::Manager calculation
  /// has not finished.
  static bool CopyEventWeights();

  /// Copy the histogram contents from the Cache::Manager into the GUNDAM
  /// histogram objects.  This will block if the Cache::Manager calculation
  /// has not finished.
  static bool CopyHistogramContents();

  /// Validate the local copy of the histogram contents against the last
  /// weight calculation.  The quiet parameter controls how much output
  /// happens during the dump.  The default is pretty loud when there is a
  /// problem.
  static bool ValidateHistogramContents(int quiet=1);

  /// This returns the index of the parameter in the cache.  If the parameter
  /// isn't defined, this will return a negative value.  This is meant as a
  /// convenience function for use outside of Cache::Manager (and probably
  /// isn't used).
  static int ParameterIndex(const Parameter* fp);

  /// Update the cache with the event and spline information.  This is called
  /// as part of Build.  It forages all of the information from the original
  /// sample list and event dials.
  bool Update(SampleSet& sampleSet, EventDialCache& eventDialCache);

  /// Fill the cache for the current iteration.  This needs to be called
  /// before the cached weights can be used.  This is used in Propagator.cpp.
  bool Fill();

  /// Return the approximate allocated memory used by the Cache.  This memory
  /// is mirrored on both the CPU and GPU.
  [[nodiscard]] std::size_t GetResidentMemory() const {return fTotalBytes;}

private:

  // Hold the configuration that will be used to construct the manager
  // (singleton).  This information was originally passed as arguments to
  // the constructor, but it became to complex and prone to mistakes since
  // C++ parameters cannot be named, and the order of a dozen or more
  // integers is easy to scramble.
  struct Configuration {
    // The number of results that are going to be calculated.  There is
    // one "result", or weight per event.
    int events{0};

    // The number of parameteters in the fit
    int parameters{0};

    // The number of histogram bins in the final histogram.
    int histBins{0};

    // The option for how the space should be allocated that is
    // passed to the weight calculation classes.
    std::string spaceOption{"space"};

    // The number of normalization parameters
    int norms{0};

    // The number of shift dials that have been applied.  These are
    // applied to the initial event weight outside of Cache::Manager and
    // are counted here for informational purposes.
    int shifts{0};

    // The parameters for the dial type CompactSpline (i.e. the
    // Catmull-Rom) splines
    int compactSplines{0}; // The number of splines
    int compactPoints{0};  // The number of knots used in the splines

    // The parameters for the dial type MonotonicSpline (i.e. the
    // Catmul-Rom splines with monotonic conditions applied).
    int monotonicSplines{0};   // The number of splines
    int monotonicPoints{0};    // The data reserved for the splines.

    // The parameters for the dial type UniformSpline (i.e. a spline with
    // values, and slopes at uniform abcissas).
    int uniformSplines{0};     // The number of splines
    int uniformPoints{0};      // The data reserved for the splines.

    // The parameters for the dial type GeneralSpline (i.e. a spline with
    // values and slopes at non-uniform abcissas).
    int generalSplines{0};  // The number of splines
    int generalPoints{0};   // The amount of data reserved for the splines

    // The parameters for the dial type LightGraph (i.e. a graph for
    // linear interpolation at non-uniform abcissas).
    int graphs{0};          // The number of graphs
    int graphPoints{0};     // The amount of data reserved for the graphs

    // The parameters for the dial type Bilinear (i.e. a bilinear
    // surface).
    int bilinear{0};       // The number of bilinear surfaces
    int bilinearPoints{0}; // The amount of data reserved for the surfaces

    // The parameters for the dial type Bicubic (i.e. a bicubic
    // surface).
    int bicubic{0};       // The number of bicubic surfaces
    int bicubicPoints{0}; // The amount of data reserved for the surfaces

    // The parameters for the dial type Tabulated
    int tabulated{0};     // The number of tabulated dials
    int tabulatedPoints{0};   // The number of entries in all the tables
    std::map<const std::vector<double>*, int> tables; // The offsets of each lookup table.

  };

  // This is a singleton, so the constructor is private.
  explicit Manager(const Cache::Manager::Configuration& config);

  /// The cache for parameter weights (on the GPU).
  std::unique_ptr<Cache::Parameters> fParameterCache;

  /// The cache for event weights.
  std::unique_ptr<Cache::Weights> fWeightsCache;

  /// The cache for the normalizations
  std::unique_ptr<Cache::Weight::Normalization> fNormalizations;

  /// The cache for the compact (Catmull-Rom) splines
  std::unique_ptr<Cache::Weight::CompactSpline> fCompactSplines;

  /// The cache for the monotonic splines
  std::unique_ptr<Cache::Weight::MonotonicSpline> fMonotonicSplines;

  /// The cache for the uniform splines
  std::unique_ptr<Cache::Weight::UniformSpline> fUniformSplines;

  /// The cache for the general splines
  std::unique_ptr<Cache::Weight::GeneralSpline> fGeneralSplines;

  /// The cache for the general splines
  std::unique_ptr<Cache::Weight::Graph> fGraphs;

  /// The cache for the general splines
  std::unique_ptr<Cache::Weight::Bilinear> fBilinear;

  /// The cache for the general splines
  std::unique_ptr<Cache::Weight::Bicubic> fBicubic;

  /// The cache for the precalculated weight tables.
  std::unique_ptr<Cache::Weight::Tabulated> fTabulated;

  /// The cache for the summed histgram weights
  std::unique_ptr<Cache::HistogramSum> fHistogramsCache;

  // The rough size of all the caches.
  std::size_t fTotalBytes;

  // Set to true when the cache needs an update.
  bool fUpdateRequired{true};

  // Keep track of the SampleSet and EventDialCache that are part of the
  // cache. This cannot be changed.
  SampleSet* fSampleSet{nullptr};

  // Keep track of the SampleSet and EventDialCache that are part of the
  // cache.  This cannot be changed.
  EventDialCache* fEventDialCache{nullptr};

public:
  virtual ~Manager() = default;

  // Provide "internal" references to the GPU cache.
  Cache::Parameters& GetParameterCache() {return *fParameterCache;}
  Cache::Weights&    GetWeightsCache() {return *fWeightsCache;}
  Cache::HistogramSum& GetHistogramsCache() {return *fHistogramsCache;}
};

// An MIT Style License

// Copyright (c) 2022 Clark McGrew

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#endif
