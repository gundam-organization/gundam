#ifndef CacheTabulated_hxx_seen
#define CacheTabulated_hxx_seen

#include "CacheWeights.h"
#include "WeightBase.h"

#include "hemi/array.h"

class SplineDial;

#include <TSpline.h>

#include <cstdint>
#include <memory>
#include <vector>
#include <map>

namespace Cache {
  namespace Weight {
    class Tabulated;
  }
}

/// A class to apply the tabulated interpolation weight parameter to the cached
/// event weights.
class Cache::Weight::Tabulated:
    public Cache::Weight::Base {
public:

  // Construct the class.  This should allocate all the memory on the host
  // and on the GPU.  The "results" are the total number of results to be
  // calculated (one result per event, often >1E+6).  The "parameters" are
  // the number of input parameters that are used (often ~1000).  The
  // parameters are not used by this class since the weight table is filled
  // before the Cache::Manager starts.  The dials are the total entries that
  // need to be reserved (typically one or two per event).  The tableSpace
  // is the number of elements in the precalculated weight tables that will
  // need to be copied to the GPU.  The map of tables holds the offset of each
  // table in the table data that is copied to the GPU.
  Tabulated(Cache::Weights::Results& results,
            Cache::Parameters::Values& parameters,
            std::size_t dials,
            std::size_t tableSpace,
            const std::map<const std::vector<double>*,int>& tables);

  virtual ~Tabulated() = default;

  /// Reinitialize the cache.  This puts it into a state to be refilled, but
  /// does not deallocate any memory.
  virtual void Reset() override;

  // Apply the kernel to the event weights.
  virtual bool Apply() override;

  /// Add the data for a single dial.  The "table" must exist in the map of
  /// tables.
  void AddData(int resultIndex,
               const std::vector<double>* table,
               int index,
               double fraction);

  /// Get the number of event-by-event entries reserved for the tabulated
  /// splines
  std::size_t GetReserved() const {return fReserved;}

  /// Return the number of entries used for the tabulated splines
  std::size_t GetUsed() const {return fUsed;}

  /// Get the space reserved for the tables.
  std::size_t GetDataReserved() const { return fDataReserved; }

  /// Get the space used by the the tables.
  std::size_t GetDataUsed() const { return fDataUsed; }

private:

  // The number of tabulated dials that have been reserved, and used.
  std::size_t fReserved;
  std::size_t fUsed;

  ///////////////////////////////////////////////////////////////////////
  /// An array of indices into the results that go for each surface.
  /// This is copied from the CPU to the GPU once, and is then constant.
  std::unique_ptr<hemi::Array<int>> fResult;

  /// An array of indices for the dial in the tables. This is copied from
  /// the CPU to the GPU once, and is then constant.
  std::unique_ptr<hemi::Array<int>> fIndex;

  /// An array of fractional components for the dial in the tables. This is
  /// copied from the CPU to the GPU once, and is then constant.
  std::unique_ptr<hemi::Array<WEIGHT_BUFFER_FLOAT>> fFraction;

  /// An array for the data in the tables.  This is copied from the CPU to
  /// the GPU for each iteration.
  std::size_t    fDataReserved;
  std::size_t    fDataUsed;
  std::unique_ptr<hemi::Array<WEIGHT_BUFFER_FLOAT>> fData;

  // The offsets for each table in the data.
  const std::map<const std::vector<double>*, int> fTables;
};

// An MIT Style License

// Copyright (c) 2024 Clark McGrew

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

// Local Variables:
// mode:c++
// c-basic-offset:4
// End:
#endif
