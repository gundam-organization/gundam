/// @file Bilinear.h
/// @brief Declares Bilinear, a two-parameter surface dial that interpolates a
///        TH2 grid bilinearly.
/// @ingroup DialDictionary

#ifndef GUNDAM_BILINEAR_INTERPOLATION_H
#define GUNDAM_BILINEAR_INTERPOLATION_H

#include "DialBase.h"
#include "DialUtils.h"
#include "DialInputBuffer.h"

#include <vector>
#include <utility>

/// Manage a bilinear interpolation for a grid of points.
///
/// A dial is the response function of one or more fit parameters: it turns
/// the current parameter values into a weight factor for an event or a bin.
/// This one depends on two parameters. DialCollection::makeSurfaceDial()
/// creates it for `Surface` dial sets whose options (`dialSubType`) are
/// exactly `Bilinear`, from a TH2. The first dial input (input buffer
/// element 0) runs along the histogram's X axis and the second along its Y
/// axis. The dial is then stored in its DialCollection.
///
/// @par Interpolation
/// The knots are the bin centres of the TH2 (under- and overflow bins are not
/// used) and the knot values are the bin contents. Bins may have different
/// widths, so the knots need not be uniformly spaced on either axis. Inside
/// a grid cell the response is interpolated linearly along X on the two
/// bounding rows, then linearly along Y between them. It always stays
/// between the four corner values of the cell. It is continuous, but its
/// first derivatives can jump across cell edges.
///
/// @par Data layout
/// `_splineData_` (returned by getDialData()) holds, with `nx` and `ny` the
/// numbers of X and Y bins:
/// - `[0]`: `nx` and `[1]`: `ny` (stored as doubles);
/// - `[2 .. 2+nx-1]`: X positions of the knots;
/// - `[2+nx .. 2+nx+ny-1]`: Y positions of the knots;
/// - `[2+nx+ny + ix*ny + iy]`: value of the knot at X index `ix` and Y index
///   `iy` (X-major order).
///
/// This is the layout that evalResponse() unpacks for
/// CalculateBilinearInterpolation(); the GPU weight calculator
/// Cache::Weight::Bilinear receives the same block.
///
/// @par Grid size
/// CalculateBilinearInterpolation() documents at least 2 and at most 16
/// knots per axis. buildDial() does not check this.
///
/// @par Outside the knot range
/// The knot range ends at the centres of the first and last bins. When
/// extrapolation is not allowed (the class default), each input is clamped
/// to its own axis range, so the response is flat between an outer bin
/// centre and the histogram edge. When it is allowed, the edge cell's linear
/// interpolation is continued beyond the range.
/// Dials made by DialCollection::makeDial() extrapolate unless the dial
/// set sets `allowDialExtrapolation` to false.
/// @note The Cache::Manager GPU path runs the same kernel on getDialData()
///       but passes the parameter value unclamped: it never checks the
///       extrapolation flag, so it always extrapolates.
///
/// @par Thread safety
/// Once built, the dial is only read by evalResponse(), so it can be
/// evaluated from several threads at once.
///
/// @see Bicubic
/// @ingroup DialDictionary
class Bilinear : public DialBase {

public:
  /// Creates an empty surface. Call buildDial() before evaluating it.
  Bilinear() = default;
  /// Destroys the dial; it owns only its data vectors.
  ~Bilinear() override = default;

  /// Copies this dial, including its grid data, bounds and extrapolation
  /// flag. Used by the polymorphic dial container (DialBaseObject) when
  /// dials are copied.
  /// @return A new Bilinear equal to this one, owned by the caller.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Bilinear>(*this); }

  /// Gives the type name of this dial. Cache::Manager::Build() uses it to
  /// count the dials of each type and size its GPU buffers.
  /// @return `"Bilinear"`.
  [[nodiscard]] std::string getDialTypeName() const override {return {"Bilinear"};}

  /// Evaluates the bilinear interpolation at the current values of the two
  /// parameters.
  /// @param input_ Input buffer of the dial. Element 0 is the X-axis
  ///        parameter and element 1 the Y-axis parameter, after any
  ///        mirroring.
  /// @return The interpolated response. It is not clamped here; the response
  ///         limits of the dial set are applied by DialResponseSupervisor.
  /// @pre buildDial() has been called.
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  // other overrides

  /// Sets whether evalResponse() may extrapolate beyond the outer bin
  /// centres. DialCollection::makeDial() calls it with the dial set's
  /// `allowDialExtrapolation` option, which is true when not configured.
  /// @param allowExtrapolation If false, each input is clamped to its axis
  ///        range. If true, the edge cell is extrapolated linearly.
  void setAllowExtrapolation(bool allowExtrapolation) override { _allowExtrapolation_ = allowExtrapolation; }

  /// Tells whether evalResponse() extrapolates beyond the knot range.
  /// @return The value given to setAllowExtrapolation(); false by default.
  [[nodiscard]] bool getAllowExtrapolation() const override { return _allowExtrapolation_; }

  /// Gives the packed grid data (layout in the class description).
  /// Cache::Manager copies it to the GPU.
  /// @return A reference to `_splineData_`; empty before buildDial().
  [[nodiscard]] const std::vector<double>& getDialData() const override { return _splineData_; }

  /// Packs the bin centres and bin contents of a TH2 into `_splineData_` and
  /// records the range of each axis. Called by
  /// DialCollection::makeSurfaceDial().
  /// @param h2_ Histogram whose bin centres are the knots and whose bin
  ///        contents are the knot values. It is only read. (Named `h2` in
  ///        this declaration and `h2_` in the definition.)
  /// @pre The dial is empty. This is not checked: a second call appends a
  ///      second data block and a second pair of bounds, which
  ///      evalResponse() ignores.
  /// @pre The histogram has at least 2 bins on each axis.
  void buildDial(const TH2& h2);


protected:
  /// Whether evalResponse() extrapolates outside the knot range (true) or
  /// clamps its inputs to that range (false).
  bool _allowExtrapolation_{false};

  /// A block of data to calculate the spline values.  This must be filled for
  /// the Cache::Manager to work, and provides the input for spline calculation
  /// functions that can be shared between the CPU and the GPU.
  /// Holds the grid sizes, the knot positions and the knot values (see the
  /// class description).
  std::vector<double> _splineData_{};

  /// The vector of input parameter bounds.
  /// Element 0 is the range of X bin centres and element 1 the range of Y
  /// bin centres; evalResponse() clamps its inputs to them when
  /// extrapolation is not allowed.
  std::vector<GenericToolbox::Range> _splineBounds_;
};


// The guard name in the comment below is a leftover: this closes
// GUNDAM_BILINEAR_INTERPOLATION_H.
#endif //GUNDAM_COMPACTSPLINE_H
