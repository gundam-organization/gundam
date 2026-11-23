//
// Created by Adrien BLANCHET on 13/04/2022.
//

/// @file DialBase.h
/// @brief Declares DialBase, the abstract interface shared by every dial type.
/// @ingroup DialDictionary

#ifndef GUNDAM_DIALBASE_H
#define GUNDAM_DIALBASE_H

#include "DialInputBuffer.h"

#include <vector>
#include <string>
#include <memory>

// should be thread safe -> add lock?
// any number of inputs (provided doubles) -> set input size
// fast -> no checks while eval

// as light as possible: minimize members
// keep in mind: class size is n*8 bytes (64 bits chunks)
// virtual layer adds 8 bytes for vftable pointer (virtual function table)
// adds members memory ->
// https://stackoverflow.com/questions/9439240/sizeof-class-with-int-function-virtual-function-in-c

/// Abstract base class of every dial: a response function that turns the
/// current values of one or more fit parameters into a weight factor.
///
/// Each subclass implements one shape: a normalisation (Norm), a constant
/// (Shift), a piecewise-linear graph (Graph, RootGraph), cubic splines
/// (CompactSpline, MonotonicSpline, UniformSpline, GeneralSpline,
/// RootSpline), 2D surfaces (Bilinear, Bicubic), formulas (RootFormula,
/// Polynomial, CompiledLibDial) and table lookups (Tabulated, Kriged).
///
/// DialCollection creates the dials (DialCollection::makeDial() and its
/// helpers; TabulatedDialFactory and KrigedDialFactory for table dials) and
/// wraps each one in a DialInterface. The DialInterface pairs the dial with a
/// DialInputBuffer (the parameter values to read) and a
/// DialResponseSupervisor (which clamps the response). EventDialCache then
/// multiplies the clamped responses of all dials attached to an event into
/// the event weight.
///
/// Event-by-event collections can hold one dial per event, so subclasses
/// keep as few data members as possible (see the design notes above the
/// class).
///
/// @par Ownership
/// A DialInterface owns its dial through a DialBaseObject, which holds a
/// std::shared_ptr and copies the dial with clone(). Identical binned dials
/// may share one object (DialInterface::shareDial()), so a dial must not be
/// modified once its collection is initialized.
///
/// @par Thread safety
/// evalResponse() is const. During Propagator::reweightEvents() several
/// threads may evaluate the same dial at once (a binned or normalisation
/// dial is shared by many events), so an implementation must not change the
/// dial's state while evaluating.
///
/// @note The section banner inside the class ("Pass information to the dial
///       so that it can build it's internal information...") does not match
///       the current design: DialBase declares no build methods. Each
///       subclass has its own `buildDial()` or setter methods.
///
/// @see DialInterface, DialCollection, DialInputBuffer, EventDialCache
/// @ingroup DialDictionary
class DialBase {
public:
  /// Creates an empty dial. Subclasses fill their shape through their own
  /// setters or `buildDial()` methods.
  DialBase() = default;

  /// Destroys the dial. Virtual so that a dial can be deleted through a
  /// DialBase pointer.
  virtual ~DialBase() = default;

  // virtual layer + 8 bytes

  /// Construct a copy of this dial.  Needed for PolymorphicObject.
  ///
  /// DialBaseObject calls it when a handle, and so a DialInterface, is
  /// copied; CompositeDial::bind() also calls it to keep its own copy of each
  /// node dial. Every subclass implements it with its own copy constructor.
  /// @return A new dial of the same concrete type, holding a copy of this
  ///         dial's data.
  [[nodiscard]] virtual std::unique_ptr<DialBase> clone() const = 0;

  /// Return the name of the dial type (simple local RTTI).
  ///
  /// Used in summaries and error messages (DialInterface::getSummary()), and
  /// by Cache::Manager::Build(), which compares the start of the name with
  /// the GPU-supported types to count the dials of each type.
  /// @return The type name, usually the class name (e.g. `"Graph"`).
  [[nodiscard]] virtual std::string getDialTypeName() const = 0;

  /// Evaluate the dial response at the set of parameter values in DialInputBuffer.
  ///
  /// The response is the factor applied to the weight of every event (or
  /// bin) the dial is attached to: 1 leaves the weight unchanged.
  /// DialInterface::evalResponse() passes it through the collection's
  /// DialResponseSupervisor (clamping to `minDialResponse` and
  /// `maxDialResponse`), and EventDialCache::reweightEntry() multiplies the
  /// responses of all dials of an event.
  ///
  /// During reweighting, EventDialCache only calls this when the input
  /// buffer's revision changed (DialInputBuffer::getRevision()); otherwise
  /// it reuses the cached response.
  ///
  /// @param input_ Current input values, one per parameter the dial reads,
  ///        in the order the dial expects (`input_.getInputBuffer()[i]`).
  ///        They are the parameter values after mirroring, if the collection
  ///        enables it. Some dials (Shift, Tabulated, Kriged) ignore them.
  /// @return The weight factor for these inputs.
  /// @pre The buffer holds at least as many values as the dial reads.
  ///      Implementations do not check this, to stay fast.
  [[nodiscard]] virtual double evalResponse(const DialInputBuffer& input_) const = 0;

  /// Allow extrapolation of the data. The default should be to forbid extrapolation.
  ///
  /// Only matters for dials defined on a finite input range (graphs,
  /// splines, surfaces, Polynomial). DialCollection::makeDial(const TObject*)
  /// calls it with the collection's `allowDialExtrapolation` option. This
  /// default implementation does nothing.
  /// @param allow_ True to evaluate the shape outside its defined range;
  ///        false to keep the response at its end-of-range value. What
  ///        "outside" means depends on the subclass.
  virtual void setAllowExtrapolation(bool allow_){}

  /// Tells whether this dial evaluates its shape outside its defined range.
  /// @return The flag set by setAllowExtrapolation(). This default
  ///         implementation always returns false.
  /// @note DialCollection compares it when it looks for identical dials to
  ///       share (DialCollection::deduplicateBinnedDials(),
  ///       DialCollection::storeEventDial()).
  [[nodiscard]] virtual bool getAllowExtrapolation() const {return false;}

  /// Return the retained knot positions in dial-input coordinates.
  /// Dials without knots return an empty vector.
  ///
  /// Overridden by the graph and spline dials. Used by the Python interface
  /// through DialInterface::getSourceKnotPositions().
  /// @return The knot positions; empty here.
  [[nodiscard]] virtual std::vector<double> getSourceKnotPositions() const { return {}; }

  /// Dial summary describing its content
  ///
  /// DialInterface::getSummary() appends it when called with
  /// `shallow_ = false`.
  /// @return A human-readable description of the dial's data. Empty in this
  ///         default implementation.
  [[nodiscard]] virtual std::string getSummary() const { return {}; };

  /////////////////////////////////////////////////////////////////////////
  // Pass information to the dial so that it can build it's internal
  // information.  New overloads should be added as we have classes of dials
  // (e.g. multi-dimensional dials).
  /////////////////////////////////////////////////////////////////////////

  /// Return the data used by the dial to calculate the output values. The
  /// specific data contained in the vector depends on the derived class.
  ///
  /// Cache::Manager reads it to copy graph, spline and surface dials to the
  /// GPU (Cache::Manager::Build() and Cache::Manager::Update()). Subclasses
  /// with such a GPU counterpart override it.
  /// @return Reference to the packed data. It stays valid while the dial
  ///         exists and is not rebuilt.
  /// @warning This default implementation logs an error. In builds with
  ///          NDEBUG defined it then throws std::runtime_error; otherwise it
  ///          returns a reference to an empty vector.
  [[nodiscard]] virtual const std::vector<double>& getDialData() const;

};

#endif //GUNDAM_DIALBASE_H
// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
