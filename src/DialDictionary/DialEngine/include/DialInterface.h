//
// Created by Adrien Blanchet on 29/11/2022.
//

/// @file DialInterface.h
/// @brief Declares DialInterface, which ties one dial to its input buffer and
///        response supervisor, and the DialBaseObject handle that owns a dial.
/// @ingroup DialDictionary

#ifndef GUNDAM_DIALINTERFACE_H
#define GUNDAM_DIALINTERFACE_H

#include "DialBase.h"
#include "DialInputBuffer.h"
#include "DialResponseSupervisor.h"

#include "SampleSet.h"
#include "ParameterSet.h"

#include <string>
#include <memory>
#include <vector>
#include <utility>

/// Copies retain the historical deep-copy semantics. Sharing is explicit in
/// DialInterface::shareDial, for immutable dial definitions only.
///
/// Owning handle to a dial of any concrete DialBase type. The dial is held in
/// a `std::shared_ptr<DialBase>` (public member `dialPtr`). Copying the
/// handle deep-copies the dial with DialBase::clone(), and moving it
/// transfers the dial. Two handles point to the same dial only after
/// DialInterface::shareDial(), which DialCollection uses for identical
/// binned dials, or when DialCollection sets `dialPtr` directly.
/// @ingroup DialDictionary
class DialBaseObject {
public:
  /// Creates an empty handle.
  DialBaseObject() = default;
  /// Takes ownership of a dial.
  /// @param dial_ The dial; may be null, giving an empty handle.
  explicit DialBaseObject(std::unique_ptr<DialBase> dial_): dialPtr(std::move(dial_)) {}
  /// Holds a copy of a dial.
  /// @param dial_ Dial copied with DialBase::clone().
  explicit DialBaseObject(const DialBase& dial_): dialPtr(dial_.clone()) {}
  /// Deep-copies the other handle's dial.
  /// @param other_ Handle to copy; if it is empty, this one is empty too.
  DialBaseObject(const DialBaseObject& other_){
    if( other_.dialPtr ){ dialPtr = other_.dialPtr->clone(); }
  }
  /// Replaces the dial with a deep copy of the other handle's dial.
  /// @param other_ Handle to copy; if it is empty, this one becomes empty.
  /// @return This handle.
  DialBaseObject& operator=(const DialBaseObject& other_){
    if( this != &other_ ){
      dialPtr = other_.dialPtr ? std::shared_ptr<DialBase>(other_.dialPtr->clone()) : nullptr;
    }
    return *this;
  }
  /// Takes the dial of another handle, which is left empty.
  DialBaseObject(DialBaseObject&&) noexcept = default;
  /// Takes the dial of another handle, which is left empty.
  /// @return This handle.
  DialBaseObject& operator=(DialBaseObject&&) noexcept = default;

  /// Gives the dial.
  /// @return A reference to the dial.
  /// @pre The handle is not empty.
  DialBase& operator*() const { return *dialPtr; }
  /// Gives access to the dial's members.
  /// @return The dial pointer; null for an empty handle.
  DialBase* operator->() const { return dialPtr.get(); }
  /// Gives the dial pointer.
  /// @return The dial pointer; null for an empty handle.
  [[nodiscard]] DialBase* get() const { return dialPtr.get(); }

  /// The dial. Shared between handles only through
  /// DialInterface::shareDial() or a direct assignment.
  std::shared_ptr<DialBase> dialPtr;
};


/// Ties one dial (a response function of one or more fit parameters) to the
/// parameter values it reads and to the clamp applied to its result.
///
/// DialInterface holds the dial itself (a DialBaseObject) and non-owning
/// pointers to the DialInputBuffer that supplies the dial's parameter values,
/// to the DialResponseSupervisor that clamps its response, and, for binned
/// dials, to the Bin the dial applies to. evalResponse() combines them:
/// `supervisor.process(dial.evalResponse(inputBuffer))`.
///
/// @par Ownership
/// DialCollection owns its DialInterface objects by value
/// (DialCollection::getDialInterfaceList()); how many it has depends on the
/// dial definition, for example one per dial bin, or one per event for
/// event-by-event dials (events whose dials are identical can share one, see
/// DialCollection::storeEventDial()). The input buffer, supervisor and bin
/// belong to the
/// same DialCollection, which sets the pointers in
/// DialCollection::setupDialInterfaceReferences(). EventDialCache keeps raw
/// pointers to DialInterface objects.
///
/// @par Thread safety
/// evalResponse() is const and is called from several threads during
/// reweighting, often on the same object when several events share a dial.
///
/// This class is size critical and should not be used as a base class (no
/// virtual methods.
/// @see DialCollection, DialInputBuffer, DialResponseSupervisor, EventDialCache
/// @ingroup DialDictionary
class DialInterface {

public:
  /// Creates an interface with no dial and null pointers. DialCollection
  /// fills it with setDial() and DialCollection::setupDialInterfaceReferences().
  DialInterface() = default;

  /// Stores a copy of a dial.
  /// @param dial_ Handle holding the dial. The dial is deep-copied with
  ///        DialBase::clone(), so the caller keeps its own. An empty handle
  ///        leaves this interface without a dial.
  void setDial(const DialBaseObject& dial_){ _dial_ = dial_; }
  /// Takes a dial handle without copying the dial.
  /// @param dial_ Handle to move from; it is left empty.
  void setDial(DialBaseObject&& dial_){ _dial_ = std::move(dial_); }
  /// Share the immutable curve; bin, inputs and response supervisor stay local.
  ///
  /// Makes this interface use the same dial object as `other_`.
  /// DialCollection::deduplicateBinnedDials() calls it for bins whose dials
  /// have identical content.
  /// @param other_ Interface whose dial is shared. Neither dial may be
  ///        modified afterwards.
  void shareDial(const DialInterface& other_){ _dial_.dialPtr = other_._dial_.dialPtr; }
  /// Sets the buffer the dial reads its parameter values from.
  /// @param inputBufferRef Non-owning pointer. The buffer must not move while
  ///        this interface is used.
  void setInputBufferRef(const DialInputBuffer *inputBufferRef){ _inputBufferPtr_ = inputBufferRef; }
  /// Sets the clamp applied to the dial response.
  /// @param responseSupervisorRef Non-owning pointer. Must be set before
  ///        evalResponse() is called.
  void setResponseSupervisorRef(const DialResponseSupervisor *responseSupervisorRef){ _responseSupervisorPtr_ = responseSupervisorRef; }
  /// Sets the bin this dial applies to; only used for printouts.
  /// @param dialBinRef Non-owning pointer into the collection's dial
  ///        binning. Left null when the collection has no dial binning.
  void setDialBinRef(const Bin *dialBinRef){ _dialBinRef_ = dialBinRef; }

  /// Gives mutable access to the dial handle.
  ///
  /// DataDispenser and DialCollection::storeEventDial() use it to install
  /// event-by-event dials: they assign a `std::unique_ptr` to
  /// `getDial().dialPtr` directly, instead of calling setDial(), which would
  /// clone the dial.
  /// @return The handle that owns the dial.
  auto& getDial(){ return _dial_; }
  /// Gives read-only access to the dial handle.
  /// @return The handle that owns the dial.
  const auto& getDial() const{ return _dial_; }

  /// Return the input buffer containing the connection to the Parameter(s)
  /// used by this dial.  The number of Parameters contained in the input
  /// buffer musts mach the number expected by the specialization of the
  /// DialBase.
  /// @return Non-owning pointer to the buffer; null until set.
  [[nodiscard]] const DialInputBuffer *getInputBufferRef() const {return _inputBufferPtr_;}

  /// Hacky way of tweaking input buffer manually. Used for python interface
  ///
  /// Casts away the constness of the stored pointer.
  /// @return The input buffer, writable.
  /// @pre The input buffer is set.
  DialInputBuffer& getInputBuffer() {return *((DialInputBuffer *) _inputBufferPtr_);}

  /// Get the dial calculation method.  The dial will need one or more
  /// Parameter inputs, and the number *must* match the number and order of
  /// the Parameters in the DialInputBuffer.
  /// @return Non-owning pointer to the dial; null if no dial was set.
  [[nodiscard]] DialBase* getDialBaseRef() const {return _dial_.get();}

  /// Get the DialResponseSupervisor. This conditions the return value of the
  /// dial (normally truncates between zero and the maximum response).
  /// @return Non-owning pointer to the supervisor; null until set.
  /// @note DialCollection sets the minimum to 0 unless `minDialResponse` is
  ///       configured (no minimum for composite nodes and variable dials),
  ///       and sets no maximum unless `maxDialResponse` is configured.
  [[nodiscard]] const DialResponseSupervisor* getResponseSupervisorRef() const {return _responseSupervisorPtr_;}

  /// Get the data bin definition for the dial.
  /// @return Non-owning pointer to the bin; null for dials without binning.
  /// @note EventDialCache::buildReferenceCache() copies it into the
  ///       event-specific interfaces of composite and event-formula dials.
  [[nodiscard]] const Bin* getDialBinRef() const {return _dialBinRef_;}

  /// Gives the knot positions of the dial (DialBase::getSourceKnotPositions()).
  /// Exposed to the Python interface.
  /// @return The knot positions; empty if there is no dial or it has no knots.
  [[nodiscard]] std::vector<double> getSourceKnotPositions() const {
    if( not _dial_.get() ){ return {}; }
    return _dial_->getSourceKnotPositions();
  }

  /// Evaluates the dial at the cached input values and clamps the result.
  ///
  /// Called by EventDialCache::DialResponseCache::update() during
  /// reweighting, and by getSummary().
  /// @return `supervisor.process(dial.evalResponse(inputBuffer))`.
  /// @pre The dial, the input buffer and the supervisor are set; none of
  ///      them is checked. The input buffer must be up to date
  ///      (DialInputBuffer::update() after the last parameter change).

  [[nodiscard]] double evalResponse() const { return _responseSupervisorPtr_->process(_dial_->evalResponse(*_inputBufferPtr_)); }
  /// Describes the dial for debug printouts.
  /// @param shallow_ If false, a new line and the dial's own
  ///        DialBase::getSummary() are appended. Also passed on to the input
  ///        buffer, bin and supervisor summaries.
  /// @return `TYPE: input(...) applyOn(...) supervisor(...) response=VALUE`,
  ///         where the parts for null pointers are left out. The response
  ///         is left out too when it cannot be evaluated: no input buffer or
  ///         supervisor, a CompositeDial not bound yet, or a RootFormula
  ///         whose bound inputs do not match the buffer size.
  /// @pre The dial is set: its type name is always read.
  [[nodiscard]] std::string getSummary(bool shallow_=true) const;

private:
  // owner of
  /// The dial (response function), owned by this interface.
  DialBaseObject _dial_{};

  // non owner of
  /// Buffer of parameter values the dial reads; owned by the DialCollection.
  const DialInputBuffer* _inputBufferPtr_{nullptr};
  /// Clamp applied to the dial response; owned by the DialCollection.
  const DialResponseSupervisor* _responseSupervisorPtr_{nullptr};
  /// Bin the dial applies to, owned by the DialCollection's dial binning;
  /// null when the collection has no dial binning.
  const Bin* _dialBinRef_{nullptr}; // for printout

};


#endif //GUNDAM_DIALINTERFACE_H
