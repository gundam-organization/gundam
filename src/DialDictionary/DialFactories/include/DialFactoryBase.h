/// @file DialFactoryBase.h
/// @brief Declares DialFactoryBase, the interface of the factories that create
///        one event-by-event dial per event while the data are loaded.
/// @ingroup DialDictionary

#ifndef DIAL_FACTORY_BASE_H_SEEN
#define DIAL_FACTORY_BASE_H_SEEN
#include <ConfigUtils.h>
#include <DialCollection.h>

class Event;
class DialBase;

/// A mostly abstract base class for the dial factories.
///
/// A dial is a response function that turns fit-parameter values into a
/// weight for an event. Other event-by-event dials are read from the input
/// trees (one spline or graph per event). A dial factory instead *computes*
/// the dial of each event from the event's variables and a recipe it holds
/// itself (for example a lookup table filled by an external library). The
/// implementations are TabulatedDialFactory (`dialType: Tabulated`) and
/// KrigedDialFactory (`dialType: Kriged`).
///
/// A factory is stored as the collection data (DialCollection::CollectionData)
/// of the DialCollection it serves, so the collection can hand it to the data
/// loader without knowing its concrete type.
///
/// @par Ownership
/// DialCollection::initializeDialsWithTabulation() and
/// DialCollection::initializeDialsWithKriging() create the factory and append
/// it to the collection's list of collection data, which holds it through a
/// `std::shared_ptr`. Copies of the DialCollection share the same factory.
///
/// @par Life cycle
/// While the data are loaded, DataDispenser::loadEvent() gets the factory with
/// `getCollectionData<DialFactoryBase>(0)` and calls makeDial() for every
/// event that passes the collection's apply condition, once for each sample
/// the event is filled into.
///
/// @par Thread safety
/// DataDispenser::loadEvent() runs on several threads, so makeDial() can be
/// called concurrently. Both implementations serialise it with
/// GundamGlobals::getGlobalMutEx().
///
/// @see TabulatedDialFactory, KrigedDialFactory, DialCollection
/// @ingroup DialDictionary
class DialFactoryBase : public DialCollection::CollectionData {
public:
    /// Creates an empty factory. Derived classes do the set-up in their own
    /// constructors.
    DialFactoryBase() = default;

    /// Destroys the factory. Virtual, like the DialCollection::CollectionData
    /// destructor, so the owning collection can delete it through its base.
    virtual ~DialFactoryBase() = default;

    /// Create an event-by-event weighting dial for an event.
    ///
    /// @param event Event to build the dial for. The factory reads some of
    ///        its variables by name; the collection requests them for loading
    ///        with DialCollection::addExtraLeafName().
    /// @return A new dial owned by the caller (DataDispenser wraps it in a
    ///         `std::unique_ptr`), or `nullptr` when the factory gives this
    ///         event no dial, in which case the collection does not reweight
    ///         the event.
    [[nodiscard]] virtual DialBase* makeDial(const Event& event) = 0;

};
#endif
