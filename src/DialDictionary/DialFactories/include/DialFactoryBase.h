#ifndef DIAL_FACTORY_BASE_H_SEEN
#define DIAL_FACTORY_BASE_H_SEEN
#include <ConfigUtils.h>
#include <DialCollection.h>

class Event;
class DialBase;

/// A mostly abstract base class for the dial factories.
class DialFactoryBase : public DialCollection::CollectionData {
public:
    DialFactoryBase() = default;
    virtual ~DialFactoryBase() = default;

    /// Create an event-by-event weighting dial for an event.
    [[nodiscard]] virtual DialBase* makeDial(const Event& event) = 0;

};
#endif
