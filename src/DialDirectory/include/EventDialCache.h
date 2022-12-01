//
// Created by Adrien Blanchet on 01/12/2022.
//

#ifndef GUNDAM_EVENTDIALCACHE_H
#define GUNDAM_EVENTDIALCACHE_H

#include "PhysicsEvent.h"
#include "DialInterface.h"

#include "vector"
#include "utility"

class EventDialCache {

public:
  EventDialCache() = default;

private:
  std::vector<std::pair<PhysicsEvent*, std::vector<DialInterface*>>> _cache_;

};


#endif //GUNDAM_EVENTDIALCACHE_H
