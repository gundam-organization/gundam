//
// Created by Nadrino on 06/03/2024.
//

#ifndef GUNDAM_EVENT_UTILS_H
#define GUNDAM_EVENT_UTILS_H

#include "GenericToolbox.Utils.h"

#include <iostream>
#include <string>


namespace EventUtils{

  struct Indices{

    // source
    int dataset{-1}; // which DatasetDefinition?
    long long entry{-1}; // which entry of the TChain?

    // destination
    int sample{-1}; // this information is lost in the EventDialCache manager
    int bin{-1}; // which bin of the sample?

    [[nodiscard]] std::string getSummary() const;
    friend std::ostream& operator <<( std::ostream& o, const Indices& this_ ){ o << this_.getSummary(); return o; }
  };

  struct Weights{
    double base{1};
    double current{1};

    void resetCurrentWeight(){ current = base; }
    [[nodiscard]] std::string getSummary() const;
    friend std::ostream& operator <<( std::ostream& o, const Weights& this_ ){ o << this_.getSummary(); return o; }
  };

}


#endif //GUNDAM_EVENT_UTILS_H
