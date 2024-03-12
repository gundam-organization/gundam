//
// Created by Nadrino on 23/06/2022.
//

#ifndef GUNDAM_JOINT_PROBABILITY_BASE_H
#define GUNDAM_JOINT_PROBABILITY_BASE_H

#include "Sample.h"
#include "JsonBaseClass.h"

#include <string>

namespace JointProbability{

  class JointProbabilityBase : public JsonBaseClass {

  public:
    // simple rtti, makes the class purely virtual
    [[nodiscard]] virtual std::string getType() const = 0;

    // two choices -> either override bin by bin llh or global eval function
    [[nodiscard]] virtual double eval( const Sample &sample_, int bin_ ) const{ return 0; }

    // classic binned llh. Could be overriden to introduce correlations for instance.
    [[nodiscard]] virtual double eval( const Sample &sample_ ) const{
      double out{0};
      int nBins = int(sample_.getBinning().getBinList().size());
      for( int iBin = 0; iBin < nBins; iBin++ ){ out += this->eval(sample_, iBin); }
      return out;
    }

  };
}


#endif // GUNDAM_JOINT_PROBABILITY_BASE_H
