#include <stdlib.h>
#include <cstdlib>

#include <iostream>

#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"
#include "TMatrix.h"
#include "TMatrixT.h"

#include "T2KReWeight.h"
#include "T2KSyst.h"

#include "T2KGenieReWeight.h" 
#include "T2KGenieUtils.h"

#include "T2KNeutReWeight.h"
#include "T2KNeutUtils.h"

#include "T2KJNuBeamReWeight.h"

#ifdef __T2KRW_OAANALYSIS_ENABLED__
#include "ND__NRooTrackerVtx.h"
#endif

// For weight storer class
#include "T2KWeightsStorer.h"

#include "SK__h1.h"

//added
#include "T2KNIWGReWeight.h"
#include "T2KNIWGUtils.h"
