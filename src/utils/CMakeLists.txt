set(SRCFILES
        src/BinManager.cc
        src/CalcChisq.cc
        src/EigenDecomp.cc
        src/OptParser.cc
        src/PlotStyle.cc
        src/ToyThrower.cc
        src/XsecDial.cc
        src/GlobalVariables.cc
        src/GeneralizedFitBin.cc
        src/DataBinSet.cpp
        src/DataBin.cpp
        src/JsonUtils.cpp
        src/YamlUtils.cpp
        )
set(HEADERS
        include/BinManager.hh
        include/CalcChisq.hh
        include/EigenDecomp.hh
        include/OptParser.hh
        include/PlotStyle.hh
        include/ToyThrower.hh
        include/XsecDial.hh
        include/GlobalVariables.h
        include/GeneralizedFitBin.h
        include/DataBin.h
        include/DataBinSet.h
        include/JsonUtils.h
        include/YamlUtils.h
        include/YamlUtils.impl.h
        )

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  add_library(xslfutils STATIC ${SRCFILES})
else(CMAKE_BUILD_TYPE MATCHES RELEASE)
  add_library(xslfutils SHARED ${SRCFILES})
endif()

set_target_properties(xslfutils PROPERTIES VERSION
  "${xsLLhFitter_VERSION_MAJOR}.${xsLLhFitter_VERSION_MINOR}.${xsLLhFitter_VERSION_REVISION}")
set_target_properties(xslfutils PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

install(TARGETS xslfutils DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} xslfutils PARENT_SCOPE)
