set(SRCFILES XsecFitter.cc
    #${CMAKE_CURRENT_BINARY_DIR}/XSecFitterDict.cxx #The ROOT dictionary gen function appends .cxx
    )
set(HEADERS XsecFitter.hh)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/anaevents/src
  ${CMAKE_SOURCE_DIR}/fitparam/src ${CMAKE_SOURCE_DIR}/utils/src)

#ROOT_GENERATE_DICTIONARY(${CMAKE_CURRENT_BINARY_DIR}/XSecFitterDict XsecFitter.hh
#  LINKDEF LinkDef.h)

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  add_library(xsecfit STATIC ${SRCFILES})
else(CMAKE_BUILD_TYPE MATCHES RELEASE)
  add_library(xsecfit SHARED ${SRCFILES})
endif()

set_target_properties(xsecfit PROPERTIES VERSION
  "${xsLLhFitter_VERSION_MAJOR}.${xsLLhFitter_VERSION_MINOR}.${xsLLhFitter_VERSION_REVISION}")
set_target_properties(xsecfit PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})
# target_include_directories(xsecfit PUBLIC ${CMAKE_SOURCE_DIR}/anaevents/src
#   ${CMAKE_SOURCE_DIR}/fitparam/src ${CMAKE_SOURCE_DIR}/utils/src)

install(TARGETS xsecfit DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} xsecfit PARENT_SCOPE)
