set(SRCFILES ND280Fitter.cc)
set(HEADERS ND280Fitter.hh)

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  add_library(nd280Fit STATIC ${SRCFILES})
else(CMAKE_BUILD_TYPE MATCHES RELEASE)
  add_library(nd280Fit SHARED ${SRCFILES})
endif()

set_target_properties(
        nd280Fit PROPERTIES VERSION
        "${xsLLhFitter_VERSION_MAJOR}.${xsLLhFitter_VERSION_MINOR}.${xsLLhFitter_VERSION_REVISION}"
)

set_target_properties(
        nd280Fit PROPERTIES LINK_FLAGS
        ${ROOT_LD_FLAGS}
)

target_include_directories(
        nd280Fit PUBLIC
        ${CMAKE_SOURCE_DIR}/anaevents/src
        ${CMAKE_SOURCE_DIR}/fitparam/src
        ${CMAKE_SOURCE_DIR}/utils/src
        ${CMAKE_SOURCE_DIR}/include
)

install(TARGETS nd280Fit DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} nd280Fit PARENT_SCOPE)
