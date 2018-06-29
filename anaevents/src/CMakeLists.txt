set(SRCFILES AnaSample.cc AnyTreeMC.cc)
set(HEADERS AnaSample.hh AnyTreeMC.hh)

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  add_library(anaevents STATIC ${SRCFILES})
else(CMAKE_BUILD_TYPE MATCHES RELEASE)
  add_library(anaevents SHARED ${SRCFILES})
endif()

set_target_properties(anaevents PROPERTIES VERSION
  "${xsLLhFitter_VERSION_MAJOR}.${xsLLhFitter_VERSION_MINOR}.${xsLLhFitter_VERSION_REVISION}")
set_target_properties(anaevents PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})
target_include_directories(anaevents PUBLIC ${CMAKE_SOURCE_DIR}/include)

install(TARGETS anaevents DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} anaevents PARENT_SCOPE)
