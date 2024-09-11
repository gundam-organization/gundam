set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/EventTreeWriter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Propagator.cpp
    )

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/EventTreeWriter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Propagator.h
)

#ROOT_GENERATE_DICTIONARY(
#        GundamPropagatorDict
#        ${HEADERS}
#        LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/PropagatorLinkDef.h
#        MODULE GundamDatasetManager
#)
#
#
#list(APPEND SRCFILES GundamPropagatorDict.cxx)
#list(APPEND HEADERS  GundamPropagatorDict.h)


if( USE_STATIC_LINKS )
  add_library( GundamPropagator STATIC ${SRCFILES})
else()
  add_library( GundamPropagator SHARED ${SRCFILES})
endif()


target_include_directories( GundamPropagator
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if( WITH_CACHE_MANAGER )
  set(CACHE_MANAGER_LIBRARIES "GundamCache")
else()
  set(CACHE_MANAGER_LIBRARIES "")
endif()

target_link_libraries( GundamPropagator
    PUBLIC
    GundamDialDictionary
    ${CACHE_MANAGER_LIBRARIES}
)

#set_target_properties( GundamPropagator PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS GundamPropagator DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
#install(FILES ${HEADERS} DESTINATION include)
