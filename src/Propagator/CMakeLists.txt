
set( LIB_NAME GundamPropagator )

set(SRCFILES
    src/Propagator.cpp
    src/ParScanner.cpp
    )

set(HEADERS
    include/Propagator.h
    include/ParScanner.h
)

#ROOT_GENERATE_DICTIONARY(
#        ${LIB_NAME}Dict
#        ${HEADERS}
#        LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/PropagatorLinkDef.h
#        MODULE GundamDatasetManager
#)
#
#
#list(APPEND SRCFILES ${LIB_NAME}Dict.cxx)
#list(APPEND HEADERS  ${LIB_NAME}Dict.h)


if( USE_STATIC_LINKS )
  add_library( ${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library( ${LIB_NAME} SHARED ${SRCFILES})
endif()


target_include_directories( ${LIB_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

if( WITH_CACHE_MANAGER )
  set(CACHE_MANAGER_LIBRARIES "GundamCache")
else()
  set(CACHE_MANAGER_LIBRARIES "")
endif()

target_link_libraries( ${LIB_NAME} PUBLIC
    GundamDatasetManager
    GundamDialDictionary
    ${CACHE_MANAGER_LIBRARIES}
)

#set_target_properties( ${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS ${LIB_NAME} DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
#install(FILES ${HEADERS} DESTINATION include)
