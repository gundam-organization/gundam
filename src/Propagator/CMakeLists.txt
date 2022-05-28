set(SRCFILES
        src/Propagator.cpp
)

set(HEADERS
        include/Propagator.h
)

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library( GundamPropagator STATIC ${SRCFILES})
else()
  add_library( GundamPropagator SHARED ${SRCFILES})
endif()

target_include_directories( GundamPropagator PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

if( WITH_CACHE_MANAGER )
  target_link_libraries(  GundamPropagator
          GundamFitParameters
          GundamFitSamples
          GundamDatasetManager
          GundamCache
          ${ROOT_LIBRARIES}
          )
else()
  target_link_libraries(  GundamPropagator
          GundamFitParameters
          GundamFitSamples
          GundamDatasetManager
          ${ROOT_LIBRARIES}
          )
endif()

#set_target_properties( GundamPropagator PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS  GundamPropagator DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

