set(SRCFILES
  src/FitterEngine.cpp
  )
set(HEADERS
  include/FitterEngine.h
  )
      
if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(GundamFitter STATIC ${SRCFILES})
else()
  add_library(GundamFitter SHARED ${SRCFILES})
endif()

target_include_directories(GundamFitter PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries( GundamFitter PUBLIC
        GundamPropagator
        ${ROOT_LIBRARIES}
        )

#set_target_properties( GundamFitter PROPERTIES VERSION "${GUNDAM_VERSION_STRING}" )

install(TARGETS GundamFitter DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

