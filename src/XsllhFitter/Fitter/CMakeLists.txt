set(SRCFILES
  src/ND280Fitter.cpp
  src/XsecFitter.cc
  )
set(HEADERS
  include/ND280Fitter.h
  include/XsecFitter.hh
  )
      
if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(xsllhFitter STATIC ${SRCFILES})
else()
  add_library(xsllhFitter SHARED ${SRCFILES})
endif()

target_include_directories(xsllhFitter PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries( xsllhFitter PUBLIC
        GundamErrorPropagator
        ${ROOT_LIBRARIES}
        )

#set_target_properties( xsllhFitter PROPERTIES VERSION "${GUNDAM_VERSION_STRING}" )

install(TARGETS xsllhFitter DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

