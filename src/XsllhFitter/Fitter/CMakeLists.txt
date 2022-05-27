set(SRCFILES
  src/ND280Fitter.cpp
  src/XsecFitter.cc
  )
set(HEADERS
  include/ND280Fitter.h
  include/XsecFitter.hh
  )
      
if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(XsllhFitter STATIC ${SRCFILES})
else()
  add_library(XsllhFitter SHARED ${SRCFILES})
endif()

target_include_directories(XsllhFitter PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries( XsllhFitter PUBLIC  XsllhFitterErrProp ${ROOT_LIBRARIES} )

#set_target_properties( XsllhFitter PROPERTIES VERSION "${GUNDAM_VERSION_STRING}" )

install(TARGETS XsllhFitter DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

