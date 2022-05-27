set(SRCFILES
        src/FitObj.cc
        src/XsecCalc.cc
)

set(HEADERS
        include/FitObj.hh
        include/XsecCalc.hh
)

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(XsllhFitterErrProp STATIC ${SRCFILES})
else()
  add_library(XsllhFitterErrProp SHARED ${SRCFILES})
endif()

target_include_directories(XsllhFitterErrProp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries( XsllhFitterErrProp XsllhFitterAnaevents XsllhFitterParameters ${ROOT_LIBRARIES} )

#set_target_properties(XsllhFitterErrProp PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS XsllhFitterErrProp DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

