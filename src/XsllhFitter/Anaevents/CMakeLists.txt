# Samples

set( SRCFILES
        src/AnaSample.cc
        src/AnaTreeMC.cc
        src/AnaEvent.cpp
        )

if( USE_STATIC_LINKS )
  add_library(XsllhFitterAnaevents STATIC ${SRCFILES})
else()
  add_library(XsllhFitterAnaevents SHARED ${SRCFILES})
endif()

# Make sure the current directories are available for the later
# compilation.
target_include_directories(XsllhFitterAnaevents PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries( XsllhFitterAnaevents PUBLIC XsllhFitterUtils GundamUtils ${ROOT_LIBRARIES} )

#set_target_properties( XsllhFitterAnaevents PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS XsllhFitterAnaevents DESTINATION lib)

