set(SRCFILES
        src/FitterEngine.cpp
        src/LikelihoodInterface.cpp
        src/MinimizerBase.cpp
        src/MinimizerInterface.cpp
        src/MCMCInterface.cpp
  )

if( USE_STATIC_LINKS )
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
