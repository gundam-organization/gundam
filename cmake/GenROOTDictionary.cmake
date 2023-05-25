#dummy parameter is for compatibility with the ROOT exposed macro.
function(ROOT_GENERATE_DICTIONARY OutputDictName Header LINKDEFDUMMY LinkDef)

  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  string(REPLACE ";" ";-I" LISTDIRINCLUDES "-I${incdirs}")
  string(REPLACE " " ";" LISTCPPFLAGS "${CMAKE_CXX_FLAGS}")

  #ROOT5 CINT cannot handle it.
  list(REMOVE_ITEM LISTCPPFLAGS "-std=c++11")

  cmessage( STATUS "ROOTDICTGEN -- LISTCPPFLAGS: ${LISTCPPFLAGS}")
  cmessage( STATUS "ROOTDICTGEN -- LISTINCLUDES: ${LISTDIRINCLUDES}")
  #Learn how to generate the Dict.cxx and Dict.hxx
  add_custom_command(
    OUTPUT "${OutputDictName}.cxx" "${OutputDictName}.h"
    COMMAND rootcint
    ARGS -f ${OutputDictName}.cxx -c
    -p ${LISTDIRINCLUDES} ${LISTCPPFLAGS} ${Header} ${LinkDef}
    )
endfunction()

cmessage( STATUS "Added ROOT_GENERATE_DICTIONARY CMake macro.")
