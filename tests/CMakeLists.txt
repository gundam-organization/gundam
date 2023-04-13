# Run the gundam-tests.sh script for the test target (i.e. make test)
#
#  This places the output of the test script in
#  ${CMAKE_BINARY_DIR}/Testing/Temporary/LastTest.log and the output
#  of the individual gundam tests in to
#  ${CMAKE_BINARY_DIR}/tests/output.<date>
#
# The COMMAND is a little complicated since the testing script needs
# to run in the source directory, but puts the output in a
# sub-directory of the binary directory.
#
add_test(NAME gundam-tests
         COMMAND "sh" "-c" "(cd ${CMAKE_SOURCE_DIR}/tests && ./gundam-tests.sh -f -a ${CMAKE_CURRENT_BINARY_DIR}/output.$(date +%Y-%m-%d-%H%M%S) )" )
