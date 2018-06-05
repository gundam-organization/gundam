#!/bin/sh

#We will set up to build in a subdir of the source tree
#If it was sourced as . setup.sh then you can't scrub off the end... assume that
#we are in the correct directory.
if ! echo "${BASH_SOURCE}" | grep --silent "/"; then
  SETUPDIR=$(readlink -f $PWD)
else
  SETUPDIR=$(readlink -f ${BASH_SOURCE%/*})
fi
export XSLLHFITTER=${SETUPDIR}

BUILDSETUP="${XSLLHFITTER}/build/$(uname)/setup.sh"

echo "[INFO]: XSLLHFITTER root expected at: ${XSLLHFITTER}"

source ${XSLLHFITTER}/cmake/CMakeSetup.sh

if [ ! -e ${BUILDSETUP} ]; then
  echo "[INFO]: Cannot find build setup script where expected: ${BUILDSETUP}"
else
  echo "[INFO]: Sourcing build setup script."
  source ${BUILDSETUP}
  echo "[INFO]: \$ which calcXsec: $(which calcXsec)"
fi

unset SETUPDIR
unset BUILDSETUP
