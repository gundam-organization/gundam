#!/bin/sh

if [ "${1}" != "force" ] && hash cmake; then
  CMAKEV=$(cmake --version | head -1 | sed "s/cmake version //")
  echo "[INFO]: cmake version ${CMAKEV} lives at: $(which cmake)"
  return 0
fi


if [ ! "$XSLLHFITTER" ]; then
  echo "[ERROR]: Please source setup.sh from the root of this package."
  return 1
fi

if ! cd ${XSLLHFITTER}; then
  echo "[ERROR]: Failed to cd to \${XSLLHFITTER}: ${XSLLHFITTER}"
  return 1
fi

cd cmake;
if [ ! -e "CMAKE" ];then
  echo "[INFO]: Checking out ND280 CMake..."
  if [ ! ${CVSROOT} ]; then
    echo "[ERROR]: Is CVS set up correctly? \${CVSROOT}: ${CVSROOT}"
    return 1
  fi
  if ! cvs co CMAKE; then
    echo "[ERROR]: Failed. Is CVS set up correctly? \${CVSROOT}: ${CVSROOT}"
    return 1
  fi
fi

cd CMAKE;
CMAKEROOT=$(readlink -f .)
CMAKERELEASE=$(ls cmake-*.tar.gz | sed "s/^cmake-\(.*\)\.tar\.gz$/\1/")

if [ ! -e "$(uname)/cmake-${CMAKERELEASE}" ]; then
  mkdir -p $(uname)
  cd $(uname)
  tar -xzf ${CMAKEROOT}/cmake-${CMAKERELEASE}.tar.gz
  cd cmake-${CMAKERELEASE}
else
  cd $(uname)/cmake-${CMAKERELEASE}
fi

if [ ! -e "bin/cmake" ]; then
  ./bootstrap
  if ! make; then
    unset CMAKEROOT
    unset CMAKERELEASE
    return 1
  fi
fi

CMAKEBIN=$(readlink -f bin)

if ! [[ ":$PATH:" == *":${CMAKEBIN}:"* ]]; then
  export PATH=${CMAKEBIN}:$PATH
fi

CMAKEV=$(cmake --version | head -1 | sed "s/cmake version //")
echo "[INFO]: cmake version ${CMAKEV} lives at: $(which cmake)"

unset CMAKEV
unset CMAKEROOT
unset CMAKERELEASE
unset CMAKEBIN

return 0
