#! /bin/bash

THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
builtin cd ${THIS_SCRIPT_DIR} || exit 1;


for arg in "$@"
do
  if [ $arg == "--show-versions" ]; then
    echo "Get tag list..."
    git tag --sort=creatordate
    exit 0;
  elif [ $arg == "-v" ]; then
    shift
    if [[ -n $1 ]]; then
      echo "Updating to version: $1"
      git checkout $1 && git submodule update --init --remote
    else
      echo "You should provide a version after -v"
    fi
    exit 0;
  elif [ $arg == "--latest" ]; then
    echo "Updating repo.."
    git pull origin main # updates repo
    LATEST_VERSION=$(git describe --tags `git rev-list --tags --max-count=1`)
    echo "Checking out latest version: $LATEST_VERSION"
    git checkout $LATEST_VERSION
    git submodule update --init --remote
    exit 0;
  elif [ $arg == "--head" ]; then
    echo "Checking out main branch..."
    git pull origin main # updates repo
    git checkout main && git submodule update --init --remote
    exit 0;
  fi
done


echo "Simple script that help users to checkout a given version of GUNDAM"
echo "Usage:"
echo "--latest: checkout the latest tagged version"
echo "--head: checkout the main branch"
echo "--show-versions: printout all available versions"
echo "-v: checkout a given version"


#builtin cd - || exit 1;
