#! /bin/bash

THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
builtin cd ${THIS_SCRIPT_DIR} || exit 1;


for arg in "$@"
do
  if [ $arg == "--show-versions" ]; then
    echo "Get tag list..."
    git tag --sort=creatordate
    exit 0;
  elif [ $arg == "--show-branches" ]; then
    echo "Get branches list..."
    git fetch --all
    git branch -a --sort=committerdate
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
  elif [ $arg == "-b" ]; then
    shift
    if [[ -n $1 ]]; then
      echo "Updating to branch: $1"
      git checkout $1
      if [[ "$1" == "remotes/origin/"* ]]; then
        # in case of remotes, checkout will be pointing at a commit hash without being attached to a branch.
        # Need to checkout the real branch name now:
        git checkout "${1#"remotes/origin/"}"
      fi
      git submodule update --init --remote
    else
      echo "You should provide a version after -b"
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
    git checkout main && git submodule update --init --remote
    git pull origin main # updates repo
    exit 0;
  fi
done


echo "**********************************************************************"
echo "* Simple script that help users to checkout a given version of GUNDAM"
echo "* Usage:"
echo "* --latest: checkout the latest tagged version"
echo "* --head: checkout the main branch"
echo "* --show-branches: printout all available versions"
echo "* --show-versions: printout all available versions"
echo "* -v: checkout a given version"
echo "* -b: checkout a given branch"
echo "**********************************************************************"

#builtin cd - || exit 1;
