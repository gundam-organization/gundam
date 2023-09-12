#! /bin/bash

PROJECT_NAME="GUNDAM"

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
  elif [ $arg == "--fix-submodules" ]; then
    echo "Re-initializing submodules..."
    git submodule deinit --all -f
    git submodule sync
    git submodule update --init --remote --recursive
    exit 0;
  elif [ $arg == "-v" ]; then
    shift
    if [[ -n $1 ]]; then
      echo "Updating to version: $1"
      git checkout $1
      git submodule sync
       git submodule update --init --remote --recursive
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
      git submodule sync
       git submodule update --init --remote --recursive
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
    git submodule sync
     git submodule update --init --remote --recursive
    exit 0;
  elif [ $arg == "--head" ]; then
    echo "Checking out main branch..."
    git checkout main
    git pull origin main # updates repo
    git submodule sync
     git submodule update --init --remote --recursive
    exit 0;
  elif [ $arg == "--up" ]; then
    echo "Updating..."
    git pull
    git submodule sync
    git submodule update --init --remote --recursive
    exit 0;
  fi
done


echo "**********************************************************************"
echo "* Simple script that help users to checkout a given version of $PROJECT_NAME"
echo "* Usage:"
echo "* --up: Update with remote (staying on your current branch)"
echo "* --latest: checkout the latest tagged version"
echo "* --head: checkout the main branch"
echo "* --fix-submodules: Reinitialize submodules"
echo "* --show-branches: printout all available versions"
echo "* --show-versions: printout all available versions"
echo "* -v: checkout a given version"
echo "* -b: checkout a given branch"
echo "**********************************************************************"

#builtin cd - || exit 1;
