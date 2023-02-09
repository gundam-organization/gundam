# Validations for that take a little extra time.

This directory contains scripts that will be run as part of the
gundam-tests.sh extended testing. These tests check things that take a
bit of extra time (for example, simple MCMC checks), and should be
tested frequently, but for which there are not enough resources to run
as part of regular pushes.

Each script should finish in a well under a minute, and all of the
scripts should finish in few minutes.  See the gundam-tests.sh script
for documentation on the file naming convention.

This file should exist so that git will create the regular-tests
subdirectory
