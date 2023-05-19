# Validation scripts that run during continuous integration

This directory contains scripts that will be run as part of the
gundam-tests.sh fast tests.  These tests are run as part of continuous
integration, and should test critical features that block a commit on
failure.  Each script should finish in a "few" seconds, and all of the
tests should finish in well under half a minute.  See the
gundam-tests.sh script for documentation on the file naming
convention.

This file should exist so that git will create the fast-tests
subdirectory
