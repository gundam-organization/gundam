# Validations for that should be run before a push or pull-request

This directory contains scripts that will be run as part of the
gundam-tests.sh regular testings.  These tests are not run as part of
continuous integration, but should be run locally before making a
push/pull-request.  These tests are checking critical features and the
fit precision.  They could be run during continuous integration, but
are placed here to limit the total CI time.

Each script should finish in a "few" seconds, and all of the scripts
should finish in a couple minutes.  See the gundam-tests.sh script for
documentation on the file naming convention.

This file should exist so that git will create the regular-tests
subdirectory
