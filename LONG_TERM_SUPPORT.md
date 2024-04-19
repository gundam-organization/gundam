# Long Term Support Procedure for LTS/1.8.x

# Creating a new patch release.

To create a new patch release.

1. Make sure that you are at the head of the LTS/1.8.x branch.

1. Edit ChangeLog to explain all of the changes.
 
1. Compile from scratch, and check for errors.

1. Run the full tests, and make sure they pass.

1. Commit the final ChangeLog.  The ChangeLog should note what has
changed, and which issues are addressed.

1. Push the branch to github.

1. Create a draft release.  The draft release message should include a
copy of the changes in the ChangeLog.

1. Get concensus between GUNDAM coordinators that the patch is ready

1. Make the tag.  The tag should be named `1.8.<n+1>` where `n` is the
number of the last patch.

# Fixing an Issue

The procedure for merging work into the long term support branch is:

1. Make sure that an issue has been created, and has been labeled with
LTS/1.8.  If necessary, edit the issue title to include the version
where the issue is first seen.  This is usually the latest long term
support patch.  For example, the title might be "GUNDAM 1.8.0: Catmull
Rom extrapolation is not linear".

1. Create a branch to propose a solution.  The branch should be named
LTS/issue<N>/<topic>, and should be rooted on LTS/1.8.x.  For example
`LTS/issue486/SplineExtrapolation`

1. Describe how to cause the problem in the issue report.  This should
include instructions on how to reproduce the error.  If possible, add
a test that shows the issue, and which should fail. Be sure to add the
script that will fail to
[./tests/EXPECTED_FAILURES](./tests/EXPECTED_FAILURES). If the problem
is to complex test in a script, make sure the instructions in the
issue are complete.

1. Discuss fixing the issue with the GUNDAM group, and make sure it
should be fixed.

1. Commit the failing test and push the issue branch to github

1. Create a draft pull request.

1. Fix the issue.  Try to make all of your commits "atomic" and check
that the code compiles and runs after each commit.  Push your commits
to the issue branch frequently.  Make sure that your commit messages
are very clear, and are sufficient to be used as a ChangeLog. Most
fixes may only require one commit.

1. When the issue is fixed, and the test script is working correctly,
remove the test script from EXPECTED_FAILURES.

1. Discuss the proposed fix with the GUNDAM group.  When there is
concensus, note that in the Pull Request discussion.

1. Update the issue to note that it will be closed by the pull request.

1. A GUNDAM coordinator (preferably, not the same person that has
fixed the issue) will merge the branch.

1. Test that the HEAD of LTS/1.8.x passes the tests, and that the
issue is fixed.

1. Close the issue.
