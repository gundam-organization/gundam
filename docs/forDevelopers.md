---
layout: default
title: Development policy 
next_page: "faq.html"
---

# Editing the code

- The `main` branch is the official HEAD of GUNDAM.
- Developers might make a [fork](https://github.com/gundam-organization/gundam/fork) of the `main` branch on their own GitHub account.
- Developments should happen in a dedicated branch with a descriptive name of  
  the feature you are developing. We recommend to tag your branches this way:
  - `fix/myFix`: for addressing specific issues with the code.
  - `feature/myFeature`: for adding specific feature.
  - `doc/myDoc`: for documentation additions.
  - `experimental/myBranch`: for your own implementation tests.
    Note that no experimental branch are meant to be merged with `main`.
    This means you are free to commit and implement whatever you want in those branches.
    Those are just placeholders for you to identify which `feature` should be implemented.
- Commit messages must be explicit.
- Commit content must contain a few modifications to the code.


# Logging

> Note: existing log messages are still being updated to follow these guidelines
> (see [#854](https://github.com/gundam-organization/gundam/issues/854)),
> so you may find older code that doesn't match yet.

GUNDAM prints messages through the [simple-cpp-logger](https://github.com/gundam-organization/simple-cpp-logger) macros
(`LogError`, `LogAlert`, `LogWarning`, `LogInfo`, `LogDebug`, `LogTrace`).
Pick the level based on what the message means:

| Severity | Color      | Context                                                                                                                                                              | Implementation                                       |
|----------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| ERROR    | Red        | Something IS wrong.                                                                                                                                                  | Stop runtime and hint the user what should be changed |
| ALERT    | Purple     | Something is wrong, but maybe the user knows what they're doing (e.g. using parameter limits with Eigen decomposition). User should know about it. Purple catches the eye. | Always printed out                                   |
| WARNING  | Yellow     | Important message the user might need to check during the runtime                                                                                                     |                                                      |
| INFO     | Green      | Generic info that indicates the state of the runtime                                                                                                                  |                                                      |
| DEBUG    | Blue       | Info for devs                                                                                                                                                        | Only printed on DEBUG builds                         |
| TRACE    | Light blue | Dev spamming printouts                                                                                                                                               | Never implemented on the main branch                 |


# Merging to the official repository

- First of all, create a dedicated entry on the [Issue tracking page](https://github.com/gundam-organization/gundam/issues).
- Create a pull request (PR) of the branch from your fork into `main`.
- Copy-paste the associated issue link in the comment section of the PR.
- All the CI tests must be successful before merging.


# Licence and rights

- Usage of the forked code is regulated by the code license.
- Share of the code is regulated by the code license.
