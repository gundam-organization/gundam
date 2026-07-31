#!/usr/bin/env python3

import sys
from array import array
from pathlib import Path


def write_root_files(main_path: Path, friend_path: Path) -> None:
    import uproot

    with uproot.recreate(main_path) as root_file:
        tree = root_file.mktree("events", {"event": "int32"})
        tree.extend({"event": array("i", [0, 1, 2, 3])})

    with uproot.recreate(friend_path) as root_file:
        tree = root_file.mktree("friend_events", {"friendValue": "float64"})
        tree.extend({"friendValue": array("d", [0.25, 0.75, 1.25, 1.75])})


def build_config(main_path: Path, friend_path: Path) -> str:
    return f"""
fitterEngineConfig:
  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: PoissonLLH
    dataSetList:
      - name: FriendTreeSample
        isEnabled: true
        model:
          filePathList:
            - name: main
              path: "{main_path}:events"
              friendList:
                - name: friend
                  path: "{friend_path}:friend_events"
          variableDict:
            - name: X
              expr: "friend.friendValue"
  propagatorConfig:
    sampleSetConfig:
      sampleList:
        - name: X
          isEnabled: true
          binning: {{ binningDefinition: [{{ name: X, edges: [0, 1, 2] }}] }}
          dataSets: [FriendTreeSample]
"""


def main() -> int:
    import GUNDAM

    work_dir = Path.cwd()
    main_path = work_dir / "214FriendTrees-main.root"
    friend_path = work_dir / "214FriendTrees-friend.root"
    write_root_files(main_path, friend_path)

    GUNDAM.setRuntimeWorkingDirectory(str(work_dir))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)

    config_builder = GUNDAM.ConfigUtils.ConfigBuilder()
    config_builder.setConfigFromYamlString(build_config(main_path, friend_path))
    config = GUNDAM.ConfigUtils.ConfigReader(config_builder.getConfig())
    config.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition("fitterEngineConfig"))

    engine = GUNDAM.FitterEngine()
    engine.setConfig(config.fetchValueConfigReader("fitterEngineConfig"))
    engine.configure()
    likelihood = engine.getLikelihoodInterface()
    likelihood.initialize()
    likelihood.propagateAndEvalLikelihood()

    sample = likelihood.getModelPropagator().getSampleSet().getSampleList()[0]
    contents = [bin_content.sumWeights for bin_content in sample.getHistogram().getBinContentList()]
    if contents != [2.0, 2.0]:
        print(f"FAIL: expected [2.0, 2.0] from friend.friendValue, got {contents}")
        return 1

    print("SUCCESS: friend-tree branch is available through its configured alias.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
