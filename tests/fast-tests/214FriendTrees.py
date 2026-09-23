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
        tree = root_file.mktree("friend_offsets", {"friendValue": "float64"})
        tree.extend({"friendValue": array("d", [0.5, 0.5, 0.0, 0.0])})


def build_config(main_path: Path, friend_list: str, expression: str) -> str:
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
{friend_list}
          variableDict:
            - name: X
              expr: "{expression}"
  propagatorConfig:
    sampleSetConfig:
      sampleList:
        - name: X
          isEnabled: true
          binning: {{ binningDefinition: [{{ name: X, edges: [0, 1, 2] }}] }}
          dataSets: [FriendTreeSample]
"""


def check_histogram(config_yaml: str, expected: list[float]) -> None:
    import GUNDAM

    config_builder = GUNDAM.ConfigUtils.ConfigBuilder()
    config_builder.setConfigFromYamlString(config_yaml)
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
    if contents != expected:
        raise AssertionError(f"Expected {expected}, got {contents}")


def main() -> int:
    import GUNDAM

    work_dir = Path.cwd()
    main_path = work_dir / "214FriendTrees-main.root"
    friend_path = work_dir / "214FriendTrees-friend.root"
    write_root_files(main_path, friend_path)

    GUNDAM.setRuntimeWorkingDirectory(str(work_dir))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)

    legacy_friend = f"""
                - name: friend
                  path: "{friend_path}:friend_events"
"""
    grouped_friends = f"""
                - name: xsec_syst_friends
                  path: "{friend_path}"
                  treeList: [friend_events, friend_offsets]
"""
    check_histogram(build_config(main_path, legacy_friend, "friend.friendValue"), [2.0, 2.0])
    check_histogram(
        build_config(main_path, grouped_friends, "friend_events.friendValue + friend_offsets.friendValue"),
        [1.0, 3.0],
    )
    check_histogram(
        build_config(main_path, legacy_friend + grouped_friends, "friend.friendValue + friend_offsets.friendValue"),
        [1.0, 3.0],
    )
    disabled_friends = f"""
                - name: disabled_friends
                  path: "{work_dir / '214FriendTrees-missing.root'}"
                  isEnabled: false
                  treeList: [missing_tree1, missing_tree2]
"""
    check_histogram(build_config(main_path, legacy_friend + disabled_friends, "friend.friendValue"), [2.0, 2.0])

    print("SUCCESS: legacy, grouped, mixed and disabled friend trees behave as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
