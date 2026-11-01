#!/usr/bin/env python3
"""Historical friend paths preserve their results alongside grouped declarations."""
from importlib import import_module
from pathlib import Path
import sys

sys.dont_write_bytecode = True
fixture = import_module('310ModelFriendTrees')


def main():
    import GUNDAM
    work = Path.cwd()
    main_path = work / '820CompatibilityFriendTrees-main.root'
    friend_path = work / '820CompatibilityFriendTrees-friend.root'
    fixture.write_root_files(main_path, friend_path)
    GUNDAM.setRuntimeWorkingDirectory(str(work))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)
    legacy = f'                - name: friend\n                  path: "{friend_path}:friend_events"\n'
    grouped = f'''                - name: grouped
                  path: "{friend_path}"
                  treeList: [friend_events, friend_offsets]
'''
    old = fixture.check_histogram(fixture.build_config(main_path, legacy, 'friend.friendValue'), [2., 2.])
    current = fixture.check_histogram(
        fixture.build_config(main_path, grouped, 'friend_events.friendValue'), [2., 2.])
    assert old == current, (old, current)
    fixture.check_histogram(fixture.build_config(
        main_path, legacy + grouped, 'friend.friendValue + friend_offsets.friendValue'), [1., 3.])
    disabled = f'''                - name: disabled
                  path: "{work / 'missing.root'}"
                  isEnabled: false
                  treeList: [missing_tree1, missing_tree2]
'''
    fixture.check_histogram(fixture.build_config(main_path, legacy + disabled, 'friend.friendValue'), old)
    print('SUCCESS: historical friend declarations preserve bins, mixed inputs and disabled friends.')


if __name__ == '__main__':
    main()
