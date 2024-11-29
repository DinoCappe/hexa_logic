# Change Log

All notable changes to the "hivemind" project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Nothing new.

## [v1.2.0] - 2024/11/29

- Add Minmax with alpha-beta pruning agent.
- Minor internal code changes, including minor improvements to Board and its move cache.
- Added 3 new engine options:
  * `StrategyWhite`  
    Which AI agent to use when the engine plays white.
  * `StrategyBlack`  
    Which AI agent to use when the engine plays black.
  * `NumThreads`  
    Currently does nothing, might be implemented/removed in the future.

## [v1.1.0] - 2024/11/25

- Fixes for Ladybug and Soldier Ant moves.
- Addition of random playing agent.
- Minor internal code changes.
- Addition of CHANGELOG.md.

## [v1.0.0] - 2024/11/24

- First release.
- Fully functional game engine.
- Documentation (README.md and with Sphinx).
- Prebuilt `.exe` file.

[Unreleased]: https://github.com/crystal-spider/hivemind
[README]: https://github.com/crystal-spider/hivemind#readme

[v1.2.0]: https://github.com/crystal-spider/hivemind/releases?q=1.2.0
[v1.1.0]: https://github.com/crystal-spider/hivemind/releases?q=1.1.0
[v1.0.0]: https://github.com/crystal-spider/hivemind/releases?q=1.0.0
