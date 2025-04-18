# Hivemind

## Description

[UHP](https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol)-compliant [Hive](https://en.wikipedia.org/wiki/Hive_(game)) game engine in Python.  
The game engine logic is - sometimes loosely, sometimes strongly - inspired from [Mzinga Engine](https://github.com/jonthysell/Mzinga).

The engine comes with different AI agent configurations. More on this below.

The repository includes builds for both a fully-fledged documentation generated with Sphinx and the ready-to-use `HivemindEngine.exe`.

## Documentation

The source code is fully documented with Docstrings in [reST](https://docutils.sourceforge.io/rst.html).

The structured documentation can be generated with [Sphinx](https://www.sphinx-doc.org/en/master/).  
A working build of the documentation is already included under [`docs/build/`](/docs/build/).  
To view it, simply open [`index.html`](/docs/build/html/index.html) with a browser.

To build the documentation yourself, simply run the following command under [`docs/`](/docs/):
```powershell
make html
``` 

## Setup

Setting up the environment is pretty easy:

1. Install [Anaconda](https://www.anaconda.com/download/success).
2. Open the project root directory and run the following command:
   ```powershell
   conda create --name <env> --file requirements.txt
   ```
   `<env>` can be any name you want.

The suggested IDE is [Visual Studio Code](https://code.visualstudio.com/), and settings for it are included.

## Usage

There are two ways to use this Hive engine:

1. Run [`engine.py`](/src/engine.py) from the command line or with VSCode and start using the console to interact with it.  
   The engine will be fully functional, but there won't be any graphical interface.
2. Use the included `HivemindEngine.exe` (or build it yourself) along with [MzingaViewer](https://github.com/jonthysell/Mzinga/wiki/MzingaViewer).  
   To do this, move `HivemindEngine.exe` into the same directory as `MzingaViewer.exe` and then follow the instructions [here](https://github.com/jonthysell/Mzinga/wiki/BuildingAnEngine), specifically `step 2 > iii`.

To build the `HivemindEngine.exe` yourself, simply run the following command in the project root:
```powershell
pyinstaller ./src/engine.py --name HivemindEngine --noconsole --onefile
```

## AI

There are currently 2 implemented AI strategies:

1. Random: the agent plays random moves.
2. Minmax: the agent plays moves following a Minmax policy with alpha-beta pruning and a custom node (game state) evaluation.

A third implementation will come in the future that will leverage machine learning.
