.. Hivemind documentation master file, created by
   sphinx-quickstart on Sun Nov 24 21:23:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hivemind documentation
======================

.. Add your content using ``reStructuredText`` syntax. See the
   `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
   documentation for details.

Description
-----------

| `Link UHP <https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol>`_-compliant `Link Hive <https://en.wikipedia.org/wiki/Hive_(game)>`_ game engine in Python.  
| The game engine logic is - sometimes loosely, sometimes strongly - inspired from `Link Mzinga Engine <https://github.com/jonthysell/Mzinga>`_.
|
| Currently, it's a WIP. Future plans include designing an AI to play the game.
|
| The repository includes builds for both a fully-fledged documentation generated with Sphinx and the ready-to-use ``HivemindEngine.exe``.

Setup
-----

Setting up the environment is pretty easy:

1. Install `Link Anaconda <https://www.anaconda.com/download/success>`_.
2. Open the project root directory and run the following command:

   .. code:: powershell

      conda create --name <env> --file requirements.txt

   ``<env>`` can be any name you want.

The suggested IDE is `Link Visual Studio Code <https://code.visualstudio.com/>`_, and settings for it are included.

Usage
-----

There are two ways to use this Hive engine:

1. | Run ``engine.py`` from the command line or with VSCode and start using the console to interact with it.
   | The engine will be fully functional, but there won't be any graphical interface.
2. | Use the included ``HivemindEngine.exe`` (or build it yourself) along with `Link MzingaViewer <https://github.com/jonthysell/Mzinga/wiki/MzingaViewer>`_.
   | To do this, move ``HivemindEngine.exe`` into the same directory as ``MzingaViewer.exe`` and then follow the instructions `Link here <https://github.com/jonthysell/Mzinga/wiki/BuildingAnEngine>`_, specifically ``step 2 > iii``.

To build the ``HivemindEngine.exe`` yourself, simply run the following command in the project root:

.. code:: powershell

   pyinstaller ./src/engine.py --name HivemindEngine --noconsole --onefile

Contents
--------

.. toctree::
   :maxdepth: 1

   engine
   board
   game
   enums
