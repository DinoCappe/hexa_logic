# ♟️ Hive AlphaZero Agent  
*A deep reinforcement learning framework for the board game Hive, inspired by AlphaZero.*

---

## 🎯 Overview  
This project was developed within the [**Scuola Ortogonale Program**](https://www.elicsir.it/scuola-ortogonale) by the **Elicsir Foundation**, an initiative supporting selected students from Italian universities in their academic journey.  

The goal was to design an agent capable of playing **Hive** competitively — first outperforming a random player, then reaching human-level play — by combining **Monte Carlo Tree Search (MCTS)** with **deep reinforcement learning**, following a simplified [**AlphaZero-style**](https://suragnair.github.io/posts/alphazero.html) approach.

The system re-implements the *AlphaZero* pipeline (*self-play → neural network training → arena evaluation*) adapted to Hive’s unique rules and **hexagonal topology**, using the [**Mzinga**](https://github.com/jonthysell/Mzinga) engine as the backend for rule enforcement and move validation.

---

## ⚙️ Architecture  
- **Search:** Monte Carlo Tree Search with upper confidence bound (PUCT) exploration.  
- **Policy & Value Network:** 8-layer CNN operating on a 4-plane binary board encoding (queen/others × white/black), with tile-relative move encoding.  
- **Training Loop:** Alternates between *self-play* and *supervised updates* of the network using cross-entropy (policy) and MSE (value) losses.  
- **Self-Play Parallelization:** Multi-process CPU pool generating episodes concurrently, with **GPU inference servers** for batched network evaluations.  
- **Distributed Execution:** Designed to run efficiently on multi-GPU clusters (DISI HPC), with synchronized queues and fault-tolerant inference.

---

## 🧠 Learning Pipeline  

### 1️⃣ Pre-Training  
The network is first trained on a corpus of **5k human games** (2020–2025) using supervised imitation learning.  
- Parallel parsing and sharded datasets.  
- Data augmentation via board symmetries (rotations, flips).  
- Efficient loaders with `uint8` / `float16` compression for large-scale training.

### 2️⃣ Self-Play Reinforcement Learning  
- Multiple CPU workers simulate matches using the current policy/value network.  
- GPU inference servers compute batched predictions.  
- Training data generated in continuous cycles.  
- Arena evaluation retains the best-performing model.

### 3️⃣ Containerized Deployment  
The final trained agent (`best.pth.tar`) is packaged in a **Docker container** with the Mzinga engine for reproducible evaluation.

---

## 📈 Results  
- Stable end-to-end training pipeline (pre-training → self-play → evaluation).  
- Generated hundreds of thousands of self-play examples within 24 hours on the cluster.  
- The network learns non-trivial positional heuristics and achieves a **~60% win rate** vs. random agent.  
- Awarded the **“Lee Sedol Prize”** by the *Elicsir Foundation* for innovative use of reinforcement learning in Hive.

---

## 🧩 Implementation Highlights  

| Component | Description |
|------------|-------------|
| **Board Encoding** | 4-plane binary representation (queen/others × color). |
| **Move Encoding** | Tile-relative indexing (28 × 14 × 7 + 1 = 2745 possible moves). |
| **Network** | 8-layer CNN predicting policy and value jointly. |
| **Optimization** | Adam optimizer, learning rate scheduling, early stopping on arena results. |
| **Parallelism** | Multiprocessing + GPU inference queues (NCCL/Gloo backend). |
| **Frameworks** | PyTorch • AlphaZero General • Mzinga • Docker • Python 3.10 |

---

## Usage

There are two ways to use this Hive engine:

1. Run [`engine.py`](/src/engine.py) from the command line or with VSCode and start using the console to interact with it.  
   The engine will be fully functional, but there won't be any graphical interface.
2. Use the included `HexalogicEngine.exe` (or build it yourself) along with [MzingaViewer](https://github.com/jonthysell/Mzinga/wiki/MzingaViewer).  
   To do this, move `HexalogicEngine.exe` into the same directory as `MzingaViewer.exe` and then follow the instructions [here](https://github.com/jonthysell/Mzinga/wiki/BuildingAnEngine), specifically `step 2 > iii`.

To build the `HexalogicEngine.exe` yourself, simply run the following command in the project root:
```powershell
pyinstaller ./src/engine.py --name HexalogicEngine --noconsole --onefile
```

---

## 📚 References  
- Goede et al., *The Cost of Reinforcement Learning for Game Engines: The AZ-Hive Case Study (2022).*  
- Surag Nair, *AlphaZero General Framework.*  
- Jon Thysell, *Mzinga: Open-Source Hive Engine.*

---

## 👥 Authors  
**Nancy Kalaj** & **Ludovico Cappellato**  
Developed for the *Scuola Ortogonale 2024/25* by the *Elicsir Foundation*.  
Awarded the **“Lee Sedol Prize”** for innovative use of deep reinforcement learning.
