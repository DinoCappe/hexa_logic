import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import dotdict
from board import Board
import torch.optim as optim
import numpy as np
from numpy.typing import NDArray
import os
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import multiprocessing as mp
import glob
from typing import List, Tuple, Optional
import bisect
from collections import OrderedDict
import math, time
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import get_worker_info

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

TrainingExample = Tuple[np.ndarray, np.ndarray, float]

def setup_distributed(train_ddp: bool = False):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = "nccl" if (torch.cuda.is_available() and train_ddp) else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return local_rank
    
def make_human_loader(
    shard_dir: str,
    batch_size: int,
    num_workers: int = 4,
    distributed: bool = False,
    max_shards: Optional[int] = None,
    cache_size: int = 2,
    pin_memory: bool = True,
    verbose: bool = False
):

    rank0_verbose = verbose
    if distributed and dist.is_available() and dist.is_initialized():
        rank0_verbose = (dist.get_rank() == 0)

    ds = HumanGameDataset(
        shard_dir=shard_dir,
        max_shards=max_shards,
        cache_size=cache_size,
        dtype_boards=torch.float32,
        dtype_pi=torch.float32,
        dtype_v=torch.float32,
        verbose=rank0_verbose
    )

    sampler = DistributedSampler(
        ds,
        shuffle=False,         # <— critical
        drop_last=True
    ) if distributed else None
    
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,              # try 2–4 per rank
        pin_memory=pin_memory,                # (self.device.type=='cuda')
        persistent_workers=(num_workers > 0),
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context="forkserver",
    )
    return ds, loader, sampler
    
class HumanGameDataset(Dataset):
    def __init__(
        self,
        shard_dir: str,
        max_shards: Optional[int] = None,
        cache_size: int = 2,          # keep small (1–2)
        dtype_boards: torch.dtype = torch.float32,  # target dtype on GPU; CPU we’ll store uint8
        dtype_pi: torch.dtype = torch.float32,      # target dtype on GPU; CPU we’ll store float16
        dtype_v: torch.dtype = torch.float32,       # target dtype on GPU; CPU we’ll store float16
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.shard_paths: List[str] = sorted(glob.glob(os.path.join(shard_dir, "*.pt")))
        if max_shards is not None:
            self.shard_paths = self.shard_paths[:max_shards]
        if not self.shard_paths:
            raise RuntimeError(f"No .pt shards found in {shard_dir}")

        # Build length index per shard (read size only)
        self.shard_sizes: List[int] = []
        for p in self.shard_paths:
            lst = torch.load(p, map_location="cpu", weights_only=False)
            self.shard_sizes.append(len(lst))
        self.cum_sizes: List[int] = np.cumsum(self.shard_sizes).tolist()

        self.cache: "OrderedDict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
        self.cache_size = max(1, cache_size)

        # target dtypes (used later on GPU casts)
        self.dtype_boards = dtype_boards
        self.dtype_pi = dtype_pi
        self.dtype_v = dtype_v

    def __len__(self) -> int:
        return self.cum_sizes[-1]

    def _load_shard(self, shard_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.shard_paths[shard_idx]
        if path in self.cache:
            packed = self.cache.pop(path); self.cache[path] = packed
            return packed

        t0 = time.perf_counter()
        lst = torch.load(path, map_location="cpu", weights_only=False)
        dt = time.perf_counter() - t0

        # ----- PACK ONCE (compact CPU dtypes) -----
        # boards: keep as uint8 to save RAM (4*26*26 bytes per sample)
        bs, ps, vs = zip(*lst)  # lists of np arrays / floats
        boards_np = np.stack(bs, axis=0)                    # (N, C, H, W), typically uint8 already
        pis_np    = np.stack(ps, axis=0).astype(np.float16) # (N, A), store as fp16 on CPU
        vals_np   = np.asarray(vs, dtype=np.float16)        # (N,)

        boards = torch.from_numpy(boards_np)                         # uint8
        pis    = torch.from_numpy(pis_np)                             # float16
        vals   = torch.from_numpy(vals_np)                            # float16

        packed = (boards.contiguous(), pis.contiguous(), vals.contiguous())
        self.cache[path] = packed

        # LRU eviction
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        if self.verbose:
            wi = get_worker_info()
            wid = wi.id if wi is not None else 0
            if wid == 0:
                print(f"[dataset] shard {shard_idx:03d} loaded in {dt:.2f}s "
                      f"({boards.size(0)} examples) cache={len(self.cache)}/{self.cache_size}",
                      flush=True)
        return packed

    def __getitem__(self, idx: int):
        # locate shard
        shard_idx = bisect.bisect_right(self.cum_sizes, idx)
        prev_cum = 0 if shard_idx == 0 else self.cum_sizes[shard_idx - 1]
        local_idx = idx - prev_cum

        boards, pis, vals = self._load_shard(shard_idx)
        # return *tensors* directly (no per-item numpy→torch)
        return boards[local_idx], pis[local_idx], vals[local_idx]

class SelfPlayDataset(Dataset):
    def __init__(self, examples: list[tuple[NDArray[np.float64], NDArray[np.float64], float]], board_size: int):
        self.boards, self.pis, self.zs = zip(*examples)
        # pre‐encode any Board objects into raw numpy arrays
        self.encoded = []
        for b in self.boards:
            if isinstance(b, np.ndarray):
                self.encoded.append(b)
            else:
                # b is a Board instance
                self.encoded.append(b.encode_board(grid_size=board_size))
        self.encoded = np.stack(self.encoded)              # shape (N,C,H,W)
        self.pis = np.stack(self.pis)                  # shape (N,action_size)
        self.zs = np.array(self.zs, dtype=np.float32) # shape (N,)

    def __len__(self):
        return len(self.zs)

    def __getitem__(self, idx):
        # return torch tensors
        return (
            torch.from_numpy(self.encoded[idx]).float(),
            torch.from_numpy(self.pis[idx]).float(),
            torch.tensor(self.zs[idx]).float(),
        )

class NNetWrapper:
    def __init__(self, board_size: tuple[int, int], action_size: int, args: dotdict):
        """
        Args:
            board_size: (board_x, board_y) dimensions. For instance, (14, 14).
            action_size: The number of possible moves.
            args: An arguments object with hyperparameters (e.g., num_channels, num_layers, dropout, cuda, etc.).
        """
        self.args = args
        self.distributed = args.distributed
        self.ddp_wrapped = False

        if self.distributed:
            self.local_rank = setup_distributed(train_ddp=args.train_ddp)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.device = torch.device("cuda" if args.cuda else "cpu")
        
        self.board_size = board_size
        self.action_size = action_size
        self.nnet = HiveNNet(board_size, action_size,
            num_channels=args.num_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            )
        
        self.nnet.to(self.device)
    
    def predict(self, board: Board):
        """
        Given a Board, encode it and return the network's output:
          - A policy vector (as log probabilities)
          - A value estimate (scalar)
        """
        grid_size = self.board_size[0]
        encoded = board.encode_board(grid_size=grid_size)

        self.nnet.eval()
        with torch.no_grad():
            x = torch.as_tensor(encoded, dtype=torch.float32, device=self.device).unsqueeze(0)
            pi, v = self.nnet(x)
        return torch.exp(pi).detach().cpu().numpy()[0], v.detach().cpu().numpy()[0]
    
    def _is_dist(self) -> bool:
        return bool(self.ddp_wrapped) and torch.distributed.is_available() and torch.distributed.is_initialized()

    def _rank0(self) -> bool:
        return (not self._is_dist()) or dist.get_rank() == 0

    def train_from_examples(self, examples):
        """
        Self-play style: train from an in-memory list of (board_np, pi_np, value_float).
        """
        if not examples:
            print("[train_from_examples] no examples; skipping")
            return

        ds = SelfPlayDataset(examples, board_size=self.board_size[0])
        sampler = DistributedSampler(ds) if self._is_dist() else None
        loader = DataLoader(
            ds,
            batch_size=self.args.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=4 if self._is_dist() else 2,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=True,
        )

        model = self.nnet
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)

        rank = (dist.get_rank() if (self.distributed and dist.is_available() and dist.is_initialized()) else 0)
        logging.info(f"[R{rank}] STARTING TRAINING")

        for epoch in range(1, self.args.epochs + 1):
            if sampler is not None: sampler.set_epoch(epoch)
            model.train()
            total_pi = total_v = nb = 0

            logging.info(f"[R{rank}]  → Epoch {epoch} starting")
            print(f"[R{rank}][TRAIN] Epoch {epoch} ...")

            for boards, pis, vals in loader:
                boards = boards.to(self.device, non_blocking=True)
                pis = pis.to(self.device, non_blocking=True)
                vals = vals.to(self.device, non_blocking=True)

                out_pi, out_v = model(boards)          # out_pi is log-softmax in your net
                loss_pi = -(pis * out_pi).sum(dim=1).mean()
                loss_v = F.mse_loss(out_v.view(-1), vals)
                loss = loss_pi + loss_v

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_pi += float(loss_pi.item())
                total_v  += float(loss_v.item())
                nb += 1

            if self._rank0():
                print(f"[train_from_examples] epoch {epoch}/{self.args.epochs} "
                    f"π={total_pi/max(nb,1):.4f} v={total_v/max(nb,1):.4f}")

    def pretrain_from_shards(
        self,
        shard_dir: str,
        epochs: int | None = None,
        batch_size: int | None = None,
        num_workers: int = 4,
        max_shards: int | None = None,
        cache_size: int = 2,
        log_every: int = 100,
        dry_run: int = 0,   # set >0 to iterate N batches without model compute
    ):
        def _rank():
            return dist.get_rank() if dist.is_initialized() else 0
        def _world():
            return dist.get_world_size() if dist.is_initialized() else 1

        epochs = epochs or self.args.epochs
        batch_size = batch_size or self.args.batch_size

        # ---------- Count total examples on disk (once per rank) ----------
        shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*.pt")))
        if max_shards is not None:
            shard_paths = shard_paths[:max_shards]
        total_examples = 0
        for i, p in enumerate(shard_paths):
            try:
                lst = torch.load(p, map_location="cpu", weights_only=False)
            except TypeError:
                lst = torch.load(p, map_location="cpu")
            n = len(lst)
            total_examples += n
            if i < 3 or i == len(shard_paths) - 1:
                print(f"[rank {_rank()}] [pretrain] shard {i:03d} size={n}  ({os.path.basename(p)})", flush=True)
        print(f"[rank {_rank()}] [pretrain] shards={len(shard_paths)}  total_examples={total_examples} "
            f"avg/shard={total_examples/max(1,len(shard_paths)):.1f}", flush=True)

        rank0_verbose = (not self._is_dist()) or (dist.get_rank() == 0)

        # ---------- Build loader ----------
        _, loader, sampler = make_human_loader(
            shard_dir=shard_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            distributed=self._is_dist(),
            max_shards=max_shards,
            cache_size=cache_size,
            pin_memory=(self.device.type == "cuda"),
            verbose=rank0_verbose
        )

        ds_len = len(loader.dataset)
        per_rank = sampler.num_samples if sampler is not None else ds_len
        num_batches = len(loader)
        print(f"[rank {_rank()}] [pretrain] start → epochs={epochs}  batch={batch_size}  "
            f"workers={num_workers}  world_size={_world()}  "
            f"dataset_len={ds_len}  per_rank_samples={per_rank}  batches/epoch={num_batches}",
            flush=True)

        # ---------- Dry run: measure loader only ----------
        if dry_run > 0:
            t0 = time.perf_counter()
            cnt = 0
            for i, (boards, pis, vals) in enumerate(loader, start=1):
                cnt += boards.size(0)
                if i >= dry_run:
                    break
            dt = time.perf_counter() - t0
            sps = (cnt * _world()) / max(1e-6, dt)
            print(f"[rank {_rank()}] [pretrain] DRY RUN: {i} batches, {cnt} samples "
                f"in {dt:.2f}s → {sps:.0f} samp/s (global).", flush=True)
            return

        model = self.nnet
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)

        # Speed knobs
        torch.backends.cudnn.benchmark = True

        # ---- AMP (only enabled on CUDA) ----
        use_amp = (self.device.type == "cuda")
        scaler = GradScaler('cuda', enabled=use_amp)

        for epoch in range(1, epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()

            print(f"[rank {_rank()}] [pretrain] epoch {epoch}/{epochs} …", flush=True)

            total_pi = total_v = nb = 0
            b_per_rank = 0
            t_batch_prev = time.perf_counter()
            t_window_start = t_batch_prev
            i_at_window_start = 0
            first_logged = False

            try:
                for i, (boards, pis, vals) in enumerate(loader, start=1):
                    data_time = time.perf_counter() - t_batch_prev

                    if not first_logged:
                        print(f"[rank {_rank()}] [pretrain] first batch: "
                            f"boards={tuple(boards.shape)},{boards.dtype} "
                            f"pis={tuple(pis.shape)},{pis.dtype} "
                            f"vals={tuple(vals.shape)},{vals.dtype}", flush=True)
                        first_logged = True

                    boards = boards.to(self.device, non_blocking=True)
                    pis = pis.to(self.device, non_blocking=True)
                    vals = vals.to(self.device, non_blocking=True)

                    t_compute_start = time.perf_counter()
                    # ----- forward (AMP autocast) -----
                    with autocast('cuda', enabled=use_amp):
                        out_pi, out_v = model(boards.float())   # out_pi is log-softmax, out_v is [-1,1]

                    # ----- losses in fp32 for stability -----
                    pis_f  = pis.float()
                    vals_f = vals.float()
                    out_pi_f = out_pi.float()
                    out_v_f  = out_v.float()

                    loss_pi = -(pis_f * out_pi_f).sum(dim=1).mean()
                    loss_v  = F.mse_loss(out_v_f.view(-1), vals_f)
                    loss    = loss_pi + loss_v

                    optimizer.zero_grad(set_to_none=True)

                    # ----- backward + step (AMP scaler) -----
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    compute_time = time.perf_counter() - t_compute_start
                    t_batch_prev = time.perf_counter()

                    total_pi += float(loss_pi.item())
                    total_v  += float(loss_v.item())
                    nb += 1
                    b_per_rank += boards.size(0)

                    if i == 1 or i % log_every == 0 or i == num_batches:
                        steps = i - i_at_window_start
                        dt = max(1e-6, t_batch_prev - t_window_start)
                        sps = steps * boards.size(0) * _world() / dt
                        eta_min = (num_batches - i) * (dt / steps) / 60.0
                        print(
                            f"[rank {_rank()}] [pretrain] e{epoch} b{i}/{num_batches} "
                            f"π={loss_pi.item():.4f} v={loss_v.item():.4f} "
                            f"avgπ={total_pi/max(nb,1):.4f} avgv={total_v/max(nb,1):.4f} "
                            f"data={data_time*1000:.1f}ms step={compute_time*1000:.1f}ms "
                            f"sps={sps:.0f} ETA~{eta_min:.1f}m",
                            flush=True
                        )
                        t_window_start = t_batch_prev
                        i_at_window_start = i

            except Exception as e:
                print(f"[rank {_rank()}] [pretrain] ERROR mid-epoch at batch {nb+1}: {e}", flush=True)
                raise

            print(f"[rank {_rank()}] [pretrain] epoch {epoch} done → "
                f"avgπ={total_pi/max(nb,1):.4f} avgv={total_v/max(nb,1):.4f} "
                f"samples_seen_per_rank={b_per_rank}", flush=True)

        if self._rank0():
            self.save_checkpoint(folder=self.args.checkpoint, filename="pretrain_last.pth.tar")
            print(f"[rank 0] [pretrain] saved checkpoint → "
                f"{os.path.join(self.args.checkpoint,'pretrain_last.pth.tar')}", flush=True)

    def train(self, examples=None, pretrain_dir=None, **kwargs):
        """
        Back-compat convenience:
        - if pretrain_dir is given -> call pretrain_from_shards
        - else -> train_from_examples
        """
        if pretrain_dir is not None:
            return self.pretrain_from_shards(pretrain_dir, **kwargs)
        return self.train_from_examples(examples)

    def save_checkpoint(self, folder: str='checkpoint', filename: str='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)

        # If the model is wrapped by DDP, pull out the underlying module
        if hasattr(self.nnet, 'module'):
            state_dict = self.nnet.module.state_dict()
        else:
            state_dict = self.nnet.state_dict()

        torch.save({'state_dict': state_dict}, filepath)

    def load_checkpoint(self, folder: str='checkpoint', filename: str='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)  # type: ignore
        state_dict = checkpoint['state_dict']

        # If we're wrapped by DDP, load into the underlying module;
        # otherwise load directly.
        if hasattr(self.nnet, 'module'):
            self.nnet.module.load_state_dict(state_dict)
        else:
            self.nnet.load_state_dict(state_dict)


class HiveNNet(nn.Module):
    def __init__(self, board_size: tuple[int, int], action_size: int, 
                 num_channels: int=256, num_layers: int=8, dropout: float=0.3):
        """
        Args:
            board_size (tuple): (board_x, board_y) dimensions.
            action_size (int): Number of possible moves.
            num_channels (int): Number of convolutional filters.
            num_layers (int): Number of convolutional layers (supported: 4, 6, 8).
            dropout (float): Dropout rate.
        """
        super().__init__()  # type: ignore
        self.board_x, self.board_y = board_size
        self.action_size = action_size
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.adapt = nn.AdaptiveAvgPool2d((14, 14))
        
        in_channels = 4

        # First convolutional layer.
        self.conv1 = nn.Conv2d(in_channels, self.num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)

        # Second convolutional layer.
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        
        # Additional convolutional layers.
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _i in range(2, self.num_layers):
            self.convs.append(nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=1))
            self.bns.append(nn.BatchNorm2d(self.num_channels))
        
        flattened_size = self.num_channels * 14 * 14  # thanks to AdaptiveAvgPool2d((2,2))
        
        # Fully connected layers.
        self.fc1 = nn.Linear(flattened_size, 4096)
        self.fc_bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc_bn2 = nn.BatchNorm1d(2048)
        
        # Policy head.
        self.fc_policy = nn.Linear(2048, self.action_size)
        # Value head.
        self.fc_value = nn.Linear(2048, 1)

    def forward(self, s: torch.Tensor):
        """
        Forward pass.
        Args:
            s (torch.Tensor): Input board tensor. Its shape should be 
                              (batch_size, board_x, board_y) for simple representations,
                              or include channel information for spatial planes.
        Returns:
            (pi, v): where pi is the log-softmax over actions and v is the board value.
        """
        batch_size = s.size(0)
        in_channels = 4
        s = s.view(batch_size, in_channels, self.board_x, self.board_y)
        
        # Convolutional layers.
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        for conv, bn in zip(self.convs, self.bns):
            s = F.relu(bn(conv(s)))
        
        # Flatten.
        s = self.adapt(s)
        s = s.view(batch_size, -1)
        
        # Fully connected layers.
        s = F.relu(self.fc_bn1(self.fc1(s)))
        s = F.dropout(s, p=self.dropout, training=self.training)
        s = F.relu(self.fc_bn2(self.fc2(s)))
        s = F.dropout(s, p=self.dropout, training=self.training)
        
        # Policy head.
        pi = self.fc_policy(s)
        pi = F.log_softmax(pi, dim=1)
        
        # Value head.
        v = self.fc_value(s)
        v = torch.tanh(v)
        
        return pi, v
