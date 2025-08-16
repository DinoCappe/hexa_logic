import time, queue, os
from multiprocessing import Process
import torch
from torch.amp import autocast

class InferenceServer(Process):
    """
    A single CUDA process that owns the model and batches requests from CPU workers.
    Requests arrive via in_q as tuples: (req_id:str, board_tensor:torch.Tensor[uint8 or float32, CxHxW]).
    Responses are sent on out_q as tuples: (req_id, pi_cpu:torch.Tensor[A], v_cpu:torch.Tensor[1]).
    """
    def __init__(
        self,
        model_ctor,                # callable: () -> nn.Module  (no CUDA yet inside)
        checkpoint_path: str,      # path to state_dict checkpoint
        device: torch.device,      # e.g. torch.device('cuda:0')
        in_q, out_q,               # multiprocessing queues
        max_batch: int = 128,
        max_wait_ms: float = 3.0,
        use_amp: bool = True,
        log_prefix: str = "[GPU-SERVER]"
    ):
        super().__init__()
        self.model_ctor = model_ctor
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.in_q = in_q
        self.out_q = out_q
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.use_amp = use_amp
        self.log_prefix = log_prefix

    def _load_checkpoint(self, model):
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            print(f"{self.log_prefix} loaded checkpoint: {self.checkpoint_path}", flush=True)
        else:
            print(f"{self.log_prefix} no checkpoint at: {self.checkpoint_path} (using random weights)", flush=True)

    def run(self):
        torch.cuda.set_device(self.device.index if hasattr(self.device, "index") else self.device)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        # Construct model on CPU, then move to CUDA
        model = self.model_ctor()
        model.eval()
        self._load_checkpoint(model)
        model.to(self.device)
        torch.set_grad_enabled(False)

        # Main batching loop
        while True:
            item = self.in_q.get()
            if item is None:
                break

            ids, xs, reply_qs = [], [], []

            # Support (rid, x) [old] and (rid, x, reply_q) [new]
            if len(item) == 3:
                rid, x, rq = item
                reply_qs.append(rq)
            else:
                rid, x = item
                reply_qs.append(None)
            ids.append(rid); xs.append(x)

            t0 = time.perf_counter()
            while len(xs) < self.max_batch and (time.perf_counter() - t0) * 1000.0 < self.max_wait_ms:
                try:
                    item = self.in_q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    break
                if len(item) == 3:
                    rid, x, rq = item
                    reply_qs.append(rq)
                else:
                    rid, x = item
                    reply_qs.append(None)
                ids.append(rid); xs.append(x)

            x = torch.stack(xs, dim=0).to(self.device, non_blocking=True).float()
            # print(f"{self.log_prefix} batch={x.shape[0]} item={tuple(x.shape[1:])}", flush=True)

            with torch.inference_mode(), autocast(device_type="cuda", enabled=self.use_amp):
                pi_log, v = model(x)

            pi_cpu = pi_log.exp().detach().cpu()
            v_cpu  = v.detach().cpu().view(-1)

            for k in range(len(ids)):
                payload = (ids[k], pi_cpu[k], v_cpu[k])
                if reply_qs[k] is not None:
                    reply_qs[k].put(payload)
                else:
                    self.out_q.put(payload)

        print(f"{self.log_prefix} shutdown complete.", flush=True)
