import uuid, time
import torch

class GPUClient:
    def __init__(self, in_q, reply_q, timeout_s: float = 120.0):
        self.in_q = in_q
        self.reply_q = reply_q
        self.timeout_s = timeout_s

    def predict_encoded(self, board_tensor_cpu: torch.Tensor):
        rid = uuid.uuid4().hex
        self.in_q.put((rid, board_tensor_cpu, self.reply_q))
        t0 = time.perf_counter()
        while True:
            try:
                r_rid, pi, v = self.reply_q.get(timeout=0.5)
            except Exception:
                if time.perf_counter() - t0 > self.timeout_s:
                    raise TimeoutError("GPUClient: prediction timed out")
                continue
            if r_rid == rid:
                return pi.numpy(), float(v.item())
            
class RemoteNNet:
    def __init__(self, board_size, action_size, gpu_client: GPUClient):
        self.board_size = board_size
        self.action_size = action_size
        self.gpu_client = gpu_client

    def predict(self, board):
        grid_size = self.board_size[0]
        enc = board.encode_board(grid_size=grid_size)
        x = torch.as_tensor(enc, dtype=torch.uint8).contiguous()
        pi, v = self.gpu_client.predict_encoded(x)
        return pi, v
