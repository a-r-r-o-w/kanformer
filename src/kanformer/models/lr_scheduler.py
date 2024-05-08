import torch.optim as optim


class LRScheduler(optim.lr_scheduler.LRScheduler):
    def __init__(
        self, optimizer, embedding_dim: int, warmup_steps: int = 4000
    ) -> None:
        self.embedding_dim = embedding_dim
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            (self.embedding_dim**-0.5)
            * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for _ in self.base_lrs
        ]
