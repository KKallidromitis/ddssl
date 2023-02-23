from typing import Optional
from mmcv.runner.hooks.lr_updater import annealing_cos
class CosineAnnealing:
    """CosineAnnealing LR scheduler.

    Args:
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self,
                 min_lr: Optional[float] = None,
                 min_lr_ratio: Optional[float] = None,
                 **kwargs) -> None:
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio

    def get_lr(self, base_lr: float,iter,max_iters):
        progress = iter
        max_progress = max_iters

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr  # type:ignore
        return annealing_cos(base_lr, target_lr, progress / max_progress)