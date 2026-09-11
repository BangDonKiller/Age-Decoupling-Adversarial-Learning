import os
from typing import Dict, Optional, Tuple

import torch


class CheckpointManager:
    """只保留最後一個 epoch 的模型與 projector。"""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.last_epoch_model: Optional[Tuple[int, str]] = None

    def save_epoch_model(self, epoch: int, model: torch.nn.Module) -> str:
        """保存當前 epoch 的模型權重，刪除舊的 epoch 模型。"""
        ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}_model.pth")
        torch.save({"model": model.state_dict()}, ckpt_path)
        
        # 刪除舊的 epoch 模型
        if self.last_epoch_model is not None:
            _, old_path = self.last_epoch_model
            if os.path.exists(old_path):
                os.remove(old_path)
        
        self.last_epoch_model = (epoch, ckpt_path)
        return ckpt_path

    def record_test_eer(self, epoch: int, test_eer_after: float):
        # 已移除平均統計，不再保存歷史 EER。
        return None

    def record_val_metrics(self, epoch: int, val_spk_acc: float, val_age_acc: float, val_age_auc: float):
        # 已移除平均統計，不再保存歷史 validation 指標。
        return None

    def get_last_epoch_checkpoint(self) -> Optional[Tuple[int, str]]:
        """取得最後一個 epoch 的 checkpoint 路徑。"""
        return self.last_epoch_model

    def finalize(
        self,
        model: torch.nn.Module,
        projector_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        extra_summary: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """保存最後一個 epoch 的模型和 projector（不進行 SWA）。"""
        last_ckpt = self.get_last_epoch_checkpoint()

        result = {
            "last_epoch": None,
            "last_epoch_model_path": None,
            "last_epoch_projector_path": None,
        }

        # 保存最後一個 epoch 的權重和 projector
        if last_ckpt is not None:
            last_epoch, last_ckpt_path = last_ckpt
            result["last_epoch"] = last_epoch
            result["last_epoch_model_path"] = last_ckpt_path
            
            # 保存 projector
            if projector_state_dict is not None:
                last_projector_path = os.path.join(self.checkpoint_dir, f"last_epoch_{last_epoch:03d}_projector.pth")
                torch.save(
                    {
                        "projector": projector_state_dict,
                        "epoch": last_epoch,
                    },
                    last_projector_path,
                )
                result["last_epoch_projector_path"] = last_projector_path

        return result
