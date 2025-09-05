import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from params import param
import importlib.util
import inspect
from torch.utils.tensorboard import SummaryWriter

class Save_system:
    @staticmethod
    def ensure_directories_exist():
        """
        確保所有必要的目錄存在。
        """
        os.makedirs(param.LOG_DIR, exist_ok=True)
        os.makedirs(param.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(param.SCORE_DIR, exist_ok=True)
        
    def __init__(self):
        """
        初始化保存系統，確保目錄存在。
        """
        self.ensure_directories_exist()
        print("所有必要的目錄已確保存在。")
        
        self.count = 1
        
        self.create_save_file(param.SCORE_DIR, "result")  # 創建一個初始的保存模型訓練結果文件
        self.create_save_file(param.LOG_DIR, "setup")  # 創建一個初始的保存模型參數文件
        self.create_save_file(param.FINETUNE_DIR, "finetune")  # 創建一個初始的保存微調結果文件
        self.write_parameters_to_file(param.LOG_DIR, "setup")  # 寫入參數到 setup.txt
        
    def create_save_file(self, path, filename):
        """
        在指定的目錄下創建一個保存文件。
        
        :param filename: 文件名
        :return: 完整的文件路徑
        """
    
        while True:
            file_path = os.path.join(path, f"{filename}{self.count}.txt")
            if not os.path.exists(file_path):
                break
            self.count += 1
        with open(file_path, 'w') as f:
            if filename == "result":
                f.write("Epoch, main_lr, L_id, Acc_id, cosEER, minDCF, Testacc, Precision, Recall\n")  # 寫入表頭
                # f.write("Epoch, main_lr, detach_lr, alpha, L_id, Acc_id, Total Loss, Acc_age, Loss_age, Loss_Recon, Detach_Loss_pred_ID, Detach_Acc_pred_ID\n")  # 寫入表頭
            elif filename == "finetune":
                f.write("Epoch, Loss, Accuracy, EER, minDCF, cosEER\n")
            else:
                f.write("")
        print(f"文件已創建: {file_path}")
        
    def write_result_to_file(self, path, filename, content):
        """
        將內容寫入指定的文件。
        
        :param path: 目錄路徑
        :param filename: 文件名
        :param content: 要寫入的內容
        """
        file_path = os.path.join(path, f"{filename}{self.count}.txt")
        
        if filename == "result":
            # epoch, main_lr, detach_lr, alpha, l_id, acc_id, loss_detach, acc_age, loss_age, loss_recon, detachment_ID_loss, detach_acc_ID= content
            # with open(file_path, 'a') as f:
            #     f.write(f"{epoch}, {main_lr}, {detach_lr}, {alpha:.4f}, {l_id:.4f}, {acc_id:.4f}, {loss_detach:.4f}, {acc_age:.4f}, {loss_age:.4f}, {loss_recon:.4f}, {detachment_ID_loss:.4f}, {detach_acc_ID:.4f}\n")
            epoch, main_lr, l_id, acc_id, cos_eer, min_dcf, Testacc, precision, recall = content
            with open(file_path, 'a') as f:
                f.write(f"{epoch}, {main_lr}, {l_id:.4f}, {acc_id:.4f}, {cos_eer:.4f}, {Testacc:.4f}, {precision:.4f}, {recall:.4f}\n")
        elif filename == "finetune":
            epoch, loss, accuracy, eer, min_dcf, cos_eer = content
            with open(file_path, 'a') as f:
                f.write(f"{epoch}, {loss:.4f}, {accuracy:.4f}, {eer:.4f}, {min_dcf:.4f}, {cos_eer:.4f}\n")

        print(f"結果已寫入: {file_path}")
            
    def write_parameters_to_file(self, path, filename):
        """
        將 param.py 中的參數寫入指定的 txt 文件。

        :param path: 輸出檔案目錄
        :param filename: 檔名（不含副檔名）
        """
        # 匯入 param.py 模組
        param_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'params', 'param.py'))
        spec = importlib.util.spec_from_file_location("param", param_path)
        param_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(param_module)

        # 取得所有變數（排除內建、函數、模組）
        parameters = {
            name: value for name, value in inspect.getmembers(param_module)
            if not name.startswith("__")
            and not inspect.ismodule(value)
            and not inspect.isfunction(value)
        }

        # 建立目錄
        os.makedirs(path, exist_ok=True)

        # 寫入到 txt 檔案
        file_path = os.path.join(path, f"{filename}{self.count}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            for name, value in parameters.items():
                f.write(f"{name} = {value}\n")

        print(f"參數已寫入：{file_path}")
        
    def save_model(self, model, epoch, mode, state):
        """
        保存模型的狀態字典到指定的檔案。

        :param model: 要保存的模型
        :param epoch: 當前訓練的 epoch
        """
        if mode == "pretrain":
            model_checkpoint_path = os.path.join(param.CHECKPOINT_DIR, f'pretrain_model_{epoch}.pth')
            torch.save(model.state_dict(), model_checkpoint_path)
            feature_extractor_path = os.path.join(param.CHECKPOINT_DIR, f'pretrain_feature_extractor_{epoch}.pth')
            torch.save(model.extractor.state_dict(), feature_extractor_path)
        elif mode == "finetune":
            if state == "best":
                model_checkpoint_path = os.path.join(param.CHECKPOINT_DIR, f'best_finetune_model.pth')
                torch.save(model.state_dict(), model_checkpoint_path)
                feature_extractor_path = os.path.join(param.CHECKPOINT_DIR, f'best_finetune_feature_extractor.pth')
                torch.save(model.extractor.state_dict(), feature_extractor_path)
            else:
                model_checkpoint_path = os.path.join(param.CHECKPOINT_DIR, f'last_finetune_model.pth')
                torch.save(model.state_dict(), model_checkpoint_path)
                feature_extractor_path = os.path.join(param.CHECKPOINT_DIR, f'last_finetune_feature_extractor.pth')
                torch.save(model.extractor.state_dict(), feature_extractor_path)
                
        print(f"模型已保存到：{model_checkpoint_path}")
        print(f"特徵提取器已保存到：{feature_extractor_path}")

    
# if __name__ == "__main__":
    # SS = save_system()  # 初始化保存系統，確保目錄存在並創建初始文件
    # SS.write_result_to_file(param.SCORE_DIR, "result", (1, 0.1234, 0.5678, 0.9101, 1.2345, 0.1234, 0.5678))
    # SS.write_parameters_to_file(param.LOG_DIR, "setup")  # 寫入參數到 setup.txt
    # print("保存系統已初始化。")