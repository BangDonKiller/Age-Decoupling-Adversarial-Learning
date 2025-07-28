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
        self.write_parameters_to_file(param.LOG_DIR, "setup")  # 寫入參數到 setup.txt
        # self.create_tensor_board(param.TENSOR_BOARD_DIR)  # 創建 TensorBoard 日誌目錄
        
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
                # f.write("Epoch, lr, L_id, L_age, L_grl, Total Loss, EER, minDCF\n")  # 寫入表頭
                f.write("Epoch, lr, L_id, acc_id, EER, minDCF\n")  # 寫入表頭
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
        # epoch, lr, l_id, l_age, l_grl, total_loss, eer, min_dcf = content
        epoch, lr, l_id, acc_id, eer, min_dcf = content
        with open(file_path, 'a') as f:
            # f.write(f"{epoch}, {lr}, {l_id:.4f}, {l_age:.4f}, {l_grl:.4f}, {total_loss:.4f}, {eer:.4f}, {min_dcf:.4f}\n")
            f.write(f"{epoch}, {lr}, {l_id:.4f}, {acc_id:.4f}, {eer:.4f}, {min_dcf:.4f}\n")
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
        
    def save_model(self, model, epoch):
        """
        保存模型的狀態字典到指定的檔案。

        :param model: 要保存的模型
        :param epoch: 當前訓練的 epoch
        """
        checkpoint_path = os.path.join(param.CHECKPOINT_DIR, f'model_step_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"模型已保存到：{checkpoint_path}")
        
    # def create_tensor_board(self, path):
    #     """
    #     根據 count 創建一個 TensorBoard 日誌資料夾。
        
    #     :param path: TensorBoard 根目錄
    #     :return: SummaryWriter 實例
    #     """
    #     # 根據 count 建立子資料夾，例如 ./logs_tensorboard/run1/
    #     tb_path = os.path.join(path, f"run{self.count}")
    #     os.makedirs(tb_path, exist_ok=True)
        
    #     writer = SummaryWriter(log_dir=tb_path)
    #     print(f"TensorBoard 事件文件已創建: {tb_path}")
    #     return writer
    
    # def write_tensorboard_log(self, writer, state, epoch, l_id, l_age, l_grl, total_loss, eer, min_dcf):
    #     """
    #     將數據寫入 TensorBoard 日誌。

    #     :param writer: SummaryWriter 實例
    #     :param tag: 日誌標籤
    #     :param value: 要寫入的值
    #     :param step: 步驟或 epoch 編號
    #     """
    #     if state == "train":
    #         writer.add_scalar('Loss/L_id', l_id, epoch)
    #         writer.add_scalar('Loss/L_age', l_age, epoch)
    #         writer.add_scalar('Loss/L_grl', l_grl, epoch)
    #         writer.add_scalar('Loss/Total', total_loss, epoch)
            
    #     elif state == "val":
    #         writer.add_scalar('EER', eer, epoch)
    #         writer.add_scalar('minDCF', min_dcf, epoch)

    
# if __name__ == "__main__":
    # SS = save_system()  # 初始化保存系統，確保目錄存在並創建初始文件
    # SS.write_result_to_file(param.SCORE_DIR, "result", (1, 0.1234, 0.5678, 0.9101, 1.2345, 0.1234, 0.5678))
    # SS.write_parameters_to_file(param.LOG_DIR, "setup")  # 寫入參數到 setup.txt
    # print("保存系統已初始化。")