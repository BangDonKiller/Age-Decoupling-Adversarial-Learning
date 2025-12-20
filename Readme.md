有關於 ffmpeg 說明(for Windows environment)
- 請先至 https://github.com/GyanD/codexffmpeg/releases 下載檔案 (必須是shared)
- 加入系統變數後，且裝完 torchaudio 後，請至 .venv\Lib\site-packages\torchaudio\_extension\__init__.py 修改此行

原 code: if os.name == "nt" and (3, 8) <= sys.version_info < (3, 9):
修正後: if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):