Build image
docker build -t adal .

Run container
docker run --gpus device=0 -d --rm -it --name adal -v C:/vscode/Age-Decoupling-Adversarial-Learning:/app -v C:/Dataset:/app/dataset adal

docker run --gpus device=0 -d --rm -it --name adal -v C:/Python/Master/ADAL:/app -v D:/Dataset:/app/dataset adal

顯示出目前正在執行的所有container
docker ps (+ -a 可查看包含未執行的container)

docker attach <container_name> 可連接到指定容器當前正在運行的終端機(ctrl + p + q 可維持容器
背景運行並退出，輸入exit 是強制停止)