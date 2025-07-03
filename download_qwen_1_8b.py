from modelscope import snapshot_download

# 下载 Qwen-VL-Chat-1.8B
model_dir = snapshot_download('qwen/Qwen-VL-Chat-1.8B', revision='v1.0.3')
print('模型已下载到:', model_dir)
print('请将该目录下的所有文件复制到 D:/tiktok/models/qwen/Qwen-VL-Chat-1.8B/') 