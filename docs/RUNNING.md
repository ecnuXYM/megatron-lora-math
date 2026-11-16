# 运行说明

使用 scripts/run_plain_lora_sft.sh，显式提供 MILES_DIR、MEGATRON_DIR、MODEL_HF、SFT_DATA、SAVE_DIR。它设置仓库的 Python 路径，直接启动 Miles train_async.py；DRY_RUN=1 仅打印命令。脚本不自动停止其他 Ray 进程。

历史 launcher 依赖原 Miles 安装及训练机。MILES_DIR、MODEL_HF、SFT_DATA、SAVE_DIR、LORA_RANK、LR 可覆盖；Ray runtime env 的 PYTHONPATH 也须对应新机器。own/rollout/plain_sft_rollout.py 必须处于 worker 的 Python 路径中。

复现前确认 Miles revision、Megatron-Bridge 版本与硬件。历史 launcher 保留清理 Ray 进程命令，应在独占训练环境运行。

指标查看与语法检查可在 CPU 运行；模型生成评测和训练需要对应 GPU 环境。
