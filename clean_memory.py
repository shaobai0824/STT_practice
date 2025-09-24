import gc
import os
from pathlib import Path

import numpy as np
import psutil
import torch


def cleanup_memory():
    """清理記憶體"""
    print("\n🧹 記憶體清理")
    print("-" * 20)

    # 清理 Python 記憶體
    gc.collect()

    # 清理 CUDA 記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 重設統計
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except:
            pass

    print("✅ 記憶體清理完成")
cleanup_memory()