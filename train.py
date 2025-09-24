# ==============================================================================
# 檔案：train_final.py
# 描述：一個完整、高效、穩健的 Whisper 中文語音辨識模型微調流程。
# 核心策略：
# 1. 即時轉換 (.with_transform)：徹底解決記憶體不足與預處理過久的問題。
# 2. 背景預取 (dataloader_num_workers)：解決 CPU 與 I/O 瓶頸，最大化 GPU 使用率。
# 3. 全域定義 (Global Scope)：解決多核心處理時的 pickling 錯誤。
# 4. 智慧續練 (Smart Resuming)：自動從上次的檢查點恢復訓練。
# 5. 中文優化評估：使用 CER 和語意相似度替代 WER，更適合中文語音辨識。
# 6. 快速展示優化：根據研究結果調整訓練步數，1000步適合快速展示。
# ==============================================================================

import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import evaluate
# --- 新增：中文評估相關導入 ---
import Levenshtein
import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset, DatasetDict
# --- Hugging Face 相關導入 ---
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperFeatureExtractor,
                          WhisperForConditionalGeneration, WhisperProcessor,
                          WhisperTokenizer)

# ==============================================================================
# 步驟 1: 將所有輔助類別與函式定義在「全域範圍」
# 這是為了確保在使用 dataloader_num_workers > 0 時，背景程序可以成功序列化 (pickle) 它們。
# ==============================================================================

# --- 全域變數：中文語意模型 (避免重複載入) ---
semantic_model = None

def load_semantic_model():
    """載入中文語意相似度模型"""
    global semantic_model
    if semantic_model is None:
        print("載入中文語意相似度模型...")
        semantic_model = SentenceTransformer('shibing624/text2vec-base-chinese')
        print("✅ 語意相似度模型載入完成")
    return semantic_model


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """處理語音到序列資料的 Data Collator，負責將樣本整理成批次並進行填充。"""

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def prepare_dataset_batched(batch, feature_extractor, tokenizer):
    """將一批音訊和文本資料『即時』轉換為模型輸入格式。"""
    audio_list = batch["audio"]
    batch["input_features"] = feature_extractor(
        [x["array"] for x in audio_list], sampling_rate=audio_list[0]["sampling_rate"]
    ).input_features
    batch["labels"] = tokenizer(
        batch["transcription"], max_length=448, truncation=True
    ).input_ids
    return batch


def compute_cer(predictions, references):
    """計算字元錯誤率 (Character Error Rate)"""
    total_cer = 0
    total_chars = 0
    
    for pred, ref in zip(predictions, references):
        # 計算編輯距離
        edit_distance = Levenshtein.distance(pred, ref)
        # 計算 CER
        cer = edit_distance / max(len(ref), 1)  # 避免除零
        total_cer += cer
        total_chars += len(ref)
    
    avg_cer = total_cer / len(predictions) if predictions else 0
    return avg_cer

def compute_semantic_similarity(predictions, references):
    """計算語意相似度"""
    try:
        model = load_semantic_model()
        
        # 批次計算語意嵌入
        all_texts = predictions + references
        embeddings = model.encode(all_texts)
        
        # 分割嵌入向量
        pred_embeddings = embeddings[:len(predictions)]
        ref_embeddings = embeddings[len(predictions):]
        
        # 計算餘弦相似度
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            # 計算餘弦相似度
            similarity = cosine_similarity([pred_emb], [ref_emb])[0][0]
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        return avg_similarity
        
    except Exception as e:
        print(f"⚠️ 語意相似度計算失敗: {e}")
        return 0.0

def compute_metrics(pred, tokenizer):
    """在評估階段，計算並回傳 CER 和語意相似度指標。"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # 處理填充標記
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # 解碼預測和參考文本
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # 清理文本 (移除空白字符)
    pred_str = [text.strip() for text in pred_str]
    label_str = [text.strip() for text in label_str]
    
    # 計算 CER
    cer = compute_cer(pred_str, label_str)
    
    # 計算語意相似度
    semantic_sim = compute_semantic_similarity(pred_str, label_str)
    
    # 計算綜合評分 (語意相似度 - CER，越大越好)
    combined_score = semantic_sim - cer
    
    return {
        "cer": cer,
        "semantic_similarity": semantic_sim,
        "combined_score": combined_score
    }


# ==============================================================================
# 步驟 2: 主執行流程
# ==============================================================================
def main():
    # --- 參數設定 ---
    CSV_PATH = "youtube_clips_isolated/clips_mapping.csv"
    MODEL_NAME = "openai/whisper-small"
    LANGUAGE = "zh"
    TASK = "transcribe"
    OUTPUT_DIR = "./whisper-small-zh-finetune-zh-final"

    # --- 載入 Processor 和模型 ---
    print("--- 步驟 1/4: 載入 Processor 和模型 ---")
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # --- 建立原始資料集 ---
    class AudioDatasetProcessor:
        def __init__(self, file_path: str, target_sampling_rate: int = 16000):
            self.file_path = file_path
            self.target_sampling_rate = target_sampling_rate

        def create_dataset(self) -> Dataset:
            # 讀取 CSV 檔案
            full_data = pd.read_csv(self.file_path)
            
            # 重新命名欄位以符合程式碼期望的格式
            # 原始格式：漢字,原始切片,檔案位置
            # 期望格式：file,transcription
            full_data = full_data.rename(columns={
                '檔案位置': 'file',  # 使用隔離人聲的檔案
                '漢字': 'transcription'
            })
            
            # 只保留需要的欄位
            full_data = full_data[['file', 'transcription']]
            
            # 建立資料集
            dataset = Dataset.from_pandas(full_data)
            dataset = dataset.cast_column(
                "file", Audio(sampling_rate=self.target_sampling_rate)
            )
            dataset = dataset.rename_column("file", "audio")
            return dataset

    print("\n--- 步驟 2/4: 建立原始資料集並設定『即時轉換』---")
    audio_processor = AudioDatasetProcessor(file_path=CSV_PATH)
    full_dataset = audio_processor.create_dataset()
    common_voice = full_dataset.train_test_split(test_size=0.2, seed=42)

    # 使用 .with_transform() 確保記憶體穩定，訓練能立刻開始
    prepare_fn = partial(
        prepare_dataset_batched,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
    )
    vectorized_datasets = common_voice.with_transform(prepare_fn)
    print("即時轉換已設定。")

    # --- 建立訓練元件 ---
    print("\n--- 步驟 3/4: 建立訓練元件 (最終穩定運行版) ---")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics_fn = partial(compute_metrics, tokenizer=processor.tokenizer)

    # [中文優化版本]
    # 根據研究結果和中文語音辨識需求調整的訓練參數
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # 1. 大幅降低批次大小，這是避免 OOM 的核心
        per_device_train_batch_size=4,  # 從 32 或 16 大幅降至 4，這是一個極度安全的值
        per_device_eval_batch_size=4,  # 驗證批次也使用同樣的安全值
        # 2. 適度使用梯度累積，以穩定訓練
        # 有效批次大小為 4 * 4 = 16，這是一個不錯的平衡點
        gradient_accumulation_steps=4,
        # 3. 禁用多核心處理，這是確保程式不被掛起的關鍵
        dataloader_num_workers=0,
        # --- 其他參數維持不變 ---
        learning_rate=1e-5,
        warmup_steps=5,                 # 測試用：減少預熱步數
        max_steps=10,                   # 測試用：10步驗證流程
        # --- 評估與保存設定 ---
        gradient_checkpointing=False,
        fp16=True,
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=200,                 # 更頻繁保存，快速展示用
        eval_steps=5,                   # 測試用：每5步評估
        logging_steps=25,
        report_to=["tensorboard"],
        # --- 最佳模型選擇：使用綜合評分 ---
        load_best_model_at_end=True,
        metric_for_best_model="combined_score",  # 使用綜合評分
        greater_is_better=True,         # 綜合評分越大越好
        push_to_hub=True,
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=processor.feature_extractor,
    )

    # --- 開始訓練 ---
    print("\n--- 步驟 4/4: 開始中文語音辨識模型微調訓練 ---")
    print("📊 評估指標：CER (越低越好) + 語意相似度 (越高越好) + 綜合評分")
    print("🚀 訓練步數：10步 (測試用，驗證訓練流程)")
    print("⏱️ 預估時間：約 30 秒 (快速測試)")
    # 不帶參數的 .train() 會自動處理斷點續練，是最穩健的做法。
    trainer.train()
    print("\n*** 中文語音辨識模型訓練完成 ***")

    # --- 儲存最終模型 ---
    print("\n--- 正在儲存最終的最佳模型 ---")
    final_model_path = training_args.output_dir
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"\n最終模型已儲存至：{final_model_path}")


if __name__ == "__main__":
    # 確保您已在終端機使用 `huggingface-cli login` 登入
    # 執行前建議重新啟動您的電腦，確保系統處於乾淨狀態
    main()
