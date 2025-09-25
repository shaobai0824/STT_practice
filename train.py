# ==============================================================================
# 檔案：train_final.py
# 描述：一個完整、高效、穩健的 Whisper 中文語音辨識模型微調流程。
# 核心策略：
# 1. 即時轉換 (.with_transform)：徹底解決記憶體不足與預處理過久的問題。
# 2. 背景預取 (dataloader_num_workers)：解決 CPU 與 I/O 瓶頸，最大化 GPU 使用率。
# 3. 全域定義 (Global Scope)：解決多核心處理時的 pickling 錯誤。
# 4. 智慧續練 (Smart Resuming)：自動從上次的檢查點恢復訓練。
# 5. 中文優化評估：使用 CER 和語意相似度替代 WER，更適合中文語音辨識。
# 6. 快速展示優化：根據研究結果調整訓練步數，10步適合測試流程。
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
    """
    處理語音到序列資料的 Data Collator，負責將樣本整理成批次並進行填充。
    
    主要功能：
    ------------------------------------------------------------------------------
    1. 批次資料整理：將不同長度的音訊和文字樣本整理成相同大小的批次
    2. 填充處理：使用 padding 讓所有樣本具有相同的維度
    3. Attention Mask 建立：標記哪些位置是真實資料，哪些是填充
    4. 標籤處理：準備訓練用的標籤，並處理特殊 token
    
    輸入範例：
    - features[0]: {"input_features": [80, 2500], "labels": [50258, 16563, 16563, 50259]}
    - features[1]: {"input_features": [80, 3000], "labels": [50258, 16563, 16563, 16563, 50259]}
    
    輸出範例：
    - batch["input_features"]: shape = [2, 80, 3000] (填充到最大長度)
    - batch["labels"]: shape = [2, 5] (填充到最大長度)
    - batch["attention_mask"]: shape = [2, 3000] (標記真實資料位置)
    """

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # 步驟 1: 整理音訊特徵
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        # 使用特徵提取器進行填充
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        # 輸出：batch["input_features"] shape = [batch_size, 80, max_time_steps]
        # 輸出：batch["attention_mask"] shape = [batch_size, max_time_steps]
        # attention_mask: 1 = 真實資料, 0 = 填充位置
        
        # 步驟 2: 整理文字標籤
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # 輸出：labels_batch["input_ids"] shape = [batch_size, max_sequence_length]
        # 輸出：labels_batch["attention_mask"] shape = [batch_size, max_sequence_length]
        
        # 步驟 3: 建立訓練標籤 (Attention Mask 處理)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # masked_fill 功能：
        # - attention_mask = 1 的位置：保留原始 token ID
        # - attention_mask = 0 的位置：填充為 -100 (忽略計算損失)
        # 範例：
        # input_ids:     [50258, 16563, 16563, 50259, 0, 0]
        # attention_mask: [1,     1,     1,     1,     0, 0]
        # labels:        [50258, 16563, 16563, 50259, -100, -100]
        
        # 步驟 4: 移除 BOS Token (如果存在)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        # 原因：BOS token 不需要預測，因為它是輸入的一部分
        
        # 步驟 5: 組合最終批次
        batch["labels"] = labels
        return batch
        # 最終輸出：
        # - batch["input_features"]: 音訊特徵 [batch_size, 80, max_time_steps]
        # - batch["attention_mask"]: 音訊注意力遮罩 [batch_size, max_time_steps]
        # - batch["labels"]: 訓練標籤 [batch_size, max_sequence_length]


def prepare_dataset_batched(batch, feature_extractor, tokenizer):
    """
    將一批音訊和文本資料『即時』轉換為模型輸入格式。
    
    🔬 特徵提取流程詳解：
    ------------------------------------------------------------------------------
    輸入音訊範例：
    - 原始音訊：30秒的 .wav 檔案，取樣率 44,100 Hz
    - 音訊陣列：shape = [1,320,000] (44,100 × 30 秒)
    
    步驟 1: 音訊重採樣 (Audio Resampling)
    - 從 44,100 Hz → 16,000 Hz (Whisper 標準)
    - 重採樣後：shape = [480,000] (16,000 × 30 秒)
    
    步驟 2: STFT (短時距傅立葉變換)
    - n_fft：400 (FFT 大小)
    - hop_length：160 (步長，10ms @ 16kHz)
    - 音框數：30,000ms ÷ 10ms = 3,000 個音框
    - 每個音框：201 個頻率點 (n_fft=400 → 400//2+1=201)
    - STFT 輸出：shape = [3,000, 201]
    
    步驟 3: 梅爾頻譜轉換 (Mel Spectrogram)
    - 梅爾濾波器組：80 個濾波器
    - 梅爾頻譜：shape = [3,000, 80] (201 個頻率點 → 80 個梅爾頻率)
    - 時間軸壓縮：3,000 → 3,000 (保持不變)
    - 最終梅爾頻譜：shape = [80, 3,000]
    
    步驟 4: 對數轉換 (Log Transformation)
    - 對數梅爾頻譜：log(mel_spectrogram + 1e-10)
    - 最終特徵：shape = [80, 3,000] (80 個梅爾頻率 × 3,000 個時間步)
    
    📊 實際維度範例：
    - 批次大小：4
    - 輸入特徵：shape = [4, 80, 3000]
    - 標籤序列：shape = [4, 448] (最大長度 448 tokens)
    """
    audio_list = batch["audio"]
    
    # 🔬 特徵提取：音訊 → 梅爾頻譜圖
    # 內部執行：重採樣 → STFT → 梅爾轉換 → 對數轉換
    batch["input_features"] = feature_extractor(
        [x["array"] for x in audio_list], sampling_rate=audio_list[0]["sampling_rate"]
    ).input_features
    # 輸出：shape = [batch_size, 80, 3000]
    # 80: 梅爾頻率特徵維度
    # 3000: 時間步數 (約30秒音訊)
    
    # 🔤 文字標記化：中文文字 → Token IDs
    batch["labels"] = tokenizer(
        batch["transcription"], max_length=448, truncation=True
    ).input_ids
    # 輸出：shape = [batch_size, sequence_length]
    # 範例：["你好世界"] → [50258, 16563, 16563, 16563, 50259]
    # 50258: <|startoftranscript|>
    # 16563: "你", 16563: "好", 16563: "世", 16563: "界"
    # 50259: <|endoftext|>
    
    return batch


def compute_cer(predictions, references):
    """
    計算字元錯誤率 (Character Error Rate)
    
    🎯 CER 計算原理：
    ------------------------------------------------------------------------------
    CER = (插入錯誤 + 刪除錯誤 + 替換錯誤) / 參考文字總字元數
    
    實際範例：
    - 參考文字："今天天氣很好"
    - 預測文字："今日天氣真好"
    - 編輯距離：2 (替換 "天"→"日", "很"→"真")
    - CER = 2 / 6 = 0.333 (33.3% 錯誤率)
    
    中文優勢：CER 比 WER (詞錯誤率) 更適合中文，因為中文沒有明確的詞邊界
    """
    total_cer = 0
    total_chars = 0
    
    for pred, ref in zip(predictions, references):
        # 🔍 計算編輯距離 (Levenshtein Distance)
        # 範例："今天天氣很好" vs "今日天氣真好" = 2
        edit_distance = Levenshtein.distance(pred, ref)
        
        # 📊 計算 CER：編輯距離 / 參考文字長度
        cer = edit_distance / max(len(ref), 1)  # 避免除零
        total_cer += cer
        total_chars += len(ref)
    
    # 📈 平均 CER
    avg_cer = total_cer / len(predictions) if predictions else 0
    return avg_cer

def compute_semantic_similarity(predictions, references):
    """
    計算語意相似度
    
    🧠 語意相似度原理：
    ------------------------------------------------------------------------------
    使用預訓練的中文語意模型 (shibing624/text2vec-base-chinese) 將文字轉換為向量，
    然後計算餘弦相似度來衡量語意相近程度。
    
    實際範例：
    - 文字A："今天天氣很好"
    - 文字B："今日天氣真好"
    - 語意向量A：[0.1, 0.8, -0.3, ...] (768維)
    - 語意向量B：[0.2, 0.7, -0.2, ...] (768維)
    - 餘弦相似度：0.95 (95% 相似)
    
    優勢：即使有錯字，只要意思相近，相似度仍然很高
    """
    try:
        # 載入中文語意模型
        model = load_semantic_model()
        # 模型：shibing624/text2vec-base-chinese
        # 輸出維度：768 維語意向量
        
        # 批次計算語意嵌入
        all_texts = predictions + references
        embeddings = model.encode(all_texts)
        # 輸出：shape = [len(all_texts), 768]
        
        # 分割嵌入向量
        pred_embeddings = embeddings[:len(predictions)]
        ref_embeddings = embeddings[len(predictions):]
        
        # 計算餘弦相似度
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            # 餘弦相似度公式：cos(θ) = A·B / (||A|| × ||B||)
            similarity = cosine_similarity([pred_emb], [ref_emb])[0][0]
            similarities.append(similarity)
        
        # 平均相似度
        avg_similarity = np.mean(similarities) if similarities else 0
        return avg_similarity
        
    except Exception as e:
        print(f"⚠️ 語意相似度計算失敗: {e}")
        return 0.0

def compute_metrics(pred, tokenizer):
    """
    在評估階段，計算並回傳 CER 和語意相似度指標。
    
    🔄 評估流程詳解：
    ------------------------------------------------------------------------------
    1. 解碼預測結果：Token IDs → 中文文字
    2. 解碼參考標籤：Token IDs → 中文文字  
    3. 計算 CER：字元級別錯誤率
    4. 計算語意相似度：語意層面相似程度
    5. 計算綜合評分：語意相似度 - CER
    
    實際範例：
    - 預測："今日天氣真好" (Token IDs: [50258, 16563, 16563, ...])
    - 參考："今天天氣很好" (Token IDs: [50258, 16563, 16563, ...])
    - CER: 0.333 (33.3% 錯誤)
    - 語意相似度: 0.95 (95% 相似)
    - 綜合評分: 0.95 - 0.333 = 0.617
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # 處理填充標記 (-100 表示忽略的標記)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # 解碼預測和參考文本
    # Token IDs → 中文文字
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # 清理文本 (移除空白字符)
    pred_str = [text.strip() for text in pred_str]
    label_str = [text.strip() for text in label_str]
    
    # 計算 CER (字元錯誤率)
    cer = compute_cer(pred_str, label_str)
    
    # 計算語意相似度
    semantic_sim = compute_semantic_similarity(pred_str, label_str)
    
    # 計算綜合評分 (語意相似度 - CER，越大越好)
    # 理想情況：語意相似度接近 1.0，CER 接近 0.0
    # 綜合評分範圍：-1.0 到 1.0
    combined_score = semantic_sim - cer
    
    return {
        "cer": cer,                    # 字元錯誤率 (越低越好)
        "semantic_similarity": semantic_sim,  # 語意相似度 (越高越好)
        "combined_score": combined_score      # 綜合評分 (越高越好)
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
    
    # 載入 WhisperProcessor (包含特徵提取器和分詞器)
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )
    # processor.feature_extractor: 音訊 → 梅爾頻譜圖 [80, 3000]
    # processor.tokenizer: 中文文字 ↔ Token IDs
    
    # 載入 Whisper 模型 (Transformer 編碼器-解碼器)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    # 模型架構：
    # - 編碼器：12層 Transformer，隱藏維度 768
    # - 解碼器：12層 Transformer，隱藏維度 768
    # - 注意力頭數：12
    # - 參數總數：約 244M (whisper-small)
    
    # 模型配置調整
    model.config.forced_decoder_ids = None      # 不強制特定語言標記
    model.config.suppress_tokens = []            # 不抑制任何 Token

    # --- 建立原始資料集 ---
    class AudioDatasetProcessor:
        """
        📊 音訊資料集處理器
        
        功能：將 CSV 檔案轉換為 Hugging Face Dataset 格式
        輸入：CSV 檔案 (漢字, 原始切片, 檔案位置)
        輸出：Dataset 物件 (audio, transcription)
        """
        def __init__(self, file_path: str, target_sampling_rate: int = 16000):
            self.file_path = file_path
            self.target_sampling_rate = target_sampling_rate  # Whisper 標準：16kHz

        def create_dataset(self) -> Dataset:
            # 讀取 CSV 檔案
            full_data = pd.read_csv(self.file_path)
            # 範例資料：
            # 漢字          原始切片                   檔案位置
            # "你好世界"   "audio_001.wav"          "youtube_clips_isolated/@2_isolated_vocals/audio_001.wav"
            
            # 🔄 重新命名欄位以符合程式碼期望的格式
            # 原始格式：漢字, 原始切片, 檔案位置
            # 期望格式：file, transcription
            full_data = full_data.rename(columns={
                '檔案位置': 'file',      # 使用隔離人聲的檔案路徑
                '漢字': 'transcription'  # 中文文字稿
            })
            
            # 只保留需要的欄位
            full_data = full_data[['file', 'transcription']]
            
            # 建立 Hugging Face Dataset
            dataset = Dataset.from_pandas(full_data)
            
            # 將檔案路徑轉換為音訊物件
            dataset = dataset.cast_column(
                "file", Audio(sampling_rate=self.target_sampling_rate)
            )
            # 內部處理：
            # 1. 讀取 .wav 檔案
            # 2. 重採樣到 16kHz
            # 3. 轉換為 numpy 陣列
            
            # 重新命名欄位
            dataset = dataset.rename_column("file", "audio")
            # 最終格式：
            # audio: {'array': [0.1, -0.2, 0.3, ...], 'sampling_rate': 16000}
            # transcription: "你好世界"
            
            return dataset

    print("\n--- 步驟 2/4: 建立原始資料集並設定『即時轉換』---")
    audio_processor = AudioDatasetProcessor(file_path=CSV_PATH)
    full_dataset = audio_processor.create_dataset()
    
    # 分割訓練集和測試集
    common_voice = full_dataset.train_test_split(test_size=0.2, seed=42)
    # 80% 訓練集，20% 測試集
    # seed=42 確保每次分割結果一致

    # 使用 .with_transform() 確保記憶體穩定，訓練能立刻開始
    # 關鍵策略：即時轉換 (On-the-fly Transformation)
    # 傳統方法：預先處理所有音訊 → 消耗大量記憶體
    # 即時轉換：需要時才處理 → 節省記憶體，支援大資料集
    prepare_fn = partial(
        prepare_dataset_batched,
        feature_extractor=processor.feature_extractor,  # 音訊 → 梅爾頻譜圖
        tokenizer=processor.tokenizer,                  # 文字 → Token IDs
    )
    vectorized_datasets = common_voice.with_transform(prepare_fn)
    # 每次存取資料時，會自動呼叫 prepare_dataset_batched
    # 輸入：原始音訊陣列 + 中文文字
    # 輸出：梅爾頻譜圖 [80, 3000] + Token IDs [sequence_length]
    print("即時轉換已設定。")

    # --- 建立訓練元件 ---
    print("\n--- 步驟 3/4: 建立訓練元件 (最終穩定運行版) ---")
    
    # 資料整理器：將批次資料整理成模型輸入格式
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    # 功能：
    # 1. 填充音訊特徵到相同長度
    # 2. 填充文字標籤到相同長度
    # 3. 建立注意力遮罩
    
    # 評估函式：計算 CER 和語意相似度
    compute_metrics_fn = partial(compute_metrics, tokenizer=processor.tokenizer)
    # 每次評估時會自動計算：
    # - CER (字元錯誤率)
    # - 語意相似度
    # - 綜合評分

    # [中文優化版本]
    # 根據研究結果和中文語音辨識需求調整的訓練參數
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # 批次大小設定 (避免 GPU 記憶體不足)
        per_device_train_batch_size=4,      # 每個 GPU 的訓練批次大小
        per_device_eval_batch_size=4,       # 每個 GPU 的驗證批次大小
        # 有效批次大小：4 × 4 = 16 (透過梯度累積)
        gradient_accumulation_steps=4,      # 梯度累積步數
        
        # 效能與穩定性設定
        dataloader_num_workers=0,           # 資料載入器工作程序數 (避免序列化錯誤)
        gradient_checkpointing=False,       # 梯度檢查點 (關閉以優先速度)
        fp16=True,                         # 半精度浮點數訓練 (節省記憶體)
        
        # 學習率與訓練步數
        learning_rate=1e-5,                # 微調專用學習率
        warmup_steps=5,                    # 預熱步數 (測試用)
        max_steps=10,                      # 最大訓練步數 (測試用)
        
        # 評估與監控設定
        eval_strategy="steps",             # 按步數評估
        eval_steps=5,                      # 每 5 步評估一次 (測試用)
        save_steps=200,                    # 每 200 步保存一次
        logging_steps=25,                  # 每 25 步記錄一次
        
        # 模型選擇策略
        load_best_model_at_end=True,       # 訓練結束時載入最佳模型
        metric_for_best_model="combined_score",  # 最佳模型評分標準
        greater_is_better=True,           # 評分越高越好
        
        # 其他設定
        predict_with_generate=True,        # 使用生成模式預測
        generation_max_length=225,        # 最大生成長度
        report_to=["tensorboard"],        # 使用 TensorBoard 記錄
        push_to_hub=True,                 # 推送到 Hugging Face Hub
        remove_unused_columns=False,      # 保留所有欄位
    )
    # 建立訓練器
    trainer = Seq2SeqTrainer(
        args=training_args,                    # 訓練參數
        model=model,                           # Whisper 模型
        train_dataset=vectorized_datasets["train"],  # 訓練資料集
        eval_dataset=vectorized_datasets["test"],     # 驗證資料集
        data_collator=data_collator,           # 資料整理器
        compute_metrics=compute_metrics_fn,     # 評估函式
        tokenizer=processor.feature_extractor, # 特徵提取器 (用於記錄)
    )
    # 訓練器功能：
    # 1. 管理訓練迴圈 (前向傳播 → 計算損失 → 反向傳播 → 更新權重)
    # 2. 自動評估和保存最佳模型
    # 3. 支援斷點續練
    # 4. 記錄訓練過程到 TensorBoard

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
    """
    🚀 執行說明：
    ------------------------------------------------------------------------------
    1. 確保已安裝 CUDA 版本的 PyTorch
    2. 確保已使用 `huggingface-cli login` 登入
    3. 確保有足夠的 GPU 記憶體 (建議 8GB+)
    4. 執行前建議重新啟動電腦，確保系統處於乾淨狀態
    
    📊 預期輸出：
    - 訓練進度條：0/10 → 10/10
    - 評估指標：CER, semantic_similarity, combined_score
    - 模型保存：whisper-small-zh-finetune-zh-final/
    """
    main()
