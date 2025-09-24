# ==============================================================================
# 檔案：compare_models.py
# 描述：載入微調後的模型與原始模型，並對同一段音訊進行辨識以比較效果。
# ==============================================================================
import torch
from datasets import Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def compare_whisper_models(
    fine_tuned_model_path, original_model_name, test_audio_path, language="zh"
):
    """
    比較微調模型和原始模型的辨識結果。

    Args:
        fine_tuned_model_path (str): 您本地訓練好的模型資料夾路徑。
        original_model_name (str): Hugging Face Hub 上的原始模型名稱。
        test_audio_path (str): 用於測試的音訊檔案路徑。
        language (str): 目標語言。
    """
    # 檢查是否有可用的 GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"將使用設備：{device}")

    # --- 1. 載入微調後的模型和對應的 Processor ---
    print(f"\n--- 正在載入您的微調模型自：{fine_tuned_model_path} ---")
    fine_tuned_processor = WhisperProcessor.from_pretrained(fine_tuned_model_path)
    fine_tuned_model = WhisperForConditionalGeneration.from_pretrained(
        fine_tuned_model_path
    ).to(device)

    # --- 2. 載入原始的 OpenAI 模型和對應的 Processor ---
    print(f"\n--- 正在載入原始模型：{original_model_name} ---")
    original_processor = WhisperProcessor.from_pretrained(original_model_name)
    original_model = WhisperForConditionalGeneration.from_pretrained(
        original_model_name
    ).to(device)

    # --- 3. 準備測試音訊 ---
    print(f"\n--- 正在載入並處理測試音訊：{test_audio_path} ---")
    # 使用 datasets.Audio 來載入並重新取樣到 16kHz
    audio_dataset = Audio(sampling_rate=16000)
    audio_input = audio_dataset.decode_example(
        audio_dataset.encode_example(test_audio_path)
    )

    # 使用您的微調模型的 Processor 來準備輸入特徵
    # (兩個模型的 Processor 其實是一樣的，用哪個都可以)
    input_features = fine_tuned_processor(
        audio_input["array"],
        sampling_rate=audio_input["sampling_rate"],
        return_tensors="pt",
    ).input_features.to(device)

    # --- 4. 進行推論 (Inference) ---
    print("\n--- 正在進行辨識，請稍候... ---")

    # 使用微調模型進行預測
    predicted_ids_fine_tuned = fine_tuned_model.generate(
        input_features, language="zh", task="transcribe"
    )
    transcription_fine_tuned = fine_tuned_processor.batch_decode(
        predicted_ids_fine_tuned, skip_special_tokens=True
    )[0]

    # 使用原始模型進行預測
    predicted_ids_original = original_model.generate(input_features)
    transcription_original = original_processor.batch_decode(
        predicted_ids_original, skip_special_tokens=True
    )[0]

    # --- 5. 打印比較結果 ---
    print("\n================== 辨識結果比較 ==================")
    print(f"原始 Whisper-small 模型辨識結果：")
    print(f"==> {transcription_original}\n")

    print(f"您的微調模型辨識結果：")
    print(f"==> {transcription_fine_tuned}\n")

    # print(f"原本的中文:")
    # print("你怎麼這麼厲害")
    print("======================================================")


if __name__ == "__main__":
    # --- 請修改以下參數 ---

    # 您訓練好的模型的本地資料夾路徑
    # 這應該是您在 TrainingArguments 中設定的 output_dir
    MY_FINE_TUNED_MODEL_PATH = (
        "shaobai880824/whisper-small-zh-finetune-zh-final"  # 或者 prototype 版本
    )

    # 原始模型的名稱
    ORIGINAL_MODEL_NAME = "openai/whisper-small"

    # 您用來測試的音訊檔案路徑
    TEST_AUDIO_FILE = (
        r"test_wav\audio_571349684816969848.m4a"  # 請替換成您自己的檔案名
    )

    # 執行比較
    compare_whisper_models(
        fine_tuned_model_path=MY_FINE_TUNED_MODEL_PATH,
        original_model_name=ORIGINAL_MODEL_NAME,
        test_audio_path=TEST_AUDIO_FILE,
    )
