# main_with_vocal_isolation.py
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal

import imageio_ffmpeg
import pandas as pd
from pydub import AudioSegment
from pytube import YouTube

# 核心處理套件
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

# 設定 pydub 使用 imageio-ffmpeg 下載的 ffmpeg 可執行檔
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()


# --- 資料類別定義 (Data Class Definition) ---
@dataclass
class ClipperConfig:
    youtube_url: str
    output_dir: str = "output_from_subtitles"
    language_codes: List[str] = field(
        default_factory=lambda: ["zh-TW", "zh-Hant", "zh", "zh-Hans"]
    )
    output_format: Literal["audio", "video"] = "audio"
    perform_vocal_isolation: bool = True  # <<< 新增：是否執行人聲分離的開關


@dataclass
class ClippedSegment:
    index: int
    text: str
    start_time: float
    end_time: float
    duration: float
    original_clip_path: str  # <<< 修改：保留原始片段路徑
    processed_clip_path: str  # <<< 新增：指向最終處理過的檔案路徑（可能是純化後的人聲）


# --- 核心處理類別 (Core Processor Class) ---
class YouTubeSubtitleClipper:
    def __init__(self, config: ClipperConfig):
        self.config = config
        self.output_path = Path(self.config.output_dir)
        self.source_media_path: Path | None = None
        # 輸出到指定子資料夾
        self.clips_path = self.output_path / "@1_original_clips"
        self.isolated_path = self.output_path / "@2_isolated_vocals"
        self.demucs_output_path = self.output_path / "temp_demucs_output"
        self.results: List[ClippedSegment] = []
        self.csv_rows: List[Dict[str, str]] = []  # 供輸出符合規格的 CSV
        self.video_id: str | None = None

    def _ensure_dir_exists(self) -> None:
        self.output_path.mkdir(exist_ok=True)
        self.clips_path.mkdir(exist_ok=True)
        if self.config.perform_vocal_isolation:
            self.isolated_path.mkdir(exist_ok=True)
            self.demucs_output_path.mkdir(exist_ok=True)
        print(f"輸出目錄結構已建立於: {self.output_path}")

    # <<< 新增：執行人聲分離的函式
    def _isolate_vocals(self, audio_path: Path):
        """
        使用 Demucs 對指定的音訊檔案進行人聲分離。
        返回 (vocals_path, denoised_vocals_path)。任一失敗則為 None。
        """
        print(f"  - [AI] 正在分離人聲: {audio_path.name}")
        try:
            # 動態偵測 GPU，並告訴 Demucs 使用對應裝置
            try:
                import torch  # 延遲載入，避免非必要依賴

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

            # Demucs 預設會在輸出目錄中建立一個名為 'htdemucs' 的子目錄
            model_name = "htdemucs"  # 這是 Demucs v4 的預設模型名稱
            expected_output_dir = self.demucs_output_path / model_name / audio_path.stem

            # 呼叫 Demucs 命令列工具
            command = [
                sys.executable,
                "-m",
                "demucs.separate",
                "-d",
                device,
                # 使用 4 stems（vocals、drums、bass、other）以獲得更乾淨的人聲
                "-o",
                str(self.demucs_output_path),
                str(audio_path),
            ]

            # 使用 subprocess.run，並捕獲輸出
            result = subprocess.run(command, check=True, capture_output=True, text=True)

            # 找到分離出的人聲檔案 (vocals.wav)
            vocals_file = expected_output_dir / "vocals.wav"
            if vocals_file.exists():
                # 將檔案移動並重新命名到我們的最終目錄
                final_vocal_path = self.isolated_path / f"{audio_path.stem}_vocals.wav"
                shutil.move(str(vocals_file), str(final_vocal_path))
                print(f"  - [成功] 人聲已儲存至: {final_vocal_path.name}")
                # 不進行降噪以節省計算時間
                return final_vocal_path, None
            else:
                print(f"  - [錯誤] Demucs 執行完畢，但找不到 vocals.wav 檔案。")
                print(f"  - Demucs stdout: {result.stdout}")
                print(f"  - Demucs stderr: {result.stderr}")
                return None, None
        except subprocess.CalledProcessError as e:
            print(f"  - [錯誤] Demucs 執行失敗。返回碼: {e.returncode}")
            print(f"  - stdout: {e.stdout}")
            print(f"  - stderr: {e.stderr}")
            return None, None
        except Exception as e:
            print(f"  - [錯誤] 執行人聲分離時發生未知錯誤: {e}")
            return None, None
        finally:
            # 清理 Demucs 產生的暫存資料夾
            if expected_output_dir.exists():
                shutil.rmtree(expected_output_dir.parent)

    def _slice_wav_ffmpeg(
        self, input_path: Path, start_s: float, duration_s: float, output_path: Path
    ) -> bool:
        """使用 ffmpeg 快速切片為 wav（重編碼為 pcm_s16le）。"""
        try:
            ffmpeg_exe = AudioSegment.converter
            cmd = [
                ffmpeg_exe,
                "-y",
                "-ss",
                f"{start_s:.3f}",
                "-t",
                f"{duration_s:.3f}",
                "-i",
                str(input_path),
                "-acodec",
                "pcm_s16le",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  - [切片失敗] ffmpeg stderr:\n{e.stderr}")
            return False
        except Exception as e:
            print(f"  - [切片例外] {e}")
            return False

    def run(self) -> None:
        # ... (_get_transcript, _download_media 函式與之前相同，此處省略)
        # 完整的 _get_transcript, _download_media 函式請參考上一版本的回答
        self._ensure_dir_exists()
        transcript = self._get_transcript()
        if not transcript:
            return
        if not self._download_media() or not self.source_media_path:
            return

        print(f"\n--- 開始根據 {len(transcript)} 句字幕進行切割（優化版） ---")

        # 只讀取一次音訊長度供邊界檢查
        main_audio = AudioSegment.from_file(self.source_media_path)
        total_ms = len(main_audio)

        # 若開啟人聲分離，先對整段音訊做一次分離（同時得到原始人聲與降噪人聲）
        vocals_full = None
        vocals_full_denoised = None  # 已關閉降噪
        if self.config.perform_vocal_isolation:
            v_path, d_path = self._isolate_vocals(self.source_media_path)
            vocals_full = v_path
            vocals_full_denoised = d_path
            if v_path is None and d_path is None:
                print("⚠️ 人聲分離失敗，改用原始音訊切片。")

        created_count = 0
        skipped_indices: List[int] = []

        # 平行切片以加速 CPU 使用率（ffmpeg 多工）
        max_workers = max(2, min(8, (os.cpu_count() or 2)))
        futures = []
        per_index: Dict[int, Dict[str, str]] = {}

        def submit_job(input_path: Path, out_path: Path):
            return executor.submit(
                self._slice_wav_ffmpeg, input_path, start_time, duration_sec, out_path
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, entry in enumerate(transcript):
                start_time = float(entry["start"])  # 秒
                duration = float(entry["duration"])  # 秒
                text = entry["text"]
                end_time = start_time + duration

                start_ms, end_ms = int(start_time * 1000), int(end_time * 1000)
                if start_ms >= total_ms:
                    skipped_indices.append(i)
                    continue
                end_ms = min(end_ms, total_ms)
                if end_ms <= start_ms:
                    skipped_indices.append(i)
                    continue

                duration_sec = (end_ms - start_ms) / 1000.0
                per_index[i] = {
                    "漢字": text,
                    "原始切片": "",
                    "分離人聲": "",
                    "降噪": "",
                }

                # 原始切片（加入 video_id 以避免不同影片重名）
                suffix = f"{self.video_id or 'video'}_{i:04d}.wav"
                orig_out = self.clips_path / f"clip_{suffix}"
                futures.append(
                    (
                        i,
                        "原始切片",
                        submit_job(self.source_media_path, orig_out),
                        orig_out,
                    )
                )

                # 分離人聲切片（未降噪）
                if vocals_full is not None:
                    iso_out = (
                        self.isolated_path
                        / f"clip_{self.video_id or 'video'}_{i:04d}_vocals.wav"
                    )
                    futures.append(
                        (i, "分離人聲", submit_job(vocals_full, iso_out), iso_out)
                    )

                # 已停用降噪切片以節省時間

            for i, key, fut, out_path in futures:
                ok = fut.result()
                if ok:
                    per_index[i][key] = str(Path(out_path).resolve())
                    if key == "原始切片":
                        created_count += 1

            for i in sorted(per_index.keys()):
                row = per_index[i]
                self.csv_rows.append(row)

        self._export_to_csv()
        # 匹配性檢查
        if created_count == len(transcript):
            print(f"\n✅ 所有字幕均已對應到音檔：{created_count}/{len(transcript)}")
        else:
            print(
                f"\n⚠️ 有部分字幕未能對應到音檔：{created_count}/{len(transcript)}，跳過索引：{skipped_indices}"
            )

    # ... (_get_transcript, _download_media, _export_to_csv 函式與之前相同)
    # 為了簡潔，此處省略，請從上一版本複製過來
    def _get_transcript(self) -> List[Dict[str, Any]] | None:
        try:
            video_id = self.config.youtube_url.split("v=")[1].split("&")[0]
            self.video_id = video_id
            print(f"正在為影片 ID: {video_id} 尋找字幕...")
            ytt_api = YouTubeTranscriptApi()
            fetched_transcript = ytt_api.fetch(
                video_id, languages=self.config.language_codes
            )
            transcript = fetched_transcript.to_raw_data()
            print(f"成功獲取 {len(transcript)} 句字幕。")
            return transcript
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            print(f"錯誤：無法獲取此影片的字幕。 ({e})")
            return None
        except Exception as e:
            print(f"獲取字幕時發生未知錯誤: {e}")
            return None

    def _download_media(self) -> bool:
        try:
            print(f"正在從 YouTube 下載來源媒體: {self.config.youtube_url}")

            # 僅保留 v 參數，避免 playlist 或其他參數造成 400
            url = self.config.youtube_url
            if "v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
                url = f"https://www.youtube.com/watch?v={video_id}"

            # 使用 video_id 命名，避免多影片彼此覆寫
            vid = self.video_id or (
                url.split("v=")[1].split("&")[0] if "v=" in url else "video"
            )
            base_pattern = f"source_{vid}"
            # 預設目標（若實際副檔名不同，稍後會自動偵測）
            self.source_media_path = self.output_path / f"{base_pattern}.m4a"
            all_existing = list(self.output_path.glob(f"{base_pattern}.*"))
            # 過濾掉臨時 .part 檔
            existing = [p for p in all_existing if not p.name.endswith(".part")]
            if existing:
                # 選擇最新修改時間的檔案
                existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                self.source_media_path = existing[0]
                print(f"來源媒體檔案已存在: {self.source_media_path}，跳過下載。")
                return True
            # 如果只有 .part 檔，刪除後重新下載
            part_files = [p for p in all_existing if p.name.endswith(".part")]
            if part_files:
                for p in part_files:
                    try:
                        p.unlink()
                        print(f"已移除不完整檔案: {p}")
                    except Exception:
                        pass

            # 方案 A: 先嘗試 pytube 純音訊
            try:
                yt = YouTube(url)
                stream = (
                    yt.streams.filter(only_audio=True, mime_type="audio/mp4").first()
                    or yt.streams.get_audio_only()
                )
                if stream is None:
                    raise RuntimeError("找不到可用的音訊串流。")
                tmp_name = "source_audio_tmp"
                stream.download(output_path=str(self.output_path), filename=tmp_name)
                # 推測副檔名
                guessed_ext = (
                    stream.mime_type.split("/")[-1] if stream.mime_type else "m4a"
                )
                tmp_path = self.output_path / f"{tmp_name}.{guessed_ext}"
                if tmp_path.exists():
                    tmp_path.replace(self.source_media_path)
                print(f"來源媒體已成功下載至: {self.source_media_path}")
                return True
            except Exception as e:
                print(f"pytube 下載失敗，改用 yt-dlp 後備方案。原因: {e}")

            # 方案 B: 後備使用 yt-dlp 下載最佳音訊（多組參數逐一嘗試）
            ytdlp_attempts = [
                [
                    sys.executable,
                    "-m",
                    "yt_dlp",
                    "--no-part",
                    "--continue",
                    "--no-playlist",
                    "-f",
                    "ba[ext=m4a]/bestaudio/best",
                    "-o",
                    str(self.output_path / f"{base_pattern}.%(ext)s"),
                    url,
                ],
                [
                    sys.executable,
                    "-m",
                    "yt_dlp",
                    "--no-part",
                    "--continue",
                    "--no-playlist",
                    "--force-ipv4",
                    "--geo-bypass",
                    "-f",
                    "ba[ext=m4a]/bestaudio/best",
                    "-o",
                    str(self.output_path / f"{base_pattern}.%(ext)s"),
                    url,
                ],
                [
                    sys.executable,
                    "-m",
                    "yt_dlp",
                    "--no-part",
                    "--continue",
                    "--no-playlist",
                    "--extractor-args",
                    "youtube:player_client=android",
                    "-f",
                    "ba[ext=m4a]/bestaudio/best",
                    "-o",
                    str(self.output_path / f"{base_pattern}.%(ext)s"),
                    url,
                ],
                [
                    sys.executable,
                    "-m",
                    "yt_dlp",
                    "--no-part",
                    "--continue",
                    "--no-playlist",
                    "--cookies-from-browser",
                    "chrome",
                    "-f",
                    "ba[ext=m4a]/bestaudio/best",
                    "-o",
                    str(self.output_path / f"{base_pattern}.%(ext)s"),
                    url,
                ],
                [
                    sys.executable,
                    "-m",
                    "yt_dlp",
                    "--no-part",
                    "--continue",
                    "--no-playlist",
                    "--cookies-from-browser",
                    "edge",
                    "-f",
                    "ba[ext=m4a]/bestaudio/best",
                    "-o",
                    str(self.output_path / f"{base_pattern}.%(ext)s"),
                    url,
                ],
            ]

            for idx, cmd in enumerate(ytdlp_attempts, start=1):
                try:
                    result = subprocess.run(
                        cmd, check=True, capture_output=True, text=True
                    )
                    downloaded = [
                        p
                        for p in self.output_path.glob(f"{base_pattern}.*")
                        if not p.name.endswith(".part")
                    ]
                    if not downloaded:
                        raise RuntimeError("yt-dlp 執行成功但未找到輸出檔案。")
                    self.source_media_path = downloaded[0]
                    print(
                        f"來源媒體已成功下載至: {self.source_media_path} (yt-dlp 嘗試 #{idx})"
                    )
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"yt-dlp 嘗試 #{idx} 失敗，stderr:\n{e.stderr}\n---")
                except Exception as e:
                    print(f"yt-dlp 嘗試 #{idx} 例外：{e}\n---")

            return False
        except Exception as e:
            print(f"下載媒體時發生錯誤: {e}")
            return False

    def _export_to_csv(self) -> None:
        # 依需求輸出：欄位為「漢字」、「原始切片」、「分離人聲」、「降噪」（降噪目前停用，欄位保留為空）
        if not self.csv_rows:
            print("沒有可匯出的資料。")
            return
        df = pd.DataFrame(
            self.csv_rows, columns=["漢字", "原始切片", "分離人聲", "降噪"]
        )
        csv_path = self.output_path / "clips_mapping.csv"
        write_header = not csv_path.exists()
        # 以附加模式寫入，避免覆蓋既有內容
        df.to_csv(
            csv_path, mode="a", header=write_header, index=False, encoding="utf-8-sig"
        )
        print(f"\n處理完成！已追加寫入至: {csv_path}")


# --- 程式進入點 (Entry Point) ---
if __name__ == "__main__":
    TARGET_URL = "https://www.youtube.com/watch?v=7A984ezu26s&list=PL_OjiqlIsqlK1NlhFzQ8ycOLPj0NWHDK1&index=29&ab_channel=%E6%B0%91%E8%A6%96%E6%88%B2%E5%8A%87%E9%A4%A8FormosaTVDramas"

    config = ClipperConfig(
        youtube_url=TARGET_URL,
        output_dir="youtube_clips_isolated",
        output_format="audio",  # 人聲分離只對 audio 有意義
        perform_vocal_isolation=True,  # <<< 設定為 True 來啟動 AI 去背景音
    )

    # 警告使用者 CPU 會很慢
    if config.perform_vocal_isolation:
        try:
            import torch

            if not torch.cuda.is_available():
                print("\n" + "=" * 50)
                print("警告：未偵測到 CUDA GPU。Demucs 將在 CPU 上執行。")
                print("      處理速度將會非常非常緩慢。")
                print("=" * 50 + "\n")
        except ImportError:
            pass  # 如果連 torch 都沒有，demucs-cpu 會處理

    clipper = YouTubeSubtitleClipper(config)
    clipper.run()
