# Basic tools
import io
import json
import os
import tarfile
from typing import Any, Dict, Generator, Tuple

import opencc

# Torch and audio processing related
import torchaudio
from torchaudio.functional import resample

# --- 設定區 ---
TAR_PATH = "/mnt/data_hdd/.cache/huggingface/datasets/Emilia/ZH/ZH-B000000.tar"  # please change to your Emilia tar path
OUTPUT_DIR = "./debug_dataset"                 # 輸出 debug 資料的資料夾
TARGET_SR = 16000                              # 目標採樣率 (配合 EnCodec/Moshi)
MAX_SAMPLES = 10                               # 只抓 20 筆來測試
converter = opencc.OpenCC('s2twp.json')          # 簡體轉繁體的轉換器 (如果需要)
# ----------------

def transcode_mp3_to_wav_bytes(
    mp3_data: bytes, 
    target_sr: int = 24000
) -> Tuple[bytes, float]:
    """
    將原始 MP3 的 bytes 資料轉碼為指定採樣率的 WAV bytes。
    全程在記憶體中運作，不涉及硬碟 I/O。

    Args:
        mp3_data (bytes): MP3 檔案的二進位內容。
        target_sr (int): 目標採樣率 (預設 24000)。

    Returns:
        wav_bytes (bytes): 轉碼後的 WAV 二進位內容。
        duration (float): 音訊長度(秒)。
    """
    # 1. 使用 BytesIO 將 bytes 偽裝成檔案物件供 torchaudio 讀取
    with io.BytesIO(mp3_data) as input_io:
        waveform, sample_rate = torchaudio.load(input_io, format="mp3")

    # 2. 轉為單聲道 (若為立體聲)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 3. 重採樣 (Resample)
    if sample_rate != target_sr:
        waveform = resample(waveform, sample_rate, target_sr)

    # 4. 將 Tensor 寫入 BytesIO buffer (模擬存檔)
    output_buffer = io.BytesIO()
    torchaudio.save(output_buffer, waveform, target_sr, format="wav", bits_per_sample=16, encoding="PCM_S")
    
    # 5. 獲取 bytes 並重置游標
    output_buffer.seek(0)
    wav_bytes = output_buffer.read()
    
    duration = waveform.shape[1] / target_sr
    return wav_bytes, duration

def iter_emilia_tar(tar_path: str) -> Generator[Tuple[str, bytes, bytes], None, None]:
    """
    生成器 (Generator)：讀取 Emilia 原始 Tar 檔，自動配對 MP3 與 JSON。
    
    Yields:
        sample_id (str): 檔案名稱 ID (不含副檔名)。
        mp3_bytes (bytes): MP3 的內容。
        json_bytes (bytes): JSON 的內容。
    """
    data_buffer = {}
    
    # 使用 "r|*" 模式進行串流讀取，即使 Tar 檔很大也不會卡記憶體
    with tarfile.open(tar_path, "r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue

            file_name = os.path.basename(member.name)
            base_name, ext = os.path.splitext(file_name)

            f = tar.extractfile(member)
            if f is None: continue
            content = f.read()

            # 暫存至 buffer
            if base_name not in data_buffer:
                data_buffer[base_name] = {}

            if ext == ".mp3":
                data_buffer[base_name]["mp3"] = content
            elif ext == ".json":
                data_buffer[base_name]["json"] = content

            # 檢查配對是否完成
            if "mp3" in data_buffer[base_name] and "json" in data_buffer[base_name]:
                yield (
                    base_name, 
                    data_buffer[base_name]["mp3"], 
                    data_buffer[base_name]["json"]
                )
                # 處理完畢後釋放記憶體
                del data_buffer[base_name]


def save_debug_sample(
    output_dir: str, 
    sample_id: str, 
    wav_bytes: bytes, 
    metadata: Dict[str, Any],
    mode : str | None = None,
) -> None:
    """
    [Debug Mode] 將單筆樣本寫入硬碟 (散檔)。
    通常用於檢查音質或 Metadata 正確性。
    """
    if mode == "mfa":   
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.join(output_dir, sample_id)
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. 寫入 WAV 實體檔案
    wav_path = os.path.join(output_dir, f"{sample_id}.wav")
    with open(wav_path, "wb") as f:
        f.write(wav_bytes)

    # 2. 寫入 JSON 實體檔案
    if mode == "mfa":
        lab_path = os.path.join(output_dir, f"{sample_id}.lab")
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(metadata["text"])  # 如果需要，轉為繁體中文
    else:
        json_path = os.path.join(output_dir, f"{sample_id}.json")
        metadata["text"] = converter.convert(metadata["text"])  # 如果需要，轉為繁體中文
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


def write_webdataset_sample(
    tar_handle: tarfile.TarFile, 
    sample_id: str, 
    wav_bytes: bytes, 
    metadata: Dict[str, Any]
) -> None:
    """
    [Streaming Mode] 將單筆樣本寫入已開啟的 Tar 檔 (WebDataset 格式)。
    適合大規模訓練資料集製作。
    """
    # 1. 準備 WAV 的 TarInfo
    wav_info = tarfile.TarInfo(name=f"{sample_id}.wav")
    wav_info.size = len(wav_bytes)
    tar_handle.addfile(wav_info, io.BytesIO(wav_bytes))

    # 2. 準備 JSON 的 bytes
    json_str = json.dumps(metadata, ensure_ascii=False)
    json_bytes = json_str.encode("utf-8")

    # 3. 準備 JSON 的 TarInfo
    json_info = tarfile.TarInfo(name=f"{sample_id}.json")
    json_info.size = len(json_bytes)
    tar_handle.addfile(json_info, io.BytesIO(json_bytes))

if __name__ == "__main__":
    pair_generator = iter_emilia_tar(TAR_PATH)
    for idx, (sample_id, mp3_bytes, json_bytes) in enumerate(pair_generator):
        if idx >= MAX_SAMPLES:
            break

        # 1. 轉碼 MP3 為 WAV bytes
        wav_bytes, duration = transcode_mp3_to_wav_bytes(mp3_bytes, TARGET_SR)

        # 2. 解析 JSON Metadata
        metadata = json.loads(json_bytes.decode("utf-8"))
        metadata["duration"] = duration  # 可以選擇將音訊長度加入 Metadata

        # 3. [Debug] 寫入實體檔案以供檢查
        save_debug_sample(OUTPUT_DIR, sample_id, wav_bytes, metadata)

        # 4. [Streaming] 寫入 WebDataset Tar (如果需要)
        # with tarfile.open("output_webdataset.tar", "w") as tar_handle:
        #     write_webdataset_sample(tar_handle, sample_id, wav_bytes, metadata)