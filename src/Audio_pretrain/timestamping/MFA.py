import gc
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Set

import opencc
import textgrid
from pypinyin import Style, load_phrases_dict, pinyin
from transformers import AutoTokenizer

# 假設這些從你的模組匯入
from Audio_pretrain.timestamping.emilia_unpackage import (
    iter_emilia_tar,
    save_debug_sample,
    transcode_mp3_to_wav_bytes,
    write_webdataset_sample,
)

# --- 設定區 ---
TAR_PATH = "/mnt/data_hdd/.cache/huggingface/datasets/Emilia/ZH/ZH-B000000.tar"
TEMP_DIR_ZH = "./mfa_outputs/temp/zh"
TEMP_DIR_EN = "./mfa_outputs/temp/en"
OUTPUT_DIR = "./mfa_outputs/results"
OUTPUT_TAR_ROOT = "/mnt/data_ssd/Emilia/ZH"  # 最終輸出 tar 的根目錄
CHECKPOINT_FILE = "./mfa_outputs/processed_ids.txt" # 紀錄已成功對齊的 ID
LOG_FILE = "./mfa_outputs/processing.log"
TARGET_SR = 16000
MAX_SAMPLES = 2000  # 增加樣本數以展示 Resume 威力
CUSTOM_PHRASES = {}
converter = opencc.OpenCC('s2twp.json')

# --- Logger 設定 ---
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Emilia_MFA")

# --- Tokenizer 初始化與原地替換 ---
def get_custom_tokenizer():
    tok = AutoTokenizer.from_pretrained("MediaTek-Research/Llama-Breeze2-3B-Instruct")
    return tok

tokenizer = get_custom_tokenizer()

def load_checkpoints(file_path: str) -> Set[str]:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_checkpoint(file_path: str, sample_id: str):
    with open(file_path, "a") as f:
        f.write(f"{sample_id}\n")

# ... [保留你的 clean_text 和 convert_to_mfa_pinyin 函數] ...

def clean_text(text):
    text = re.sub(r"[^\w\s\u4e00-\u9fa5]", " ", text)
    return " ".join(text.split())

def convert_to_mfa_pinyin(text):
    if CUSTOM_PHRASES: load_phrases_dict(CUSTOM_PHRASES)
    clean = clean_text(text)
    raw_pinyins = pinyin(clean, style=Style.TONE3)
    final_pinyins = []
    for p in raw_pinyins:
        if p[0] != ' ' and not p[0][-1].isdigit():
            final_pinyins.append(p[0] + '5')
        else:
            final_pinyins.append(p[0])
    return " ".join(final_pinyins)

def run_mfa(temp_dir: str, out_dir: str, model_name: str):
    logger.info(f"Step 2: 執行 MFA 對齊. 輸入: {temp_dir}")
    cmd = [
        "mfa", "align", str(temp_dir), model_name, model_name, str(out_dir),
        "--clean", "--beam", "40", "--retry-beam", "150", "--j", "4"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("MFA 對齊批次執行成功")
    except subprocess.CalledProcessError as e:
        logger.error(f"MFA 失敗: {e.stderr}")
        raise e

def textgrid_to_moshi_format(tg_path, frame_rate=12.5):
    # 使用 reserved_4 作為預設 pad (對應上面替換後的 <|pad|>)
    PAD_TOKEN = "<|reserved_special_token_4|>" 
    EPAD_TOKEN = "<|reserved_special_token_5|>"
    
    tg = textgrid.TextGrid.fromFile(tg_path)
    words_tier = tg[0]
    total_duration = tg.maxTime
    total_frames = int(total_duration * frame_rate)
    
    moshi_sequence = [PAD_TOKEN] * total_frames
    
    for interval in words_tier:
        start_frame = int(interval.minTime * frame_rate)
        end_frame = min(int(interval.maxTime * frame_rate), total_frames)
        text = interval.mark.strip()
        
        if text:
            # 第一幀放文字，之後放 <pad> (這裡原本初始化就是 pad，所以只需放文字)
            moshi_sequence[min(start_frame, total_frames-1)] = converter.convert(text)
            # 如果是停頓前，標記為 epad
            if start_frame > 0 and moshi_sequence[start_frame - 1] == PAD_TOKEN:
                moshi_sequence[start_frame - 1] = EPAD_TOKEN
                
    return moshi_sequence

def language_ratio(text):
    # 移除空白
    text = text.replace(" ", "")
    if not text: return "Empty"
    
    chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    total = len(text)
    cn_rate = chinese_chars / total
    en_rate = english_chars / total
    
    if cn_rate > 0.5:
        return "zh"
    elif en_rate > 0.8: # 英文通常包含空格與標點，比例設高一點
        return "en"
    return "hybrid"

def purge_mfa_working_dirs():
    """專門清理導致長跑崩潰的資料庫與暫存，保留模型檔案"""
    mfa_root = Path("~/Documents/MFA").expanduser()
    
    # 根據你的 ls 結果，這些是會產生 zh.db 與暫存的地方
    # 當你處理英文時，請把 "en" 也加入
    dirs_to_purge = ["zh", "en", "temp", "joblib_cache"]
    
    for d in dirs_to_purge:
        target = mfa_root / d
        if target.exists():
            try:
                shutil.rmtree(target)
                print(f"🧹 已清除對齊快取: {target.name}")
            except Exception as e:
                # 有時因為權限或鎖定會失敗，我們記錄下來
                print(f"⚠️ 無法清理 {target.name}: {e}")

    # 另外特別檢查 command_history 這種會緩慢增長的檔案
    history_file = mfa_root / "command_history.yaml"
    if history_file.exists():
        history_file.unlink()

if __name__ == "__main__":
    processed_ids = load_checkpoints(CHECKPOINT_FILE)
    logger.info(f"已從 Checkpoint 載入 {len(processed_ids)} 筆資料，將跳過已處理項。")
    
    
    all_files = Path("/mnt/data_hdd/.cache/huggingface/datasets/Emilia/ZH").glob("*.tar")
    for tar_path in all_files:
        if str(tar_path) in processed_ids:
            logger.info(f"跳過已處理檔案: {tar_path.name}")
            continue
        
        # 清除長期快取
        purge_mfa_working_dirs()
        
        logger.info(f"開始處理檔案: {tar_path.name}")
        pair_generator = iter_emilia_tar(tar_path=tar_path)
        
        # 建立臨時資料夾（僅存放本次需要處理的）
        if os.path.exists(TEMP_DIR_ZH): shutil.rmtree(TEMP_DIR_ZH)
        os.makedirs(TEMP_DIR_ZH, exist_ok=True)
        if os.path.exists(TEMP_DIR_EN): shutil.rmtree(TEMP_DIR_EN)
        os.makedirs(TEMP_DIR_EN , exist_ok=True)
        if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        current_batch_ids = {
            "zh": [],
            "en": [],
        }
        
        
        file_counter = 0
        # --- Step 1: 收集尚未處理的數據 ---
        for idx, (sample_id, mp3_bytes, json_bytes) in enumerate(pair_generator):
            # Debugging : use fewer samples
            # if (len(current_batch_ids["zh"]) + len(current_batch_ids["en"])) >= MAX_SAMPLES:
            #     break

            # logger.info(f"正在預處理: {sample_id}")
            wav_bytes, duration = transcode_mp3_to_wav_bytes(mp3_bytes, TARGET_SR)
            metadata = json.loads(json_bytes.decode("utf-8"))

            metadata["text"] = clean_text(metadata["text"])
            lang = language_ratio(metadata["text"])
            metadata["language"] = lang
            metadata["duration"] = duration
            speaker = metadata["speaker"]

            
            if lang == "zh":
                current_batch_ids["zh"].append(sample_id)
                save_debug_sample(os.path.join(TEMP_DIR_ZH, speaker), sample_id, wav_bytes, metadata, mode="mfa")
            elif lang == "en":
                current_batch_ids["en"].append(sample_id)
                save_debug_sample(os.path.join(TEMP_DIR_EN, speaker), sample_id, wav_bytes, metadata, mode="mfa")
            file_counter += 1
            # --- Step 2: 執行 MFA ---
            if file_counter>= MAX_SAMPLES:
                if Path(TEMP_DIR_ZH).exists() and any(Path(TEMP_DIR_ZH).iterdir()):
                    run_mfa(TEMP_DIR_ZH, OUTPUT_DIR, model_name="mandarin_mfa")
                if Path(TEMP_DIR_EN).exists() and any(Path(TEMP_DIR_EN).iterdir()):
                    run_mfa(TEMP_DIR_EN, OUTPUT_DIR, model_name="english_us_arpa")
                
                # 更新快取資料夾（僅存放本次需要處理的）
                if os.path.exists(TEMP_DIR_ZH): shutil.rmtree(TEMP_DIR_ZH)
                os.makedirs(TEMP_DIR_ZH, exist_ok=True)
                if os.path.exists(TEMP_DIR_EN): shutil.rmtree(TEMP_DIR_EN)
                os.makedirs(TEMP_DIR_EN , exist_ok=True)
                
                # 重新初始化counter
                file_counter = 0

        # --- Step 2: 執行 MFA ---
        if Path(TEMP_DIR_ZH).exists() and any(Path(TEMP_DIR_ZH).iterdir()):
            run_mfa(TEMP_DIR_ZH, OUTPUT_DIR, model_name="mandarin_mfa")
        if Path(TEMP_DIR_EN).exists() and any(Path(TEMP_DIR_EN).iterdir()):
            run_mfa(TEMP_DIR_EN, OUTPUT_DIR, model_name="english_us_arpa")
        
        moshi_tokens ={}
        # --- Step 3: 轉換與 Tokenize ---
        merge_ids = current_batch_ids["zh"] + current_batch_ids["en"]
        for sample_id in merge_ids:
            speaker = sample_id[:len(sample_id)-len(sample_id.split("_")[-1])-1]  # 假設 ID 格式為 speaker_XXX
            textgrid_root = os.path.join(OUTPUT_DIR, speaker)
            textgrid_path = os.path.join(textgrid_root, f"{sample_id}.TextGrid")
            
            if not os.path.exists(textgrid_path):
                logger.warning(f"跳過 {sample_id}: MFA 未生成 TextGrid (可能對齊失敗)")
                continue

            # logger.info(f"正在轉換格式並儲存結果: {sample_id}")
            moshi_sequence = textgrid_to_moshi_format(textgrid_path)
            
            token_list = []
            # 使用你的長度補償邏輯
            pad_remove_counter = 0
            for token in moshi_sequence:
                if pad_remove_counter > 0:
                    pad_remove_counter -= 1
                    continue
                # add_special_tokens=False 確保不會噴出 BOS [128000]
                ids = tokenizer.encode(token, add_special_tokens=False)
                pad_remove_counter = len(ids) - 1
                token_list.extend(ids)

            # 驗證與存檔邏輯
            try:
                assert len(token_list) == len(moshi_sequence), f"長度不匹配 {len(token_list)} vs {len(moshi_sequence)}"
                # 這裡可以加入你寫入 WebDataset 或 Final JSON 的邏輯
            except AssertionError as e:
                logger.error(f"{sample_id} 處理失敗: {str(e)}")
                token_list = []  # 或者其他的 fallback 策略
            moshi_tokens[sample_id] = token_list
                
        logger.info(f"完成處理檔案: {tar_path.name}，成功對齊 {len(merge_ids)} 筆資料。")
        
        
        # --- Step 4: Save as tar file ---
        pair_generator = iter_emilia_tar(tar_path=tar_path)
        output_tar_path = os.path.join(OUTPUT_TAR_ROOT, f"{tar_path.stem}.tar")
        with tarfile.open(output_tar_path, "w") as tar_handle:
            for idx, (sample_id, mp3_bytes, json_bytes) in enumerate(pair_generator):
                if sample_id not in merge_ids:
                    # logger.info(f"跳過 {sample_id}，因為它不在本批次的對齊結果中。")
                    continue
                
                # 1. 轉碼 MP3 為 WAV bytes
                wav_bytes, duration = transcode_mp3_to_wav_bytes(mp3_bytes, TARGET_SR)

                # 2. 解析 JSON Metadata
                metadata = json.loads(json_bytes.decode("utf-8"))
                metadata["duration"] = duration  # 可以選擇將音訊長度加入 Metadata
                if sample_id in current_batch_ids["zh"]:
                    metadata["language"] = "zh"
                elif sample_id in current_batch_ids["en"]:
                    metadata["language"] = "en"
                else:
                    continue  # 理論上不應該發生，因為前面已經過濾過了
                metadata["padded_tokens"] = moshi_tokens.get(sample_id, [])
                if not metadata["padded_tokens"]:
                    logger.warning(f"{sample_id} 沒有成功生成 padded_tokens，將在後續檢查中被標記。")
                    continue
                # else:
                #     logger.info(f"{sample_id} 成功生成 padded_tokens，將被包含在輸出中。")
                
                # 3. [Streaming] 寫入 WebDataset Tar (如果需要)
                write_webdataset_sample(tar_handle, sample_id, wav_bytes, metadata)
        save_checkpoint(CHECKPOINT_FILE, str(tar_path))
        gc.collect()  # 嘗試釋放記憶體，尤其是在處理大型檔案後
    logger.info("本批次任務執行完畢。")