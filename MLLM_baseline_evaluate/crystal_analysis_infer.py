#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# prepare: pip install openai>=1.3.0 dashscope zhipuai pandas tqdm
# Configure provider credentials through environment variables.

import os
import re
import json
import base64
import mimetypes
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import concurrent.futures
from tqdm import tqdm
import time

import pandas as pd

# ─────────────────────────── 第三方 SDK ──────────────────────────── #
# try:
#     import openai
# except ImportError:
#     openai = None

try:
    from dashscope import MultiModalConversation
except ImportError:
    MultiModalConversation = None  # noqa: N816

# try:
#     from zhipuai import ZhipuAI
# except ImportError:
#     ZhipuAI = None
# ------------------------------------------------------------------ #

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)


# ╔═══════════════════════════ 工具函数 ════════════════════════════╗
def get_image_files(directory: str, exts=(".png", ".jpg", ".jpeg", ".bmp", ".gif")) -> List[Path]:
    dir_path = Path(directory)
    if not dir_path.exists():
        logging.error("图像目录 %s 不存在", directory)
        return []
    if not dir_path.is_dir():
        logging.error("%s 不是有效目录", directory)
        return []
    return [dir_path / f for f in os.listdir(directory) if f.lower().endswith(exts)]


def require_api_key(name: str, default: Optional[str] = None) -> str:
    key = os.getenv(name, default)
    if not key:
        logging.critical("Environment variable %s not set!", name)
        raise SystemExit(1)
    return key


def guess_mime_type(fp: Path) -> str:
    mt, _ = mimetypes.guess_type(fp.name)
    return mt or "application/octet-stream"


def encode_image_to_base64(fp: Path) -> str:
    return base64.b64encode(fp.read_bytes()).decode("utf-8")


def extract_material_id(image_path: Path) -> Optional[str]:
    """从图片文件名中提取material_id（orthogonal_和_supercell之间的字符串）"""
    pattern = r"orthogonal_(.*?)_supercell_.*\.jpg"
    match = re.search(pattern, image_path.name)
    if match:
        return match.group(1)
    #logging.warning("无法从文件名 %s 中提取material_id", image_path.name)
    return None


def load_materials_properties(csv_path: str) -> pd.DataFrame:
    """加载材料属性CSV文件"""
    try:
        df = pd.read_csv(csv_path)
        logging.info("成功加载材料属性数据，共 %d 条记录", len(df))
        return df
    except Exception as e:
        logging.error("加载CSV文件 %s 失败: %s", csv_path, e)
        return pd.DataFrame()


def get_material_elements(materials_df: pd.DataFrame, material_id: str) -> Optional[str]:
    """从材料数据框中获取指定material_id的elements信息"""
    if materials_df.empty:
        return None
    
    try:
        row = materials_df[materials_df['material_id'] == material_id]
        if not row.empty:
            return row['elements'].iloc[0]
        logging.warning("在CSV中未找到material_id: %s", material_id)
        return None
    except Exception as e:
        logging.error("获取elements信息失败: %s", e)
        return None


# ╔══════════════════════════─ API 调用封装 ─═══════════════════════╗
def call_dashscope(api_key: str, model: str, image_fp: Path, prompt: str) -> Optional[str]:
    if MultiModalConversation is None:  # local env 没装 dashscope
        logging.error("dashscope SDK not installed")
        return None
    image_uri = f"file://{image_fp.resolve()}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"image": image_uri},
                {"text": prompt.rstrip()},
            ],
        },
    ]
    try:
        rsp: Dict[str, Any] = MultiModalConversation.call(
            api_key=api_key, model=model, messages=messages
        )
        content = (
            rsp.get("output", {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", [{}])
        )
        # content 可能是 list[dict] 或 str
        if isinstance(content, list):
            return content[0].get("text")
        return content
    except Exception as exc:
        logging.error("DashScope error: %s", exc)
        return None


def call_openai(api_key: str, model: str, image_fp: Path, prompt: str) -> Optional[str]:
    if openai is None:
        logging.error("openai SDK not installed")
        return None
    openai.api_key = api_key
    mime = guess_mime_type(image_fp)
    image_b64 = encode_image_to_base64(image_fp)
    image_url = f"data:{mime};base64,{image_b64}"

    try:
        # 兼容 openai==1.x 与 0.x
        if hasattr(openai, "chat"):  # >=1.0.0
            chat = openai.chat.completions.create
            rsp = chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt.rstrip()},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
            )
            return rsp.choices[0].message.content
        # fallback 到 0.x
        rsp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt.rstrip()},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        )
        return rsp.choices[0].message["content"]
    except Exception as exc:
        logging.error("OpenAI error: %s", exc)
        return None


def call_zhipuai(api_key: str, model: str, image_fp: Path, prompt: str) -> Optional[str]:
    if ZhipuAI is None:
        logging.error("ZhipuAI SDK not installed")
        return None
    mime = guess_mime_type(image_fp)
    image_b64 = encode_image_to_base64(image_fp)
    url = f"data:{mime};base64,{image_b64}"

    try:
        client = ZhipuAI(api_key=api_key)
        rsp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": prompt.rstrip()},
                    ],
                }
            ],
        )
        msg = rsp.choices[0].message
        # 新版 SDK ⇒ msg.content，旧版 ⇒ msg.text
        return getattr(msg, "content", None) or getattr(msg, "text", None)
    except Exception as exc:
        logging.error("ZhipuAI error: %s", exc)
        return None


# ╔═══════════════════════ 解析 JSON 输出 ═════════════════════════╗
_JSON_RE = re.compile(r"\{.*}", re.S)


def parse_model_output(text: Optional[str]) -> Tuple[Optional[str], ...]:
    """始终返回 (crystal_type, energy_above_hull, formation_energy, band_gap, error_type)"""
    if not text:
        return None, None, None, None, "empty_output"

    m = _JSON_RE.search(text)
    if not m:
        return None, None, None, None, "json_not_found"

    try:
        data = json.loads(m.group(0))
        return (
            data.get("crystal_type"),
            data.get("energy_above_hull"),
            data.get("formation_energy"),
            data.get("band_gap"),
            None,  # error_type
        )
    except json.JSONDecodeError:
        return None, None, None, None, "json_decode_error"


# ╔══════════════════════ 主批处理逻辑 ════════════════════════════╗
def process_single_image(cfg, img, prompt, materials_df=None):
    model, platform, api_key = cfg["name"], cfg["platform"].lower(), cfg["api_key"]
    api_key = require_api_key(api_key) if not api_key.startswith("${") else api_key[2:-1]
    
    # 获取元素信息并添加到提示中
    enriched_prompt = prompt
    if materials_df is not None:
        material_id = extract_material_id(img)
        if material_id:
            elements = get_material_elements(materials_df, material_id)
            if elements:
                enriched_prompt = f"晶体元素: {elements}\n\n{prompt}"
                logging.info("   为 %s 添加了元素信息: %s", img.name, elements)
    
    raw = None
    if platform == "dashscope":
        raw = call_dashscope(api_key, model, img, enriched_prompt)
    elif platform == "openai":
        raw = call_openai(api_key, model, img, enriched_prompt)
    elif platform == "zhipuai":
        raw = call_zhipuai(api_key, model, img, enriched_prompt)
    else:
        logging.error("未知平台 %s", platform)
        return None
        
    ct, e_hull, f_energy, gap, err = parse_model_output(raw)
    return dict(
        model=model,
        platform=platform,
        image_file=str(img),
        material_id=extract_material_id(img),
        crystal_type=ct,
        energy_above_hull=e_hull,
        formation_energy=f_energy,
        band_gap=gap,
        error_type=err,
        raw_output=raw,
    )


def test_models_on_images(
    image_dir: str,
    prompt: str,
    models_cfg: List[Dict[str, str]],
    materials_csv_path: Optional[str] = None,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    imgs = get_image_files(image_dir)
    if not imgs:
        logging.warning("目录 %s 中未找到图像", image_dir)
        return []

    # 加载材料属性数据
    materials_df = None
    if materials_csv_path:
        materials_df = load_materials_properties(materials_csv_path)
        if materials_df.empty:
            logging.warning("未能加载材料属性数据，将使用原始提示")

    results: List[Dict[str, Any]] = []
    start_time = time.time()
    
    tasks = []
    for cfg in models_cfg:
        model, platform = cfg["name"], cfg["platform"].lower()
        logging.info("▶ 测试模型=%s  平台=%s", model, platform)
        
        for img in imgs:
            tasks.append((cfg, img, prompt, materials_df))
    
    total_tasks = len(tasks)
    logging.info("共有 %d 个测试任务 (模型数=%d × 图像数=%d)", 
                total_tasks, len(models_cfg), len(imgs))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, *task) for task in tasks]
        
        for future in tqdm(
            concurrent.futures.as_completed(futures), 
            total=len(futures),
            desc="处理进度"
        ):
            result = future.result()
            if result:
                results.append(result)
    
    elapsed = time.time() - start_time
    logging.info("所有任务完成，耗时: %.2f 秒，平均每个任务: %.2f 秒", 
                elapsed, elapsed/total_tasks if total_tasks else 0)
    
    return results


def save_results_csv(rows: List[Dict[str, Any]], csv_fp: str = "crystal_analysis_results.csv") -> None:
    if not rows:
        logging.warning("No rows to save.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(csv_fp, index=False, encoding="utf-8")
    logging.info("Results saved to %s (%d rows)", csv_fp, len(df))


# ╔══════════════════════════─ main ─═════════════════════════════╗
if __name__ == "__main__":
    IMAGE_DIR = "./evaluate_example/img"  # 修改为实际图像目录
    MATERIALS_CSV = "./evaluate_example/materials_properties.csv"  # 材料属性CSV
    
    PROMPT = (
        "你将收到一张晶体结构图像，请分析该结构的晶体类型，并根据图像信息估算下列参数：\n"
        "- crystal_type\n- energy_above_hull (eV/atom)\n- formation_energy (eV/atom)\n- band_gap (eV)\n"
        "请仅以最小化后的 JSON 格式返回，不要包含解释性文字。"
    )

    MODELS = [
        {
            "name": "qwen2.5-vl-7b-instruct",  # 修改为实际使用的模型
            "platform": "dashscope",
            "api_key": "DASHSCOPE_API_KEY"
        },
        # {
        #     "name": "gpt-4o",
        #     "platform": "openai",
        #     "api_key": "OPENAI_API_KEY",
        # },
        # {
        #     "name": "glm-4v",
        #     "platform": "zhipuai",
        #     "api_key": "ZHIPU_API_KEY",
        # },
    ]
    
    # 设置并行线程数，可以根据需要调整
    MAX_WORKERS = 4
    
    all_rows = test_models_on_images(IMAGE_DIR, PROMPT, MODELS, 
                                     materials_csv_path=MATERIALS_CSV,
                                     max_workers=MAX_WORKERS)
    save_results_csv(all_rows)
