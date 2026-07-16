import os
import json
import argparse
import json5
import re
import sys
import traceback
from typing import Any, Dict
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print
from pipline_framework import (
    denoise_patch_inference_tool,
    template_match_tool,
    stem2cif_tool,
    property_prediction_tool
)
import warnings
from scipy.optimize import OptimizeWarning
warnings.filterwarnings("ignore", category=OptimizeWarning)


# -----------------------------------------------------------------------
# 全局默认（可被 main 覆盖）
# -----------------------------------------------------------------------
_PIPELINE_DEFAULTS: Dict[str, str] = {
    "weight_path": os.environ.get("AUTOMAT_WEIGHT_PATH", "MOE_model_weights/moe_model.ckpt"),
    "label_dir": os.environ.get("AUTOMAT_LABEL_DIR", "data_generation/label"),
    "metadata_csv": os.environ.get("AUTOMAT_METADATA_CSV", "baseline/property.csv"),
    "work_root": os.environ.get("AUTOMAT_WORK_ROOT", "harness_runs/current"),
}


# ========== 工具注册 ==========
@register_tool('denoise_patch_inference_tool')
class DenoisePatchTool(BaseTool):
    description = '对STEM大图进行去噪和patch重建，返回重建图片路径'
    parameters = [
        {'name': 'image_path', 'type': 'string', 'description': 'STEM大图路径', 'required': True},
        {'name': 'weight_path', 'type': 'string', 'description': '去噪模型权重路径', 'required': True},
        {'name': 'work_root', 'type': 'string', 'description': '工作目录', 'required': True},
        {'name': 'device', 'type': 'string', 'description': '推理设备', 'required': False, 'default': 'cuda'}
    ]
    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        result = denoise_patch_inference_tool(**args)
        return json5.dumps(result, ensure_ascii=False)

@register_tool('template_match_tool')
class TemplateMatchTool(BaseTool):
    description = '对去噪后图片做模板匹配，返回最佳label路径和元素信息'
    parameters = [
        {'name': 'recon_png', 'type': 'string', 'description': '去噪后图片路径', 'required': True},
        {'name': 'label_dir', 'type': 'string', 'description': '模板匹配label目录', 'required': True},
        {'name': 'metadata_csv', 'type': 'string', 'description': '材料元素元数据CSV', 'required': True},
        {'name': 'user_message', 'type': 'string', 'description': '用户补充说明', 'required': True},
        {'name': 'work_root', 'type': 'string', 'description': '工作目录', 'required': True}
    ]
    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        result = template_match_tool(**args)
        return json5.dumps(result, ensure_ascii=False)

@register_tool('stem2cif_tool')
class Stem2CifTool(BaseTool):
    description = '将label图片和元素类型转换为CIF结构，返回CIF路径'
    parameters = [
        {'name': 'label_path', 'type': 'string', 'description': 'label图片路径', 'required': True},
        {'name': 'elements', 'type': 'array', 'description': '元素类型列表', 'required': True},
        {'name': 'work_root', 'type': 'string', 'description': '工作目录', 'required': True},
        {'name': 'max_atoms', 'type': 'integer', 'description': '最大原子数', 'required': False, 'default': 50},
        {'name': 'max_shrink_iter', 'type': 'integer', 'description': '最大缩减迭代次数', 'required': False, 'default': 4}
    ]
    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        result = stem2cif_tool(**args)
        return json5.dumps(result, ensure_ascii=False)

@register_tool('property_prediction_tool')
class PropertyPredictionTool(BaseTool):
    description = '对CIF结构进行物性预测，返回能量、力、应力等'
    parameters = [
        {'name': 'cif_path', 'type': 'string', 'description': 'CIF结构路径', 'required': True},
        {'name': 'work_root', 'type': 'string', 'description': '工作目录', 'required': True},
        {'name': 'noise_amp', 'type': 'number', 'description': '扰动幅度', 'required': False, 'default': 0.05},
        {'name': 'relax_steps', 'type': 'integer', 'description': '松弛步数', 'required': False, 'default': 500},
        {'name': 'device', 'type': 'string', 'description': '推理设备', 'required': False, 'default': 'cuda'}
    ]
    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        result = property_prediction_tool(**args)
        return json5.dumps(result, ensure_ascii=False)

# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 多轮对话Agent（Qwen-Agent范式）")
    parser.add_argument('--model', type=str, default='qwen-plus-2025-04-28')
    parser.add_argument('--model_server', type=str, default='http://localhost:8000/v1')
    parser.add_argument(
        '--api_key',
        type=str,
        default=os.environ.get('AUTOMAT_LLM_API_KEY') or os.environ.get('DASHSCOPE_API_KEY'),
        help='LLM API key; prefer AUTOMAT_LLM_API_KEY or DASHSCOPE_API_KEY.',
    )
    parser.add_argument('--weight_path', type=str, default=_PIPELINE_DEFAULTS["weight_path"])
    parser.add_argument('--label_dir', type=str, default=_PIPELINE_DEFAULTS["label_dir"])
    parser.add_argument('--metadata_csv', type=str, default=_PIPELINE_DEFAULTS["metadata_csv"])
    parser.add_argument('--work_root', type=str, default=_PIPELINE_DEFAULTS["work_root"], help="一次性 pipeline 的结果保存目录")
    args = parser.parse_args()

    llm_cfg = {
        'model': args.model,
        # 'model_server': args.model_server,
        'api_key': args.api_key,
        'generate_cfg': {
            'top_p': 0.8,
            # This parameter will affect the tool-call parsing logic. Default is False:
            # Set to True: when content is `<think>this is the thought</think>this is the answer`
            # Set to False: when response consists of reasoning_content and content
            # 'thought_in_content': True,
            
            # tool-call template: default is nous (recommended for qwen3):
            # 'fncall_prompt_type': 'nous'
            
            # Maximum input length, messages will be truncated if they exceed this length, please adjust according to model API:
            # 'max_input_tokens': 58000
        }
    }

    system_instruction = (
        "你是材料科学智能Agent，能够对STEM表征图像进行结构重建与物性分析。"
        "你可以调用如下工具完成去噪、模板匹配、结构重建、物性预测等任务。"
        "每次用户输入后，你应根据需求判断是否需要调用工具，"
        "并在需要时自动选择合适的工具和参数。"
        "你可以多轮对话，支持用户补充说明、追问、结果解释等。"
        f"全局参数：weight_path={args.weight_path}，label_dir={args.label_dir}，metadata_csv={args.metadata_csv}，work_root={args.work_root}"
    )

    tools = [
        'denoise_patch_inference_tool',
        'template_match_tool',
        'stem2cif_tool',
        'property_prediction_tool'
    ]

    bot = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        function_list=tools
    )

    # Step 4: Run the agent as a chatbot.
    messages = []
    print("🔹 Qwen2.5-VL Agent 启动，输入 exit/quit 退出。")
    
    while True:
        # For example, enter the query "draw a dog and rotate it 90 degrees".
        query = input('\nuser query: ')
        # Append the user query to the chat history.
        messages.append({'role': 'user', 'content': query})
        response = []
        response_plain_text = ''
        print('bot response:')
        for response in bot.run(messages=messages):
            # Streaming output.
            response_plain_text = typewriter_print(response, response_plain_text)
        
        # Append the bot responses to the chat history.
        messages.extend(response)


if __name__ == "__main__":
    main()
