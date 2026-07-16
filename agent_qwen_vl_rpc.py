from __future__ import annotations

import os
import json
import argparse
import json5
import re
import sys
import traceback
import inspect
from pathlib import Path
from typing import Any, Dict

from agent_qwen import AgentQwenHarness, default_registry, normalize_harness_request
from agent_qwen.defaults import PIPELINE_DEFAULTS
from agent_qwen.runtime_skills import SkillContext, dumps_tool_result

_AUTOMAT_ROOT = Path(__file__).resolve().parent
_AUTOMAT_SRC = _AUTOMAT_ROOT / "src"
if str(_AUTOMAT_SRC) not in sys.path:
    sys.path.insert(0, str(_AUTOMAT_SRC))
_AUTOMAT_PREPROCESS = _AUTOMAT_SRC / "preprocess_model"
if str(_AUTOMAT_PREPROCESS) not in sys.path:
    sys.path.insert(0, str(_AUTOMAT_PREPROCESS))
_AUTOMAT_SR = _AUTOMAT_PREPROCESS / "SR_model"
if str(_AUTOMAT_SR) not in sys.path:
    sys.path.insert(0, str(_AUTOMAT_SR))

# -----------------------------------------------------------------------
# 关键：避免 pipline_framework 或其依赖在 import 时抢先 parse_args()
# 做法：临时清空 argv，再导入，随后恢复 argv
# -----------------------------------------------------------------------
_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]

try:
    from qwen_agent.agents import Assistant
    from qwen_agent.tools.base import BaseTool, register_tool
    from qwen_agent.utils.output_beautify import typewriter_print
except ModuleNotFoundError:
    Assistant = None
    typewriter_print = None

    class BaseTool:
        pass

    def register_tool(_name):
        def decorator(cls):
            return cls
        return decorator

try:
    import pipline_framework as _pipeline_framework

    denoise_patch_inference_tool = getattr(_pipeline_framework, "denoise_patch_inference_tool", None)
    assess_image_path = getattr(_pipeline_framework, "assess_image_path", None)
    template_match_tool = getattr(_pipeline_framework, "template_match_tool", None)
    stem2cif_tool = getattr(_pipeline_framework, "stem2cif_tool", None)
    reconstruct_from_denoised_img = getattr(_pipeline_framework, "reconstruct_from_denoised_img", None)
    property_prediction_tool = getattr(_pipeline_framework, "property_prediction_tool", None)
    _PIPELINE_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    denoise_patch_inference_tool = None
    assess_image_path = None
    template_match_tool = None
    stem2cif_tool = None
    reconstruct_from_denoised_img = None
    property_prediction_tool = None
    _PIPELINE_IMPORT_ERROR = exc

# 恢复 argv，保证我们自己的 argparse 能看到 --rpc 等参数
sys.argv = _ORIG_ARGV

import warnings
from scipy.optimize import OptimizeWarning
warnings.filterwarnings("ignore", category=OptimizeWarning)


def _missing_pipeline_payload() -> str:
    return json5.dumps(
        {
            "success": False,
            "error": (
                "pipline_framework dependencies are not importable in this environment: "
                f"{_PIPELINE_IMPORT_ERROR!r}"
            ),
        },
        ensure_ascii=False,
    )


def _filter_call_kwargs(func, params: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return params
    return {key: value for key, value in params.items() if key in signature.parameters}


def _message_with_elements(user_message: str, elements: list[str] | None) -> str:
    if not elements:
        return user_message
    if "元素" in user_message or "element" in user_message.lower():
        return user_message
    return f"{user_message} 元素: {' '.join(elements)}".strip()


# -----------------------------------------------------------------------
# 全局默认（可被 main 覆盖）
# -----------------------------------------------------------------------
_PIPELINE_DEFAULTS: Dict[str, str] = {
    "weight_path": PIPELINE_DEFAULTS["weight_path"],
    "label_dir": PIPELINE_DEFAULTS["label_dir"],
    "metadata_csv": PIPELINE_DEFAULTS["metadata_csv"],
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
        if denoise_patch_inference_tool is None:
            return _missing_pipeline_payload()
        args = json5.loads(params)
        result = denoise_patch_inference_tool(**args)
        return json5.dumps(result, ensure_ascii=False)


# @register_tool('template_match_tool')
# class TemplateMatchTool(BaseTool):
#     description = '对去噪后图片做模板匹配，返回最佳label路径和元素信息'
#     parameters = [
#         {'name': 'recon_png', 'type': 'string', 'description': '去噪后图片路径', 'required': True},
#         {'name': 'label_dir', 'type': 'string', 'description': '模板匹配label目录', 'required': True},
#         {'name': 'metadata_csv', 'type': 'string', 'description': '材料元素元数据CSV', 'required': True},
#         {'name': 'user_message', 'type': 'string', 'description': '用户补充说明', 'required': True},
#         {'name': 'work_root', 'type': 'string', 'description': '工作目录', 'required': True}
#     ]
#     def call(self, params: str, **kwargs) -> str:
#         args = json5.loads(params)
#         result = template_match_tool(**args)
#         return json5.dumps(result, ensure_ascii=False)


@register_tool('template_match_tool')
class TemplateMatchTool(BaseTool):
    # 修改描述，提示 LLM 传入元素
    description = '对去噪后图片做模板匹配，返回最佳label路径和元素信息。如果已知元素类型，请务必传入 elements 参数。'
    
    parameters = [
        {'name': 'recon_png', 'type': 'string', 'description': '去噪后图片路径', 'required': True},
        {'name': 'label_dir', 'type': 'string', 'description': '模板匹配label目录', 'required': True},
        {'name': 'metadata_csv', 'type': 'string', 'description': '材料元素元数据CSV', 'required': True},
        {'name': 'user_message', 'type': 'string', 'description': '用户补充说明', 'required': True},
        {'name': 'work_root', 'type': 'string', 'description': '工作目录', 'required': True},
        # === 新增：显式暴露 elements 参数给 LLM ===
        {'name': 'elements', 'type': 'array', 'description': '元素列表，例如 ["Si", "O"]', 'required': False}
    ]

    def call(self, params: str, **kwargs) -> str:
        if template_match_tool is None:
            return _missing_pipeline_payload()
        args = json5.loads(params)
        args["user_message"] = _message_with_elements(args.get("user_message", ""), args.get("elements"))
        result = template_match_tool(**_filter_call_kwargs(template_match_tool, args))
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
        if stem2cif_tool is None:
            return _missing_pipeline_payload()
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
        if property_prediction_tool is None:
            return _missing_pipeline_payload()
        args = json5.loads(params)
        result = property_prediction_tool(**args)
        return json5.dumps(result, ensure_ascii=False)


@register_tool('direct_reconstruct_tool')
class DirectReconstructTool(BaseTool):
    description = '绕过模板匹配，使用FFT→候选窗口→最小单胞重建的直通管线；输入去噪后图像+元素(+可选配位)'
    parameters = [
        {'name': 'denoised_img', 'type': 'string', 'description': '去噪后的STEM图像路径', 'required': True},
        {'name': 'elements', 'type': 'array', 'description': '元素列表，如 ["Zr","N","Cl"]', 'required': True},
        {'name': 'coord', 'type': 'object', 'description': '可选配位关系，如 {"Zr":1,"N":1,"Cl":1}', 'required': False},
        {'name': 'pixel_size', 'type': 'number', 'description': '像素尺寸(Å/px)，影响FFT物理单位', 'required': False, 'default': 0.10},
        {'name': 'top_n', 'type': 'integer', 'description': '候选窗口数量(top-N)', 'required': False, 'default': 3},
        {'name': 'out_dir', 'type': 'string', 'description': '输出目录', 'required': False, 'default': 'pipeline_out'}
    ]
    def call(self, params: str, **kwargs) -> str:
        if reconstruct_from_denoised_img is None:
            return _missing_pipeline_payload()
        args = json5.loads(params)
        denoised_img = args['denoised_img']
        elements = args['elements']
        print(elements)
        coord = args.get('coord')
        pixel_size = args.get('pixel_size', 0.10)
        top_n = int(args.get('top_n', 3))
        out_dir = args.get('out_dir', 'pipeline_out')
        final_cif, cell, basis_atoms = reconstruct_from_denoised_img(
            denoised_img=denoised_img,
            user_elements=elements,
            coord=coord,
            pixel_size=pixel_size,
            top_n=top_n,
            out_dir=out_dir
        )
        result = {
            'final_cif': final_cif,
            'cell': cell,
            'basis_atoms': basis_atoms
        }
        return json5.dumps(result, ensure_ascii=False)


@register_tool('assess_image_path_tool')
class AssessImagePathTool(BaseTool):
    description = '评估图像路径的有效性和图像质量，返回评估结果'
    parameters = [
        {'name': 'path', 'type': 'string', 'description': '图像路径', 'required': True},
    ]
    def call(self, params: str, **kwargs) -> str:
        if assess_image_path is None:
            return _missing_pipeline_payload()
        args = json5.loads(params)
        result = assess_image_path(**args)
        return json5.dumps(result, ensure_ascii=False)


@register_tool('confidence_estimation_tool')
class ConfidenceEstimationTool(BaseTool):
    description = '对去噪STEM图像进行原子/晶格置信度估计，输出不确定性量化JSON报告'
    parameters = [
        {'name': 'image_path', 'type': 'string', 'description': '去噪后STEM图像路径', 'required': True},
        {'name': 'elements', 'type': 'array', 'description': '元素列表，例如 ["Mo", "S"]', 'required': True},
        {'name': 'pixel_size', 'type': 'number', 'description': '像素尺寸 Å/pixel', 'required': False, 'default': 0.10},
        {'name': 'output', 'type': 'string', 'description': '输出JSON路径', 'required': False},
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        result = default_registry().run("confidence_estimation_tool", SkillContext(), args)
        return dumps_tool_result(result)


def _extract_artifacts_from_text(text: str) -> Dict[str, Any]:
    """
    尽量不改变你现有 pipeline：这里做一个轻量的“路径提取”，方便 meta-agent 串联后续 agent。
    只要 assistant 输出里出现了 .cif/.png/.jpg 等路径，就收集到 artifacts。
    """
    if not text:
        return {}

    # 简单路径正则：支持绝对路径和相对路径
    path_pat = r'((?:/[^ \n\r\t]+)|(?:\./[^ \n\r\t]+)|(?:[^ \n\r\t]+))'
    exts = ('.cif', '.vasp', '.poscar', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.xyz', '.json', '.csv')

    candidates = re.findall(path_pat, text)
    found = []
    for c in candidates:
        for e in exts:
            if c.lower().endswith(e):
                found.append(c)
                break

    # 去重保序
    seen = set()
    uniq = []
    for x in found:
        if x not in seen:
            uniq.append(x)
            seen.add(x)

    return {"files": uniq} if uniq else {}


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 多轮对话Agent（Qwen-Agent范式）")

    # 保留你原有参数
    parser.add_argument('--model', type=str, default='qwen-plus-2025-04-28')
    parser.add_argument('--model_server', type=str, default='http://localhost:8000/v1')
    parser.add_argument(
        '--api_key',
        type=str,
        default=os.environ.get('AUTOMAT_LLM_API_KEY') or os.environ.get('DASHSCOPE_API_KEY'),
        help='LLM API key (prefer AUTOMAT_LLM_API_KEY or DASHSCOPE_API_KEY).',
    )
    parser.add_argument('--weight_path', type=str, default=_PIPELINE_DEFAULTS["weight_path"])
    parser.add_argument('--label_dir', type=str, default=_PIPELINE_DEFAULTS["label_dir"])
    parser.add_argument('--metadata_csv', type=str, default=_PIPELINE_DEFAULTS["metadata_csv"])
    parser.add_argument('--work_root', type=str, default=PIPELINE_DEFAULTS["work_root"],
                        help="一次性 pipeline 的结果保存目录")

    # RPC 开关：与 echo_agent_entry.py 一致（stdin JSON -> stdout JSON -> exit）
    parser.add_argument('--rpc', action='store_true', help="RPC mode: read one JSON request from stdin and output JSON to stdout, then exit.")
    parser.add_argument('--list_skills', action='store_true', help="列出 agent_qwen harness runtime skills 后退出。")
    parser.add_argument('--list_lines', action='store_true', help="列出 agent_qwen harness lines 后退出。")
    parser.add_argument('--harness', action='store_true', help="直接使用 agent_qwen harness line，而不是进入 LLM agent。")
    parser.add_argument('--harness_line', '--workflow', dest='harness_line', choices=('classic', 'direct', 'compare'), default='classic', help="harness line 类型。")
    parser.add_argument('--dry_run', action='store_true', help="harness dry-run，不加载模型/不调用 MatterSim。")
    parser.add_argument('--skip_property', action='store_true', help="harness line 跳过 MatterSim 物性预测。")
    parser.add_argument('--run_confidence', action='store_true', help="harness line 增加 uncertainty/confidence 估计。")
    parser.add_argument('--image', type=str, default=None, help="classic/compare line 输入 STEM 原图。")
    parser.add_argument('--denoised_img', type=str, default=None, help="direct line 输入去噪图。")
    parser.add_argument('--elements', nargs='*', default=None, help="harness 已知元素列表。")

    # 用 parse_known_args 更稳，避免上游残留参数影响
    args, _unknown = parser.parse_known_args()

    if args.list_skills:
        print(json.dumps(default_registry().describe(), ensure_ascii=False, indent=2))
        return
    if args.list_lines:
        print(json.dumps(AgentQwenHarness.from_defaults().list_lines(), ensure_ascii=False, indent=2))
        return

    # 直接 harness 模式：把旧流程显式拆成 denoise/template/stem2cif/property/confidence runtime tools。
    # RPC 下可从 stdin JSON 的 context 中传 image_path / denoised_img / elements / harness_line。
    rpc_mode = bool(args.rpc) or (not sys.stdin.isatty())
    if args.harness:
        req: Dict[str, Any] = {}
        if rpc_mode:
            line = sys.stdin.readline()
            req = json.loads(line) if line else {}
        context_rpc = req.get("context", {}) if isinstance(req, dict) else {}
        query_rpc = req.get("query", "") if isinstance(req, dict) else ""
        merged_context = dict(context_rpc)
        if args.image and not merged_context.get("image_path") and not merged_context.get("image"):
            merged_context["image_path"] = args.image
        if args.denoised_img and not merged_context.get("denoised_img"):
            merged_context["denoised_img"] = args.denoised_img
        if args.elements and not merged_context.get("elements"):
            merged_context["elements"] = args.elements
        intake = normalize_harness_request(
            query_rpc,
            merged_context,
            default_workflow=args.harness_line,
            default_dry_run=args.dry_run,
            default_skip_property=args.skip_property,
            default_run_confidence=args.run_confidence,
        )
        if not intake.ready:
            resp = {
                "agent_id": "characterization_recon_agent",
                **intake.need_input_payload(),
            }
            print(json.dumps(resp, ensure_ascii=False, default=str))
            return
        harness = AgentQwenHarness.from_defaults(
            work_root=intake.work_root or args.work_root,
            weight_path=intake.weight_path or args.weight_path,
            label_dir=intake.label_dir or args.label_dir,
            metadata_csv=intake.metadata_csv or args.metadata_csv,
            device=intake.device,
            dry_run=intake.dry_run,
        )
        if intake.harness_line == "direct":
            result = harness.run_line(
                "direct",
                denoised_img=intake.denoised_img or "",
                elements=intake.elements,
                coord=intake.coord,
                run_property=not intake.skip_property,
                run_confidence=intake.run_confidence,
            )
            resp = {"ok": result.ok, "agent_id": "characterization_recon_agent", "intake": intake.as_context(), "result": result.__dict__}
        elif intake.harness_line == "compare":
            result = harness.run_line(
                "compare",
                image_path=intake.image_path or "",
                user_message=intake.user_message,
                elements=intake.elements,
                run_property=not intake.skip_property,
                run_confidence=intake.run_confidence,
            )
            resp = {"ok": result.ok, "agent_id": "characterization_recon_agent", "intake": intake.as_context(), "result": result.__dict__}
        else:
            result = harness.run_line(
                "classic",
                image_path=intake.image_path or "",
                user_message=intake.user_message,
                elements=intake.elements,
                run_property=not intake.skip_property,
                run_confidence=intake.run_confidence,
            )
            resp = {"ok": result.ok, "agent_id": "characterization_recon_agent", "intake": intake.as_context(), "result": result.__dict__}
        print(json.dumps(resp, ensure_ascii=False, default=str))
        return

    if Assistant is None:
        resp = {
            "ok": False,
            "agent_id": "characterization_recon_agent",
            "error": "qwen_agent is not installed; use --harness for workflow execution or install qwen_agent for LLM mode.",
        }
        if bool(args.rpc) or (not sys.stdin.isatty()):
            print(json.dumps(resp, ensure_ascii=False))
            return
        raise ModuleNotFoundError(resp["error"])

    llm_cfg = {
        'model': args.model,
        # 如果你要走本地 OpenAI-compat，就打开这一行，并确保 meta/sub 一致
        # 'model_server': args.model_server,
        'api_key': args.api_key,
        'generate_cfg': {'top_p': 0.8},
    }

    system_instruction = (
        "你是材料科学智能Agent，能够对STEM表征图像进行结构重建与物性分析。"
        "你可以调用如下工具,包括图像评估、去噪、模板匹配、结构重建、物性预测等工具。"
        "当用户提供去噪后图像与元素信息时，你可以直接根据情况选择不同方式进行重建；"
        "若需要先验或验证，也可选择模板匹配以辅助确定元素与结构先验；若无先验结构候选或者重建后的质量足够好时，也可直接进行最小单胞结构重建。"
        "在需要量化不确定性时，可以调用 confidence_estimation_tool 输出原子级与晶格级置信度报告。"
        "每次用户输入后，你应根据需求自主选择是否以及如何组合调用工具，"
        "并在需要时自动选择合适的工具和参数。"
        "允许多轮对话，支持用户补充说明、追问、结果解释等。"
        f"全局参数：weight_path={args.weight_path}，label_dir={args.label_dir}，metadata_csv={args.metadata_csv}，work_root={args.work_root}"
    )

    tools = [
        'denoise_patch_inference_tool',
        'assess_image_path_tool',
        'template_match_tool',
        'stem2cif_tool',
        'property_prediction_tool',
        'direct_reconstruct_tool',
        'confidence_estimation_tool'
    ]

    bot = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        function_list=tools
    )

    # -------------------------------------------------------------------
    # RPC 模式：与 echo_agent_entry.py 的调用协议一致
    # 额外做一层鲁棒：若 stdin 不是 tty（通常意味着被 meta 调用），也强制进入 RPC
    # -------------------------------------------------------------------
    if rpc_mode:
        resp = None
        try:
            line = sys.stdin.readline()
            if not line:
                resp = {
                    "ok": False,
                    "agent_id": "characterization_recon_agent",
                    "error": "empty stdin: no JSON request received"
                }
            else:
                req = json.loads(line)
                query_rpc = req.get("query", "")
                context_rpc = req.get("context", {})
                session_id = req.get("session_id", "")

                # 把 context 作为“结构化前置信息”注入给 LLM，让它进一步拆分任务并决定调用工具
                if context_rpc:
                    query_text = (
                        "CONTEXT_JSON:\n"
                        + json.dumps(context_rpc, ensure_ascii=False, indent=2)
                        + "\n\nTASK:\n"
                        + query_rpc
                    )
                else:
                    query_text = query_rpc

                messages = [{'role': 'user', 'content': query_text}]
                response = []
                for response in bot.run(messages=messages):
                    # RPC 严禁向 stdout 打流式内容
                    pass
                messages.extend(response)

                last_text = ""
                for m in reversed(messages):
                    if m.get("role") == "assistant":
                        last_text = m.get("content", "")
                        break

                artifacts = _extract_artifacts_from_text(last_text)

                resp = {
                    "ok": True,
                    "agent_id": "characterization_recon_agent",
                    "session_id": session_id,
                    "result": {
                        "summary": last_text,
                        "artifacts": artifacts
                    }
                }

        except Exception as e:
            resp = {
                "ok": False,
                "agent_id": "characterization_recon_agent",
                "error": repr(e),
                "traceback": traceback.format_exc()[:4000]
            }

        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return

    # -------------------------------------------------------------------
    # 交互模式：保持你原来的用户交互形式
    # -------------------------------------------------------------------
    messages = []
    print("🔹 Qwen2.5-VL Agent 启动，输入 exit/quit 退出。")
    while True:
        query = input('\nuser query: ').strip()
        if query in {"exit", "quit"}:
            break

        messages.append({'role': 'user', 'content': query})
        response = []
        response_plain_text = ''
        print('bot response:')
        for response in bot.run(messages=messages):
            response_plain_text = typewriter_print(response, response_plain_text)

        messages.extend(response)


if __name__ == "__main__":
    main()
