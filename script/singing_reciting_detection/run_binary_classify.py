#!/usr/bin/env python3
# pyright: reportUnusedCallResult=false, reportImplicitStringConcatenation=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportAny=false
"""
语音评论「唱歌/朗诵」二分类批量评测脚本
- 每次请求处理 20 条数据（可通过 --batch-size 调整）
- 支持续传（断点续传），中断后重新运行自动跳过已处理行
- 运行结束后自动计算正确率并生成 diff 文件
- 纯标准库，无需安装第三方依赖

使用方法:
  python3 binary_classify.py
  python3 binary_classify.py --batch-size 10
"""

import csv
import json
import os
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
import pathlib

def _load_env():
    """从 .env 文件加载环境变量（无需 python-dotenv）"""
    env_path = pathlib.Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()

# ============ 配置 ============
API_KEY = os.environ.get("WQ_API_KEY", "")
ENDPOINT_ID = "ep-d77edo-1782128122468111756"
BASE_URL = "https://wanqing-api.corp.kuaishou.com/api/gateway/v1"
BATCH_SIZE = 20
MAX_RETRIES = 5
RETRY_DELAY = 5

# 默认路径
DEFAULT_PROMPT = "/Users/para_fish66/Desktop/语音评论/prompt/singing_reciting_detection/版本6.txt"
DEFAULT_INPUT = "/Users/para_fish66/Desktop/语音评论/data/testing_data/singing_reciting_detection/评测集0606.csv"
DEFAULT_EVAL_DIR = "/Users/para_fish66/Desktop/语音评论/data/output_data/singing_reciting_detection/eval_data"
DEFAULT_DIFF_DIR = "/Users/para_fish66/Desktop/语音评论/data/output_data/singing_reciting_detection/diff_data"
# ==============================


def load_prompt_template(prompt_path: str) -> str:
    """加载 prompt 模板"""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_csv_data(csv_path: str) -> list[dict[str, str]]:
    """加载评测集 CSV"""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_batch_prompt(prompt_template: str, batch: list[dict[str, str]]) -> str:
    """将 prompt 模板与批量数据组合成完整请求 prompt"""
    lines = prompt_template.strip().split("\n")
    cut_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("## 待标注样本"):
            cut_index = i
            break

    if cut_index is not None:
        system_part = "\n".join(lines[:cut_index])
    else:
        system_part = prompt_template

    samples_block = f"## 待标注样本（共 {len(batch)} 条）\n\n"
    for i, row in enumerate(batch):
        voice_id = row.get("voice_resource_id", "")
        asr_text = row.get("voice_asr_text", "")
        samples_block += f"### 样本 {i + 1}\n"
        samples_block += f"voiceId: {voice_id}\n"
        samples_block += f"voice_asr_text: {asr_text}\n\n"

    samples_block += (
        "## 输出要求\n"
        "请对以上每条样本分别输出分类结果，返回一个 JSON 数组，"
        "数组中每个元素对应一条样本（按顺序对应样本1、样本2、...），格式如下：\n"
        "```json\n"
        "[\n"
        '  {"voiceId": "<原样回填>", "voice_asr_text": "<原样回填>", "label": 0或1, "label_name": "唱歌/朗诵"或"非唱歌/朗诵", "reason": "<不超过80字的中文理由>"},\n'
        "  ...\n"
        "]\n"
        "```\n"
        "请确保返回的是合法的 JSON 数组，不要包含 markdown 标记或其他额外文字。"
    )

    return system_part.strip() + "\n\n" + samples_block


def call_api(prompt: str) -> str:
    """调用 AI API（使用 urllib，无需第三方库）"""
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({
        "model": ENDPOINT_ID,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 16384,
    }).encode("utf-8")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
                msg = data["choices"][0]["message"]
                # DeepSeek 模型可能将正式输出放在 content，
                # 思考过程放在 reasoning_content；若 content 为空则尝试 reasoning_content
                content = msg.get("content", "") or ""
                if not content.strip():
                    rc = msg.get("reasoning_content", "") or ""
                    if rc.strip():
                        print("  [注意] content 为空，尝试从 reasoning_content 提取")
                        content = rc
                if not content.strip():
                    raise ValueError("API 返回内容为空")
                return content
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            print(f"  [重试 {attempt}/{MAX_RETRIES}] HTTP {e.code}: {err_body[:200]}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise
        except (urllib.error.URLError, OSError, ConnectionResetError, ValueError) as e:
            # OSError 包含 ConnectionResetError, BrokenPipeError 等网络异常
            print(f"  [重试 {attempt}/{MAX_RETRIES}] 网络错误: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * 2)  # 网络错误加倍等待
            else:
                raise
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"  [重试 {attempt}/{MAX_RETRIES}] 响应解析失败: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise
    raise RuntimeError("API 调用失败，已用尽重试次数")


def parse_api_response(content: str, batch: list[dict[str, str]]) -> list[dict[str, str]]:
    """解析 API 返回的 JSON 数组，映射回原始数据行"""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    results = None
    try:
        results = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            try:
                results = json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

    if results is None:
        print("  [警告] JSON 解析失败，本批次结果将被标记为解析错误")
        return [
            {
                "voice_resource_id": row.get("voice_resource_id", ""),
                "voice_asr_text": row.get("voice_asr_text", ""),
                "pred_label": "-1",
                "pred_label_name": "解析错误",
                "pred_reason": "API 返回格式异常，无法解析",
            }
            for row in batch
        ]

    if isinstance(results, dict):
        results = [results]

    parsed = []
    for i, row in enumerate(batch):
        if i < len(results) and isinstance(results[i], dict):
            r = results[i]
            pred_label = str(r.get("label", "-1"))
            pred_label_name = r.get("label_name", "解析错误")
            pred_reason = r.get("reason", "")
        else:
            pred_label = "-1"
            pred_label_name = "解析错误"
            pred_reason = f"返回结果数量不足（期望{len(batch)}条，实际{len(results)}条）"

        parsed.append({
            "voice_resource_id": row.get("voice_resource_id", ""),
            "voice_asr_text": row.get("voice_asr_text", ""),
            "pred_label": pred_label,
            "pred_label_name": pred_label_name,
            "pred_reason": pred_reason,
        })

    return parsed


def load_progress(output_path: str) -> set[str]:
    """加载已处理的 voice_resource_id 集合（断点续传）"""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("voice_resource_id", "")
            if vid:
                done.add(vid)
    return done


def append_results(output_path: str, results: list[dict[str, str]], write_header: bool = False):
    """追加写入结果到输出 CSV"""
    fieldnames = [
        "voice_resource_id", "voice_asr_text",
        "pred_label", "pred_label_name", "pred_reason"
    ]
    with open(output_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)


def compute_accuracy_and_diff(input_path: str, eval_path: str, diff_dir: str):
    """
    运行结束后：
    1. 计算正确率（将 pred_label 与原始 CSV 中的 真实标签二分类 对比）
    2. 重点计算 label=1（唱歌/朗诵）的精确率和召回率
    3. 生成 diff 文件（仅包含预测错误的行）
    """
    # 加载原始数据（含真实标签）
    original = load_csv_data(input_path)
    truth_map = {}
    for row in original:
        vid = row.get("voice_resource_id", "")
        truth_label = row.get("真实标签二分类", "").strip()
        intent = row.get("意图表达", "").strip()
        asr = row.get("voice_asr_text", "")
        if vid:
            truth_map[vid] = {
                "true_label": truth_label,
                "intent": intent,
                "asr_text": asr,
            }

    # 加载预测结果
    preds = load_csv_data(eval_path)

    total = 0
    correct = 0
    diff_rows = []

    # ===== 混淆矩阵统计（以 label=1 为正例）=====
    tp1 = 0  # 真实1，预测1（True Positive）
    fp1 = 0  # 真实0，预测1（False Positive）
    fn1 = 0  # 真实1，预测0（False Negative）
    tn1 = 0  # 真实0，预测0（True Negative）
    real_1_total = 0  # 真实为1的总数
    pred_1_total = 0  # 预测为1的总数

    for pred in preds:
        vid = pred.get("voice_resource_id", "")
        pred_label = pred.get("pred_label", "").strip()
        pred_label_name = pred.get("pred_label_name", "").strip()
        pred_reason = pred.get("pred_reason", "").strip()

        if vid not in truth_map:
            continue

        truth = truth_map[vid]
        true_label = truth["true_label"]

        # 跳过预测异常的行
        if pred_label == "-1":
            total += 1
            diff_rows.append({
                "voice_resource_id": vid,
                "voice_asr_text": pred.get("voice_asr_text", ""),
                "true_label": true_label,
                "pred_label": "异常",
                "true_label_name": truth["intent"],
                "pred_label_name": pred_label_name,
                "pred_reason": pred_reason,
                "error_type": "API异常",
            })
            continue

        total += 1
        if pred_label == true_label:
            correct += 1

        # 混淆矩阵
        if true_label == "1":
            real_1_total += 1
            if pred_label == "1":
                tp1 += 1
                pred_1_total += 1
            else:
                fn1 += 1
        else:  # true_label == "0"
            if pred_label == "1":
                fp1 += 1
                pred_1_total += 1
            else:
                tn1 += 1

        if pred_label != true_label:
            true_name = "唱歌/朗诵" if true_label == "1" else "非唱歌/朗诵"
            diff_rows.append({
                "voice_resource_id": vid,
                "voice_asr_text": pred.get("voice_asr_text", ""),
                "true_label": true_label,
                "pred_label": pred_label,
                "true_label_name": true_name,
                "pred_label_name": pred_label_name,
                "pred_reason": pred_reason,
                "error_type": "误判",
            })

    # 计算正确率
    accuracy = correct / total * 100 if total > 0 else 0

    # label=1 的精确率与召回率
    precision_1 = tp1 / (tp1 + fp1) * 100 if (tp1 + fp1) > 0 else 0
    recall_1 = tp1 / (tp1 + fn1) * 100 if (tp1 + fn1) > 0 else 0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    # 生成 diff 文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_stem = Path(input_path).stem
    diff_path = os.path.join(diff_dir, f"{input_stem}_diff_{timestamp}.csv")

    diff_fieldnames = [
        "voice_resource_id", "voice_asr_text",
        "true_label", "pred_label",
        "true_label_name", "pred_label_name",
        "pred_reason", "error_type",
    ]
    os.makedirs(diff_dir, exist_ok=True)
    with open(diff_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=diff_fieldnames)
        writer.writeheader()
        writer.writerows(diff_rows)

    # 打印统计
    print(f"\n{'=' * 60}")
    print(f"📊 准确率统计")
    print(f"{'=' * 60}")
    print(f"   总评测条数:  {total}")
    print(f"   正确条数:    {correct}")
    print(f"   错误条数:    {total - correct}")
    print(f"   准确率:      {accuracy:.2f}%")
    print(f"")
    print(f"   --- 唱歌/朗诵(1) 核心指标 ---")
    print(f"   真实为1总数:  {real_1_total}")
    print(f"   预测为1总数:  {pred_1_total}")
    print(f"   TP(真实1→预测1): {tp1}")
    print(f"   FP(真实0→预测1): {fp1}")
    print(f"   FN(真实1→预测0): {fn1}")
    print(f"   精确率(Precision1): {tp1}/{tp1+fp1} = {precision_1:.2f}%")
    print(f"   召回率(Recall1):     {tp1}/{tp1+fn1} = {recall_1:.2f}%")
    print(f"   F1:                 {f1_1:.2f}%")

    print(f"")
    print(f"📁 diff 文件: {diff_path}")
    print(f"   共 {len(diff_rows)} 条预测错误记录")

    # 同时将准确率写入一个摘要文件
    summary_path = os.path.join(diff_dir, f"{input_stem}_summary_{timestamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"评测集: {input_path}\n")
        f.write(f"Prompt: {DEFAULT_PROMPT}\n")
        f.write(f"模型: {ENDPOINT_ID}\n")
        f.write(f"\n")
        f.write(f"总评测条数: {total}\n")
        f.write(f"正确条数:   {correct}\n")
        f.write(f"错误条数:   {total - correct}\n")
        f.write(f"准确率:     {accuracy:.2f}%\n")
        f.write(f"\n")
        f.write(f"--- 唱歌/朗诵(1) 核心指标 ---\n")
        f.write(f"真实为1总数:  {real_1_total}\n")
        f.write(f"预测为1总数:  {pred_1_total}\n")
        f.write(f"TP(真实1→预测1): {tp1}\n")
        f.write(f"FP(真实0→预测1): {fp1}\n")
        f.write(f"FN(真实1→预测0): {fn1}\n")
        f.write(f"精确率(Precision): {tp1}/{tp1+fp1} = {precision_1:.2f}%\n")
        f.write(f"召回率(Recall):     {tp1}/{tp1+fn1} = {recall_1:.2f}%\n")
        f.write(f"F1:                 {f1_1:.2f}%\n")

    print(f"📁 摘要文件: {summary_path}")

    return accuracy, diff_path


def main():
    parser = argparse.ArgumentParser(description="语音评论二分类批量评测脚本")
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT,
        help="Prompt 模板文件路径"
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT,
        help="评测集 CSV 文件路径"
    )
    parser.add_argument(
        "--eval-dir", type=str, default=DEFAULT_EVAL_DIR,
        help="评测结果输出目录"
    )
    parser.add_argument(
        "--diff-dir", type=str, default=DEFAULT_DIFF_DIR,
        help="diff 文件输出目录"
    )
    parser.add_argument(
        "--run-name", type=str, default="第四次评测（prompt版本6）",
        help="本次评测名称（默认: 第四次评测（prompt版本6））"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"每次请求处理的数据条数（默认 {BATCH_SIZE}）"
    )
    args = parser.parse_args()

    # 如果没有指定 run_name，使用默认值
    # （已通过 argparse default 设置）

    # 构造输出子目录：eval_data/<run_name>/, diff_data/<run_name>/
    input_stem = Path(args.input).stem
    eval_subdir = os.path.join(args.eval_dir, args.run_name)
    diff_subdir = os.path.join(args.diff_dir, args.run_name)
    os.makedirs(eval_subdir, exist_ok=True)
    os.makedirs(diff_subdir, exist_ok=True)

    eval_path = os.path.join(eval_subdir, f"{input_stem}_result.csv")

    # ======== 断点续传：检查 diff_data/<run_name>/ 下是否已有 summary ========
    # 如果 diff_data/<run-name>/ 下已有匹配的 summary 文件，说明本次评测已完成，直接跳过
    if os.path.exists(diff_subdir):
        try:
            all_files = os.listdir(diff_subdir)
        except Exception:
            all_files = []
        summary_files = [f for f in all_files
                         if f.startswith(input_stem) and "_summary_" in f and f.endswith(".txt")]
        if summary_files:
            print(f"⏭️  检测到 diff_data/{args.run_name}/ 下已有 summary 文件，本轮评测已完成，跳过 API 调用")
            print(f"   已有 summary: {summary_files[0]}")
            return

    # 加载 prompt 和数据
    print(f"📄 加载 Prompt: {args.prompt}")
    prompt_template = load_prompt_template(args.prompt)

    print(f"📊 加载评测集: {args.input}")
    all_rows = load_csv_data(args.input)
    total = len(all_rows)
    print(f"   共 {total} 条数据")
    print(f"🏷️  本次评测名称: {args.run_name}")
    print(f"   结果输出到: {eval_subdir}")
    print(f"   diff 输出到: {diff_subdir}")

    # 断点续传（逐行跳过已处理数据）
    done_ids = load_progress(eval_path)
    already_done = len(done_ids)
    if already_done > 0:
        print(f"♻️  检测到已有结果文件，已处理 {already_done} 条，将跳过")
    else:
        print(f"🆕 未检测到结果文件，从头开始处理")

    need_header = not os.path.exists(eval_path)
    batch_size = args.batch_size
    processed_this_run = 0
    error_count = 0

    # ======== 批量评测 ========
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = all_rows[start:end]

        remaining_batch = [row for row in batch if row.get("voice_resource_id", "") not in done_ids]
        if not remaining_batch:
            continue

        batch_num = start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(
            f"\n🔄 批次 {batch_num}/{total_batches} | "
            f"原始范围 [{start + 1}-{end}] | 实际处理 {len(remaining_batch)} 条"
        )

        prompt = build_batch_prompt(prompt_template, remaining_batch)

        try:
            raw_response = call_api(prompt)
        except Exception as e:
            print(f"  ❌ API 调用彻底失败: {e}")
            print(f"  ⚠️  本批次将不写入结果，下次运行时会重试")
            error_count += len(remaining_batch)
            # 不写入失败数据，不加入 done_ids，这样续传时这些行会被重新处理
            continue

        parsed = parse_api_response(raw_response, remaining_batch)
        append_results(eval_path, parsed, write_header=need_header)
        need_header = False

        for item in parsed:
            done_ids.add(item["voice_resource_id"])
            processed_this_run += 1

        labels = [p["pred_label"] for p in parsed]
        pos = labels.count("1")
        neg = labels.count("0")
        err = labels.count("-1")
        print(f"  ✅ 完成: 唱歌/朗诵={pos}, 非唱歌/朗诵={neg}, 异常={err}")
        print(f"   📈 本次运行已处理: {processed_this_run} | 累计已处理: {len(done_ids)}/{total}")

        time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"🏁 评测运行完成！")
    print(f"   总计: {total} 条")
    print(f"   累计已处理: {len(done_ids)} 条")
    print(f"   本次运行处理: {processed_this_run} 条")
    print(f"   异常: {error_count} 条")
    print(f"   结果文件: {eval_path}")

    # ======== 计算正确率 & 生成 diff ========
    if len(done_ids) < total:
        missing = total - len(done_ids)
        print(f"\n⚠️  还有 {missing} 条数据未处理，正确率统计将基于已处理数据")
    
    print(f"\n📊 开始计算正确率并生成 diff 文件...")
    _ = compute_accuracy_and_diff(args.input, eval_path, diff_subdir)


if __name__ == "__main__":
    main()
