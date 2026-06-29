#!/usr/bin/env python3
# pyright: reportUnusedCallResult=false, reportImplicitStringConcatenation=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false
"""
语音评论「唱歌/朗诵」二分类空白数据标注脚本
- 对空白数据（无真实标签）进行批量预测标注
- 每次请求处理 20 条数据（可通过 --batch-size 调整）
- 支持续传（断点续传），中断后重新运行自动跳过已处理行
- 运行结束后输出预测统计摘要
- 纯标准库，无需安装第三方依赖

使用方法:
  python3 run_binary_annotate.py
  python3 run_binary_annotate.py --batch-size 10
  python3 run_binary_annotate.py --limit 500
"""

import csv
import json
import os
import time
import argparse
import urllib.request
import urllib.error
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
DEFAULT_INPUT = "/Users/para_fish66/Desktop/语音评论/data/new_bare_data/空白数据0626_1.csv"
DEFAULT_OUTPUT_DIR = "/Users/para_fish66/Desktop/语音评论/data/output_data/singing_reciting_detection/test_eval_data"
DEFAULT_RUN_NAME = "第二次空白标注（空白数据0626_1 501-1000）"
DEFAULT_LIMIT = 500
DEFAULT_OFFSET = 500
# ==============================


def load_prompt_template(prompt_path: str) -> str:
    """加载 prompt 模板"""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_csv_data(csv_path: str, limit: int = 0, offset: int = 0) -> list[dict[str, str]]:
    """加载空白数据 CSV，可选跳过前 offset 行、限制行数"""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < offset:
                continue
            if limit > 0 and len(rows) >= limit:
                break
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
            print(f"  [重试 {attempt}/{MAX_RETRIES}] 网络错误: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * 2)
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


def write_summary(output_dir: str, run_name: str, prompt_path: str, input_path: str,
                   total: int, pos_count: int, neg_count: int, err_count: int, limit: int, offset: int = 0):
    """生成标注统计摘要"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"标注时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"空白数据: {input_path}\n")
        f.write(f"Prompt: {prompt_path}\n")
        f.write(f"模型: {ENDPOINT_ID}\n")
        f.write(f"本次标注名称: {run_name}\n")
        f.write(f"\n")
        f.write(f"数据范围: 第 {offset + 1} ~ {offset + limit} 条\n")
        f.write(f"实际处理: {total} 条\n")
        f.write(f"\n")
        f.write(f"--- 预测分布 ---\n")
        f.write(f"唱歌/朗诵(pred=1): {pos_count}\n")
        f.write(f"非唱歌/朗诵(pred=0): {neg_count}\n")
        f.write(f"异常(pred=-1): {err_count}\n")
        if total > 0:
            f.write(f"正例比例: {pos_count}/{total} = {pos_count/total*100:.2f}%\n")

    print(f"📁 摘要文件: {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="语音评论「唱歌/朗诵」二分类空白数据标注脚本")
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT,
        help="Prompt 模板文件路径"
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT,
        help="空白数据 CSV 文件路径"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="标注结果输出目录"
    )
    parser.add_argument(
        "--run-name", type=str, default=DEFAULT_RUN_NAME,
        help=f"本次标注名称（默认: {DEFAULT_RUN_NAME}）"
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"只处理前 N 条数据（默认 {DEFAULT_LIMIT}）"
    )
    parser.add_argument(
        "--offset", type=int, default=DEFAULT_OFFSET,
        help=f"跳过前 N 行数据（默认 {DEFAULT_OFFSET}）"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"每次请求处理的数据条数（默认 {BATCH_SIZE}）"
    )
    args = parser.parse_args()

    # 构造输出子目录
    output_subdir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_subdir, exist_ok=True)

    input_stem = pathlib.Path(args.input).stem
    eval_path = os.path.join(output_subdir, f"{input_stem}_result.csv")

    # ======== 断点续传：检查是否已有 summary ========
    if os.path.exists(output_subdir):
        try:
            all_files = os.listdir(output_subdir)
        except Exception:
            all_files = []
        summary_files = [f for f in all_files if f.startswith("summary_") and f.endswith(".txt")]
        if summary_files:
            print(f"⏭️  检测到已有 summary 文件，本轮标注已完成，跳过 API 调用")
            print(f"   已有 summary: {summary_files[0]}")
            return

    # 加载 prompt 和数据
    print(f"📄 加载 Prompt: {args.prompt}")
    prompt_template = load_prompt_template(args.prompt)

    print(f"📊 加载空白数据: {args.input}")
    print(f"   偏移: 跳过前 {args.offset} 行")
    print(f"   限制: 取 {args.limit} 条")
    all_rows = load_csv_data(args.input, limit=args.limit, offset=args.offset)
    total = len(all_rows)
    print(f"   实际加载: {total} 条数据")
    print(f"🏷️  本次标注名称: {args.run_name}")
    print(f"   结果输出到: {output_subdir}")

    # 断点续传
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
    total_pos = 0
    total_neg = 0
    total_err = 0

    # ======== 批量标注 ========
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
        total_pos += pos
        total_neg += neg
        total_err += err
        print(f"  ✅ 完成: 唱歌/朗诵={pos}, 非唱歌/朗诵={neg}, 异常={err}")
        print(f"   📈 本次运行已处理: {processed_this_run} | 累计已处理: {len(done_ids)}/{total}")

        time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"🏁 标注运行完成！")
    print(f"   数据范围: 第 {args.offset + 1} ~ {args.offset + args.limit} 条")
    print(f"   实际处理: {total} 条")
    print(f"   累计已处理: {len(done_ids)} 条")
    print(f"   本次运行处理: {processed_this_run} 条")
    print(f"   异常: {error_count} 条")
    print(f"")
    print(f"   --- 预测分布 ---")
    print(f"   唱歌/朗诵(pred=1): {total_pos}")
    print(f"   非唱歌/朗诵(pred=0): {total_neg}")
    print(f"   异常(pred=-1): {total_err}")
    if total > 0:
        print(f"   正例比例: {total_pos}/{total} = {total_pos/total*100:.2f}%")
    print(f"   结果文件: {eval_path}")

    # ======== 生成摘要 ========
    print(f"\n📊 生成标注摘要...")
    write_summary(output_subdir, args.run_name, args.prompt, args.input,
                  len(done_ids), total_pos, total_neg, total_err, args.limit, args.offset)


if __name__ == "__main__":
    main()
