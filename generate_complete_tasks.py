import json
import os
import random
from pathlib import Path
from openai import OpenAI

def build_prompt(tree_path: str, seed: int | None = None) -> str:
    tree = json.loads(Path(tree_path).read_text())
    labels = [c["label"] for c in tree.get("children", [])]

    if not labels:
        raise ValueError("No labels found in tree.")

    if seed is not None:
        random.seed(seed)

    count = random.randint(5, 7)
    sample = random.sample(labels, count)

    prompt = f"""给定一个以“客厅（living room）”为根节点的场景树。示例物体如下：
{", ".join(sample)}

可用的原子动作（atomic actions）：
- move（移动一段距离）
- move2（移动到目标物体）
- rotation（原地旋转视角）
- pick_up（拾取物体）
- put_down（放下物体）
- open（打开，如门/柜子/冰箱等）
- close（关闭，如门/柜子/冰箱等）
- turn_on（开启设备）
- turn_off（关闭设备）

任务：
生成一个“完整且真实的层次化日常居家任务树”（hierarchical daily routine task tree）。

该任务树需要按照如下层级进行分解：
高层日常任务意图
→ 子任务意图
→ 更细粒度子任务意图（如有需要）
→ 原子动作

核心语义要求：

1. 意图继承关系（Intent inheritance）：
- 父节点意图必须总结其子节点意图的关键信息。
- 子节点意图必须在父节点基础上展开，提供更具体的操作细节。
- 整个层级结构应体现由粗到细（coarse-to-fine）的逐步细化过程。

2. 子任务意图必须包含动作级信息：
- 子任务意图不能是模糊标签。
- 必须描述其后续原子动作中将执行的关键操作。
- 例如，不要使用“整理物品”，应改为：
  “拿起个人物品并将其放入储物柜中”。
- 子任务意图应尽可能体现关键动作语义，例如：
  移动（move）、打开（open）、关闭（close）、拾取（pick_up）、放下（put_down）、开启（turn_on）、关闭（turn_off）等。

3. 高层意图必须包含子任务的关键信息：
- 根节点的高层意图应为自然语言描述，覆盖整个任务流程的主要组成部分。
- 应总结各个子任务，而不是过于泛化。
- 例如，不要使用“晚间例行任务”，应改为：
  “回到家中，确保入口安全，整理个人物品，并启动必要的家用设备以准备晚间环境”。

4. 层级对齐（Alignment between levels）：
- 每一个中间节点的意图必须由其子节点支撑。
- 每一组子节点的行为序列必须能够具体实现其父节点意图。
- 子任务不得引入与父任务无关的目标。

基本要求：
- 根节点：必须且仅包含一个高层日常任务意图。
- 中间节点：只能是意图/子目标节点（不能包含动作）。
- 叶子节点：必须全部为原子动作节点。
- 分解过程必须符合时间顺序，并具备逻辑一致性。
- 任务应尽可能基于场景中的物体（grounded）。
- 叶子节点只能使用提供的原子动作集合。
- 结构应简洁但完整。

意图撰写规则：
- 意图必须使用自然语言描述，而非简单标签。
- 每个意图应同时包含：
  (a) 要完成的目标（what）
  (b) 实现方式（how）
- 高层意图应总结子任务。
- 低层意图应反映其下对应的实际动作。
- 避免使用模糊表达，例如：
  “整理一下”、“准备房间”、“处理厨房事务”
- 推荐使用明确表达，例如：
  “走到柜子前，打开柜门，并将帽子放入其中以保持物品整洁”

输出格式：
仅返回合法 JSON，格式如下：

{{
  "id": "root",
  "type": "task",
  "intent": "总结子任务关键信息的高层日常任务意图",
  "children": [
    {{
      "id": "subtask_1",
      "type": "subtask",
      "intent": "包含关键动作信息的子任务意图",
      "children": [
        {{
          "id": "subtask_1_1",
          "type": "subtask",
          "intent": "紧密反映底层原子动作的细粒度子任务意图",
          "children": [
            {{
              "id": "action_1",
              "type": "action",
              "action": "move2",
              "target": "fridge",
              "notes": "移动到冰箱位置"
            }}
          ]
        }}
      ]
    }}
  ]
}}
"""
    return prompt

def main():
    tree_path = "/Users/han/Desktop/TongBench/env_data/label_tree.json"
    prompt = build_prompt(tree_path, seed=42)  # seed 可选
    output_path = Path("/Users/han/Desktop/TongBench/outputs/complete_tasks.json")

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )

    completion = client.chat.completions.create(
        model="qwen3.5-27b",
        messages=[{"role": "user", "content": prompt}],
    )

    content = completion.choices[0].message.content
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(content)
    print(f"saved to {output_path}")

if __name__ == "__main__":
    main()
