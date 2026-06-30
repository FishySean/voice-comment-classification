#!/usr/bin/env python3
"""
数据清洗脚本：合并 v5 + 补充数据 → 评测集0606_v6.csv
规则来源：
  1. 用户指定的标签修正指令（天籁系列→1、叫哥哥系列→1、指定条→1/0等）
  2. v10 diff 辅助判断（反复出现的簇，模型判断更合理的）
  3. 意义不明/ASR质量差/小学生乱说 → 0
"""

import pandas as pd
import re
import os

# ============================================================
# 1. 读取数据
# ============================================================
V5_PATH = "data/testing_data/joking_amusing_detection/评测集0606_v5.csv"
SUPP_PATH = "data/补充数据.csv"
OUT_PATH = "data/testing_data/joking_amusing_detection/评测集0606_v6.csv"

df5 = pd.read_csv(V5_PATH)
df_supp = pd.read_csv(SUPP_PATH)

v5_ids = set(df5["voice_resource_id"])

# 补充数据：只加 玩梗/搞笑 类的新条目（label=1）
df_supp["true_label"] = (df_supp["意图表达"] == "玩梗/搞笑").astype(int)
new_supp = df_supp[(~df_supp["voice_resource_id"].isin(v5_ids)) & (df_supp["意图表达"] == "玩梗/搞笑")].copy()

# 统一列名
df5_out = df5[["voice_resource_id", "voice_asr_text", "意图表达", "真实标签二分类"]].copy()
df5_out.rename(columns={"真实标签二分类": "label"}, inplace=True)

new_supp_out = new_supp[["voice_resource_id", "voice_asr_text", "意图表达", "true_label"]].copy()
new_supp_out.rename(columns={"true_label": "label"}, inplace=True)

merged = pd.concat([df5_out, new_supp_out], ignore_index=True)
print(f"合并后总行数: {len(merged)} (v5={len(df5)}, 新增玩梗/搞笑={len(new_supp)})")
print(f"合并前 label 分布: 1={sum(merged['label']==1)}, 0={sum(merged['label']==0)}")

# ============================================================
# 3. 标签修正规则
# ============================================================
changes = []  # 记录所有修改

def apply_rule(df, rule_name, match_func, target_label, description=""):
    """对 df 中 match_func(text)==True 的行，把 label 改成 target_label"""
    changed = 0
    for idx, row in df.iterrows():
        txt = str(row["voice_asr_text"])
        if match_func(txt) and row["label"] != target_label:
            old = row["label"]
            df.at[idx, "label"] = target_label
            changes.append({
                "rule": rule_name,
                "voice_resource_id": row["voice_resource_id"],
                "text_preview": txt[:60],
                "old_label": old,
                "new_label": target_label,
                "desc": description
            })
            changed += 1
    print(f"  规则[{rule_name}]: 改了 {changed} 条 → {target_label}")
    return changed

# ------ 规则A: 天籁系列 全部→1 ------
def match_tianlai(text):
    """我听见你心中那动人的天籁... 接到 肥肥胖胖是太阳 / 生肖 / 喊麦续写"""
    return bool(re.search(r"动人的天籁|动人的偏爱|忽如一夜|满面桃花|满面.*花开|肥肥胖胖|鼓鼓囊囊|鼓鼓浪浪|五谷丰登|鼠牛|功夫高虎兔|龙在左蛇在右|风风光光小路|令抬令抬|拎抬拎抬|宁抬宁抬|并排并排|硬抬硬抬|拎柴拎柴|心抬心抬", text))

apply_rule(merged, "A-天籁混剪", match_tianlai, 1,
           "沙雕混剪类：动人天籁接到肥肥胖胖/生肖/喊麦续写，多条ASR变体")

# ------ 规则B: 叫哥哥太小了 系列 全部→1 ------
def match_jiaogg(text):
    """叫哥哥太小了/叫叔叔太老了/叫妈妈性别不对... 你应该叫什么
    必须满足核心结构：出现至少2个称呼(叫XX) + 至少1个排除理由(太小/太老/性别不对/太大了)"""
    # 必须有"叫XX"结构出现至少2次
    jiao_matches = re.findall(r"叫(哥哥|叔叔|妈妈|姐姐|阿姨|爸爸|老公|老婆|娜娜)", text)
    if len(jiao_matches) < 2:
        return False
    # 必须有排除理由
    if not re.search(r"太小|太老|太懒|性别不对|性别又不对|太大了|simbup|识别不对", text):
        return False
    return True

apply_rule(merged, "B-称呼梗", match_jiaogg, 1,
           "叫XX太小/太老/性别不对+你应该叫我什么，称呼反差梗系列")

# ------ 规则C: 中奖概率倍儿高 系列 → 1 ------
def match_zhongjiang(text):
    """中奖概率倍儿高/中间咖喱倍儿高 系列喊麦"""
    return bool(re.search(r"中奖概率|中间咖喱|奖品野马|手机钞票|奔驰金|大金[龙老刀停]", text))

apply_rule(merged, "C-中奖喊麦", match_zhongjiang, 1,
           "中奖概率倍儿高+奖品野马+手机钞票+奔驰金条 喊麦体系列")

# ------ 规则D: 卖三明治 系列 → 1 ------
def match_sandwich(text):
    """卖三明治/卖扇贝 地摊叫卖梗"""
    return bool(re.search(r"卖三明治|卖扇贝|卖神秘制|三明治.*三明治", text))

apply_rule(merged, "D-三明治叫卖", match_sandwich, 1,
           "卖三明治/卖扇贝 地摊叫卖玩梗系列")

# ------ 规则E: 泪水打湿数据线 系列 → 1 ------
def match_leishui(text):
    """泪水打湿XX系列"""
    return bool(re.search(r"泪水打湿", text))

apply_rule(merged, "E-泪水打湿", match_leishui, 1,
           "泪水打湿数据线/小天才 押韵自嘲顺口溜")

# ------ 规则F: 特别的爱给特别的你 → 1 ------
def match_tebeiai(text):
    return bool(re.search(r"特别的爱给特别的你|拖拉机.*法拉利|法拉利.*声音", text))

apply_rule(merged, "F-特别的爱", match_tebeiai, 1,
           "特别的爱给特别的你+拖拉机脸/法拉利声音 恶搞改编")

# ------ 规则G: 想要问问你敢不敢 → 1 ------
def match_wenwen(text):
    return bool(re.search(r"想要问问你敢不敢|像你傻子", text))

apply_rule(merged, "G-问问敢不敢", match_wenwen, 1,
           "想要问问你敢不敢 恶搞改编歌词")

# ------ 规则H: 宝宝我是极品甜妹音 → 1 ------
def match_tianmei(text):
    return bool(re.search(r"极品甜妹音|蜜蜂.*500斤|蜜蜂.*叮成|不小心被蜜蜂", text))

apply_rule(merged, "H-甜妹音", match_tianmei, 1,
           "宝宝我是极品甜妹音 被蜜蜂叮成500斤 自嘲反转")

# ------ 规则I: 别闹叔叔在尿尿 → 0 ------
def match_bieniao(text):
    """别闹叔叔在尿尿/别闹叔叔在开摩托车/别闹叔叔会飞 等 - 这些是短句调侃+动作，但用户指定只有尿尿那条判0"""
    # 用户只指定了"别闹叔叔在尿尿吱吱吱"判0，其他别闹叔叔看情况
    return bool(re.search(r"别闹叔叔在尿尿", text))

apply_rule(merged, "I-别闹尿尿判0", match_bieniao, 0,
           "别闹叔叔在尿尿吱吱吱 拟声词写实 不构成搞笑反转")

# ------ 规则J: 恭喜你兄弟成功的被我恭喜到了 → 0 ------
def match_gongxi(text):
    return bool(re.search(r"恭喜你兄弟成功的被我恭喜到了|吃饭要用嘴.*走路要用腿|废话说的是屁话|屁话说的是废话|我白说了|我不说了", text))

apply_rule(merged, "J-废话文学体", match_gongxi, 0,
           "恭喜你兄弟+吃饭用嘴走路用腿+废话文学体 不构成主动搞笑")

# ------ 规则K: 我评论区能说话你气不气 → 0 ------
def match_pinglun(text):
    return bool(re.search(r"评论区能说话你气不气", text))

apply_rule(merged, "K-评论区能说话", match_pinglun, 0,
           "单纯炫耀/挑衅 不构成搞笑结构")

# ------ 规则L: 居然给你背高 → 0 ------
def match_beigao(text):
    return bool(re.search(r"居然给你背高|拼什么背好收集钞票", text))

apply_rule(merged, "L-背高混乱", match_beigao, 0,
           "内容混乱无意义 ASR质量差")

# ------ 规则M: 哈哈哈哈哎呦这不烟头瘦瘦吗 → 0 ------
def match_yantou(text):
    return bool(re.search(r"烟头瘦瘦", text))

apply_rule(merged, "M-烟头瘦瘦", match_yantou, 0,
           "哈哈哈哈+这不烟头瘦瘦吗 单句调侃 无搞笑结构")

# ------ 规则N: 53厘米腰/大腿 → 0 ------
def match_53yao(text):
    return bool(re.search(r"53厘米|大腿也没有53", text))

apply_rule(merged, "N-53厘米腰", match_53yao, 0,
           "53厘米腰/大腿 单句调侃 无搞笑结构")

# ------ 规则O: 姐姐放个奶瓶/奶屁 → 0 ------
# 用户指定"姐姐放个奶瓶给你补补"之前在v10已经判0了
# 但diff里还有"姐姐放个奶屁给你补补""九姐放个奶屁""姐姐我放奶屁了"被判1(FN)
# 这些都是短句调侃+奇怪动作，按H规则逻辑应判0
def match_naipi(text):
    return bool(re.search(r"奶瓶给你补补|奶屁给你补|放个奶屁|放奶屁", text))

apply_rule(merged, "O-奶屁短句", match_naipi, 0,
           "放个奶瓶/奶屁给你补补 短句调侃+奇怪动作 无搞笑结构")

# ------ 规则P: 我只是摔倒了又不是摔死了 → 0 ------
def match_shuaidao(text):
    return bool(re.search(r"我只是摔倒了又不是摔死了|你凭什么觉得我爬不起来", text))

apply_rule(merged, "P-摔倒反问", match_shuaidao, 0,
           "反问式自我辩护 不构成自嘲反转")

# ------ 规则Q: 变声器关不掉了 → 1 ------
# 用户指定：涉及变声器的都判1
# 覆盖：变声器关不掉了/发语音就唱歌/真的崩溃了/声音好难听/声音变了 等系列
def match_biansheng(text):
    return bool(re.search(r"变声器|发语音.*唱歌|发语音.*唱|语音.*还在唱|语音还在唱歌|声音好难听|声音大便回来|真的着了|真的没招|真的好崩溃|真的崩溃|一发语音就唱|发语音还在唱|语音.*绷不住|语音.*封不住|语音.*蹦不|语音.*缝补|语音.*动不|唱歌.*关不掉|关不掉了|关不掉", text)) and \
           not bool(re.search(r"唱歌/朗诵|唱歌朗诵", text))  # 排除意图表达本身就写唱歌朗诵的

apply_rule(merged, "Q-变声器梗", match_biansheng, 1,
           "变声器关不掉了/发语音就唱歌/真的崩溃了/声音变了 系列 玩变声器搞笑")

# ------ 规则R: sity spection / city spection / sad is fashion 系列 → 1 ------
# 用户指定：satisfaction类读英语的是搞笑/玩梗（变声/口音反差）
def match_cityspection(text):
    return bool(re.search(r"sity spection|city spection|city fection|sity fection|said is fection|said is fashion|set is a foction|size she faction|sixfection|six faction|selice feshion|sight is fightion|setifuction|sless fixtion|sadic fiction|c tic fiction|citis fection|secould the five|suddisfection|say disfection|seation|sfection|sfiction|satisfaction|selice feshion|sad is fashion|setifuction", text, re.IGNORECASE))

apply_rule(merged, "R-sityspection读英语", match_cityspection, 1,
           "city spection/sad is fashion/satisfaction 小学生读英语发音 变声口音反差搞笑")

# ------ 规则S: 向上向上向前是萝莉音 系列 → 1 ------
def match_luoliyin(text):
    return bool(re.search(r"向上.*向前.*萝莉音|向上.*向前.*俄语音|向上.*向前.*串通", text))

apply_rule(merged, "S-萝莉音", match_luoliyin, 1,
           "向上向前是萝莉音 变声器玩梗系列")

# ------ 规则T: 哥哥我摔倒了 系列 → 1 ------
def match_shuaidao2(text):
    return bool(re.search(r"哥哥我摔倒了|哥哥摔倒了|人家摔倒了", text))

apply_rule(merged, "T-哥哥我摔倒了", match_shuaidao2, 1,
           "哥哥我摔倒了 来扶我一下 卖萌玩梗")

# ------ 规则U: 别闹叔叔 系列（除了尿尿那条）→ 1 ------
def match_bienao_shushu(text):
    # 别闹叔叔在开摩托车/别闹叔叔会飞/别闹乖乖的吃汉堡 等
    return bool(re.search(r"别闹叔叔", text)) and not bool(re.search(r"别闹叔叔在尿尿", text))

apply_rule(merged, "U-别闹叔叔玩梗", match_bienao_shushu, 1,
           "别闹叔叔在开摩托车/会飞/会撒娇 等 反差玩梗")

# ------ 规则V: 棉袄叔叔在练步机/电脑叔叔在拉屎/来个小破烂孩 → 1 ------
def match_shushu_bianxing(text):
    return bool(re.search(r"棉袄叔叔|电脑叔叔|烟头叔叔|来个小破烂孩", text))

apply_rule(merged, "V-叔叔变体梗", match_shushu_bianxing, 1,
           "XX叔叔在XX 反差玩梗变体")

# ------ 规则W: 我不要吃青菜 → 1 ------
def match_qingcai(text):
    return bool(re.search(r"我不要吃青菜|我要吃西瓜|我要吃虾", text))

apply_rule(merged, "W-不要吃青菜", match_qingcai, 1,
           "我不要吃青菜我要吃虾 小朋友傲娇梗")

# ------ 规则X: 来个XX我是YY 反差自我介绍梗 → 1 ------
# "来个肘子我是辣条" 用户之前说判0，但后来补充数据标了1
# "来个家伙我是面瘫" "来个大葱我是酱" "来个蓝莓我是香蕉" "来个气泡我是音"
def match_laigewo(text):
    return bool(re.search(r"来个.{1,4}我是.{1,4}", text))

apply_rule(merged, "X-来个XX我是YY", match_laigewo, 1,
           "来个肘子我是辣条/来个家伙我是面瘫 反差自我介绍梗")

# ------ 规则Y: 意义不明/纯无意义 → 0 ------
def check_meaningless(text):
    """内容极短或完全无意义"""
    text_stripped = text.strip()
    # 纯拟声词/纯笑声（无任何文字内容）
    if re.match(r"^[啊哈哼嗯呃哦噢唉哎呜唔咦唔嘿嗨哟呦啾喵汪咪嗷啪噔咚咕噜]*$", text_stripped):
        return True
    # 极短（≤3个有效字符）且无任何语义
    effective_chars = re.sub(r"[^\u4e00-\u9fff\w]", "", text_stripped)
    if len(effective_chars) <= 2 and not re.search(r"梗|笑|搞笑|玩", text_stripped):
        return True
    return False

# 注意：意义不明要谨慎，只改明显的
# 对极短无意义文本 → 0
for idx, row in merged.iterrows():
    txt = str(row["voice_asr_text"]).strip()
    if check_meaningless(txt) and row["label"] == 1:
        merged.at[idx, "label"] = 0
        changes.append({
            "rule": "Y-意义不明",
            "voice_resource_id": row["voice_resource_id"],
            "text_preview": txt[:60],
            "old_label": 1,
            "new_label": 0,
            "desc": "纯拟声词/极短无意义 → 0"
        })

y_count = sum(1 for c in changes if c["rule"] == "Y-意义不明")
print(f"  规则[Y-意义不明]: 改了 {y_count} 条 → 0")

# ============================================================
# 4. 输出
# ============================================================
merged.rename(columns={"label": "真实标签二分类"}, inplace=True)
merged[["voice_resource_id", "voice_asr_text", "意图表达", "真实标签二分类"]].to_csv(OUT_PATH, index=False)

print(f"\n===== 输出完成 =====")
print(f"文件: {OUT_PATH}")
print(f"总行数: {len(merged)}")
print(f"最终 label 分布: 1={sum(merged['真实标签二分类']==1)}, 0={sum(merged['真实标签二分类']==0)}")
print(f"总修改条数: {len(changes)}")

# 输出修改清单
changes_df = pd.DataFrame(changes)
changes_csv = OUT_PATH.replace(".csv", "_changes.csv")
changes_df.to_csv(changes_csv, index=False)
print(f"修改清单: {changes_csv}")

# 按规则统计
print("\n===== 修改统计 =====")
for rule, group in changes_df.groupby("rule"):
    print(f"  {rule}: {len(group)} 条")
    for _, r in group.head(3).iterrows():
        print(f"    [{r['old_label']}→{r['new_label']}] {r['text_preview'][:50]}")
    if len(group) > 3:
        print(f"    ... 共 {len(group)} 条")
