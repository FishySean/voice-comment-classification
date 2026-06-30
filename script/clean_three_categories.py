#!/usr/bin/env python3
"""
为 opinion / questioning / social 三类清洗评测集
将 label=1 中混入的玩梗台词改为 0
输出 v3 版本评测集
"""
import pandas as pd
import re
import os

# ============== 通用玩梗模板排除规则 ==============
# 这些 ASR 文本是玩梗/搞笑模板，被误标成了真观点/真提问/真社交，需要改成 0
JOKING_RULES = [
    # 称呼反差梗
    (r"叫哥哥太小|叫叔叔太老|叫妈妈性别|你应该叫|你想想你应该叫", "称呼反差梗"),
    # 组队邀请玩梗模板
    (r"队友有麦打不打|拍王者段子有人吗|拍段子有人", "组队邀请玩梗模板"),
    # 撒娇梗
    (r"哥哥我摔倒|人家摔倒了.*扶|快来扶一下", "撒娇玩梗"),
    # 中奖喊麦
    (r"中奖概率倍儿高|奖品野马|手机钞票|满倍儿好|倍儿高|倍儿好", "中奖喊麦"),
    # 萝莉音
    (r"向上.*向上.*萝莉音|向上.*向前.*萝莉|萝莉是最最最|萝莉最厉害", "萝莉音变声器梗"),
    # 变声器
    (r"变声器.*关|变声器关不掉|关不掉的爱|发语音.*唱歌|语音.*缝补|语音.*绷不住|九块九.*变声器", "变声器玩梗"),
    # 撒娇问句
    (r"不再依赖姐姐算长大|不再依赖你算长大", "撒娇问句"),
    (r"想叔叔了吗|想我了吗.*小女朋友", "撒娇问句"),
    # 甜妹音/蜜蜂
    (r"对抗路.*甜妹|甜妹音.*蜜蜂|极品甜妹音|蜜蜂.*500斤|蜜蜂.*五百斤|被蜜蜂叮", "甜妹音玩梗"),
    # 改编歌词梗
    (r"特别的爱给特别的你|拖拉机.*法拉利", "改编歌词梗"),
    (r"想要问问你敢不敢|江南自有江南雨", "改编歌词梗"),
    # 三明治
    (r"卖三明治|三明治.*三明治|脑瓜里面全都是三明治", "三明治叫卖玩梗"),
    # 奶屁
    (r"奶屁给你补|姐姐放个奶|放奶屁|姐姐放个奶片", "奶屁短句玩梗"),
    # 摔倒反问
    (r"我只是摔倒了又不是摔死", "摔倒反问玩梗"),
    # 评论区炫耀
    (r"评论区能说话.*气不气", "评论区炫耀玩梗"),
    # 天籁
    (r"动人的天籁|动人的偏爱|肥肥胖胖|忽如一夜|满面桃花", "天籁混剪玩梗"),
    # 泪水
    (r"泪水打湿", "泪水打湿玩梗"),
    # 别闹
    (r"别闹叔叔", "别闹叔叔玩梗"),
    # 叔叔变体
    (r"棉袄叔叔|电脑叔叔|烟头叔叔", "叔叔变体玩梗"),
    # 不要吃青菜
    (r"我不要吃青菜|我要吃虾|我要吃西瓜", "不要吃青菜玩梗"),
    # 来个XX我是YY
    (r"来个.{1,4}我是.{1,4}", "来个XX我是YY玩梗"),
    # satisfaction
    (r"sity spection|city spection|city fection|sad is fashion|satisfaction|setifuction|sfection|sfiction", "satisfaction玩梗"),
    # 烟头瘦瘦/53厘米
    (r"烟头瘦瘦|53厘米", "短句调侃玩梗"),
    # 废话文学
    (r"恭喜你兄弟成功|废话说的是屁话|屁话说的是废话", "废话文学玩梗"),
    # 露露复兴
    (r"露露复兴", "露露复兴玩梗"),
    # 算姐姐
    (r"算姐姐没有", "算姐姐玩梗"),
]

# 元提问/元惊呼：对快手能否发语音的好奇/惊呼，不算真观点/真提问/真社交
META_RULES = [
    (r"快手.*能用语音|快手.*可以发语音|快手.*发语音了|快手不是可以发语音|快手居然可以发语音|快手也能发语音|为什么.*快手语音|快手什么时候能发语音|快手竟然可以发语音", "快手元提问/惊呼"),
    (r"评论区.*真的.*语音|评论区.*没有一个人语音", "评论区元提问"),
    (r"我去.*快手.*能发语音|我靠.*快手.*能发语音|我靠.*快手.*可以发语音", "快手元惊呼"),
]


def apply_clean(df, src_path):
    """把 label=1 中匹配到玩梗/元提问的样本改成 0"""
    changes = []
    for idx, row in df.iterrows():
        if row["真实标签二分类"] != 1:
            continue
        txt = str(row["voice_asr_text"])
        # 玩梗规则
        for pat, desc in JOKING_RULES + META_RULES:
            if re.search(pat, txt):
                changes.append({
                    "voice_resource_id": row["voice_resource_id"],
                    "text_preview": txt[:80],
                    "old_label": 1, "new_label": 0,
                    "rule": desc, "pattern": pat
                })
                df.at[idx, "真实标签二分类"] = 0
                break
    return changes


CASES = [
    ('opinion', 'data/testing_data/opinion_expression_detection/评测集0606_v2.csv',
     'data/testing_data/opinion_expression_detection/评测集0606_v3.csv'),
    ('question', 'data/testing_data/questioning_chatting_detection/评测集0606_v2.csv',
     'data/testing_data/questioning_chatting_detection/评测集0606_v3.csv'),
    ('social', 'data/testing_data/social_interaction_detection/评测集0606_v2.csv',
     'data/testing_data/social_interaction_detection/评测集0606_v3.csv'),
]

for name, in_path, out_path in CASES:
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    df = pd.read_csv(in_path)
    print(f"读入: {in_path}, 总行数={len(df)}")
    print(f"清洗前 label 分布: 1={(df['真实标签二分类']==1).sum()}, 0={(df['真实标签二分类']==0).sum()}")
    changes = apply_clean(df, in_path)
    df.to_csv(out_path, index=False)
    print(f"清洗后 label 分布: 1={(df['真实标签二分类']==1).sum()}, 0={(df['真实标签二分类']==0).sum()}")
    print(f"修改条数: {len(changes)}")
    print(f"输出: {out_path}")
    if changes:
        changes_df = pd.DataFrame(changes)
        changes_csv = out_path.replace(".csv", "_changes.csv")
        changes_df.to_csv(changes_csv, index=False)
        print(f"修改清单: {changes_csv}\n")
        for c in changes:
            print(f"  [{c['rule']}] {c['text_preview']}")
