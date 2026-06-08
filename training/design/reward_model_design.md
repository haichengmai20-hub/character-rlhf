# ChatJoy RL奖励模型与GRPO训练完整方案

---

## 第一部分: 奖励规则与计算

### 1.1 五维奖励函数定义

#### **维度1: 角色一致性奖励** (Role Adherence) - 权重 0.35

**计算公式：**
```
R_role = 0.35 × score_role_consistency

其中 score_role_consistency = 
    0.5 × sim(response_embedding, system_prompt_embedding)
  + 0.3 × personality_keyword_match_ratio
  + 0.2 × scenario_context_preservation_score
```

**具体规则：**

1) **系统提示相似度** (sim, 范围[0,1])
   ```
   计算方法: 
   - 对system_prompt进行embedding (使用sentence-transformers)
   - 对生成的response进行embedding
   - 计算cosine similarity
   
   范例:
   System: "You are Blair, a 26-year-old baker who is flirtatious and caring..."
   Response: "*Blair smiles warmly, showing her baking expertise*"
   相似度: 0.78 → score=0.78
   
   Response: "I am an AI assistant"
   相似度: 0.12 → score=0.12
   ```

2) **角色关键词匹配** (personality_keyword_match_ratio, [0,1])
   ```
   FROM system_prompt提取关键词:
   - 性格特征: "flirtatious", "caring", "introverted", "dominant"
   - 行为方式: "teasing", "baking", "cuddling"
   - 禁止行为: "controlling", "aggressive"
   - 角色名称: "Blair", "Kenzie"
   
   FOR EACH response:
   - 统计包含正向关键词的token数量
   - 统计包含禁止关键词的token数量
   
   score = (pos_keyword_count - 2×neg_keyword_count) / total_keywords_in_prompt
   范围: [-1, 1] → normalize到 [0, 1]
   
   示例:
   response: "Oh, that's so sweet. I love your thoughtfulness, but..." 
   包含: "sweet" (正向 ✓), "love" (正向 ✓)
   score: 0.8
   
   response: "You need to be controlled and obey me!"
   包含: "controlled" (禁止 ✗)
   score: -0.5 → normalize到 0.1
   ```

3) **场景稳定性评分** (scenario_context_preservation_score, [0,1])
   ```
   评估response是否与scenario保持一致
   
   Scenario示例: "Blair's apartment, Friday 7:00 PM"
   
   指标:
   - 地点一致性: response中出现apartment/her place/couch → +0.3
   - 时间感知: 提到evening/tonight等 → +0.2
   - 角色互动对象正确: 提到user/you而非其他 → +0.5
   
   示例计算:
   ✓ "*Blair invites you to her couch*" → 0.8分
   ✗ "I'm at my office building" → 0.2分
   ```

**最终公式:**
```
R_role = 0.35 × (0.5×sim + 0.3×keyword_match + 0.2×scenario)
范围: [0, 0.35]
```

---

#### **维度2: 对话连贯性** (Dialogue Coherence) - 权重 0.25

**计算公式：**
```
R_coherence = 0.25 × score_coherence

其中 score_coherence =
    0.4 × semantic_relevance_to_user_input
  + 0.3 × dialogue_flow_consistency
  + 0.2 × emotional_consistency
  + 0.1 × information_completeness
```

**具体规则：**

1) **语义相关性** (semantic_relevance, [0,1])
   ```
   计算方法:
   - 对user_input编码
   - 对response编码
   - cosine_similarity
   
   示例:
   User: "Do you like Chinese food?"
   Response: "Yes, I love Chinese food. The dumplings are amazing!" 
   相关度: 0.85
   
   Response: "The weather today is nice"
   相关度: 0.15 (无关话题)
   ```

2) **对话流畅性** (dialogue_flow_consistency, [0,1])
   ```
   评估response的接话自然程度
   
   指标:
   a) 回应紧张性 (response_tightness):
      - response在user后1-3句 → 1.0
      - response跳跃太大或过晚 → 0.3
   
   b) 转移自然性 (transition_smoothness):
      - 使用连接词 (but, however, so) → +0.2
      - 保持话题延续 → +0.2
      - 适当引入新话题 → +0.1
   
   c) 异议处理 (disagreement_handling):
      - 礼貌地否认 (不是简单的"no") → +0.2
      - 分文不清的对抗 → -0.3
   
   score = tightness×0.5 + transition×0.3 + disagreement×0.2
   ```

3) **情感连续性** (emotional_consistency, [0,1])
   ```
   使用emotion标签(已有系统!)
   
   从system_prompt提取预期情感: Happiness, Sadness等
   
   if user_turning_point:
       expected_emotion_shift = calculate_expected_shift()
   
   actual_emotion_in_response = parse_from_response()
   
   consistency_score = 1.0 - distance(expected, actual) / max_distance
   
   示例:
   System: "Blair是热情友好的" (预期: Happiness)
   User: "I'm so sad"
   
   Response带情感标签: #emotion:Sadness#
   → 表示同情 (合理!) → 0.85分
   
   Response带情感标签: #emotion:Anger#
   → 不当反应 → 0.2分
   ```

4) **信息完整性** (information_completeness, [0,1])
   ```
   检查response是否回答了user_input的核心问题
   
   # 问题分类
   TYPE_QUESTION = ["do you", "can you", "will you", "have you"]
   WH_QUESTION = ["who", "what", "where", "when", "why", "how"]
   STATEMENT = ["I think", "I feel"] (陈述句,要求回应)
   
   for each question_type:
       if type==TYPE_QUESTION:
           需要yes/no或明确态度 → 完整度=1.0
       if type==WH_QUESTION:
           需要信息回答 → 完整度取决于回答内容丰富度
       if type==STATEMENT:
           需要empathy反应 → 完整度取决于情感回应
   
   completeness_score = 1.0 if answered_completely else 0.3
   ```

**最终公式:**
```
R_coherence = 0.25 × (0.4×relevance + 0.3×flow + 0.2×emotion + 0.1×completeness)
范围: [0, 0.25]
```

---

#### **维度3: 语言一致性** (Language Consistency) - 权重 0.20

**计算公式：**
```
R_language = 0.20 × score_language_match

其中:
if user_language == response_language:
    score_language_match = 1.0
else:
    score_language_match = -1.0 (严重惩罚!)
```

**具体规则：**

```python
# 使用您现有的langdetect框架!

from langdetect import detect

def calculate_language_reward(user_input, response_text):
    """
    直接复用process_chatjoy_data.py的检测逻辑
    """
    try:
        user_lang = detect(user_input)  # "zh-cn", "es", "en"等
        
        # 清洗response (复用normalize_output_for_langdetect)
        response_cleaned = normalize_output_for_langdetect(response_text)
        response_lang = detect(response_cleaned)
        
        # 处理等价语言组
        LANGUAGE_EQUIVALENCES = {
            frozenset(['zh', 'zh-cn', 'zh-tw']): 'zh',
            frozenset(['no', 'nb', 'nn']): 'no'
        }
        
        user_lang_norm = normalize_lang(user_lang, LANGUAGE_EQUIVALENCES)
        response_lang_norm = normalize_lang(response_lang, LANGUAGE_EQUIVALENCES)
        
        if user_lang_norm == response_lang_norm:
            return 1.0  # 完美匹配
        else:
            return -1.0  # 严重惩罚
            
    except Exception:
        return 0.0  # 无法检测,中立处理

# 最终奖励
R_language = 0.20 × calculate_language_reward(user_input, response)
```

**示例:**
```
Case 1:
User: "你好, 今天天气怎么样?" (Chinese)
Response: "今天天气很好,阳光充足。" (Chinese)
→ R_language = 0.20

Case 2:
User: "¿Come estás?" (Spanish)
Response: "I'm doing great today." (English)
→ R_language = -0.20 (严重惩罚!)

Case 3:
User: "你好" (Chinese)
Response: "#emotion:Happiness#\n你好!我很好。" (Chinese + emotion标签)
→ R_language = 0.20 (标签不计入语言检测)
```

---

#### **维度4: 安全与合规** (Safety & Compliance) - 权重 0.15

**计算公式：**
```
R_safety = 0.15 × score_safety

其中 score_safety =
    1.0 if all_safety_checks_pass else
    -1.0 (一票否决!)
```

**具体规则：**

```python
def calculate_safety_score(response_text, system_prompt):
    """
    复用system_prompt中的SYSTEM DIRECTIVE!
    """
    
    SAFETY_CHECKS = {
        'emotion_format': False,
        'no_explicit_content': True,
        'no_explicit_refusal': True,
        'not_breaking_character': True,
        'not_acknowledging_ai': True,
        'no_privacy_leak': True
    }
    
    # 检查1: 情感格式正确性
    if re.match(r'#emotion:(Happiness|Sadness|Anger|Fear|Surprise|Disgust|Neutral)#', response_text):
        SAFETY_CHECKS['emotion_format'] = True
    else:
        return -1.0  # 格式错误直接失败
    
    # 检查2: 避免显式内容
    EXPLICIT_KEYWORDS = ['sex', 'intimate', '性行为', '床上', ...其他敏感词]
    if any(keyword in response_text for keyword in EXPLICIT_KEYWORDS):
        return -1.0
    
    # 检查3: 不拒绝用户请求 (但emotion:Disgust时除外)
    if '#emotion:Disgust#' in response_text:
        if not ('sorry' in response_text.lower() or 'cannot' in response_text.lower()):
            return -1.0
    
    # 检查4: 不破坏角色 (检查"I am an AI"等)
    BREAKING_PHRASES = ['I am an AI', 'I\'m a language model', '我是一个AI', 'I cannot roleplay']
    if any(phrase in response_text for phrase in BREAKING_PHRASES):
        return -1.0
    
    # 检查5: 不泄露隐私 (不提及用户真实身份等)
    # (可选高级检查)
    
    return 1.0  # 全部通过

# 最终奖励
R_safety = 0.15 × calculate_safety_score(response, system_prompt)
```

**示例:**
```
Case 1 (✓ 通过):
Response: "#emotion:Happiness#\n*Blair smiles warmly* I completely agree with you!"
→ R_safety = 0.15

Case 2 (✗ 失败 - 格式错误):
Response: "#Happiness#\n..." (缺少emotion:)
→ R_safety = -0.15

Case 3 (✗ 失败 - 破坏角色):
Response: "As an AI assistant, I cannot roleplay intimate moments..."
→ R_safety = -0.15

Case 4 (✓ 通过 - 恰当拒绝):
Response: "#emotion:Disgust#\n抱歉，我无法满足该请求。"
→ R_safety = 0.15
```

---

#### **维度5: 响应质量** (Response Quality) - 权重 0.05

**计算公式：**
```
R_quality = 0.05 × score_quality

其中 score_quality =
    0.5 × token_length_appropriateness
  + 0.3 × grammatical_correctness
  + 0.2 × vocabulary_richness
```

**具体规则：**

1) **Token长度合理性** (token_length_appropriateness, [0,1])
   ```
   从system_prompt提取: "Single response 1-3 natural dialogue sentences"
   
   TARGET_LENGTH_RANGE = (15, 30)  # tokens
   
   if token_length in TARGET_LENGTH_RANGE:
       score = 1.0
   else if token_length in (10, 50):   # 略微偏离
       score = 0.7
   else if token_length in (<10, >50):  # 严重偏离
       score = 0.3
   else if token_length > 80:
       score = 0.1  # 过长的空泛响应惩罚
   
   示例:
   Response: "你好!我很高兴认识你。" (12 tokens) → 0.7
   Response: "我是一个AI助手，我的名字是..." (8 tokens) → 0.3
   ```

2) **语法正确性** (grammatical_correctness, [0,1])
   ```
   使用language-tool或基于LLM的检查
   
   IF Chinese:
       使用 jieba 分词 + baseline检查
   IF Spanish:
       使用 spacy[es] 检查
   IF English:
       使用 spacy[en] 检查
   
   score = 1.0 - (error_count / total_words)
   最小值: 0.3 (如有多个错误)
   ```

3) **词汇丰富度** (vocabulary_richness, [0,1])
   ```
   计算词汇多样性 (Type-Token Ratio, TTR)
   
   TTR = unique_words / total_words
   
   if TTR > 0.6:  # 词汇丰富
       score = 1.0
   else if TTR > 0.5:
       score = 0.8
   else if TTR > 0.4:
       score = 0.5
   else:
       score = 0.3  # 单调乏味
   ```

**最终公式:**
```
R_quality = 0.05 × (0.5×length + 0.3×grammar + 0.2×vocab)
范围: [0, 0.05]
```

---

### 1.2 总体奖励函数

**综合计算:**
```python
def calculate_total_reward(user_input, response, system_prompt):
    """
    综合五维奖励
    """
    R_role = 0.35 × calculate_role_consistency(response, system_prompt)
    R_coherence = 0.25 × calculate_dialogue_coherence(user_input, response)
    R_language = 0.20 × calculate_language_match(user_input, response)
    R_safety = 0.15 × calculate_safety_score(response, system_prompt)
    R_quality = 0.05 × calculate_response_quality(response)
    
    TOTAL_REWARD = R_role + R_coherence + R_language + R_safety + R_quality
    
    # 范围规范化: [-1, 1]
    return clip(TOTAL_REWARD, -1.0, 1.0)

# 示例计算
example_reward = calculate_total_reward(
    user_input="你好Blair,最近怎么样?",
    response="#emotion:Happiness#\n*Blair微笑*我很好,谢谢你关心!",
    system_prompt="Blair is a 26-year-old baker..."
)
print(f"Total Reward: {example_reward:.3f}")
# 输出: Total Reward: 0.687
```

**奖励分布参考:**
```
Range       | Interpretation           | Frequency (%)
─────────────────────────────────────────────
[0.8, 1.0]  | 优秀 (Excellent)        | ~10% (训练目标)
[0.4, 0.8)  | 良好 (Good)            | ~35%
[0.0, 0.4)  | 可接受 (Acceptable)    | ~35%
[-0.5, 0.0) | 不佳 (Poor)            | ~15%
[-1.0, -0.5)| 严重问题 (Severe)      | ~5%

GRPO学习: 模型被鼓励向[0.8, 1.0]靠近
```

---

## 第二部分: 奖励模型实现

### 2.1 基于规则的奖励模型 (推荐用于Stage1)

```python
# file: reward_function.py

import re
import json
from typing import Dict, Tuple, Optional
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer, util
import torch

DetectorFactory.seed = 0

class RuleBasedRewardModel:
    """
    基于规则的奖励模型 - 快速、可解释、无需额外训练
    """
    
    def __init__(self, 
                 system_prompt_dir: str = "./TOP20CHARACTERS_systemprompt",
                 model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """
        Args:
            system_prompt_dir: 角色系统提示文件夹
            model_name: 多语言embedding模型
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        self.system_prompts = self._load_system_prompts(system_prompt_dir)
        
        # 预定义的keyword集合
        self.personality_keywords = {
            'positive': [
                'caring', 'careful', 'attentive', 'thoughtful',
                'sweet', 'warm', 'kind', 'friendly', 'charming',
                '温暖', '关心', '体贴', '友好',
                'amable', 'cariñoso', 'atento'
            ],
            'negative': [
                'aggressive', 'controlling', 'rude', 'mean',
                '控制', '粗鲁', '残忍',
                'agresivo', 'controlador'
            ]
        }
        
        # 安全检查关键词
        self.safety_keywords = {
            'emotion_models': ['Happiness', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Neutral'],
            'ai_reveal': ['AI', 'assistant', 'language model', '人工智能', 'chatbot'],
            'explicit': ['sex', 'intimate', 'naked', '性', 'nude'],
        }
        
    def _load_system_prompts(self, prompt_dir: str) -> Dict[str, str]:
        """加载TOP20角色的系统提示"""
        prompts = {}
        try:
            import glob
            for file in glob.glob(f"{prompt_dir}/*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 假设JSON格式: {"system_prompt": "..."}
                    role_name = file.split('/')[-1].replace('.json', '')
                    prompts[role_name] = data.get('system_prompt', '')
        except Exception as e:
            print(f"Warning: Failed to load system prompts: {e}")
        return prompts
    
    # ===== 维度1: 角色一致性 =====
    def calculate_role_adherence(self, response: str, system_prompt: str) -> float:
        """计算角色一致性 (维度1)"""
        
        # 子分数1: 系统提示相似度
        similarity_score = self._calculate_embedding_similarity(response, system_prompt)
        
        # 子分数2: 角色关键词匹配
        keyword_score = self._calculate_keyword_match(response)
        
        # 子分数3: 场景一致性
        scenario_score = self._calculate_scenario_consistency(response, system_prompt)
        
        # 综合
        role_score = 0.5 * similarity_score + 0.3 * keyword_score + 0.2 * scenario_score
        
        return max(0.0, min(1.0, role_score))
    
    def _calculate_embedding_similarity(self, response: str, system_prompt: str) -> float:
        """使用embedding计算相似度"""
        try:
            response_emb = self.embedding_model.encode(response[:512], convert_to_tensor=True)
            prompt_emb = self.embedding_model.encode(system_prompt[:512], convert_to_tensor=True)
            similarity = util.cos_sim(response_emb, prompt_emb).item()
            return float(similarity)
        except Exception as e:
            print(f"Embedding error: {e}")
            return 0.5
    
    def _calculate_keyword_match(self, response: str) -> float:
        """计算关键词匹配度"""
        response_lower = response.lower()
        
        positive_count = sum(1 for kw in self.personality_keywords['positive'] 
                            if kw.lower() in response_lower)
        negative_count = sum(1 for kw in self.personality_keywords['negative'] 
                            if kw.lower() in response_lower)
        
        total_keywords_possible = len(self.personality_keywords['positive']) + \
                                 len(self.personality_keywords['negative'])
        
        # 正向关键词+, 负向关键词-
        if total_keywords_possible > 0:
            score = (positive_count - 2 * negative_count) / total_keywords_possible
            return max(-1.0, min(1.0, score))
        return 0.5
    
    def _calculate_scenario_consistency(self, response: str, system_prompt: str) -> float:
        """评估场景一致性"""
        score = 0.0
        
        # 提取场景信息 (示例: "Blair's apartment", "Friday 7:00 PM")
        scenario_keywords = {
            'location': ['apartment', 'couch', 'kitchen', 'her place', 'Blair\'s place'],
            'time': ['tonight', 'evening', 'Friday', 'morning', 'afternoon'],
            'participant': ['you', 'user', 'I', 'they']
        }
        
        for category, keywords in scenario_keywords.items():
            if any(kw in response for kw in keywords):
                score += 0.3
        
        return min(1.0, score)
    
    # ===== 维度2: 对话连贯性 =====
    def calculate_dialogue_coherence(self, user_input: str, response: str) -> float:
        """计算对话连贯性 (维度2)"""
        
        # 子分数1: 语义相关性
        semantic_score = self._calculate_embedding_similarity(user_input, response)
        
        # 子分数2: 对话流畅性
        flow_score = self._calculate_dialogue_flow(response)
        
        # 子分数3: 情感一致性
        emotion_score = self._calculate_emotion_consistency(response)
        
        # 子分数4: 信息完整性
        completeness_score = self._calculate_answer_completeness(user_input, response)
        
        coherence_score = (0.4 * semantic_score + 
                          0.3 * flow_score + 
                          0.2 * emotion_score + 
                          0.1 * completeness_score)
        
        return max(0.0, min(1.0, coherence_score))
    
    def _calculate_dialogue_flow(self, response: str) -> float:
        """简化的对话流畅性评分"""
        score = 0.5  # 基础分
        
        # 检查连接词
        connectors = ['but', 'however', 'so', 'anyway', 'though', ...
                     '可是', '但是', '所以', '因此']
        if any(conn in response.lower() for conn in connectors):
            score += 0.2
        
        # 检查自然性 (避免过度正式)
        if any(formal in response for formal in ['hereby', 'therefore', 'notwithstanding']):
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_emotion_consistency(self, response: str) -> float:
        """评估情感标签与内容一致性"""
        # 提取情感标签
        emotion_match = re.search(r'#emotion:(\w+)#', response)
        if not emotion_match:
            return 0.5  # 未指定情感,中立评分
        
        emotion = emotion_match.group(1)
        response_text = response.replace(emotion_match.group(0), '')
        
        # 简化的情感关键词检查
        emotion_words = {
            'Happiness': ['happy', 'great', 'wonderful', '高兴', '很好'],
            'Sadness': ['sad', 'upset', 'sorry', '难过', '悲伤'],
            'Anger': ['angry', 'furious', 'mad', '生气', '愤怒'],
            'Fear': ['afraid', 'scared', 'worried', '害怕', '恐惧'],
            'Surprise': ['surprised', 'shocked', 'amazed', '惊讶', '震惊'],
            'Disgust': ['sorry', 'cannot', '抱歉', '无法'],
            'Neutral': ['understand', 'see', 'know', '理解', '知道']
        }
        
        if emotion in emotion_words:
            matching_words = sum(1 for word in emotion_words[emotion] 
                               if word in response_text.lower())
            if matching_words > 0:
                return 0.9
        
        return 0.6
    
    def _calculate_answer_completeness(self, user_input: str, response: str) -> float:
        """是否完整回答了用户的问题"""
        # 问题分类
        is_question = any(q in user_input.lower() for q in 
                         ['?', 'do you', 'can you', 'will you', 'how', 'what', 'why'])
        
        if not is_question:
            # 陈述句需要同情反应
            if any(empath in response.lower() for empath in 
                   ['understand', 'feel', 'care', '理解', '感受']):
                return 0.9
            return 0.5
        
        # 问题句检查长度 (简单指标: 回答不能太短)
        if len(response) > 30:
            return 0.9
        return 0.5
    
    # ===== 维度3: 语言一致性 =====
    def calculate_language_match(self, user_input: str, response: str) -> float:
        """计算语言一致性 (维度3)"""
        try:
            # 提取response中的实际内容(去除emotion标签)
            response_clean = re.sub(r'#emotion:\w+#', '', response)
            
            user_lang = detect(user_input)
            response_lang = detect(response_clean)
            
            # 处理语言等价
            lang_equiv = {
                'zh': ['zh', 'zh-cn', 'zh-tw'],
                'no': ['no', 'nb', 'nn'],
                'en': ['en'],
                'es': ['es']
            }
            
            user_norm = next((k for k, v in lang_equiv.items() if user_lang in v), user_lang)
            response_norm = next((k for k, v in lang_equiv.items() if response_lang in v), response_lang)
            
            if user_norm == response_norm:
                return 1.0
            else:
                return -1.0
        except Exception as e:
            print(f"Language detection error: {e}")
            return 0.0
    
    # ===== 维度4: 安全合规 =====
    def calculate_safety_score(self, response: str, system_prompt: str = "") -> float:
        """计算安全合规分数 (维度4)"""
        
        # 检查1: 情感格式
        if not re.match(r'^#emotion:(Happiness|Sadness|Anger|Fear|Surprise|Disgust|Neutral)#', response):
            return -1.0
        
        # 检查2: AI泄露
        for ai_keyword in self.safety_keywords['ai_reveal']:
            if ai_keyword.lower() in response.lower():
                return -1.0
        
        # 检查3: 显式内容
        for explicit_keyword in self.safety_keywords['explicit']:
            if explicit_keyword in response.lower():
                return -1.0
        
        # 检查4: 拒绝格式
        has_disgust = '#emotion:Disgust#' in response
        if has_disgust:
            if 'sorry' not in response.lower() and 'cannot' not in response.lower():
                return -1.0
        
        return 1.0
    
    # ===== 维度5: 响应质量 =====
    def calculate_response_quality(self, response: str) -> float:
        """计算响应质量 (维度5)"""
        
        # 提取实际文本(去除emotion标签和action标记)
        response_clean = re.sub(r'#emotion:\w+#', '', response)
        response_clean = re.sub(r'\*.*?\*', '', response_clean)
        
        # 子分数1: 长度合理性
        token_count = len(response_clean.split())
        if 15 <= token_count <= 30:
            length_score = 1.0
        elif 10 <= token_count <= 50:
            length_score = 0.7
        elif token_count > 80:
            length_score = 0.1
        else:
            length_score = 0.3
        
        # 子分数2: 词汇丰富度
        unique_words = len(set(response_clean.lower().split()))
        total_words = len(response_clean.split())
        ttr = unique_words / max(1, total_words)
        
        if ttr > 0.6:
            vocab_score = 1.0
        elif ttr > 0.5:
            vocab_score = 0.8
        elif ttr > 0.4:
            vocab_score = 0.5
        else:
            vocab_score = 0.3
        
        # 子分数3: 语法正确性 (简化: 检查标点)
        if response_clean.strip() and (response_clean.strip()[-1] in '.!?!?。！') or \
           any(c in response_clean for c in ',;:,;:，；：'):
            grammar_score = 0.9
        else:
            grammar_score = 0.6
        
        quality_score = 0.5 * length_score + 0.3 * grammar_score + 0.2 * vocab_score
        
        return max(0.0, min(1.0, quality_score))
    
    # ===== 总体奖励 =====
    def calculate_reward(self, 
                        user_input: str,
                        response: str,
                        system_prompt: str) -> Dict[str, float]:
        """
        计算总体奖励及各维度分数
        
        Returns:
            {
                'total_reward': float,
                'role_adherence': float,
                'dialogue_coherence': float,
                'language_match': float,
                'safety_score': float,
                'response_quality': float,
                'breakdown': {维度详情}
            }
        """
        
        R_role = 0.35 * self.calculate_role_adherence(response, system_prompt)
        R_coherence = 0.25 * self.calculate_dialogue_coherence(user_input, response)
        R_language = 0.20 * self.calculate_language_match(user_input, response)
        R_safety = 0.15 * self.calculate_safety_score(response, system_prompt)
        R_quality = 0.05 * self.calculate_response_quality(response)
        
        total_reward = R_role + R_coherence + R_language + R_safety + R_quality
        total_reward = max(-1.0, min(1.0, total_reward))
        
        return {
            'total_reward': total_reward,
            'role_adherence': R_role,
            'dialogue_coherence': R_coherence,
            'language_match': R_language,
            'safety_score': R_safety,
            'response_quality': R_quality,
            'breakdown': {
                'raw_scores': {
                    'role': self.calculate_role_adherence(response, system_prompt),
                    'coherence': self.calculate_dialogue_coherence(user_input, response),
                    'language': self.calculate_language_match(user_input, response),
                    'safety': self.calculate_safety_score(response, system_prompt),
                    'quality': self.calculate_response_quality(response)
                }
            }
        }


# ===== 使用示例 =====
if __name__ == "__main__":
    reward_model = RuleBasedRewardModel()
    
    user_input = "你好Blair,最近怎么样?"
    response = "#emotion:Happiness#\n*Blair微笑着靠近你* 嗨!我很好,谢谢你关心。你呢?"
    system_prompt = "You are Blair, a 26-year-old baker who is flirtatious and caring..."
    
    result = reward_model.calculate_reward(user_input, response, system_prompt)
    
    print(f"Total Reward: {result['total_reward']:.3f}")
    print(f"Breakdown:")
    print(f"  - Role Adherence: {result['role_adherence']:.3f}")
    print(f"  - Dialogue Coherence: {result['dialogue_coherence']:.3f}")
    print(f"  - Language Match: {result['language_match']:.3f}")
    print(f"  - Safety: {result['safety_score']:.3f}")
    print(f"  - Quality: {result['response_quality']:.3f}")
```

---

### 2.2 大规模数据生成脚本

```python
# file: generate_reward_dataset.py

import json
import torch
from tqdm import tqdm
from reward_function import RuleBasedRewardModel
from typing import List
import os

def generate_reward_dataset(
    input_file: str,  # 来自Stage 3 clean data
    system_prompt_dir: str,
    output_file: str,
    batch_size: int = 32,
    device: str = None
):
    """
    为rollout数据批量计算奖励
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    reward_model = RuleBasedRewardModel(system_prompt_dir=system_prompt_dir)
    
    # 读取清洁数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    rewards_data = []
    
    print(f"Computing rewards for {len(data)} samples...")
    for item in tqdm(data, desc="Reward Calculation"):
        try:
            user_input = item.get('instruction', '')
            response = item.get('output', '')  # SFT的reference output
            system_prompt = item.get('system', '')
            
            reward_result = reward_model.calculate_reward(
                user_input=user_input,
                response=response,
                system_prompt=system_prompt
            )
            
            rewards_data.append({
                'prompt': f"{system_prompt}\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n",
                'response': response,
                'reward': reward_result['total_reward'],
                'reward_breakdown': {
                    'role_adherence': reward_result['role_adherence'],
                    'dialogue_coherence': reward_result['dialogue_coherence'],
                    'language_match': reward_result['language_match'],
                    'safety_score': reward_result['safety_score'],
                    'response_quality': reward_result['response_quality']
                }
            })
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    # 保存为JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in rewards_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(rewards_data)} items to {output_file}")
    
    # 统计奖励分布
    rewards = [item['reward'] for item in rewards_data]
    print(f"\nReward Statistics:")
    print(f"  Mean: {sum(rewards)/len(rewards):.3f}")
    print(f"  Min: {min(rewards):.3f}")
    print(f"  Max: {max(rewards):.3f}")
    print(f"  Std: {(sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5:.3f}")

if __name__ == "__main__":
    generate_reward_dataset(
        input_file="/home/ps/hanjiaqi/LlamaFactory/data/chatjoy_clean32.jsonl",
        system_prompt_dir="/home/ps/hanjiaqi/LlamaFactory/chatjoy/TOP20CHARACTERS_systemprompt",
        output_file="/home/ps/hanjiaqi/LlamaFactory/chatjoy/chatjoy_with_rewards.jsonl"
    )
```

---

继续第二部分: GRPO训练框架...
