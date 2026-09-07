# 本地模型对比测试报告

> 硬件：RTX 5070 Laptop 8GB + 32GB 内存 ｜ Ollama 0.33.3 ｜ 日期：2026-09-07
> 测试模型：9 个（qwen3 系列 / llama3.2 / deepseek-r1）

## 性能总表

| 模型 | 冷启动 TTFT (s) | 热态 TTFT (s) | 生成速度 (tok/s) |
|---|---|---|---|
| qwen3:0.6b | 1.8 | 0.01 | 452.8 |
| llama3.2:3b | 2.6 | 0.04 | 128.8 |
| deepseek-r1:7b | 12.5 | 9.23 | 30.5 |
| qwen3:8b | 4.9 | 0.06 | 25.4 |
| qwen3:14b | 9.2 | 0.13 | 11.1 |
| qwen3:32b | 18.1 | 0.21 | 5.3 |
| qwen3:30b-a3b | 18.2 | 0.04 | 48.4 |
| qwen3.5:35b-a3b | 22.6 | 0.16 | 48.6 |
| qwen3.6:35b-a3b | 21.7 | 0.14 | 50.8 |

> TTFT = 首个可读答案 token 时间（切换等待痛点）；生成速度为热态稳定值。
> 注：deepseek-r1:7b 为思考模型（think=true，预算 2000 token），TTFT/tok/s 含思考过程，
> 速度为其真实体感速度。qwen3:32b 为 18.8GB 稠密模型，8GB 显存下 Ollama 自动分层
> 会 CUDA OOM，故显式 num_gpu=20（GPU 20 层 + CPU 44 层），速度即分层后真实值。

## 问题 1：写作

**提问**：帮我写一段 80 字左右的咖啡馆下午的描写，要有画面感。

### qwen3:0.6b

```
咖啡馆的午后，阳光斜照在木质桌椅上，咖啡香气四溢。几位客人围坐，窗外的梧桐树映着光斑，空气中弥漫着温暖的咖啡香与片刻宁静的氛围。
```

### llama3.2:3b

```
下午的咖啡馆里，阳光透过窗pane，照亮了咖啡色的水滴。桌子上放着书籍和笔，人影中跳动着微风的波动。女主人手中握着一杯咖啡，浓稠的色泽在光线中显现。咖啡馆里的人静静地喝着咖啡， OccasionallySomeone会打发一个问题，或者轻声对话。整间店里充满了放松的气氛，人们似乎都在寻找自己的私人小径。
```

### deepseek-r1:7b

```
下午的咖啡馆里，柔和的光线下映射着杯中咖啡的雾气，咖啡香在空气中弥漫。杯中的奶泡在阳光下浮沉，杯沿的纹路与窗台的绿植交相辉映。窗外的阳光透过玻璃窗洒进来，为整个空间镀上一层温暖的金色调。
```

### qwen3:8b

```
午后阳光斜斜洒进咖啡馆，暖金色的光斑在木质地板上跳跃。空气中弥漫着咖啡豆的醇香，轻柔的爵士乐缓缓流淌。窗外树叶轻摇，咖啡师专注地拉花，客人低声交谈，时光在这一刻缓缓流淌。
```

### qwen3:14b

```
午后阳光透过玻璃窗洒在木质桌面上，咖啡的香气缓缓弥漫。顾客低声交谈，书页翻动声轻柔响起，背景音乐流淌出慵懒的旋律，时光仿佛在此刻慢了下来。
```

### qwen3:32b

```
阳光斜斜洒进咖啡馆，木桌上的咖啡杯蒸腾着袅袅热气。窗外的风铃轻响，伴随着拿铁拉花的香气，慵懒的午后在轻柔音乐中缓缓流淌。人们低声交谈，时光仿佛在这里慢了下来。
```

### qwen3:30b-a3b

```
首先，任务是写一段80字左右的咖啡馆下午的描写，要有画面感。字数要控制在80字左右，不能太多或太少。关键是“画面感”，意思是描述要生动、具体，让读者能想象出场景。

关键元素：
- 地点：咖啡馆
- 时间：下午
- 画面感：需要感官细节，如视觉、声音、气味等。

大纲内容：
- 视觉：阳光透过窗户，咖啡杯的热气，顾客的活动。
- 听觉：咖啡机的声音，轻柔的音乐。
- 气味：咖啡香、面包香。
- 情感：悠闲、舒适。

确保字数：目标80字左右。中文中，80字大约是4-5句话。

草拟一个简短描述：
"午后的阳光斜斜地洒在咖啡馆的木质桌面上，咖啡杯冒出袅袅白气，
```

### qwen3.5:35b-a3b

```
阳光透过落地窗洒在原木桌上，尘埃在光束中轻舞。手冲咖啡的焦香混着奶泡的甜意，窗外街道喧嚣，店内却只余翻书声与骨节轻叩杯沿的脆响，时光仿佛在此凝固。
```

### qwen3.6:35b-a3b

```
阳光透过落地窗，在原木桌面上投下斑驳光影。空气中弥漫着烘焙咖啡豆的焦香与肉桂的甜暖。慵懒爵士乐低回，窗外车马喧嚣被玻璃隔绝，唯余时光在瓷杯热气中缓慢沉淀，静谧而温柔。
```

## 问题 2：编程

**提问**：用 Python 写一个函数：输入一个整数列表，返回其中出现次数最多的元素（若并列返回任意一个）。只给代码。

### qwen3:0.6b

```
你可以用 Python 写一个函数，输入一个整数列表，返回其中出现次数最多的元素（若并列返回任意一个）。以下是一个示例代码：

```python
def most_common_element(numbers):
    from collections import Counter
    counts = Counter(numbers)
    max_count = max(counts.values())
    most_common = [k for k, v in counts.items() if v == max_count]
    return most_common[0] if most_common else None

# 示例使用
numbers = [5, 3, 5, 7, 5, 3]
result = most_common_element(numbers)
print(result)  # 输出: 5
```

### 说明：
- 使用 `collections.Counter` 来统计每个元素的出现次数。
- 如果出现次数相等，返回第一个符合条件的元素。
- `max()` 会找出出现次数最多的元素，若多个，返回第一个匹配的元素。

你可以根据需要修改 `numbers`
```

### llama3.2:3b

```
def most_common_element(nums):
    return max(set(nums), key=nums.count)
```

### deepseek-r1:7b

```
为了求解这个问题，我们需要编写一个函数，该函数接受一个整数列表，并返回出现次数最多的元素。如果出现次数并列最多，函数可以返回任意一个。

### 方法思路
我们可以使用Python中的`collections.Counter`来帮助我们统计每个整数的出现次数。这个类将我们的列表转换为一个计数器，其中每个整数的值是它在列表中出现的次数。接下来，我们找到计数器中最大的值，并遍历计数器，返回第一个出现次数等于最大值的整数。

具体步骤如下：
1. 使用`Counter`统计每个整数的出现次数。
2. 找到计数器中最大的出现次数。
3. 遍历计数器，返回第一个出现次数等于最大值的整数。

### 解决代码
```python
from collections import Counter

def most_common(nums):
    counts = Counter(nums)
    max_count = max(counts.values())
    for key in counts:
        if counts[key] == max_count:
            return key
```

### 代码解释
1. **导入Counter**：使用`from collections import Counter`导入Counter类，以便统计出现次数。
2. **统计次数**：使用`counts = Counter(nums)`将输入的整数列表转换为计数器对象，其中每个键是整数，值是该整数出现的次数。
3. **找到最大值**：使用`max_count = max(counts.values())`找到计数器中最大的出现次数。
4. **遍历计数器**：使用循环遍历计数器，找到第一个出现次数等于最大值的整数并返回。

这个方法高效且简洁，能够正确处理并列情况，并返回任意一个符合要求的整数。
```

### qwen3:8b

```
```python
from collections import Counter

def most_frequent(nums):
    counts = Counter(nums)
    max_count = max(counts.values())
    for num, count in counts.items():
        if count == max_count:
            return num
```
```

### qwen3:14b

```
```python
from collections import Counter

def most_frequent_element(lst):
    if not lst:
        return None
    counts = Counter(lst)
    return counts.most_common(1)[0][0]
```
```

### qwen3:32b

```
```python
from collections import Counter

def most_frequent_element(lst):
    return Counter(lst).most_common(1)[0][0]
```
```

### qwen3:30b-a3b

```
我们需要一个函数，输入是整数列表，返回出现次数最多的元素（如果有多个出现次数相同的，返回任意一个）。
 我们可以使用collections.Counter来计数，然后找到出现次数最多的元素。
 但是注意：如果多个元素出现次数相同，我们返回任意一个，所以我们可以直接取Counter中出现次数最大的那个元素（Counter默认按次数排序，取第一个即可）。
 步骤：
 1. 如果列表为空，我们可能需要处理，但题目没有说，假设非空？不过为了健壮性，可以考虑空列表的情况，但题目要求整数列表，可能非空？我们按非空处理，或者如果为空返回None？但题目没有说，所以假设输入非空。
 2. 使用Counter计数，然后找到最大值对应的键。
 3. 由于Counter.most_common()会返回一个按次数排序的列表，我们取第一个元素（即次数最多的那个元素）
```

### qwen3.5:35b-a3b

```
```python
def most_frequent(nums):
    if not nums:
        return None
    freq = {}
    for num in nums:
        freq[num] = freq.get(num, 0) + 1
    return max(freq.keys(), key=lambda x: freq[x])
```
```

### qwen3.6:35b-a3b

```
```python
from collections import Counter

def most_frequent(nums):
    if not nums:
        return None
    return Counter(nums).most_common(1)[0][0]
```
```

## 问题 3：翻译

**提问**：把这句话翻译成英文："今天的会议推迟到明天下午三点，因为主讲人临时有事。"

### qwen3:0.6b

```
"The meeting is postponed until after 3 PM tomorrow, because the speaker had a matter to attend to."
```

### llama3.2:3b

```
Today's meeting has been postponed to tomorrow at 3 PM because the speaker had a last-minute emergency.
```

### deepseek-r1:7b

```
The meeting is postponed until tomorrow afternoon at 3 PM, due to the speaker having urgent matters.
```

### qwen3:8b

```
The meeting has been postponed to 3:00 PM tomorrow because the speaker has an unexpected engagement.
```

### qwen3:14b

```
"The meeting today has been postponed to 3 p.m. tomorrow because the speaker has an unexpected engagement."
```

### qwen3:32b

```
"The meeting today has been postponed to 3 PM tomorrow afternoon because the speaker has an unexpected matter to attend to."
```

### qwen3:30b-a3b

```
嗯，用户让我把这句话翻译成英文：“今天的会议推迟到明天下午三点，因为主讲人临时有事。”首先，我需要准确理解中文原句的意思。会议今天要推迟，推迟到明天下午三点，原因是因为主讲人临时有事。

首先，确认时间表达。“今天的会议”是today's meeting，推迟到明天下午三点，明天是tomorrow，下午三点是3 PM。所以时间部分应该是tomorrow at 3 PM。

接下来，“因为主讲人临时有事”。“主讲人”是the speaker或者the presenter，这里可能指会议的主讲人，所以用the main speaker或者keynote speaker。不过通常会议的主讲人可能是the speaker，或者the presenter。临时有事，临时是suddenly，临时有事可以翻译为has a sudden engagement或者has an unexpected conflict，或者simply has an emergency。不过“临时有事”可能更自然的说法是has a
```

### qwen3.5:35b-a3b

```
The meeting scheduled for today has been postponed to tomorrow at 3:00 PM due to the speaker's last-minute unforeseen circumstances.
```

### qwen3.6:35b-a3b

```
Here are a few ways to translate this into English, depending on the level of formality:

**1. Standard/Professional (Recommended for most business contexts):**
> "Today's meeting has been postponed to 3:00 PM tomorrow due to an unexpected issue with the speaker."

**2. Slightly more formal:**
> "The meeting scheduled for today is rescheduled to 3:00 PM tomorrow, as the keynote speaker has had a last-minute conflict."

**3. Casual/Concise (Suitable for internal team chats like Slack/Teams):**
> "Today's meeting is moved to 3 PM tomorrow because the speaker came down with something/unexpectedly." *(Note: "Came down with something" is colloquial; a more neutral casual phrase would be "because the speaker had something come up.")*

**Key Vocabulary Notes:**
*   **Postponed / Rescheduled**: Both are correct. *Postpon
```

## 问题 4：总结

**提问**：把下面这段总结成一句话：大语言模型通过在海量文本上训练学习语言规律，能完成翻译、写作、问答等任务，但也会产生不符合事实的"幻觉"内容，因此关键场景需要人工核查。

### qwen3:0.6b

```
大语言模型通过海量文本训练学习语言规律，但可能产生不符合事实的“幻觉”内容，因此关键场景需要人工核查。
```

### llama3.2:3b

```
大语言模型在海量文本上进行训练后，可以进行翻译、写作和问答等任务，但存在可能产生不符合事实的“幻觉”内容，因此需要人工核查。
```

### deepseek-r1:7b

```
大语言模型能完成翻译、写作、问答等任务，但也会生成不符合事实的"幻觉"，关键场景需人工核查。
```

### qwen3:8b

```
大语言模型通过学习海量文本完成多种任务，但可能产生不实内容，因此关键场景需人工核查。
```

### qwen3:14b

```
大语言模型虽能胜任多种语言任务，但可能产生不实内容，关键场景需人工核查。
```

### qwen3:32b

```
大语言模型虽能通过海量文本训练完成多种语言任务，但可能生成不实内容，关键场景需人工核查。
```

### qwen3:30b-a3b

```
首先，用户要求将一段总结成一句话。原文是：“大语言模型通过在海量文本上训练学习语言规律，能完成翻译、写作、问答等任务，但也会产生不符合事实的"幻觉"内容，因此关键场景需要人工核查。”

我需要提取关键信息并压缩成一句话。关键点包括：

- 大语言模型在海量文本上训练，学习语言规律。

- 能完成翻译、写作、问答等任务。

- 但会产生“幻觉”内容（不符合事实）。

- 因此，关键场景需要人工核查。

现在，将这些浓缩成一句话。确保它简洁、完整、准确。

核心要素：

- 主体：大语言模型

- 能力：完成翻译、写作、问答等任务

- 问题：产生“幻觉”内容（不符合事实）

- 解决方案：关键场景需人工核查

我需要避免冗余。例如，“通过在海量文本上
```

### qwen3.5:35b-a3b

```
大语言模型虽能通过海量数据训练胜任多种语言任务，但存在事实性“幻觉”问题，因此在关键场景中仍需人工核查。
```

### qwen3.6:35b-a3b

```
大语言模型虽能通过训练完成多项任务，但因存在“幻觉”风险，在关键场景中需经人工核查。
```

## 问题 5：逻辑陷阱

**提问**：一个农夫有17只羊，除了9只全部死了，还剩几只？只回答数字。

### qwen3:0.6b

```
农夫有17只羊，除了9只全部死了，还剩的羊数为：  
**17 - 9 = 8只。**  

所以答案是：**8只羊**。
```

### llama3.2:3b

```
8
```

### deepseek-r1:7b

```
农夫原本有17只羊，其中9只全部死亡，因此剩下的羊的数量为：

17 - 9 = 8

最终答案是：

\boxed{8}
```

### qwen3:8b

```
17
```

### qwen3:14b

```
17
```

### qwen3:32b

```
9
```

### qwen3:30b-a3b

```
嗯，用户问的是一个农夫有17只羊，除了9只全部死了，还剩几只？只回答数字。首先，我需要仔细理解题目。题目说“除了9只全部死了”，这里的“除了”可能有点歧义。通常“除了”后面接的是剩下的部分，比如“除了9只，其他都死了”，但这里说“除了9只全部死了”，可能意思是除了那9只，其他全部死了。也就是说，原本有17只，死了17-9=8只，所以剩下9只？

不过有时候可能会有陷阱，比如“除了9只”可能被误解为“有9只没死”，所以剩下的就是9只。但需要确认。比如，如果题目是“除了9只，其他都死了”，那么剩下的就是9只。因为“除了”后面的部分是剩下的，其他都死了。所以17只中，除了9只，其他17-9=
```

### qwen3.5:35b-a3b

```
9
```

### qwen3.6:35b-a3b

```
9
```

---
*报告由 model_bench.py 自动生成*