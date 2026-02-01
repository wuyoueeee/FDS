# 基于YOLO与大模型的骨折智能诊断系统
## Intelligent Upper Limb Fracture Diagnosis System Based on Deep Learning and Multimodal Large Models

> **关键词**：YOLO, 目标检测, LLaVA, 多模态医学影像分析, 智慧医疗, 辅助诊断

---
>代码获取地址：[https://mbd.pub/o/bread/YZWbkpptag==](https://mbd.pub/o/bread/YZWbkpptag==)
## 1. 系统概述 (System Overview)

本系统旨在解决传统医学影像诊断中存在的**阅片效率低、微小骨折易漏诊**以及**基层医疗资源匮乏**等痛点。项目创新性地将**单阶段目标检测算法 (YOLO)** 与 **多模态大语言模型 (LLaVA)** 相结合，构建了一个集**毫秒级骨折定位**、**病灶分类**及**自动生成医学诊断报告**于一体的智能辅助诊断平台。

系统不仅能够精准识别肘部、腕部、指骨等多个部位的骨折特征，还能模拟专业医生的口吻提供详细的病理描述与处置建议，为临床医生提供强有力的决策支持，具有重要的学术研究价值与临床应用前景。

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4e98fbd9f63e4df592ddea3986137b6d.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b49dd6d4a5b3476da2fdeb90327e4f83.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7c0d97e8bec24d28ba2be6aa486d7f86.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/796eaa09dc40426e99af330e0ed8a223.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e63e1b581ff34121b259fe00e75a61e6.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3c5432eb7c3044d19f40b3ccaab2f82a.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0a614580287447e3adb22f834a908681.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2210e6e739f14384873c85fceaab6962.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3dd4a58266964d379059291461bf4f11.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/93f534b4094741b7bc20536265463642.jpeg#pic_center)

## 2. 数据集介绍 (Dataset Description)

本项目采用的数据集源自公开医学影像数据库，经过专业的清洗与标准化处理，专门用于上肢骨折检测任务。

### 2.1 数据集概况
- **数据来源**：Roboflow Universe (Bone Fracture Detection)
- **数据格式**：YOLO 格式 (`.txt` 标注文件 + `.jpg` 影像文件)
- **数据分布**：
  - 训练集 (Train)：用于模型特征学习与权重更新。
  - 验证集 (Validation)：用于训练过程中的超参数调整与模型评估。
  - 测试集 (Test)：用于最终模型的泛化能力测试。

### 2.2 类别定义 (Class Definitions)
数据集包含 **7个** 细分医学类别，覆盖了上肢常见的骨折类型：

| ID | 类别名称 (Class Name) | 中文医学术语 | 临床意义 |
|----|----------------------|--------------|----------|
| 0  | `elbow positive`     | 肘关节骨折   | 常见于跌倒撑地，需关注肱骨髁上区域。 |
| 1  | `fingers positive`   | 指骨骨折     | 常见于挤压伤，需区分关节内与关节外骨折。 |
| 2  | `forearm fracture`   | 前臂骨折     | 包括尺骨、桡骨干骨折，需评估移位情况。 |
| 3  | `humerus fracture`   | 肱骨主要骨折 | 大结节或外科颈骨折，多见于老年人。 |
| 4  | `humerus`            | 肱骨异常关注 | 疑似骨质改变或轻微裂纹，需重点复查。 |
| 5  | `shoulder fracture`  | 肩关节骨折   | 包括锁骨远端、肩胛骨等部位骨折。 |
| 6  | `wrist positive`     | 腕关节骨折   | Colles骨折或Smith骨折，老年人高发。 |

---

## 3. 算法原理与技术路线 (Algorithm & Methodology)

### 3.1 视觉检测核心：YOLOv8
本系统采用 UltraLytics 公司发布的 **YOLOv8 (You Only Look Once Version 8)** 作为核心视觉检测引擎。相比于前代算法（YOLOv5/v7），YOLOv8 在医学小目标检测上具有显著优势：

*   **Backbone (主干网络)**：采用改进的 `CSPDarknet53` 结构，引入 `C2f` 模块替代了 C3 模块，通过更丰富的梯度流提升了特征提取能力，有助于捕捉骨折线等细微纹理特征。
*   **Neck (颈部网络)**：沿用 `PANet (Path Aggregation Network)` 结构，增强了多尺度特征融合能力，确保系统既能检测明显的长骨骨折（大目标），也能识别指骨微裂（小目标）。
*   **Head (检测头)**：采用 **Decoupled Head (解耦头)** 设计，将分类任务与回归任务分离处理，解决了单一检测头在复杂医学场景下的任务冲突问题。
*   **Loss Function (损失函数)**：引入 `DFL (Distribution Focal Loss)` 与 `CIoU Loss`，提升了边界框回归的精度，确保病灶定位准确。
*   **Anchor-free (无锚框)**：抛弃了传统的 Anchor-based 机制，减少了超参数对模型性能的影响，更适应形态各异的骨折区域。

### 3.2 多模态分析核心：LLaVA (Large Language-and-Vision Assistant)
系统集成 **LLaVA** 作为“AI 阅片医生”。LLaVA 是一种端到端训练的多模态大模型，它连接了视觉编码器 (CLIP ViT-L/14) 与大语言模型 (Vicuna/Llama)。

*   **工作机制**：
    1.  **视觉感知**：X光图像通过 CLIP 视觉编码器被映射为视觉特征向量。
    2.  **特征对齐**：通过投影层 (Projection Layer) 将视觉特征对齐到语言模型的词嵌入空间。
    3.  **多模态推理**：YOLO 检测到的骨折区域信息（位置、类别）被转化为文本提示 (Prompt)，与视觉特征一同输入 LLM。
    4.  **报告生成**：LLM 结合医学知识库，生成包含诊断结论、病情严重度评估及建议治疗方案的结构化中文报告。

---

## 4. 系统架构设计 (System Architecture)

系统采用 **B/S (Browser/Server)** 架构，后端基于 Python 进行算法调度，前端采用 Gradio 构建交互界面。


## 5. 核心代码解析 (Core Code Analysis)

系统代码结构清晰，遵循面向对象编程 (OOP) 原则，主要包含以下核心类：

### 5.1 `src/bone_fracture_detection_system.py`
**功能**：系统的核心控制器，负责协调各个子模块的工作。

**核心逻辑 - 智能诊断流程**：
```python
def detect_image(self, image_path, conf=0.25, use_llava=False):
    # 1. 启动 YOLOv8 进行视觉检测
    results, plot_img = self.detector.predict(image_path, conf=conf, class_map=self.class_map)
    
    # 2. 解析检测结果并进行风险评估
    is_risk = False
    if results:
        # 提取骨折位置与置信度
        # ... (解析逻辑)
        is_risk = True
    
    # 3. 如果需要，调用 LLaVA 进行深度医学分析
    report = ""
    if use_llava:
        # 将视觉检测到的框(BBox)信息注入到 Prompt 中
        detection_info = {"detections": detections, "is_risk": is_risk}
        # 核心：融合视觉特征与先验知识
        analysis_result = self.analyzer.analyze_with_detection(
            image_path, 
            detection_info, 
            prompt_key='bone_fracture_analysis'
        )
        report = analysis_result.get('llava_analysis', '')
        
    # 4. 自动归档与数据库记录
    self.history.add_detection_record(...)
    
    return plot_img, detections, report, save_path, is_risk
```
- **`detect_image()`**：全流程处理函数。调用 YOLO 进行检测，根据阈值筛选结果，若发现骨折则触发 LLaVA 进行深度分析，最后将结果保存并写入数据库。
- **`detect_batch()`**：实现了多线程/批量的影像处理逻辑，并自动生成统计报表。

### 5.2 `src/yolo_detector.py`
**功能**：对 YOLOv8 模型的封装类。
- **`train()`**：集成了针对医学影像优化的训练管线，支持自动早停 (Early Stopping)、目标阈值停止 (Target Stopping) 等高级策略。
- **`predict()`**：实现了推理逻辑，并内置了针对医学影像的**中文可视化绘图**功能，解决了 OpenCV 不支持中文绘制的问题。

### 5.3 `src/llava_analyzer.py`
**功能**：LLaVA 大模型接口类。

**核心逻辑 - 多模态提示词工程 (Prompt Engineering)**：
```python
def analyze_with_detection(self, image_path: str, detection_info: dict, prompt_key: str) -> dict:
    # 1. 动态构建上下文信息
    num = len(detection_info.get('detections', []))
    detection_summary = f"YOLO模型检测结果: 在画面中检测到 {num} 个目标。\n"
    
    if num > 0:
        for det in detection_info['detections']:
            # 注入精确的坐标与置信度，引导 LLM 关注特定区域
            detection_summary += f"- {det['class']}: 坐标({det['bbox']}), 置信度: {det['conf']:.2%}\n"
    
    # 2. 获取预设的医学 Prompt (如 'bone_fracture_analysis')
    base_prompt = self.llava_config['prompts'].get(prompt_key)
    
    # 3. 构造 Chain-of-Thought (CoT) 完整提示
    full_prompt = f"""
    【前置检测信息】
    {detection_summary}

    【分析任务】
    {base_prompt}
    """
    
    # 4. 调用多模态大模型进行推理
    # ... (API 调用逻辑)
```
- **`analyze_with_detection()`**：核心接口。它不仅发送图片，还将 YOLO 检测到的先验信息（如“检测到肱骨骨折”）注入到 Prompt 中，引导大模型生成更精准的报告，有效减少了多模态幻觉 (Hallucination)。

### 5.4 `src/web_interface_advanced.py`
**功能**：基于 Gradio 的高级 Web 交互层。
- 定义了“医学蓝”主题 UI。
- 实现了单图诊断、批量筛查、实时流检测、病历管理四大功能模块的前端逻辑。

---

## 6. 环境部署与使用 (Usage)

### 6.1 环境依赖
推荐使用 Python 3.8+ 及 NVIDIA GPU (CUDA 11.8+)。
```bash
# 安装项目依赖
pip install -r requirements.txt
```

### 6.2 启动系统
```bash
# 启动 Web 可视化界面
python run_web_advanced.py
```
启动后访问：`http://localhost:7860`

### 6.3 启用 AI 报告 (可选)
需安装 Ollama 并加载 LLaVA 模型：
```bash
# 下载并运行 LLaVA
ollama run llava
```

---

## 7. 结论 (Conclusion)
本系统通过融合先进的目标检测技术与多模态大语言模型，实现了一个高效、准确且具有可解释性的骨折智能诊断平台。实验表明，该系统能有效辅助医生减少漏诊率，提升阅片效率，为智慧医疗建设提供了一种可行的技术方案。
