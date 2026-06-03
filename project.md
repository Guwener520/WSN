### 📡 项目名称：基于改进群体智能算法的 WSN 覆盖优化研究

#### 1. 核心目标
在给定监测区域内，利用**群体智能算法（SI）**自动优化 $N$ 个传感器节点的部署位置，以解决以下多目标优化问题：
*   **最大化覆盖率**：减少监测盲区。
*   **最小化冗余度**：避免节点过度重叠，节约资源。
*   **均衡能耗/连通性**：确保网络寿命和通信稳定（可选进阶目标）。

#### 2. 技术栈
*   **语言**：Python 3.x
*   **核心库**：NumPy (矩阵运算), Matplotlib (静态绘图), Plotly (交互式/动态可视化)
*   **算法库**：手动实现核心 SI 逻辑（不直接调用黑盒库，以体现课程学习成果）

#### 3. 项目内容模块

##### **模块一：WSN 覆盖数学模型搭建 **
*   **环境建模**：将监测区域网格化（如 $50 \times 50$ 或 $100 \times 100$ 网格）。
*   **感知模型**：采用**二元感知模型**（Binary Sensing Model），即距离 $\le R_s$ 则覆盖，否则不覆盖。
*   **适应度函数设计**：
    $$ Fitness = w_1 \cdot CoverageRate + w_2 \cdot (1 - RedundancyRate) + w_3 \cdot Connectivity $$
*   **参考资源利用**：
    *   参考 **[TakwaKhelifi/Memetic-WSN-Coverage-Optimization](https://github.com/TakwaKhelifi/Memetic-WSN-Coverage-Optimization)** 中的网格化处理逻辑和适应度函数定义，理解硕士论文级别的建模严谨性。

##### **模块二：群体智能算法实现与改进 **
*   **基础算法实现**：
    *   **PSO (粒子群优化)**：适合连续空间搜索，收敛速度快。
    *   **GWO (灰狼优化)** 或 **WOA (鲸鱼优化)**：模拟社会等级或包围捕食行为，平衡勘探与开发。
*   **改进策略（课程项目亮点）**：
    *   **混沌初始化**：使用 Logistic 映射生成初始种群，提高多样性。
    *   **自适应参数**：动态调整惯性权重 $w$ 或收敛因子 $a$，防止早熟收敛。
    *   **混合变异**：在迭代后期引入高斯变异或 Levy 飞行，帮助跳出局部最优。
*   **参考资源利用**：
    *   参考 **[AliAmini93/Metaheuristic-Optimization](https://github.com/AliAmini93/Metaheuristic-Optimization)** ⭐102，借鉴其 GA/ACO/PSO 的标准实现框架，确保代码结构的规范性。
    *   结合 **2024年 *Cluster Computing* 综述论文** 中的结论，选择表现较好的改进策略进行复现和对比。

##### **模块三：可视化与实验对比分析 **
*   **可视化展示**：
    *   **静态图**：最终节点部署散点图 + 覆盖热力图（Heatmap）。
    *   **动态图**：制作 GIF/视频，展示节点从随机分布到逐步优化聚集的过程（体现“群体智能”的涌现性）。
    *   **交互图**：使用 Plotly 实现可缩放、可悬停查看节点信息的仪表盘。
*   **对比实验**：
    *   **Baseline**：随机部署、网格部署。
    *   **算法间对比**：Standard PSO vs. Improved GWO vs. Standard GWO。
    *   **指标**：最佳覆盖率、平均覆盖率、标准差（稳定性）、收敛迭代次数。
*   **参考资源利用**：
    *   借鉴 **[TakwaKhelifi](https://github.com/TakwaKhelifi/Memetic-WSN-Coverage-Optimization)** 中的可视化思路，提升图表的专业度。

