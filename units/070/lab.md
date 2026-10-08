# 第070讲 实验指导书

## 1 实验目标

用已知连续结构检查二维图，而不是挑一张最漂亮的图。你将重现PCA、六组t-SNE、七组UMAP和两组Gaussian负对照，计算输入邻域可信度、近邻重合比例与两种全局距离秩相关。重点是解释指标之间为什么可能不一致。

建议安排120至180分钟。CPU执行即可，不需要GPU或外部数据下载。首次UMAP执行可能触发Numba编译，等待数秒至数十秒属于正常现象；不要把它误当成所有后续运行都会重复的算法成本。

## 2 环境与命令

数值环境固定Python3.12、NumPy2.3.5、SciPy1.17.0、scikit-learn1.8.0与umap-learn0.5.9.post2。完整依赖见requirements.txt。首次安装需联网，实验数据与代码均在本单元内。

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/my-run/result.json
python test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests.json
python -O test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests-O.json
python plots.py --report outputs/my-run/result.json --directory outputs/my-run/figures
python execute_notebook.py --out outputs/my-run/experiment.ipynb
```

需要更可控的复现时，将OPENBLAS_NUM_THREADS、OMP_NUM_THREADS和NUMBA_NUM_THREADS设为1。UMAP构造器本身也固定n_jobs=1。本次使用既有固定版本环境运行，未额外验证干净Conda安装。

PDF重建另需build-requirements.txt、package.json与系统Pango/Noto CJK/DejaVu字体。bash build_all.sh输出到outputs/rebuild，允许设置PYTHON_BIN和NODE_PATH。不会自动安装软件，不覆盖参考材料。Notebook使用新Python进程中的真实IPython InProcessKernel，Jupyter网页和外部socket内核未验证。

## 3 数据字典与评价边界

roll.csv有320行。id是合成样本编号，x0、x1、x2为观测三维坐标，t是连续卷曲参数，height是高度，arc_length为解析弧长。算法输入仅x0至x2；t只用于颜色，height与arc_length只用于内在距离核验，不得悄悄拼接成模型特征。

gaussian.csv有240行，id与x0至x5六个独立标准Gaussian数值。它来自单一分布，没有预设混合簇或类别标签。固定生成种子70021，generate_data.py可在新目录重建CSV。读取同时验证文件摘要和源生成器字节。

本实验所有点参与映射，随后在这些相同点上检查几何关系。这是探索性表示诊断，不是预测泛化测试。不要把T10或距离相关写成测试准确率。若要检验样本外变换，另建训练/测试设计，并明确t-SNE与UMAP的API能力差别。

## 4 在运行前写下配置

PCA固定2维与full SVD。t-SNE固定2维、init=random、learning_rate=auto、max_iter=750、method=exact；perplexity为5、30、80，种子0与1，共6组。UMAP固定2维、n_epochs=350、n_jobs=1；n_neighbors为5、30、80，min_dist=0.1，种子0与1，共6组；再加30邻居、种子0、min_dist=0.8一组。

所以卷曲数据记录必须有14组。Gaussian负对照另有2组t-SNE，perplexity均为30，种子0与1。所有结果都保存，不能只保留看上去最好的一张。改变任意参数时，保存成新实验，不覆盖原报告。

## 5 手算邻域指标

原位置X=(0,1,3,7)，映射位置Z=(0,3,1,7)，按数组索引0至3给点编号。k=1时，先为每点排除自身，再计算原空间最近邻和映射最近邻。原最近邻依次为1、0、1、2；映射最近邻为2、2、0、1。

交集全为空，因此R1=0。每个映射近邻在原空间排名2，因此各惩罚2-1=1，总和4。代入T公式得到1-2×4/[4×1×(8-3-1)]=0.5。请解释：T没有直接数“保留比例”，它给错误近邻按原始排名分级惩罚。

打开neighborhoods.py核对实现。ordering将自身距离设为无穷后稳定排序；trust建立排名矩阵，取二维前k邻居对应的原排名；rank-k小于0的部分截为0。k必须是非布尔整数，满足1<=k<n/2。

## 6 依次读七张图

图1先确认数据确实来自连续纸面，颜色连续，不是类别。图2看线性PCA为何把卷曲层重叠。图3在同一列比较种子，在同一行比较perplexity。图4同样检查UMAP的种子与邻居数。图5只改变min_dist，避免把多个参数变化混在一起解释。

图6用数字检查视觉判断：t-SNE种子0、perplexity30的T10约0.996195，邻域重合0.8325，而内在全局距离相关仅0.293076。图7是单Gaussian负对照，不给团块起类别名字，先记录是否看到局部空隙，再解释这种外观为何不足以支持分型。

各二维面板有独立坐标范围，颜色范围在卷曲图中统一。整体旋转、反射和缩放没有直接语义。不要比较两张图上点到纸边的距离，也不要把二维团块面积当作原分布密度。

## 7 指标复核与可接受结论

每个配置检查Z形状为320×2、所有值有限。T10与R10应在[0,1]；Spearman相关在[-1,1]。独立trust与库trustworthiness差应小于1e-12。固定版本的同一配置应可复现坐标；计时字段可能不同，普通/-O测试因此不要求计时逐位相等。

解释一个具体冲突：PCA输入空间全局距离相关较高，却没有很好恢复纸面内在距离。原因是三维直线会穿过卷曲层，而内在路径不能穿纸。再解释为什么T接近1时R仍可能低得多：新增邻居虽然不是前10，但原排名可能只略超10，惩罚相对较小。

报告可以写“在当前固定样本和参数下，某配置较好保留某种近邻关系”，不能写“算法证明存在三个真实群体”或“UMAP总比t-SNE保留全局距离”。

## 8 测试与故障排查

212项普通/-O检查覆盖完整种子映射重算、14配置与2对照、独立排名指标、库对照、弧长导数、坐标关系、指标平移/缩放不变性、输入错误及CSV/报告篡改。它们不替代额外研究问题的效度检查。

若UMAP缺失，确认安装包名umap-learn而导入名umap。若Numba首次编译慢，先观察进程而非更换数据；若线程数不同，按固定配置重跑。若图方向不同但邻域指标相同，不要立即判错；若同一版本相同种子坐标仍大幅不同，检查初始化、迭代、并行设置和依赖版本。

真实数据出现缺失值、重复值或混合量纲时，应先建立相应处理协议。本单元输入拒绝非有限数值，不会自动推断哪种填补方式合理。大量相等距离还会产生近邻并列，需要明确破同分规则；本单元稳定排序用于确定性核验。

## 9 扩展任务

扩展A增加卷曲样本数，检查跨层邻居与内在距离指标如何变化；完整保存新种子与配置。扩展B改变一个输入的单位，观察欧氏邻居图如何变化，再讨论缩放是否有物理依据。扩展C对实际想报告的结论做更多种子与参数敏感性检查，不按图像美观筛选种子。

提交时附上完整配置表、全部指标、至少一个负对照和原空间检查建议。对未验证的新样本映射或真实类别结论明确保留边界。实验的目的不是让图更好看，而是让解释经得起复核。
