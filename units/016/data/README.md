# 原创合成几何数据

points2d.csv 给出固定的二维建议点；main=(3,2) 是正文主例，其余用于内部点、负分量与零向量边界核验。各坐标均无量纲。

affine_system.csv 给出三维例子的两条等式 z1+z2=1、z2+z3=0。该例的建议点明确为(3,2,1)，不能把二维点表直接当作三维输入。两行系数线性独立。

所有数值均为原创教学构造，不包含真实个人记录或市场样本，不用于现实表现主张。投影采用欧氏平方距离；若改变尺度、权重或集合，问题和答案都会改变。

## 字段和形状

- `points2d.csv`：`point_id`是固定唯一标签；`z1`、`z2`为两个实坐标。载入结果为(4,2)，顺序为main、inside、negative、origin。
- `affine_system.csv`：`row_id`是方程标签；`a1`、`a2`、`a3`是左端系数；`rhs`是右端常数。系数矩阵形状(2,3)，右端向量形状(2,)。三维建议点为(3,2,1)。

脚本对固定行标识和数值作检查，保证课堂已知答案不会被静默改动。若要探索新数据，请另建实验，不把新结果称作本讲固定数据的验收结果。

## 可再生成输出

`outputs/projection-results.csv`共16行，四个输入分别对应box、ball、hyperplane、halfspace。v1、v2保存建议点，p1、p2保存投影点；squared_distance是欧氏平方距离，objective_half_squared_distance是其一半；feasibility_violation是非负违反量，等式使用绝对残差，球与盒子使用越界的正部。

`outputs/intersection-comparison.csv`保存交集真投影及两种单次顺序投影。certificate_at_true_projection指以真投影作为可行测试点，检查当前候选的(v−q)与(p−q)内积；错误候选得到正值，真投影为0。

`outputs/grid-diagnostics.csv`记录有限候选数、最大投影证书内积，以及网格最优平方距离减去解析最优值。网格可能没有精确解；正差距不代表解析式有误，有限网格也不能证明全域最优。

`outputs/experiment-report.json`汇总上述检查、分数算术、Jensen、非扩张抽查及边界案例。这些都是执行时生成的产物；公开输入仍只有本目录两份固定CSV。
