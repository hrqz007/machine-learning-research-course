# 四幅原创科学图

所有图由plots.py根据原始函数、图结构和experiment.run()实测生成，无外部图片。重建命令：python plots.py --directory outputs/figures。每图PNG与路径化文字SVG各一份。

1. 01_shared_graph：x=2,y=3,t=xy,q=t*t,z=q+t；保留两个t到乘法输入槽位，直接t到z路径绕下方，避免穿过节点标签。
2. 02_accumulated_paths：t梯度三个贡献6、6、1与向输入传递的39、26。
3. 03_gradient_check：实际有限差分步长误差扫描，旁列不可微ReLU零点的对称割线。
4. 04_jvp_vjp：同一显式Jacobian的两个不同乘积，标签说明输入方向与输出权重的区别。

图中数值不代表参数更新或统计估计；仅在各自给定输入点成立。已检查中文、下标、图例和节点连线，没有手工修改数值以贴近期望。
