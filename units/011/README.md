# 第011讲 线性方程与最小二乘几何

直接先修第010讲矩阵与线性变换。本讲不要求微积分或已掌握SVD。

## 学习内容

从三条记录的直线拟合出发，学习行变换、列空间、零空间、秩与解的条件，借助正交证明最小二乘与正规方程，再手算QR并比较数值求解路线。最后检查秩亏、近共线、条件数与受控扰动。

lecture.pdf 是完整讲义，lab.pdf 是独立实验步骤，answers.pdf 含15题详解，均另附可编辑的同名Markdown源文件。12幅原创PNG均被讲义引用。data/main.csv 是唯一需要读取的原始合成数据，其余案例在脚本中确定性生成。

## 运行

核验环境为Python 3.12.14、NumPy 2.3.5、float64。实验无随机数、不联网、CPU即可运行。核心依赖为NumPy；不需重画插图或安装Matplotlib才能完成实验。

在本目录运行 python experiment.py，或用Jupyter打开experiment.ipynb，重启Python 3内核后Run All。保持整个文件夹结构。

输出为outputs/experiment_report.json与outputs/method_comparison.csv。原始CSV不被修改，重跑会更新生成输出。

## 固定核对

主例参数为7/6与1/2；残差为-1/6、1/3、-1/6；SSE为1/6。秩亏复制列的最小范数解为7/6、1/4、1/4。14组形状与边界检查应通过。近共线各方法真实误差与失败均保存，不要求某种方法每一例都最差。

## 限制

数值尾数与个别临界失败可能随BLAS/LAPACK变化。教学Gram–Schmidt不是生产替代。默认秩、条件数和lstsq内部可能使用SVD，其理论下一讲展开。有限实验不是一般证明，合成数据不验证真实预测能力。

Notebook已用新Python进程中的真实ipykernel进程内内核依序执行并保留输出。未测试浏览器Jupyter UI、外进程socket传输或学习者本机Anaconda安装。资料与执行证据见source-checks.json、test-result.json。
