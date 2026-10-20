# 098 门控记忆与序列任务

按lecture → lab → experiment.ipynb → answers学习。PDF与同名Markdown对应。运行python experiment.py生成全部合成数据、曲线和结果，再运行python -m unittest -v test_experiment.py。数据许可见DATA_LICENSE.md。环境与PDF构建见lab.md。

基线实际结果：三结构在当前简单任务所有测试长度均达到100%准确率；不支持门控胜过RNN的结论。源码保留这个负面比较结果，不筛选种子。所有数值为CPU真实执行。
