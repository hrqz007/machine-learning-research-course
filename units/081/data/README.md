# 数据字典与来源

生成入口：generate_data.py。随机种子81021；dataset.json为确定性UTF-8 JSON，generation.json记录字节数与SHA256。默认重建输出至outputs/generated-data，不覆盖冻结数据。实验调用相同make_data确定性函数，不依赖联网读取。

## 原创可控机制

- synthetic/train：600个正常点，只用于模型与标准化拟合。
- synthetic/calibration：199个正常点，只选阈值。
- synthetic/normal_test：300个正常点，审计整体与正常子群误报。
- synthetic/anomaly_test：100个桥区点和100个远端点，仅评价召回与排序。
- synthetic/contaminants：额外67个桥区点，仅独立污染训练对照；不是复用测试异常。
- x：二维无量纲教学特征。group=0为概率0.8的窄正常群，均值[-2,0]、标准差[0.38,0.45]；group=1为概率0.2的宽正常群，均值[2,0]、标准差[0.9,0.8]。两维条件独立。
- kind：bridge均值[0,0]、标准差[0.16,0.2]；far均值[0,3.5]、标准差[0.55,0.35]。这些是任务正例的明确定义，不宣称来自任何真实故障。

## 真实Digits

实际数据随scikit-learn.datasets.load_digits安装提供；1797行、每行64个8×8像素值，原始整数0至16。该随包集合是UCI原数据的测试部分，本课程在其中另作自己的行级划分，不能冒称UCI原实验评估协议。

本课选数字0为正常参考、数字6为任务外新颖类。真实0共178行，随机划分90训练、45校准、43正常测试；真实6共181行，仅异常测试。x是原像素除16后的值；digit保留真实数字标签；row_id是随包原始行号。标签定义任务并用于最终评价，数字6不进入模型拟合或阈值校准。

出处：Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C50P49 。数据页面：https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits 。许可CC BY 4.0：https://creativecommons.org/licenses/by/4.0/ 。本课程修改：选类、保存原行号、划分、像素固定归一化；不修改原图像标签。scikit-learn随包说明：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html 。

限制：历史小数据、单一类别对、正常测试仅43行；没有书写者级或时间外部验证。真实正常写法的多样性必须保留，不得为消除误报而删除高分负例。
