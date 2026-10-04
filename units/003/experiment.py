"""第三讲：用基本 Python 语法重现纸笔训练。所有房屋数据均为合成。
运行 python experiment.py。不读取网络，不需要第三方包，不修改数据。
# %% 后的标题与代码同样用于对应 Notebook；按顺序从空状态执行。
"""

# %% 1 一条表达式
print("1 一条表达式")
print(2 * 70 + 10)
print(2 * (70 + 10))

# %% 2 赋值发生在执行时
area = 70
weight = 2
bias = 10
prediction = weight * area + bias
area = 80
print("只改 area，旧 prediction：", prediction)
prediction = weight * area + bias
print("重新执行预测：", prediction)

# %% 3 类型与括号
print(type(70).__name__, type(70.0).__name__, type("70").__name__)
print("数值相加：", 70 + 5)
print("文本连接：", "70" + "5")
correct_mean = (5 + 0 + 10 + 15) / 4
wrong_mean = 5 + 0 + 10 + 15 / 4
print("正确均值与错误括号：", correct_mean, wrong_mean)
print("乘方括号：", (-2) ** 2, -2 ** 2)

# %% 4 用列表保持样本配对
areas = [50, 60, 80, 90]
prices = [110, 130, 170, 190]
print("长度：", len(areas))
print("第一条：", areas[0], prices[0])
print("最后一条：", areas[-1], prices[-1])
print("中间切片：", areas[1:3])

# %% 5 共享修改与复制
original = [50, 60, 80, 90]
other_name = original
other_name[0] = 55
print("共享后的原列表：", original)
independent = original.copy()
independent[0] = 999
print("复制后分别修改：", original, independent)
# 本节的 original 与训练用 areas 是分别创建的列表，训练数据未改变。

# %% 6 一个候选的逐轮计算
weight = 1.5
bias = 40
total_error = 0
candidate_predictions = []
absolute_errors = []
for index in range(len(areas)):
    prediction = weight * areas[index] + bias
    error = abs(prediction - prices[index])
    candidate_predictions.append(prediction)
    absolute_errors.append(error)
    total_error = total_error + error
    print("索引 预测 绝对误差 累计：", index, prediction, error, total_error)
candidate_mae = total_error / len(areas)
print("候选 C 的 MAE：", candidate_mae)

# %% 7 故意写错累加器位置
wrong_total = 0
for error in absolute_errors:
    wrong_total = 0
    wrong_total = wrong_total + error
wrong_accumulator_mae = wrong_total / len(absolute_errors)
print("错误程序得到：", wrong_accumulator_mae)
print("它是否等于正确答案：", wrong_accumulator_mae == candidate_mae)

# %% 8 三个候选的嵌套循环
names = ["A", "B", "C"]
weights = [2, 2, 1.5]
biases = [0, 10, 40]
scores = []
all_predictions = []
for candidate in range(len(names)):
    total_error = 0
    predictions = []
    for index in range(len(areas)):
        prediction = weights[candidate] * areas[index] + biases[candidate]
        predictions.append(prediction)
        total_error = total_error + abs(prediction - prices[index])
    scores.append(total_error / len(areas))
    all_predictions.append(predictions)
    print(names[candidate], predictions, scores[candidate])

# %% 9 比较与选择
best_index = 0
for candidate in range(1, len(names)):
    if scores[candidate] < scores[best_index]:
        best_index = candidate
selected_weight = weights[best_index]
selected_bias = biases[best_index]
print("选中：", names[best_index], selected_weight, selected_bias)

# %% 10 冻结选择后生成测试预测
# 所有标签在教程中公开。这里仅演示计算顺序，不声称保密盲测。
test_areas = [65, 85]
test_predictions = []
for area in test_areas:
    test_predictions.append(selected_weight * area + selected_bias)
print("先保存的教学测试预测：", test_predictions)

# %% 11 最后比较教学测试标签
test_prices = [144, 174]
test_error_sum = 0
for index in range(len(test_areas)):
    test_error_sum = test_error_sum + abs(test_predictions[index] - test_prices[index])
test_mae = test_error_sum / len(test_prices)
baseline_price = sum(prices) / len(prices)
baseline_error_sum = 0
for label in test_prices:
    baseline_error_sum = baseline_error_sum + abs(baseline_price - label)
baseline_mae = baseline_error_sum / len(test_prices)
print("模型与基线测试 MAE：", test_mae, baseline_mae)

# %% 12 浮点近似与容差
import math  # 导入 Python 自带的数学工具；不下载第三方库。
floating_sum = 0.1 + 0.2
floating_difference = floating_sum - 0.3
print("保存的求和值：", floating_sum)
print("直接相等比较：", floating_sum == 0.3)
print("差值：", floating_difference)
print("指定容差比较：", math.isclose(floating_sum, 0.3, rel_tol=1e-12, abs_tol=1e-12))

# %% 13 明确检查的已知答案
# assert 表示要求条件成立；失败时停止并报错。完整测试设计将在第4讲展开。
assert prediction == 175.0  # 最后一个训练候选 C 的末条预测
assert correct_mean == 7.5
assert wrong_mean == 18.75
assert original == [55, 60, 80, 90]
assert areas == [50, 60, 80, 90]
assert candidate_predictions == [115.0, 130.0, 160.0, 175.0]
assert absolute_errors == [5.0, 0.0, 10.0, 15.0]
assert candidate_mae == 7.5
assert wrong_accumulator_mae == 3.75
assert scores == [10.0, 0.0, 7.5]
assert best_index == 1
assert test_predictions == [140, 180]
assert test_mae == 5.0
assert baseline_mae == 15.0
assert abs(floating_difference) < 1e-12
print("PASS: 15 explicit checks; synthetic teaching data only")
