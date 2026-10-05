# 第050讲教学数据

这些数据由generate_data.py原创生成，无真实个人或设备信息，无外部下载许可依赖。全部生成常数及划分规则在protocol.json事前固定；本包结果未筛选种子或删除困难面板。

## 生成机制

设备g=0至59，日期t=0至47。每份面板独立抽a_g~N(0,1.2²)，u_gt~N(0,1)，eps_gt~N(0,0.5²)。x=u+0.015t；y=2+1.4x+a+0.045t+eps。所有设备完整记录48天。目标在当日产生，复核标签在t+2可用；实际预测在t日先读取x，再预测，未读取y。

- draws.npz：a形状81×60；xnoise与eps形状81×60×48。rep0是详细主例，rep1至80是独立新面板。
- panel.csv：rep0的2880行。row_id=g×48+t，device为设备ID，day为预测日，label_available_day=day+2。x是当日可用输入，y是延迟完成的目标，oracle_mean=2+1.4x+a+0.045t只供模拟诊断，模型不得读取。
- splits.npz：八个整数数组，名称为random/group/time/group_time加_train或_test后缀。
- split_membership.csv：每个row_id在四个协议中的train/test/unused归属。
- hand.csv：四行训练、两行测试的独立纸笔例，精确手算用；设备字母A/B/C与主数据整数ID不混用。
- protocol.json：数据seed50017、划分seed50053、Bootstrap seed50059、80独立重复及全部模型与超参数。

## 信息使用

random与group在整个观察期结束、第49日标签到齐后回顾拟合。time与group_time在第32日结束拟合，训练条件day+2≤32，预测day≥33。模型不在测试期间更新，未知设备专属项为0。日历项day/47使用已声明的固定单位，不从测试样本估计缩放。

## 重生成

```bash
python generate_data.py --directory outputs/new-data
```

目标目录须为空且不能是随包data。不要日常修改原数据或hash；有意新实验应另建协议。确定性NPZ写入固定ZIP内部时间以便相同数值环境下逐字节复现。data_integrity.json校验六份固定数据文件，本说明文件不是数值输入。
