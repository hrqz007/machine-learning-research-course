"""第 002 讲：任务定义与数据生成。

本实验仅审计原创合成记录，不训练预测模型，不连接网络。
直接运行：python experiment.py
只用 Python 标准库。输出放在同目录 outputs/，不会覆盖原始数据。
"""
from pathlib import Path
from datetime import date, timedelta
from collections import Counter
from decimal import Decimal
import csv
import json

# 文件位置相对于本脚本，不依赖你从哪个文件夹启动。
BASE = Path(__file__).resolve().parent
DEFAULT_CUTOFF = date(2025, 4, 20)


def read_rows(path):
    """读 CSV。每行成为一个可按列名取值的小字典；本讲不要求掌握语法。"""
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def as_date(value):
    """空白仍是未知，不把它变成 0 或今天的日期。"""
    return date.fromisoformat(value) if value else None


def deduplicate_events(rows):
    """按事件核查副本。完全一致时仅计一次；内容冲突时停止而非擅自选一条。"""
    first = {}
    duplicates = []
    for row in rows:
        event = row['event_id']
        if event not in first:
            first[event] = row
            continue
        comparable = {k: v for k, v in row.items() if k != 'raw_row_id'}
        old = {k: v for k, v in first[event].items() if k != 'raw_row_id'}
        if comparable != old:
            raise ValueError('同一事件有冲突版本，必须人工核实：' + event)
        duplicates.append({'kept': first[event]['raw_row_id'],
                           'duplicate': row['raw_row_id'], 'event_id': event})
    return list(first.values()), duplicates


def make_label(row, cutoff=DEFAULT_CUTOFF):
    """依照讲义约定返回 1、0 或 None（未知）。

    只允许使用 cutoff 当时可用的成交记录。observed_through 表示资料包中
    已有证据证明完整核实到的日期，不随 cutoff 改动而凭空延长。
    窗口含两端。没有撤销或更正成交的合成设定下，确认正例可提前定案。
    """
    start = as_date(row['listed_at'])
    end = start + timedelta(days=30)
    if start > cutoff:
        return None
    sale = as_date(row['sale_at'])
    recorded = as_date(row['sale_recorded_at'])
    visible_sale = sale is not None and recorded is not None and recorded <= cutoff
    if visible_sale and start <= sale <= end:
        return 1
    # 不能仅因“没看见成交”就填 0：必须有完整窗口的核实证据。
    verified_through = as_date(row['observed_through'])
    if end <= cutoff and verified_through is not None and end <= verified_through <= cutoff:
        return 0
    return None


def feature_audit(rows):
    """逐样本核对四类候选字段的最早可用日期与预测日期。"""
    pairs = [('area_sqm', 'area_available_at'),
             ('list_price_wan', 'list_price_available_at'),
             ('views_30d', 'views_available_at'),
             ('fee_wan', 'fee_available_at')]
    result = {}
    for field, availability in pairs:
        eligible, future, missing = [], [], []
        for row in rows:
            event = row['event_id']
            when = as_date(row[availability])
            if row[field] == '' or when is None:
                missing.append(event)
            elif when <= as_date(row['listed_at']):
                eligible.append(event)
            else:
                future.append(event)
        result[field] = {'eligible': eligible, 'future': future, 'missing': missing}
    return result


def overlap(rows, train_ids, test_ids, key):
    """取两部分中共同出现的编号：这就是集合交集。"""
    a = {r[key] for r in rows if r['event_id'] in train_ids}
    b = {r[key] for r in rows if r['event_id'] in test_ids}
    return sorted(a & b)


def known_summary(labels):
    """分母只包括当前已知标签，不把未知塞进负例。"""
    known = [v for v in labels if v is not None]
    positive = sum(known)
    return {'known': len(known), 'positive': positive,
            'negative': len(known) - positive, 'unknown': len(labels) - len(known),
            'positive_fraction': positive / len(known) if known else None}


def validate_source(rows):
    """检查可机械验证的局部约束，不声称证明原始资料真实无误。"""
    if len({r['raw_row_id'] for r in rows}) != len(rows):
        raise ValueError('导出行号必须唯一')
    for row in rows:
        assert row['event_id'] and row['house_id'] and row['building_id']
        assert Decimal(row['area_sqm']) > 0
        assert Decimal(row['list_price_wan']) > 0
        sale, recorded = as_date(row['sale_at']), as_date(row['sale_recorded_at'])
        if sale:
            assert recorded is not None and recorded >= sale
            assert sale >= as_date(row['listed_at'])
        assert as_date(row['observed_through']) <= DEFAULT_CUTOFF


def boundary_tests(reference):
    """刻意构造边界输入，检查定义，而不是只运行常见样本。"""
    checks = []
    template = dict(reference)
    template.update(listed_at='2025-06-01', observed_through='2025-07-01')
    # 6 月 1 日后第 30 天是 7 月 1 日。
    r = dict(template, sale_at='2025-07-01', sale_recorded_at='2025-07-02')
    assert make_label(r, date(2025, 7, 3)) == 1
    checks.append('day_30_included')
    r = dict(template, sale_at='2025-07-02', sale_recorded_at='2025-07-03')
    assert make_label(r, date(2025, 7, 3)) == 0
    checks.append('day_31_excluded')
    r = dict(template, sale_at='', sale_recorded_at='', observed_through='2025-06-18')
    assert make_label(r, date(2025, 6, 20)) is None
    checks.append('short_followup_is_unknown')
    r = dict(template, sale_at='', sale_recorded_at='', observed_through='2025-07-05')
    assert make_label(r, date(2025, 7, 2)) is None
    checks.append('verification_not_yet_available')
    r = dict(template, sale_at='2025-06-10', sale_recorded_at='2025-06-25',
             observed_through='2025-06-09')
    assert make_label(r, date(2025, 6, 20)) is None
    assert make_label(r, date(2025, 6, 26)) == 1
    checks.append('delayed_record_changes_knowledge')
    r = dict(template, sale_at='2025-06-01', sale_recorded_at='2025-06-01')
    assert make_label(r, date(2025, 6, 1)) == 1
    checks.append('day_0_included')
    assert make_label(r, date(2025, 5, 31)) is None
    checks.append('prediction_not_yet_occurred')
    # 同事件冲突必须阻止静默选择。
    conflict = dict(reference, raw_row_id='CONFLICT', area_sqm='999')
    try:
        deduplicate_events([reference, conflict])
    except ValueError:
        checks.append('conflicting_duplicate_rejected')
    else:
        raise AssertionError('冲突记录没有被拦截')
    assert known_summary([None, None])['positive_fraction'] is None
    checks.append('no_known_labels_no_division_by_zero')
    return checks


def run_experiment(write_outputs=True):
    raw = read_rows(BASE / 'data' / 'raw_events.csv')
    validate_source(raw)
    events, duplicates = deduplicate_events(raw)
    labels = {r['event_id']: make_label(r) for r in events}
    expected = {r['event_id']: int(r['expected_label']) if r['expected_label'] else None
                for r in read_rows(BASE / 'data' / 'expected_labels.csv')}
    assert labels == expected, '实际标签与纸笔标准答案不一致'
    audit = feature_audit(events)
    bad_train = {'E01', 'E02', 'E03', 'E05'}
    bad_test = {r['event_id'] for r in events} - bad_train
    building_train = {r['event_id'] for r in events if r['building_id'] in {'B01','B02','B03'}}
    building_test = {r['event_id'] for r in events} - building_train
    bad_overlap = {k: overlap(events, bad_train, bad_test, k)
                   for k in ['event_id', 'house_id', 'building_id']}
    group_overlap = {k: overlap(events, building_train, building_test, k)
                     for k in ['event_id', 'house_id', 'building_id']}
    # 这是有意演示的非法信息捷径；只核验算术，不把它当成预测成绩。
    fee_rows = [r for r in events if r['fee_wan'] and r['sale_price_wan']]
    invalid_errors = [abs(Decimal(r['fee_wan']) * 100 - Decimal(r['sale_price_wan']))
                      for r in fee_rows]
    invalid_mae = sum(invalid_errors) / len(invalid_errors)
    report = {
        'task': 'synthetic_listing_30_day_sale_audit', 'cutoff': DEFAULT_CUTOFF.isoformat(),
        'raw_rows': len(raw), 'unique_events': len(events),
        'unique_houses': len({r['house_id'] for r in events}),
        'unique_buildings': len({r['building_id'] for r in events}),
        'duplicates': duplicates, 'labels': labels,
        'deduplicated_label_summary': known_summary(list(labels.values())),
        'raw_label_summary': known_summary([make_label(r) for r in raw]),
        'feature_audit': audit, 'bad_split_overlap': bad_overlap,
        'building_split_overlap': group_overlap,
        'building_test_known_labels': sum(labels[e] is not None for e in building_test),
        'temporal_split': {
            'freeze_date': '2025-03-01',
            'historical_known': [r['event_id'] for r in events
                                 if as_date(r['listed_at']) < date(2025, 3, 1)
                                 and make_label(r, date(2025, 3, 1)) is not None],
            'later_events': [r['event_id'] for r in events
                             if as_date(r['listed_at']) >= date(2025, 3, 1)]},
        'invalid_fee_shortcut': {'retrospective_rows': len(fee_rows),
                                'mae_wan': float(invalid_mae), 'valid_prediction': False},
        'boundary_tests': boundary_tests(events[0]),
        'limitations': ['synthetic_only', 'no_model_training', 'no_population_rate_claim',
                        'no_near_duplicate_detection', 'no_deployment_validation']
    }
    assert (report['raw_rows'], report['unique_events'], report['unique_houses'],
            report['unique_buildings']) == (11, 10, 9, 5)
    assert report['deduplicated_label_summary'] == {
        'known': 8, 'positive': 5, 'negative': 3, 'unknown': 2, 'positive_fraction': 0.625}
    assert audit['area_sqm']['future'] == ['E06']
    assert len(audit['views_30d']['future']) == 10
    assert len(audit['fee_wan']['future']) == 7
    assert bad_overlap['house_id'] == ['H01']
    assert bad_overlap['building_id'] == ['B01', 'B02', 'B03']
    assert all(not values for values in group_overlap.values())
    assert report['building_test_known_labels'] == 1
    assert invalid_mae == 0
    if write_outputs:
        out = BASE / 'outputs'
        out.mkdir(exist_ok=True)
        (out / 'audit_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        with (out / 'event_labels.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['event_id', 'house_id', 'building_id', 'label_30d'])
            for r in events:
                value = labels[r['event_id']]
                writer.writerow([r['event_id'], r['house_id'], r['building_id'], '' if value is None else value])
    return report


def print_report(report):
    print('原始行 / 不同事件 / 不同房屋 / 不同楼栋：',
          report['raw_rows'], report['unique_events'], report['unique_houses'], report['unique_buildings'])
    print('重复副本：R02 与 R01 指向同一次 E01')
    print('逐事件标签：', report['labels'])
    print('去重后已知正例比例：5 / 8 = 62.50%')
    print('去重前已知正例比例：6 / 9 = 66.67%')
    print('挂牌时不可用的面积：', report['feature_audit']['area_sqm']['future'])
    print('未来浏览量事件数：', len(report['feature_audit']['views_30d']['future']))
    print('未来费用事件数：', len(report['feature_audit']['fee_wan']['future']))
    print('坏划分房屋交集：', report['bad_split_overlap']['house_id'])
    print('坏划分楼栋交集：', report['bad_split_overlap']['building_id'])
    print('按楼栋隔离后的楼栋交集：', report['building_split_overlap']['building_id'])
    print('按楼栋隔离后测试侧已知标签数：', report['building_test_known_labels'])
    print('费用捷径 MAE 为 0，但 valid_prediction = False')
    print('边界检查通过数：', len(report['boundary_tests']))
    print('所有预期核对通过。这里只验证合成数据审计，不验证现实预测能力。')


if __name__ == '__main__':
    print_report(run_experiment())
