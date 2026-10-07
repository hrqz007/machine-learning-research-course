# 一级资料与使用边界

核验日期：2026-10-07。优先固定1.8版本文档，避免stable页面升级引入接口差异。

1. https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
   用途：1.8 API/defaults/external X_val and y_val, explicit categorical mask。

2. https://scikit-learn.org/1.8/modules/calibration.html
   用途：calibration data independence and proper scoring rules。

3. https://xgboost.readthedocs.io/en/stable/tutorials/model.html
   用途：second-order regularized leaf objective; pedagogical convention, not identical internal histogram implementation。

4. https://scikit-learn.org/1.8/modules/ensemble.html#histogram-based-gradient-boosting
   用途：histograms, missing and categorical support。

手算、原创合成实验与文字组织为本课程设计。XGBoost资料只说明二阶目标思想，不把本实验称为XGBoost实测。未复刻第三方完整文本。
