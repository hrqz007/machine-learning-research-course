# 数据字典

three_tokens.npz包含Q、K（3×2）、V（3×2）以及allowed（3×3，下三角布尔允许矩阵）。均为源码明确给定的教学矩阵。随机验证种子99，B=2、H=2、L=4、键维3、值维2，float64。所有随机验证可由run_checks重建。
