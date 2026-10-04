"""第 9 讲：实向量内积、范数、距离与投影的可复核小规模实验。"""
from fractions import Fraction
from pathlib import Path
import argparse
import json
import platform
import numpy as np


def reject_boolean_sources(values, name):
    # 先检查 Python 容器，避免 [True, 1] 被自动提升为整数后丢失来源类型。
    if isinstance(values, (bool, np.bool_)):
        raise ValueError(f"{name} must not contain boolean values")
    if isinstance(values, np.ndarray):
        if values.dtype.kind == "b":
            raise ValueError(f"{name} must not contain boolean arrays")
        return  # 已有数值 dtype 的数组无法恢复更早的类型历史。
    if isinstance(values, (list, tuple)):
        for item in values:
            reject_boolean_sources(item, name)


def real_array(values, name):
    reject_boolean_sources(values, name)
    a = np.asarray(values)
    if a.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers, not strings, booleans or complex values")
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return a


def vector(values, name="vector"):
    a = real_array(values, name)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must have non-empty shape (d,)")
    return a


def pair(x, y):
    x, y = vector(x, "x"), vector(y, "y")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    return x, y


def finite_result(value):
    if not np.isfinite(value).all():
        raise ArithmeticError("result is outside the finite float64 range")
    return value


def dot_checked(x, y):
    x, y = pair(x, y)
    with np.errstate(over="raise", invalid="raise"):
        result = np.sum(x * y)
    return float(finite_result(result))


def norm1(x):
    x = vector(x)
    with np.errstate(over="raise", invalid="raise"):
        value = np.abs(x).sum()
    return float(finite_result(value))


def norm2(x):
    """先缩放再平方，避免简单平方在很小或很大输入上立刻失效。"""
    x = vector(x)
    scale = float(np.max(np.abs(x)))
    if scale == 0.0:
        return 0.0
    with np.errstate(over="raise", invalid="raise"):
        result = scale * np.sqrt(np.sum((x / scale) ** 2))
    return float(finite_result(result))


def distance(x, y, order=2):
    x, y = pair(x, y)
    with np.errstate(over="raise", invalid="raise"):
        difference = x - y
    if order == 1:
        return norm1(difference)
    if order == 2:
        return norm2(difference)
    raise ValueError("this lesson supports only order=1 or order=2")


def unit_direction(x):
    """零向量拒绝；先除最大分量再归一化，不用任意阈值冒充零。"""
    x = vector(x)
    scale = float(np.max(np.abs(x)))
    if scale == 0.0:
        raise ValueError("the zero vector has no unit direction")
    scaled = x / scale
    return scaled / np.sqrt(np.sum(scaled ** 2))


def clip_roundoff_cosine(value, tolerance=1e-12):
    value = float(value)
    if not np.isfinite(value) or not 0 < tolerance < 1:
        raise ValueError("cosine must be finite and tolerance must be in (0,1)")
    if value < -1.0 - tolerance or value > 1.0 + tolerance:
        raise ValueError("cosine is outside [-1,1] beyond roundoff tolerance")
    return float(np.clip(value, -1.0, 1.0))


def cosine_checked(x, y):
    x, y = pair(x, y)
    raw = dot_checked(unit_direction(x), unit_direction(y))
    return clip_roundoff_cosine(raw)


def angle_degrees(x, y):
    return float(np.degrees(np.arccos(cosine_checked(x, y))))


def project_line(x, direction):
    """投影到非零 direction 张成的过原点直线；返回系数、投影、残差。"""
    x, y = pair(x, direction)
    if not np.any(y != 0.0):
        raise ValueError("projection direction must be nonzero")
    denominator = dot_checked(y, y)
    if denominator == 0.0:
        raise ArithmeticError("direction square underflowed; rescale the direction")
    coefficient = dot_checked(x, y) / denominator
    with np.errstate(over="raise", invalid="raise"):
        projected = coefficient * y
        residual = x - projected
    finite_result(coefficient); finite_result(projected); finite_result(residual)
    return float(coefficient), projected, residual


def batch_vectors(X, reference):
    X = real_array(X, "X")
    ref = vector(reference, "reference")
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must have shape (n,d), n>0 and d>0")
    if X.shape[1] != ref.size:
        raise ValueError("reference length must equal the feature count")
    return X, ref


def row_dots(X, reference):
    X, ref = batch_vectors(X, reference)
    with np.errstate(over="raise", invalid="raise"):
        result = (X * ref).sum(axis=1)
    finite_result(result)
    assert result.shape == (X.shape[0],)
    return result


def row_distances(X, query, scales=None):
    X, query = batch_vectors(X, query)
    if scales is None:
        scales = np.ones(query.shape)
    scales = vector(scales, "scales")
    if scales.shape != query.shape or not np.all(scales > 0):
        raise ValueError("scales must have shape (d,) and be strictly positive")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        differences = (X - query) / scales
        result = np.sqrt(np.sum(differences ** 2, axis=1))
    finite_result(result)
    assert result.shape == (X.shape[0],)
    return result


def row_projections(X, direction):
    X, y = batch_vectors(X, direction)
    if not np.any(y != 0.0):
        raise ValueError("projection direction must be nonzero")
    denominator = dot_checked(y, y)
    if denominator == 0.0:
        raise ArithmeticError("direction square underflowed; rescale the direction")
    coefficients = row_dots(X, y) / denominator
    with np.errstate(over="raise", invalid="raise"):
        projected = coefficients[:, None] * y
        residual = X - projected
    finite_result(projected); finite_result(residual)
    assert coefficients.shape == (X.shape[0],)
    assert projected.shape == residual.shape == X.shape
    return coefficients, projected, residual


def expect_error(call, error_type=ValueError):
    try:
        call()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def exact_projection():
    """整数开始，所有有理数运算保持精确，不从近似小数构造 Fraction。"""
    x = [Fraction(4), Fraction(1)]
    y = [Fraction(3), Fraction(4)]
    dot = sum(a*b for a,b in zip(x,y))
    yy = sum(v*v for v in y)
    coefficient = dot / yy
    p = [coefficient*v for v in y]
    r = [a-b for a,b in zip(x,p)]
    assert dot == 16 and coefficient == Fraction(16,25)
    assert p == [Fraction(48,25), Fraction(64,25)]
    assert r == [Fraction(52,25), Fraction(-39,25)]
    assert sum(a*b for a,b in zip(r,y)) == 0
    xx = sum(v*v for v in x); pp = sum(v*v for v in p); rr = sum(v*v for v in r)
    assert xx == 17 and pp == Fraction(256,25) and rr == Fraction(169,25)
    assert xx == pp + rr
    return {"dot": str(dot), "coefficient": str(coefficient),
            "projection": [str(v) for v in p], "residual": [str(v) for v in r],
            "squared_lengths": [str(xx),str(pp),str(rr)]}


def scale_experiment():
    Q = np.array([50., 5.])
    candidates = np.array([[51., 15.], [56., 6.]])
    factor = np.array([100., 1.])
    reference_scales = np.array([10., 5.])
    raw = row_distances(candidates, Q)
    converted_raw = row_distances(candidates*factor, Q*factor)
    scaled = row_distances(candidates, Q, reference_scales)
    converted_scaled = row_distances(candidates*factor, Q*factor, reference_scales*factor)
    assert int(np.argmin(raw)) == 1 and int(np.argmin(converted_raw)) == 0
    np.testing.assert_allclose(scaled, converted_scaled, rtol=0, atol=1e-12)
    assert int(np.argmin(scaled)) == int(np.argmin(converted_scaled)) == 1
    return {"labels": ["A","B"], "raw_m2_year": raw.tolist(),
            "raw_dm2_year": converted_raw.tolist(), "dimensionless": scaled.tolist(),
            "dimensionless_after_conversion": converted_scaled.tolist()}


def run_checks():
    checks=[]
    def passed(name):checks.append({"name":name,"passed":True})
    exact = exact_projection(); passed("exact_fraction_projection_and_energy_identity")
    x=np.array([4.,1.]); y=np.array([3.,4.])
    np.testing.assert_array_equal(x+y,[7,5]); np.testing.assert_array_equal(2*x-y,[5,-2])
    assert dot_checked(x,y)==16 and dot_checked(x,y)==dot_checked(y,x)
    np.testing.assert_allclose(dot_checked(2*x-y,y),2*dot_checked(x,y)-dot_checked(y,y),rtol=0,atol=1e-12)
    passed("vector_arithmetic_and_dot_properties")
    assert norm1(x)==5 and norm2(y)==5
    np.testing.assert_allclose(norm2(x),np.sqrt(17),rtol=0,atol=1e-12)
    np.testing.assert_allclose(norm2(x),np.linalg.norm(x,ord=2),rtol=0,atol=1e-12)
    assert distance(x,[1.,5.])==5 and distance(x,[1.,5.],order=1)==7
    assert norm2([0.,0.])==0
    passed("norms_and_known_distances")
    a,p,r=project_line(x,y)
    assert a==0.64
    np.testing.assert_allclose(p,[1.92,2.56],rtol=0,atol=1e-12)
    np.testing.assert_allclose(r,[2.08,-1.56],rtol=0,atol=1e-12)
    assert abs(dot_checked(r,y))<1e-12
    np.testing.assert_allclose(dot_checked(x,x),dot_checked(p,p)+dot_checked(r,r),rtol=0,atol=1e-12)
    passed("float_projection_residual_orthogonality_and_energy")
    # 有限组系数验证恒等式；讲稿中的代数推导才覆盖所有实数系数。
    for c in [-2.,0.,0.5,a,1.,2.]:
        left=dot_checked(x-c*y,x-c*y)
        right=dot_checked(r,r)+(a-c)**2*dot_checked(y,y)
        np.testing.assert_allclose(left,right,rtol=0,atol=1e-12)
        assert left>=dot_checked(r,r)-1e-12
    passed("nearest_point_identity_on_six_coefficients")
    for factor in [2.,-3.]:
        c2,p2,r2=project_line(x,factor*y)
        np.testing.assert_allclose(p2,p,rtol=0,atol=1e-12)
        np.testing.assert_allclose(c2,a/factor,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(project_line([0.,0.],y)[1],[0,0])
    passed("direction_rescaling_and_zero_source_projection")
    expect_error(lambda:project_line(x,[0.,0.]))
    expect_error(lambda:unit_direction([0.,0.]))
    expect_error(lambda:cosine_checked([0.,0.],y))
    expect_error(lambda:angle_degrees(x,[0.,0.]))
    passed("zero_direction_normalization_and_angle_rejected")
    np.testing.assert_allclose(unit_direction(y),[.6,.8],rtol=0,atol=1e-12)
    np.testing.assert_allclose(unit_direction(10*y),unit_direction(y),rtol=0,atol=1e-12)
    np.testing.assert_allclose(unit_direction(-y),-unit_direction(y),rtol=0,atol=1e-12)
    np.testing.assert_allclose([angle_degrees([1,0],[1,0]),angle_degrees([1,0],[0,1]),angle_degrees([1,0],[-1,0])],[0,90,180],rtol=0,atol=1e-12)
    assert clip_roundoff_cosine(1+5e-13)==1
    assert clip_roundoff_cosine(-1-5e-13)==-1
    expect_error(lambda:clip_roundoff_cosine(1.01));expect_error(lambda:clip_roundoff_cosine(np.nan))
    passed("unit_directions_angles_and_guarded_roundoff_clipping")
    X=np.array([[4.,1.],[1.,-2.],[0.,0.]])
    before=X.copy();dots=row_dots(X,y);coefs,P,R=row_projections(X,y)
    assert X.shape==(3,2) and dots.shape==coefs.shape==(3,) and P.shape==R.shape==(3,2)
    np.testing.assert_allclose(dots,[dot_checked(row,y) for row in X],rtol=0,atol=1e-12)
    np.testing.assert_allclose(P,np.array([project_line(row,y)[1] for row in X]),rtol=0,atol=1e-12)
    np.testing.assert_allclose(row_dots(R,y),[0,0,0],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(X,before)
    passed("non_square_batch_matches_loop_without_input_mutation")
    assert row_dots(X[:1],y).shape==(1,)
    assert row_projections(X[:1],y)[1].shape==(1,2)
    assert row_distances(X[:1],x).shape==(1,)
    np.testing.assert_array_equal(row_dots(np.array([[2.],[3.],[4.]]),[5.]),[10,15,20])
    passed("single_sample_and_single_feature_shapes")
    for call in [lambda:dot_checked([[4,1]],[3,4]),lambda:dot_checked([1,2],[1]),lambda:vector([]),lambda:vector([np.nan]),lambda:vector([np.inf]),lambda:vector([1+2j]),lambda:vector([True]),lambda:vector([True,1]),lambda:vector([np.bool_(True),1]),lambda:row_dots([[True,1],[2,3]],y),lambda:real_array([np.array([True,False]),np.array([1,2])],"mixed"),lambda:row_dots(X[0],y),lambda:row_dots(np.empty((0,2)),y),lambda:row_distances(X,x,[1,0]),lambda:row_distances(X,x,[-1,2]),lambda:row_distances(X,x,[1]),lambda:row_projections(X,[0,0])]:
        expect_error(call)
    passed("shape_empty_real_finite_and_scale_contracts")
    scaled=scale_experiment();passed("raw_nearest_neighbor_flip_and_dimensionless_invariance")
    assert distance([3,0],[0,0],order=1)<distance([2,2],[0,0],order=1)
    assert distance([3,0],[0,0],order=2)>distance([2,2],[0,0],order=2)
    passed("l1_l2_choice_can_reverse_nearest_candidate")
    rng=np.random.default_rng(9)
    for d in [2,3,5]:
        for _ in range(40):
            u,v,z=rng.normal(size=(3,d))
            assert abs(dot_checked(u,v))<=norm2(u)*norm2(v)+1e-12
            assert norm2(u+v)<=norm2(u)+norm2(v)+1e-12
            assert norm1(u+v)<=norm1(u)+norm1(v)+1e-12
            assert distance(u,z)<=distance(u,v)+distance(v,z)+1e-12
            np.testing.assert_allclose(distance(u,z),distance(z,u),rtol=0,atol=1e-12)
    passed("120_finite_random_checks_of_bounds_and_distance_not_a_proof")
    for factor in [2.,-2.,0.]:
        np.testing.assert_allclose(abs(dot_checked(x,factor*x)),norm2(x)*norm2(factor*x),rtol=0,atol=1e-12)
    np.testing.assert_allclose(norm2(x+2*x),norm2(x)+norm2(2*x),rtol=0,atol=1e-12)
    assert norm2(x-x)<norm2(x)+norm2(-x)
    assert distance([0],[2])**2>distance([0],[1])**2+distance([1],[2])**2
    passed("equality_cases_and_squared_distance_triangle_counterexample")
    shift=np.array([10.,-7.])
    np.testing.assert_allclose(distance(x+shift,y+shift),distance(x,y),rtol=0,atol=1e-12)
    assert not np.isclose(cosine_checked(x+shift,y+shift),cosine_checked(x,y))
    passed("translation_preserves_distance_but_not_origin_based_cosine")
    np.testing.assert_allclose(unit_direction([1e-300,0]),[1,0],rtol=0,atol=1e-12)
    expect_error(lambda:project_line(x,[1e-300,0]),ArithmeticError)
    expect_error(lambda:dot_checked([1e308],[1e308]),FloatingPointError)
    passed("extreme_float_limits_are_reported_not_silently_clipped")
    return {"unit":"009","status":"passed","python":platform.python_version(),"numpy":np.__version__,"checks":checks,"exact_fraction_results":exact,"numeric_results":{"dot":dot_checked(x,y),"norm_x":norm2(x),"norm_y":norm2(y),"projection_coefficient":a,"projection":p.tolist(),"residual":r.tolist(),"projection_squared_norm":dot_checked(p,p),"residual_squared_norm":dot_checked(r,r),"cosine":cosine_checked(x,y),"angle_degrees":angle_degrees(x,y)},"scale_experiment":scaled,"limits":["Real finite vectors only; no complex inner product implementation","Small CPU numerical checks support implementation testing, not universal mathematical proof","Direct projection formula rejects zero or numerically underflowed direction denominator","Browser Jupyter UI, socket transport and local Anaconda installation not tested"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='test-result.json')
    args=parser.parse_args();report=run_checks();path=Path(args.output)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({"status":report['status'],"checks":len(report['checks']),"numeric_results":report['numeric_results'],"output":path.name},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
