"""Public, offline checks with independent Fraction/Decimal reference routes.

Run: python test_experiment.py. All checks remain active with python -O.
No textbook figure, external data or network is used.
"""
from fractions import Fraction as F
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import random
import tempfile
from unittest.mock import patch

import numpy as np
import experiment as e


def run_checks() -> dict:
    counts = {'numeric_checks': 0, 'rejection_checks': 0, 'pre_rng_checks': 0}

    def check(condition, label):
        counts['numeric_checks'] += 1
        if not bool(condition):
            raise RuntimeError('check failed: ' + label)

    def close(actual, expected, label, tolerance=4e-12):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        check(a.shape == b.shape and bool(np.allclose(a, b, atol=tolerance, rtol=tolerance)), label)

    def rejected(action, label, pre_rng=False):
        if pre_rng:
            with patch.object(np.random, 'default_rng', side_effect=RuntimeError('RNG MUST NOT BE CREATED')) as factory:
                try:
                    action()
                except ValueError:
                    counts['rejection_checks'] += 1
                else:
                    raise RuntimeError('accepted invalid input: ' + label)
                check(factory.call_count == 0, 'no RNG: ' + label)
                counts['pre_rng_checks'] += 1
        else:
            try:
                action()
            except ValueError:
                counts['rejection_checks'] += 1
            else:
                raise RuntimeError('accepted invalid input: ' + label)

    def oracle(rows, p):
        # E[UV]-E[U]E[V], deliberately independent of the centered-Gram implementation.
        d = len(rows[0])
        mean = [sum(q * row[j] for q, row in zip(p, rows)) for j in range(d)]
        cov = [[sum(q * row[i] * row[j] for q, row in zip(p, rows)) - mean[i]*mean[j]
                for j in range(d)] for i in range(d)]
        return mean, cov

    states = [[F(x), F(z)] for x,z in [[0,0],[0,0],[1,1],[1,1],[2,1],[4,1]]]
    p = [F(1,6)] * 6
    mean, cov = oracle(states, p)
    check(mean == [F(4,3),F(2,3)], 'exact mean')
    check(cov == [[F(17,9),F(4,9)],[F(4,9),F(2,9)]], 'exact covariance')
    check(cov[0][0]*cov[1][1]-cov[0][1]**2 == F(2,9), 'exact determinant')
    current = e.finite_moments(states, p)
    close(current['mean'], mean, 'finite mean')
    close(current['covariance'], cov, 'finite covariance')
    close(e.correlation(current['covariance'])[0,1], 4/math.sqrt(34), 'finite rho')
    active = e.conditional_finite(states, p, 1, 1)
    close(active['mean'], [2,1], 'active mean')
    close(active['covariance'], [[1.5,0],[0,0]], 'active covariance')
    close(active['probabilities'], [F(1,4)]*4, 'active states retain repeated x=1')
    quiet = e.conditional_finite(states, p, 1, 0)
    close(quiet['covariance'], [[0,0],[0,0]], 'quiet covariance')
    tv = e.total_variance(states,p,0,1)
    close([tv['total'],tv['within'],tv['between']], [F(17,9),1,F(8,9)], 'total variance')
    transformed = [[x+z,x-z] for x,z in states]
    mean_t, cov_t = oracle(transformed,p)
    check(cov_t == [[3,F(5,3)],[F(5,3),F(11,9)]], 'independent transform oracle')
    close(e.linear_moments(mean,cov,[[1,1],[1,-1]],[7,-3])['covariance'], cov_t, 'linear transform')
    close(e.linear_moments(mean,cov,[[1,1],[1,-1]],[7,-3])['mean'], [mean_t[0]+7,mean_t[1]-3], 'linear shifted mean')
    for a in [[1,0],[0,1],[1,1],[1,-1],[-2,3]]:
        projected = [[sum(F(ai)*v for ai,v in zip(a,row))] for row in states]
        variance = oracle(projected,p)[1][0][0]
        quadratic = sum(F(a[i])*cov[i][j]*a[j] for i in range(2) for j in range(2))
        check(variance == quadratic and variance >= 0, 'directional variance identity')
    close(e.sample_covariance(states,0)['covariance'], cov, 'ddof0 finite empirical representation')
    close(e.sample_covariance(states,1)['covariance'], np.asarray(cov,float)*6/5, 'ddof1 factor')
    close(e.sample_covariance(states,0)['covariance'], np.cov(np.asarray(states,float),rowvar=False,ddof=0), 'NumPy explicit rowvar/ddof')
    check(np.cov(np.asarray(states,float)).shape == (6,6), 'NumPy default rowvar is a different object')
    centered = np.array(states,float)-np.mean(np.array(states,float),axis=0)
    check(np.linalg.matrix_rank(centered.T@centered) <= min(2,5), 'Gram rank bound')

    # 60 unrelated rational finite models, 2-4 dimensions; independent exact
    # moments and direct transformed-state oracle, not formula self-comparison.
    rng = random.Random(212121)
    for case in range(60):
        n,d = 5+case%4,2+case%3
        rows = [[F(rng.randrange(-6,7), rng.randrange(1,5)) for _ in range(d)] for _ in range(n)]
        raw = [rng.randrange(1,10) for _ in range(n)]
        probs = [F(v,sum(raw)) for v in raw]
        om,oc = oracle(rows,probs)
        got = e.finite_moments(rows,probs)
        close(got['mean'], om, 'random rational mean')
        close(got['covariance'], oc, 'random rational covariance')
        for i in range(d):
            for j in range(d):
                close(got['covariance'][i,j], oc[i][j], 'entrywise rational oracle')
        a = [F(rng.randrange(-3,4)) for _ in range(d)]
        proj = [[sum(x*y for x,y in zip(a,row))] for row in rows]
        ov = oracle(proj,probs)[1][0][0]
        close(e.finite_moments(proj,probs)['covariance'][0,0], ov, 'rational directional variance')
        check(ov >= 0, 'exact nonnegative directional variance')

    # Triangle monomial moments from direct symbolic iterated integration:
    # integral_0^1 integral_0^x 2*x^a*y^b dy dx = 2/((b+1)(a+b+2)).
    tri = lambda a,b: F(2,(b+1)*(a+b+2))
    ex,ey = tri(1,0),tri(0,1)
    vx,vy,cxy = tri(2,0)-ex**2,tri(0,2)-ey**2,tri(1,1)-ex*ey
    check([ex,ey,vx,vy,cxy] == [F(2,3),F(1,3),F(1,18),F(1,18),F(1,36)], 'triangle exact integrals')
    check(tri(2,0)/12 + vx/4 == vy, 'triangle total variance')
    for value in [F(1,10),F(1,2),F(3,4),F(1)]:
        out = e.triangle_conditional(value)
        close([out['mean'],out['variance'],out['density']], [value/2,value**2/12,1/value], 'triangle conditional')
    close(e.triangle_density([[.7,.3],[.3,.7],[0,0],[1,1],[-1,0]]),[2,0,2,2,0], 'triangle support incl boundary convention')

    gm = e.load_gaussian()
    close(gm['factor'], [[2,0],[1.5,math.sqrt(27)/2]], 'independent 2D Cholesky factor')
    close(gm['factor']@gm['factor'].T, [[4,3],[3,9]], 'factor reconstruct')
    close(np.linalg.det(gm['covariance']),27,'Gaussian determinant')
    eigenvalues,eigenvectors=np.linalg.eigh(gm['covariance'])
    close((eigenvectors*eigenvalues)@eigenvectors.T,gm['covariance'],'eigen reconstruction')
    close(eigenvectors.T@eigenvectors,np.eye(2),'eigenvector orthogonality')
    check(bool(np.all(eigenvalues>0)),'Gaussian eigenvalues positive')
    close(e.correlation(gm['covariance']),[[1,.5],[.5,1]],'Gaussian correlation')
    # Explicit two-coordinate rational formula compared against block solves.
    for var_x in [1,2,4,7]:
        for var_y in [2,5,9]:
            for cross in [F(-1,2),F(0),F(1,3)]:
                cmu = [F(1,3),F(-5,2)]
                cc = [[F(var_x),cross],[cross,F(var_y)]]
                for x in [F(-2),F(0),F(5)]:
                    result = e.gaussian_conditional(cmu,cc,[0],[x])
                    expected_m = cmu[1]+cross/F(var_x)*(x-cmu[0])
                    expected_v = F(var_y)-cross**2/F(var_x)
                    close(result['mean'],[expected_m],'conditional rational mean')
                    close(result['covariance'],[[expected_v]],'conditional rational variance')
                    close(result['residual_cross_covariance'],[[0]],'Gaussian residual cross covariance')
    for x in [-3,1,5]:
        result = e.gaussian_conditional([1,-2],[[4,3],[3,9]],[0],[x])
        close(result['mean'],[-2+3*(x-1)/4],'lesson conditional mean')
        close(result['covariance'],[[F(27,4)]],'lesson conditional variance')
    # Multivariate B: independent rational solve by 2 x 2 inverse in test only.
    sigma3 = [[5,1,2],[1,4,1],[2,1,3]]
    result3 = e.gaussian_conditional([1,2,3],sigma3,[1,2],[4,0])
    invbb = [[F(3,11),F(-1,11)],[F(-1,11),F(4,11)]]
    beta = [sum(F([1,2][k])*invbb[k][j] for k in range(2)) for j in range(2)]
    close(result3['mean'],[1+beta[0]*2+beta[1]*(-3)],'two-observation block mean')
    close(result3['covariance'],[[5-beta[0]-2*beta[1]]],'two-observation block variance')
    two_remaining=e.gaussian_conditional([1,2,3],sigma3,[2],[0])
    close(two_remaining['mean'],[-1,1],'two remaining coordinate means')
    close(two_remaining['covariance'],[[F(11,3),F(1,3)],[F(1,3),F(11,3)]],'two remaining Schur covariance')
    close(two_remaining['residual_cross_covariance'],[[0],[0]],'two remaining residual cross covariance')
    # Conditioning coordinate order may change, result must not.
    reorder = e.gaussian_conditional([1,2,3],sigma3,[2,1],[0,4])
    close(result3['mean'],reorder['mean'],'observed index reordering')
    close(result3['covariance'],reorder['covariance'],'observed block reordering')
    with localcontext() as context:
        context.prec=60
        pi = Decimal('3.141592653589793238462643383279502884197169399375105820974944')
        center_density = Decimal(1)/(2*pi*Decimal(27).sqrt())
    close(e.gaussian_pdf([[1,-2]],[1,-2],[[4,3],[3,9]]),[float(center_density)],'Decimal center density')
    for point in [[0,0],[5,1],[-1,-4],[1,-2]]:
        dx,dy = F(point[0]-1),F(point[1]+2)
        q = (9*dx**2-6*dx*dy+4*dy**2)/27
        expected = -math.log(2*math.pi)-.5*math.log(27)-.5*float(q)
        close(e.gaussian_logpdf([point],[1,-2],[[4,3],[3,9]]),[expected],'explicit inverse quadratic log density')
    check(np.isfinite(e.gaussian_logpdf([[1000,1000]],[0,0],[[1,0],[0,1]])[0]),'far-tail log density stays representable')

    nonlinear = e.finite_moments([[-1,1],[0,0],[1,1]],[F(1,3)]*3)
    close(nonlinear['covariance'],[[F(2,3),0],[0,F(2,9)]],'zero correlation nonlinear counterexample')
    check(F(1,3) != F(1,3)*F(1,3),'event factorization fails despite covariance zero')
    rare = e.total_variance([[0,0],[-10,1],[10,1]],[.9,.05,.05],0,1)
    close([rare['total'],rare['groups'][1]['variance']],[10,100],'conditional variance can exceed marginal')
    pp = [0.5, float(np.nextafter(np.nextafter(.5,1),1))]
    check(math.fsum(pp) == float(np.nextafter(1.,2.)), 'test input really has total 1+1ulp')
    for _ in range(5):
        # A probability total 1+1ulp is accepted with a disclosed adjustment;
        # reusing its normalized result or conditioning on an all-one event works.
        canonical,audit = e.probabilities(pp)
        check(audit['maximum_adjustment'] <= e.PROB_TOL,'bounded probability canonicalization')
        close(e.probabilities(canonical)[0],canonical,'probability roundtrip')
        cc = e.conditional_finite([[1,1],[2,1]],canonical,1,1)
        close(cc['probabilities'],canonical,'all-one conditional composability')
        pp = canonical
    normalized, _ = e.probabilities([1e-100, float(np.nextafter(1,0))])
    close(e.probabilities(normalized)[0],normalized,'tiny probability roundtrip',tolerance=1e-110)

    bad_scalars = [True,np.bool_(False),'2',2+0j,None,float('nan'),float('inf'),10**400]
    for bad in bad_scalars:
        rejected(lambda bad=bad:e.sample_finite([[0,0],[bad,1]],[1,0],n=1), 'full zero-weight state validation',True)
        rejected(lambda bad=bad:e.conditional_finite([[0,0],[bad,1]],[.5,.5],1,0), 'excluded condition state validation')
        rejected(lambda bad=bad:e.sample_gaussian([0,bad],[[1,0],[0,1]],1), 'Gaussian mean type',True)
        rejected(lambda bad=bad:e.gaussian_pdf([[0,0],[bad,1]],[0,0],[[1,0],[0,1]]), 'all density points validated')
    for data in [[[1],[2,3]], [], [1,2], np.array([[1,2]],dtype=object), np.array([[1,2]],dtype=complex), [[1e101,0]], [[1e-320,0]]]:
        rejected(lambda data=data:e.sample_finite(data,[1],1),'bad state array',True)
    for pbad in [[-.1,1.1],[.4,.4],[True,False],['.5',.5],[.5+0j,.5],[np.nan,.5],np.array([.5,.5],dtype=object),[1e-320,1]]:
        rejected(lambda pbad=pbad:e.sample_finite([[0,0],[1,1]],pbad,1),'bad probabilities',True)
    rejected(lambda:e.probabilities([1+1e-14,0]),'probability above one')
    rejected(lambda:e.probabilities([.5,.5+1e-12]),'probability total outside tolerance')
    for badcov in [[[1,2],[2,1]], [[1,.9,.9],[.9,1,-.9],[.9,-.9,1]], [[1,.1],[.2,1]], [[1,0],[0,-1]], [[1,0],[0,0]], [[1,2],[2,4]], [[1,0],[0,1e-14]], [[1,True],[True,1]], [[1,'0'],['0',1]], [[1,0j],[0j,1]], np.array([[1,0],[0,1]],dtype=object)]:
        rejected(lambda badcov=badcov:e.sample_gaussian([0]*len(badcov),badcov,1),'invalid/unsupported strict covariance',True)
    e.covariance_matrix([[1,2],[2,4]])
    check(True,'valid singular matrix accepted by PSD validator')
    rejected(lambda:e.gaussian_conditional([0,0],[[1,2],[2,4]],[0],[0]),'singular conditioning rejected')
    rejected(lambda:e.gaussian_pdf([[0,0]],[0,0],[[1,2],[2,4]]),'singular density rejected')
    rejected(lambda:e.correlation([[1,0],[0,0]]),'constant-coordinate correlation rejected')
    for ids, vals in [([],[]),([0,0],[1,2]),([0,1],[1,2]),([True],[1]),([2],[1]),([0],[]),([0],['1']),([[0]],[1])]:
        rejected(lambda ids=ids,vals=vals:e.gaussian_conditional([0,0],[[1,0],[0,1]],ids,vals),'conditional index/value contract')
    for nbad in [0,-1,True,1.0,e.MAX_SAMPLES+1]:
        rejected(lambda nbad=nbad:e.sample_gaussian_factor([0],[[1]],nbad),'bad sample size',True)
    rejected(lambda:e.sample_finite([[0]*11],[1],200000),'output sample entry limit',True)
    for seedbad in [-1,True,1.0,2**32]:
        rejected(lambda seedbad=seedbad:e.sample_triangle(10,seedbad),'bad seed',True)
    rejected(lambda:e.sample_finite([[0],[1e-200]],[.5,.5],1),'underflow even in unobserved state',True)
    rejected(lambda:e.sample_gaussian_factor([1e100],[[1]],1),'unresolvable mean/noise scale',True)
    rejected(lambda:e.sample_gaussian_factor([0],[[1e-200]],1),'factor square underflow',True)
    rejected(lambda:e.linear_moments([0],[[1e100]],[[1e150]],[0]),'conservative input range blocks potential transform overflow')
    rejected(lambda:e.linear_moments([0],[[1e-100]],[[1e-110]],[0]),'transform product underflow')
    rejected(lambda:e.gaussian_pdf([[1000,1000]],[0,0],[[1,0],[0,1]]),'PDF underflow advises logpdf')
    rejected(lambda:e.sample_covariance([[0]],1),'ddof leaves no denominator')
    rejected(lambda:e.sample_covariance([[0],[1]],True),'bool ddof rejected')
    rejected(lambda:e.triangle_conditional(0),'triangle zero boundary')
    rejected(lambda:e.triangle_conditional(1e-200),'triangle square underflow')
    rejected(lambda:e.conditional_finite([[0,0],[1,1]],[1,0],1,1),'zero model probability group')

    # Exactly constant coordinates remain exactly zero after complete input
    # validation, including when other coordinates vary or conditioning fixes a
    # coordinate. These cases formerly exposed rounding-created residuals.
    for constant in [.1, 1e-140, 3e90]:
        for n in [3, 5, 7]:
            rows = [[constant, float(i)] for i in range(n)]
            probs = [F(1, n)] * n
            fm = e.finite_moments(rows, probs)
            check(fm['mean'][0] == constant, 'constant finite mean exact')
            check(np.array_equal(fm['covariance'][0], [0,0]), 'constant finite covariance row')
            cond = e.conditional_finite(rows, probs, 0, constant)
            check(cond['mean'][0] == constant and np.array_equal(cond['covariance'][0], [0,0]), 'conditioned coordinate exact zero')
            for ddof in [0,1]:
                sm = e.sample_covariance(rows,ddof)
                check(sm['mean'][0] == constant and np.array_equal(sm['covariance'][0], [0,0]), 'constant sample coordinate exact zero')
            all_constant = e.finite_moments([[constant]]*n,probs)
            check(all_constant['covariance'][0,0] == 0, 'all constant finite variance')
            check(e.sample_covariance([[constant]]*n)['covariance'][0,0] == 0, 'all constant sample variance')
    for bad in [True,'0.1',float('nan'),float('inf')]:
        rejected(lambda bad=bad:e.finite_moments([[.1],[.1],[bad]],[.5,.5,0]), 'constant prefix zero-weight bad state')
        rejected(lambda bad=bad:e.sample_covariance([[.1],[.1],[bad]]), 'constant prefix bad sample')
        rejected(lambda bad=bad:e.conditional_finite([[.1,0],[.1,0],[bad,1]],[.5,.5,0],1,0), 'constant prefix excluded bad state')
        rejected(lambda bad=bad:e.sample_finite([[.1],[.1],[bad]],[.5,.5,0],1), 'constant prefix rejected before RNG',True)
    rejected(lambda:e.finite_moments([[.1]]*3,[.2,.2,.2]), 'constant does not bypass probability sum')

    # CSV parsers must validate every row; absent/extra cells are not ignored.
    with tempfile.TemporaryDirectory(prefix='ml021-check-') as folder:
        root=Path(folder)
        valid=(e.ROOT/'data/joint_states.csv').read_text()
        for i,badtext in enumerate([valid.replace('F,1,6,4,1','F,1,6,nan,1'), valid.replace('F,1,6,4,1','F,1,6,4,1,extra'),valid.replace('F,1,6,4,1','F,1,0,4,1'),valid.replace('F,1,6,4,1','F,1,6,4,'), valid.replace('scenario_id','state')]):
            path=root/f'bad{i}.csv';path.write_text(badtext)
            rejected(lambda path=path:e.generate_experiment(states_path=path,n=10),'invalid CSV before RNG',True)
        path=root/'bad_gaussian.csv';path.write_text((e.ROOT/'data/gaussian_model.csv').read_text().replace('y,-2,3,9','y,-2,2,9'))
        rejected(lambda:e.generate_experiment(gaussian_path=path,n=10),'second CSV before all sampling',True)
        # Validate first file, then fail second: no output mutation occurs because
        # main only calls write_results after generate_experiment has returned.

    # Deterministic generator identities do not rely on Monte Carlo convergence.
    first=e.sample_gaussian([1,-2],[[4,3],[3,9]],300,2102)
    second=e.sample_gaussian([1,-2],[[4,3],[3,9]],300,2102)
    check(np.array_equal(first['rows'],second['rows']),'same seed local deterministic')
    manual_rng=np.random.default_rng(2102)
    noise=manual_rng.standard_normal((300,2))
    manual=np.column_stack((1+2*noise[:,0],-2+1.5*noise[:,0]+math.sqrt(27)/2*noise[:,1]))
    close(first['rows'],manual,'independent explicit 2D sampling equations')
    singular=e.sample_gaussian_factor([0,0],[[1],[2]],300,2108)
    check(np.array_equal(singular['rows'][:,1],2*singular['rows'][:,0]),'singular line support')
    close(singular['covariance'],[[1,2],[2,4]],'singular covariance from explicit factor')
    triangle=e.sample_triangle(300,2103)
    check(bool(np.all((triangle[:,1]>=0)&(triangle[:,1]<=triangle[:,0])&(triangle[:,0]<=1))),'triangle sampler support')
    one=e.sample_covariance([[2,4]],0)
    close(one['covariance'],[[0,0],[0,0]],'one sample ddof0 has rank zero')
    # Ellipse-domain density quadrature. True omitted tail is known independently
    # via the standard-normal CDF; truncation is not mistaken for grid error.
    estimates=[]
    bound=6.0
    exact_retained=math.erf(bound/math.sqrt(2))**2
    for steps in [60,120]:
        axis=np.linspace(-bound,bound,steps+1)
        xx,yy=np.meshgrid(axis,axis,indexing='ij')
        white=np.column_stack((xx.ravel(),yy.ravel()))
        points=white@gm['factor'].T+gm['mean']
        density=e.gaussian_pdf(points,gm['mean'],gm['covariance']).reshape(xx.shape)*math.sqrt(27)
        estimate=float(np.trapezoid(np.trapezoid(density,axis,axis=1),axis))
        estimates.append(estimate)
    check(abs(estimates[1]-exact_retained)<abs(estimates[0]-exact_retained),'quadrature refinement reduces this observed grid error')
    check(abs(estimates[1]-exact_retained)<2e-10,'fine density-grid normalization check')
    return {'status':'passed', **counts,
            'references':'Fraction finite-model raw moments and direct projections; rational 2D/block conditionals; Decimal center density; explicit componentwise sampler',
            'quadrature':{'domain':'mu+L[-6,6]^2','steps_per_axis':[60,120],'estimates':estimates,
                          'analytic_retained_mass':exact_retained,'omitted_tail_mass':1-exact_retained,
                          'absolute_grid_errors':[abs(v-exact_retained) for v in estimates]},
            'no_assert_statements':True,'python_optimized_mode_supported':True}


if __name__=='__main__':
    print(json.dumps(run_checks(),ensure_ascii=False,indent=2))
