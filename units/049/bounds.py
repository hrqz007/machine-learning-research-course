"""Finite fixed-class IID bounded-loss bounds, with explicit assumptions."""
import math
from numeric import scalar

def radius(n,delta,hypotheses=1):
    n=scalar(n,'sample size',1,1000000,True);M=scalar(hypotheses,'hypothesis count',1,1048576,True);delta=scalar(delta,'failure probability',1e-12,1-1e-12)
    return math.sqrt((math.log(2)+math.log(M)-math.log(delta))/(2*n))

def tail(n,epsilon,hypotheses=1):
    n=scalar(n,'sample size',1,1000000,True);M=scalar(hypotheses,'hypothesis count',1,1048576,True);epsilon=scalar(epsilon,'absolute risk deviation',0,1)
    lograw=math.log(2)+math.log(M)-2*n*epsilon**2;raw=math.exp(lograw)
    return {'raw_bound':raw,'probability_bound':min(1.,raw),'log_raw_bound':lograw,'vacuous':bool(raw>=1),'assumptions':'fixed finite hypothesis class; independent identically distributed observations; losses in [0,1]'}

def agnostic_sample_requirement(epsilon,delta,hypotheses):
    eps=scalar(epsilon,'excess risk target',1e-6,1);delta=scalar(delta,'failure probability',1e-12,1-1e-12);M=scalar(hypotheses,'hypothesis count',1,1048576,True)
    raw=2*(math.log(2)+math.log(M)-math.log(delta))/eps**2
    return {'sufficient_sample_size':math.ceil(raw),'unrounded':raw,'derivation':'uniform radius <= epsilon/2 implies ERM excess risk <= epsilon','assumptions':'exact ERM over fixed finite class, IID [0,1] losses; no realizability assumed'}

def realizable_sample_requirement(epsilon,delta,hypotheses):
    eps=scalar(epsilon,'risk target',1e-6,1);delta=scalar(delta,'failure probability',1e-12,1-1e-12);M=scalar(hypotheses,'hypothesis count',1,1048576,True)
    raw=(math.log(M)-math.log(delta))/eps
    return {'sufficient_sample_size':math.ceil(raw),'unrounded':raw,'assumptions':'IID binary classification, fixed finite H contains a zero-risk rule, learner returns a consistent rule','warning':'not applicable to the noisy default finite population'}
