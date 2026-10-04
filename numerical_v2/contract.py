"""Exact-rational affine references and prospective binary32 error bounds.

Stdlib only. No checkpoints, native callbacks, allocations, optimizer, training,
runtime edits or v1 gate overrides. The caller must independently establish the
actual pinned source, tensor bytes and runtime. Passing this arithmetic contract
is not host qualification or measurement authority.
"""
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import math

U32 = Fraction(1, 2**24)
U64 = Fraction(1, 2**53)
MIN_NORMAL32 = Fraction(1, 2**126)
TANH_ABSOLUTE_BUDGET = Fraction(1, 2**23)
DECIMAL_ERROR = Fraction(1, 10**60)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def number(value):
    require(type(value) in (int, float, Fraction), 'literal finite real scalar required')
    if type(value) is float:
        require(math.isfinite(value), 'finite scalar required')
    result = Fraction(value)
    require(result.numerator.bit_length() <= 4096 and result.denominator.bit_length() <= 4096,
            'bounded rational scalar required')
    return result


def vector(values, length=None):
    require(type(values) in (list, tuple) and 1 <= len(values) <= 256,
            'bounded nonempty vector required')
    require(length is None or len(values) == length, 'vector shape differs')
    return tuple(number(x) for x in values)


def matrix(weights, width):
    require(type(weights) in (list, tuple) and 1 <= len(weights) <= 256,
            'bounded nonempty matrix required')
    return tuple(vector(row, width) for row in weights)


def gamma(operations, unit_roundoff=U32):
    require(type(operations) is int and 0 < operations <= 1024,
            'literal bounded positive operation count required')
    unit = number(unit_roundoff)
    require(0 < unit and operations*unit < 1, 'invalid roundoff or operation bound')
    return operations*unit/(1-operations*unit)


def affine_reference(weights, bias, observation):
    x = vector(observation)
    w = matrix(weights, len(x))
    b = vector(bias, len(w))
    return tuple(sum((a*v for a, v in zip(row, x)), offset)
                 for row, offset in zip(w, b))


def affine_error(weights, bias, reference_input, input_error):
    x = vector(reference_input)
    e = vector(input_error, len(x))
    require(all(v >= 0 for v in e), 'nonnegative input error required')
    w = matrix(weights, len(x))
    b = vector(bias, len(w))
    operations = 2*len(x)+2
    g = gamma(operations)
    # Products, reduction and bias may have any summation order. This count
    # covers separate multiply/add as well as FMA. A normal-number floor also
    # covers flush-to-zero instead of assuming gradual underflow.
    def radius(row, offset):
        # Include denormals-are-zero operands as well as underflowed outputs.
        # A tiny operand times a large partner is not bounded by a single floor.
        floor = (1+g)*MIN_NORMAL32*(operations + sum(abs(a) for a in row)
                    + sum(abs(v)+dv for v, dv in zip(x, e)) + len(x)*MIN_NORMAL32)
        return (sum((abs(a)*v for a, v in zip(row, e)), Fraction(0))
                + g*sum((abs(a)*(abs(v)+dv) for a, v, dv in zip(row, x, e)), abs(offset))
                + floor)
    return tuple(radius(row, offset) for row, offset in zip(w, b))


def relu_reference(values):
    return tuple(max(Fraction(0), x) for x in vector(values))


def tanh_reference(value):
    x = number(value)
    require(abs(x) <= 100, 'supported tanh reference domain exceeded')
    with localcontext(Context(prec=100, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)):
        z = Decimal(x.numerator)/Decimal(x.denominator)
        exponential = (2*z).exp()
        value = (exponential-1)/(exponential+1)
    # The surrounding +/- 1e-60 envelope exceeds conversion, exp and ratio
    # errors at 100 decimal digits on the declared [-100,100] domain.
    return Fraction(value)


def coefficient_bound(logit_reference, logit_error, bounds):
    reference = vector(logit_reference, 3)
    error = vector(logit_error, 3)
    require(all(e >= 0 for e in error), 'nonnegative logit error required')
    limits = matrix(bounds, 2)
    require(len(limits) == 3 and all(lo < hi for lo, hi in limits), 'three strict decoder bounds required')
    centers, radii = [], []
    for x, e, (lo, hi) in zip(reference, error, limits):
        center = tanh_reference(x)
        action_error = e + TANH_ABSOLUTE_BUDGET + DECIMAL_ERROR + 3*gamma(4) + 4*MIN_NORMAL32
        centers.append(lo + (center+1)*(hi-lo)/2)
        decoder_error = gamma(8, U64)*(3*abs(lo)+3*abs(hi)+1)
        radii.append((hi-lo)*action_error/2 + decoder_error)
    return tuple(centers), tuple(radii)


def within(values, centers, radii, label):
    centers = vector(centers)
    radii = vector(radii, len(centers))
    actual = vector(values, len(centers))
    require(all(e >= 0 for e in radii), 'error envelope shape/sign')
    require(all(abs(v-c) <= e for v, c, e in zip(actual, centers, radii)),
            'arithmetic error envelope exceeded: '+label)


def actor_contract(weights, observation, stages, bounds):
    require(type(weights) is dict and type(stages) is dict, 'actual layer mapping required')
    labels = ('actor.latent_pi.0', 'actor.latent_pi.2', 'actor.mu')
    required = {name+ending for name in labels for ending in ('.weight', '.bias')}
    require(set(weights) == required, 'exact inference tensor membership required')
    require(set(stages) == {*(name+'.linear' for name in labels),
                           'actor.latent_pi.0.relu', 'actor.latent_pi.2.relu',
                           'tanh', 'action', 'coefficients'}, 'exact observed stage membership required')
    value = vector(observation, 15)
    error = (Fraction(0),)*15
    reports = []
    for index, label in enumerate(labels):
        rows = matrix(weights[label+'.weight'], len(value))
        require(len(rows) == (3 if index == 2 else 256), 'frozen actor architecture differs')
        bias = weights[label+'.bias']
        next_value = affine_reference(rows, bias, value)
        next_error = affine_error(rows, bias, value, error)
        require(all(abs(x)+e < Fraction(2**128-2**104) for x, e in zip(next_value, next_error)),
                'possible binary32 affine overflow')
        # A small final result after cancellation does not exclude overflowing
        # products or partial sums. Bound every reduction by its absolute term
        # sum, including inherited input uncertainty and the underflow floor.
        magnitudes = affine_reference([[abs(x) for x in row] for row in rows],
                                      [abs(x) for x in vector(bias,len(rows))],
                                      [abs(x)+e for x,e in zip(value,error)])
        require(all(m+e < Fraction(2**128-2**104) for m,e in zip(magnitudes,next_error)),
                'possible binary32 intermediate overflow')
        error = next_error
        within(stages[label+'.linear'], next_value, error, label+'.linear')
        reports.append(dict(stage=label+'.linear', maximumErrorBound=float(max(error))))
        if index != 2:
            value = relu_reference(next_value)
            observed_linear = vector(stages[label+'.linear'])
            require(vector(stages[label+'.relu']) == relu_reference(observed_linear), 'actual ReLU differs')
            within(stages[label+'.relu'], value, error, label+'.relu')
        else:
            value = next_value
    # Validate the implementation of tanh at its ACTUAL logits independently,
    # rather than assuming that every platform kernel satisfies an ulp claim.
    actual_logit = vector(stages['actor.mu.linear'], 3)
    tanh_centers = tuple(tanh_reference(x) for x in actual_logit)
    within(stages['tanh'], tanh_centers, (TANH_ABSOLUTE_BUDGET+DECIMAL_ERROR,)*3, 'tanh kernel')
    action_center = tuple(tanh_reference(x) for x in value)
    action_radii = tuple(e + TANH_ABSOLUTE_BUDGET + DECIMAL_ERROR
                        + 3*gamma(4) + 4*MIN_NORMAL32 for e in error)
    require(all(abs(x) <= 1 for x in vector(stages['action'], 3)), 'action outside declared box')
    within(stages['action'], action_center, action_radii, 'float32 box unscale')
    centers, radii = coefficient_bound(value, error, bounds)
    coefficients = vector(stages['coefficients'], 3)
    limits = matrix(bounds, 2)
    require(all(lo <= x <= hi for x, (lo, hi) in zip(coefficients, limits)), 'coefficient outside decoder box')
    within(coefficients, centers, radii, 'decoded coefficients')
    return dict(schema='mc4-actor-arithmetic-contract-v2', status='arithmetic-validated',
                layerBounds=reports, coefficientReference=[float(x) for x in centers],
                coefficientReferenceRational=[dict(numerator=str(x.numerator), denominator=str(x.denominator))
                                              for x in centers],
                coefficientErrorBound=[float(x) for x in radii],
                coefficientBoundRational=[dict(numerator=str(x.numerator), denominator=str(x.denominator))
                                          for x in radii],
                coefficientErrorObserved=[float(abs(a-b)) for a, b in zip(coefficients, centers)],
                mathematicalInputExact=True, float32ByteIdentityRequired=False,
                errorBoundsChosenFromObservedDiscrepancy=False,
                newObjectiveCalls=0, nativeExecutionAuthorized=False)
