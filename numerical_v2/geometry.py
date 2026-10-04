"""Same five motion formulas with separately declared rounding propagation.

NumPy is supplied by the actual pinned preflight. This module never evaluates
an objective or imports a native backend. V1 baseline comparisons remain in
the caller and are not overridden here.
"""


def proposals(core, z, coefficients, np):
    x, v, p, g, r1, r2 = [z[k] for k in ('state_x', 'state_v', 'state_pbest', 'state_g', 'r1', 'r2')]
    width = z['upper']-z['lower']
    basis = np.stack((v, r1*(p-x), r2*(g-x)), axis=-1)
    base = np.einsum('ndk,k->nd', basis, coefficients)
    increment = coefficients[2]*r2*(.25*g+.75*z['state_l']-g)
    split = core.decompose(basis, increment, width)
    norm = core.row_norm(split.residual/width)
    direction, fallback = core.matched_direction(split.particle_space, z['gaussian'], norm)
    if fallback.any():
        raise ValueError('unregistered Gaussian fallback')
    full = base+increment
    removal = base+split.particle_explainable
    quarter = removal+.25*split.residual
    random = removal+.25*direction*width
    zero = norm == 0
    for value in (removal, quarter, random):
        value[zero] = full[zero]
    full_norm, quarter_norm = [core.row_norm(v/width) for v in (full, quarter)]
    if np.any((full_norm == 0) & (quarter_norm != 0)):
        raise ValueError('unmatchable whole proposal')
    ratio = np.divide(quarter_norm, full_norm, out=np.ones_like(norm), where=full_norm > 0)
    if np.any(ratio > 1+1e-10):
        raise ValueError('whole proposal enlarges step')
    whole = ratio[:, None]*full
    whole[zero] = full[zero]
    return dict(removal=removal, full=full, quarter=quarter,
                quarter_gaussian=random, whole_norm=whole), dict(
        localRanks=split.particle_space.ranks, sharedRank=split.shared_rank,
        zeroResidual=zero, basis=basis, ratio=ratio)


def raw_error_bound(core, z, reference, actual, reference_coefficients, coefficients, np):
    """Bound fixed-field motion perturbations from the actual coefficient delta.

The formula is fixed before native observations. The actor certificate separately
requires both coefficient triples to lie in its analytic arithmetic envelope.
No point or objective discrepancy is fitted to choose this bound.
"""
    x, v, p, g, r1, r2 = [z[k] for k in ('state_x', 'state_v', 'state_pbest', 'state_g', 'r1', 'r2')]
    width = z['upper']-z['lower']
    basis = np.stack((v, r1*(p-x), r2*(g-x)), axis=-1)/width[None, :, None]
    delta = np.abs(coefficients-reference_coefficients)
    increment_unit = r2*(.25*g+.75*z['state_l']-g)/width
    # Positive social coefficient preserves the Gaussian direction's scaling.
    if min(coefficients[2], reference_coefficients[2]) <= 0:
        raise ValueError('positive social coefficient required for direction scaling')
    affine = sum(core.row_norm(basis[:, :, k])*delta[k] for k in range(3))
    affine += core.row_norm(increment_unit)*delta[2]
    result = {}
    for arm in ('removal', 'full', 'quarter', 'quarter_gaussian'):
        slack = core.row_norm((2e-9+2e-12*np.abs(reference[arm]))/width)
        result[arm] = np.nextafter(affine+slack, np.inf)
    f = core.row_norm(reference['full']/width)
    q = core.row_norm(reference['quarter']/width)
    e = np.maximum(result['full'], result['quarter'])
    zero = (f == 0)
    if np.any(zero & ((q != 0) | (core.row_norm(actual['full']/width) != 0)
                      | (core.row_norm(actual['quarter']/width) != 0))):
        raise ValueError('zero whole-step denominator changed')
    if np.any((~zero) & (f <= e)):
        raise ValueError('whole-step normalization has no certified denominator margin')
    ratio = np.divide(q, f, out=np.ones_like(q), where=~zero)
    denominator = np.where(zero, 1, f-e)
    ratio_error = np.where(zero, 0, (e+ratio*e)/denominator)
    whole = ratio*e+(f+e)*ratio_error
    whole += core.row_norm((2e-9+2e-12*np.abs(reference['whole_norm']))/width)
    result['whole_norm'] = np.nextafter(whole, np.inf)
    for arm in result:
        observed = core.row_norm((actual[arm]-reference[arm])/width)
        if not np.all(np.isfinite(result[arm])) or np.any(observed > result[arm]):
            raise ValueError('motion propagation bound exceeded: '+arm)
    return result


def repair(recon, reader, z, raw, np, torch):
    n, d = raw.shape
    if (n, d) != (100, 30):
        raise ValueError('fixed motion shape differs')
    draws = dict(r1=recon.operation_draw(0, (n,d), torch), r2=recon.operation_draw(1, (n,d), torch),
                 draw_2=recon.operation_draw(2, (n,), torch))
    mask = (z['pre_stagnation'] >= 19) & (draws['draw_2'] < .3)
    slots = {'draw_2'} | ({'draw_3','draw_4','draw_5'} if mask.any() else set())
    if {k for k in z if k.startswith('draw_')} != slots:
        raise ValueError('conditional draw membership differs')
    if mask.any():
        for slot in (3,4,5):
            draws[f'draw_{slot}'] = recon.operation_draw(slot, (n,d), torch)
    for key, value in draws.items():
        reader.exact(value, z[key], key)
    clipped = np.maximum(np.minimum(raw, z['vmax']), -z['vmax'])
    before = z['state_x']+clipped
    outside = (before < z['lower']) | (before > z['upper'])
    x = np.maximum(np.minimum(before, z['upper']), z['lower'])
    v = np.where(outside, -.5*clipped, clipped)
    if mask.any():
        lower, upper, pbest = [torch.from_numpy(z[k].copy()) for k in ('lower','upper','state_pbest')]
        scale = .2*(1-26/399)**2*(upper-lower)
        angle = torch.tensor(float(np.pi), dtype=torch.float64)*(torch.from_numpy(draws['draw_3'])-.5)
        step = torch.clamp(scale*torch.tan(angle), -5*scale, 5*scale)
        selected = torch.argsort(torch.from_numpy(draws['draw_4']), dim=-1, stable=True)[..., :max(1,int(np.floor(.1*d+.5)))]
        chosen = torch.zeros_like(pbest,dtype=torch.bool)
        chosen.scatter_(-1,selected,True)
        candidate = torch.maximum(torch.minimum(pbest+torch.where(chosen,step,torch.zeros_like(step)),upper),lower).numpy()
        velocity = (draws['draw_5']-.5)*(.2*z['vmax'])
        x = np.where(mask[:,None],candidate,x)
        v = np.where(mask[:,None],velocity,v)
    return x, v, mask, outside
