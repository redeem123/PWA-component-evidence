"""Objective-free v2 state-to-motion computation; never native admission.

All input/source hashes and v1 baseline motion comparisons remain exact or use
their unchanged tolerance. Only checkpoint arithmetic and its propagated motion
distance have the separately authorized v2 numerical contract.
"""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PUBLIC_MANIFEST_SHA = '8a74a20a41836e485d8b51346527b45d7f7def22ae18c7b0dc41e85f3e5e4988'
RECONSTRUCT_SHA = 'e019712678ba1324b5aef8c61942922b21df8fd0f6218a4488341f064dd88373'
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
           'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def inventory(package):
    require(package.resolve() == package, 'physical public package required')
    raw = (package/'independent-package.json').read_bytes()
    require(sha(raw) == PUBLIC_MANIFEST_SHA, 'fixed public manifest differs')
    expected = json.loads(raw)['filesSha256']
    require(not any(p.is_symlink() for p in package.rglob('*')), 'redirected package member')
    actual = {str(p.relative_to(package)):sha(p.read_bytes()) for p in package.rglob('*') if p.is_file()}
    require(actual == {**expected, 'independent-package.json':PUBLIC_MANIFEST_SHA}, 'public inventory drift')
    return actual


def load(name, path, digest=None):
    raw = path.read_bytes()
    require(path.resolve() == path and (digest is None or sha(raw) == digest), 'dependency source differs')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(path.read_bytes() == raw, 'source changed during import')
    return module


def fraction_record(record):
    return tuple(Fraction(int(r['numerator']), int(r['denominator'])) for r in record)


def compute(package):
    package = Path(package).absolute()
    package_before = inventory(package)
    source_paths = [HERE/name for name in ('contract.py','geometry.py','preflight.py')]
    source_before = {p.name:sha(p.read_bytes()) for p in source_paths}
    contract = load('_mc4_v2_contract', HERE/'contract.py')
    geometry = load('_mc4_v2_geometry', HERE/'geometry.py')
    recon = load('_mc4_pinned_reconstructor', package/'research/measurement_portability_v1/reconstruct.py', RECONSTRUCT_SHA)
    with recon.no_measurement_imports():
        np, torch = recon.runtime()
        environment = recon.environment_binding(np, torch)
        root = package/'original'
        reader = recon.frozen_reader(root)
        core = reader.projection(root)
        model = recon.weights(root, reader, torch)
        _, prefix = reader.state(root/'fixture/d30-202609301/fork-state',25,
                                'coco-2.8.2-bare-raw-v1:bbob_f001_i201_d30')
        protocol = prefix['protocol']
        require(prefix['_seed'] == recon.SEED and protocol['population_size'] == 100
                and protocol['dimension'] == 30 and protocol['max_iterations'] == 399
                and protocol['observation_interval'] == 25 and protocol['train_agent'] is False
                and protocol['store_transitions'] is False and protocol['escape_mechanism_version'] == 'subspace-v2'
                and protocol['escape_dimension_fraction'] == .1, 'fixed saved prefix scope differs')
        observation = prefix['state']
        stages = {}
        with torch.no_grad():
            value = torch.from_numpy(observation.copy())
            for label in ('actor.latent_pi.0','actor.latent_pi.2','actor.mu'):
                value = torch.nn.functional.linear(value,model[label+'.weight'],model[label+'.bias'])
                stages[label+'.linear'] = value.numpy()[0].copy().tolist()
                if label != 'actor.mu':
                    value = torch.relu(value)
                    stages[label+'.relu'] = value.numpy()[0].copy().tolist()
            stages['tanh'] = torch.tanh(value).numpy()[0].copy().tolist()
        action = recon.actor_action(model,observation,np,torch)
        bounds = np.array([protocol[k] for k in ('w_range','c1_range','c2_range')])
        coefficients = reader.decode(action,bounds)
        stages.update(action=action.tolist(),coefficients=coefficients.tolist())
        inference_weights = {k:v.tolist() for k,v in model.items()
                             if k.startswith(('actor.latent_pi.','actor.mu.'))}
        arithmetic = contract.actor_contract(inference_weights,observation[0].tolist(),stages,bounds.tolist())
        p, fitness = prefix['best_positions'][0],prefix['best_fitness'][0]
        neighbors = (np.arange(100)[:,None]+np.array([-1,0,1])) % 100
        local = neighbors[np.arange(100),np.argmin(fitness[neighbors],axis=1)]
        draws = reader.control_draws(recon.SEED,30)
        inputs = dict(state_x=prefix['positions'][0],state_v=prefix['velocities'][0],state_pbest=p,
            state_g=p[np.argmin(fitness)],state_l=p[local],lower=prefix['_lb'][0],upper=prefix['_ub'][0],
            vmax=prefix['_vmax'][0],pre_pbest_fitness=fitness,pre_stagnation=prefix['stagnation_counter'][0],
            informer_indices=neighbors,r1=recon.operation_draw(0,(100,30),torch),
            r2=recon.operation_draw(1,(100,30),torch),gaussian=draws['gaussian'],informer_draws=draws['informers'])
        cases = {arm:reader.npz(root/'fixture/d30-202609301/early_all'/arm/'pulse.npz') for arm in recon.ARMS}
        for z in cases.values():
            for key, actual in inputs.items():
                reader.exact(actual,z[key],'v2 unchanged actual input '+key)
        z = cases['removal']
        reference_coefficients = z['coefficients']
        for item in cases.values():
            reader.exact(item['coefficients'],reference_coefficients,'common historical coefficients')
        contract.within(reference_coefficients.tolist(),fraction_record(arithmetic['coefficientReferenceRational']),
                        fraction_record(arithmetic['coefficientBoundRational']),'historical coefficient arithmetic')
        baseline = reader.proposals(core,z,'early_all',reference_coefficients,expected_tolerance=(recon.RTOL,recon.ATOL))
        actual, detail = geometry.proposals(core,z,coefficients,np)
        _, reference_detail = geometry.proposals(core,z,reference_coefficients,np)
        reader.exact(detail['localRanks'],reference_detail['localRanks'],'unchanged local rank selection')
        reader.exact(detail['zeroResidual'],reference_detail['zeroResidual'],'unchanged residual zero selection')
        require(detail['sharedRank'] == reference_detail['sharedRank'],'shared rank changed')
        motion_budget = geometry.raw_error_bound(core,z,baseline,actual,reference_coefficients,coefficients,np)
        rows = []
        for arm in recon.ARMS:
            current = cases[arm]
            # V1's original proposal, clamp, escape and coordinate comparisons
            # are still mandatory for the historical-coefficient baseline.
            reader.proposals(core,current,'early_all',reference_coefficients,expected_tolerance=(recon.RTOL,recon.ATOL))
            reference_x,reference_v,reference_escape = recon.repair(reader,current,baseline[arm],np,torch)
            x,v,escaped,outside = geometry.repair(recon,reader,current,actual[arm],np,torch)
            _,_,_,reference_outside = geometry.repair(recon,reader,current,baseline[arm],np,torch)
            reader.exact(escaped,reference_escape,'unchanged escape mask')
            reader.exact(outside,reference_outside,'unchanged boundary-damping branch')
            width = current['upper']-current['lower']
            point_radius = np.max(width)*motion_budget[arm]
            point_radius += core.row_norm(2e-9+2e-12*np.abs(current['evaluated_position']))
            point_radius = np.nextafter(point_radius,np.inf)
            observed_distance = core.row_norm(x-current['evaluated_position'])
            require(np.all(observed_distance <= point_radius),'repaired motion envelope exceeded: '+arm)
            velocity_radius = np.max(width)*motion_budget[arm]+core.row_norm(2e-9+2e-12*np.abs(reference_v))
            require(np.all(core.row_norm(v-reference_v) <= velocity_radius),'velocity envelope exceeded: '+arm)
            require(np.all(np.isfinite(x)) and np.all(x >= -5) and np.all(x <= 5),'finite native box required')
            rows.append(dict(arm=arm,points=recon.batch(np.ascontiguousarray(x,dtype='<f8'),[100,30]),
                rawProposal=recon.batch(np.ascontiguousarray(actual[arm],dtype='<f8'),[100,30]),
                velocities=recon.batch(np.ascontiguousarray(v,dtype='<f8'),[100,30]),
                cachedValues=recon.batch(np.ascontiguousarray(current['fitness'],dtype='<f8'),[100]),
                coordinateErrorRadius=point_radius.tolist(),coordinateErrorObserved=observed_distance.tolist(),
                escapedParticles=int(escaped.sum())))
        require(inventory(package) == package_before,'public input drift during computation')
        require(recon.environment_binding(np,torch) == environment,'numerical runtime drift')
        require({p.name:sha(p.read_bytes()) for p in source_paths} == source_before,'v2 source drift')
        return dict(schema='mc4-numerical-motion-preflight-v2',status='computed',scope=recon.SCOPE,
            sourcesSha256=source_before,publicPackageManifestSha256=PUBLIC_MANIFEST_SHA,
            originalPackageManifestSha256=recon.MANIFEST_SHA,checkpointSha256=recon.CHECKPOINT_SHA,
            environmentBinding=environment,arithmeticContract=arithmetic,
            actionFloat32Hex=action.tobytes().hex(),coefficientsFloat64Hex=coefficients.tobytes().hex(),
            preBestFitness=recon.batch(np.ascontiguousarray(fitness,dtype='<f8'),[100]),batches=rows,
            exactNonControllerInputs=True,originalBaselineGeometryPassed=True,
            originalBaselineTolerance=dict(rtol=recon.RTOL,atol=recon.ATOL),
            newObjectiveCalls=0,trainingTransitions=0,nativeExecutionAuthorized=False,
            targetNativeBackendQualified=False,independentTeamReplication=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    package = Path(args.package).absolute()
    require(output.resolve() == output and output.parent.is_dir() and not os.path.lexists(output),
            'new physical preflight output required')
    require(package != output and package not in output.parents,'output outside immutable package required')
    for key in THREADS:
        os.environ[key] = '1'
    result = compute(package)
    with output.open('xb') as stream:
        stream.write((json.dumps(result,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result['status'],newObjectiveCalls=0,
        maximumCoordinateDistance=max(max(row['coordinateErrorObserved']) for row in result['batches']),
        maximumCoordinateRadius=max(max(row['coordinateErrorRadius']) for row in result['batches']),
        nativeExecutionAuthorized=False)))


if __name__ == '__main__':
    main()
