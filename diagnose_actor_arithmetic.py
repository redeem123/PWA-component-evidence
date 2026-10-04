"""Observe checkpoint arithmetic without admitting or repairing a measurement.

The frozen preflight is not invoked or modified. Every checkpoint-derived stage
and its actual runtime are retained for comparison across hosts. No objective,
learner, allocation or qualification operation is available here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
MANIFEST_SHA = '8a74a20a41836e485d8b51346527b45d7f7def22ae18c7b0dc41e85f3e5e4988'
RECONSTRUCT_SHA = 'e019712678ba1324b5aef8c61942922b21df8fd0f6218a4488341f064dd88373'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def array(value):
    raw = value.tobytes()
    return dict(shape=list(value.shape), dtype=value.dtype.str, dataHex=raw.hex(),
                sha256=sha(raw), values=value.tolist())


def inventory(package):
    require(package.resolve() == package, 'physical package required')
    raw = (package/'independent-package.json').read_bytes()
    require(sha(raw) == MANIFEST_SHA, 'public manifest differs')
    expected = json.loads(raw)['filesSha256']
    actual = {str(p.relative_to(package)): sha(p.read_bytes())
              for p in package.rglob('*') if p.is_file()}
    require(not any(p.is_symlink() for p in package.rglob('*')), 'redirected package member')
    require(actual == {**expected, 'independent-package.json': MANIFEST_SHA},
            'exact public inventory required')
    return actual


def diagnose(package):
    package = Path(package).absolute()
    before = inventory(package)
    source = Path(__file__).resolve()
    source_before = sha(source.read_bytes())
    path = package/'research/measurement_portability_v1/reconstruct.py'
    require(sha(path.read_bytes()) == RECONSTRUCT_SHA, 'frozen reconstructor differs')
    spec = importlib.util.spec_from_file_location('_diagnostic_pinned_reconstructor', path)
    recon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recon)
    with recon.no_measurement_imports():
        np, torch = recon.runtime()
        environment = recon.environment_binding(np, torch)
        reader = recon.frozen_reader(package/'original')
        model = recon.weights(package/'original', reader, torch)
        _, prefix = reader.state(package/'original/fixture/d30-202609301/fork-state', 25,
                                'coco-2.8.2-bare-raw-v1:bbob_f001_i201_d30')
        protocol = prefix['protocol']
        observation = prefix['state']
        stages = {'observation': array(observation)}
        with torch.no_grad():
            value = torch.from_numpy(observation.copy())
            for label in ('actor.latent_pi.0', 'actor.latent_pi.2', 'actor.mu'):
                value = torch.nn.functional.linear(value, model[label+'.weight'], model[label+'.bias'])
                stages[label+'.linear'] = array(value.numpy().copy())
                if label != 'actor.mu':
                    value = torch.relu(value)
                    stages[label+'.relu'] = array(value.numpy().copy())
            stages['tanh'] = array(torch.tanh(value).numpy().copy())
        action = recon.actor_action(model, observation, np, torch)
        bounds = np.array([protocol[k] for k in ('w_range', 'c1_range', 'c2_range')])
        coefficients = reader.decode(action, bounds)
        stages.update(action=array(action), bounds=array(bounds), coefficients=array(coefficients))
        expected = reader.npz(package/'original/fixture/d30-202609301/early_all/removal/pulse.npz')['coefficients']
        delta = coefficients - expected
        draws = {f'slot{slot}': array(recon.operation_draw(slot, shape, torch))
                 for slot, shape in ((0, (100, 30)), (1, (100, 30)), (2, (100,)))}
        require(recon.environment_binding(np, torch) == environment, 'runtime changed')
        require(inventory(package) == before, 'package changed')
        require(sha(source.read_bytes()) == source_before, 'diagnostic source changed')
        return dict(schema='pwa-checkpoint-arithmetic-observation-v1', status='observed',
                    diagnosticSourceSha256=source_before, packageManifestSha256=MANIFEST_SHA,
                    reconstructSourceSha256=RECONSTRUCT_SHA, checkpointSha256=recon.CHECKPOINT_SHA,
                    environmentBinding=environment, stages=stages, keyedDraws=draws,
                    expectedCoefficients=array(expected), coefficientDelta=array(delta),
                    exactCoefficients=(coefficients.dtype == expected.dtype and coefficients.shape == expected.shape
                                       and coefficients.tobytes() == expected.tobytes()),
                    newObjectiveCalls=0, trainingTransitions=0, nativeExecutionAuthorized=False,
                    qualificationGranted=False, frozenChecksModified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    require(output.resolve() == output and output.parent.is_dir() and not os.path.lexists(output),
            'new physical diagnostic output required')
    package = Path(args.package).absolute()
    require(package != output and package not in output.parents, 'output outside immutable package required')
    for key in THREADS:
        os.environ[key] = '1'
    result = diagnose(package)
    with output.open('xb') as stream:
        stream.write((json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({k:result[k] for k in ('status', 'exactCoefficients', 'newObjectiveCalls', 'qualificationGranted')}))


if __name__ == '__main__':
    main()
