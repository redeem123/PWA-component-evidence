"""Read supplied v2 records and raw joins, never admit or measure anything.

Requires a separate, exact 43-file source bundle and a relocated hosted artifact
root. The reservation digest must be independently supplied, not inferred from
that artifact. Consistency of supplied records is not native provenance or an
independent-team replication. CLI writes only its result to stdout.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
from types import ModuleType


MANIFEST_SHA = 'f3a84e1db0d5faedcc89b59832b0d85ed6e72c58e4be03cdbf09626971531851'
QUALIFICATION_SHA = 'ddc69d023158977b3f81908e46ffd5c2f9f55c3b924f1704f2f65ffcd6fd764c'
SOURCE_COMMIT = 'ea3d94cbf9bdd85ef653fb0eb0150816751e0aff'
MANIFEST = 'numerical-source-bundle.json'
HERE = 'research/measurement_numerical_v2/'
FIXTURE = 'artifacts/audits/measurement-portability-fixtures-20261002/expected-fixtures.json'
GEOMETRY = 'artifacts/audits/portable-input-ledger-source-20261002-v1/expected-geometry.json'
OUTPUT = 'artifacts/qualifications/measurement_numerical_v2/native532_v2'
RECORDS = ('reservation.json', 'request.json', 'qualification.json', 'permission.json', 'claim.json')
ARMS = ('removal', 'full', 'quarter', 'quarter_gaussian', 'whole_norm')
NAMES = ('authorization.json', 'PROTOCOL.md', 'contract.py', 'geometry.py', 'preflight.py', 'plan.py',
         'policy.py', 'ledger.py', 'effect.py', 'archive.py', 'transport.py', 'sources.py',
         'qualify.py', 'admission.py', 'runner.py', 'host_job.py')
TEST_NAMES = tuple('test_measurement_numerical_' + name + '_v2.py' for name in
                   ('contract', 'plan', 'storage', 'transport', 'admission', 'runner', 'sources', 'host_job'))
FOREIGN = ('research/component_diagnosis_v2/backend.py',
           'research/measurement_portability_v1/native_transport.py',
           'research/measurement_portability_v1/host_ledger.py',
           'research/measurement_portability_v1/reconstruct.py',
           'tests/test_portable_native_transport_v1.py',
           'research/measurement_release_v1/authorization.json',
           'research/measurement_portability_v1/author_admission.py',
           'research/measurement_portability_v1/shared_claim.py',
           'research/measurement_portability_v1/measurement_plan.py',
           'research/measurement_portability_v1/fixtures.py',
           'research/measurement_portability_v1/call_capture.py',
           'research/component_measurement_execution_v1/execution_common.py',
           'research/component_measurement_execution_v1/execution_gates.py')
SOURCE_PATHS = frozenset([HERE + name for name in NAMES] + ['tests/' + name for name in TEST_NAMES] + list(FOREIGN))
EXTRAS = frozenset((FIXTURE, GEOMETRY, 'qualification/receipt.json',
                    'qualification/tests.stdout.log', 'qualification/tests.stderr.log'))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def encoded(value):
    return (canonical(value) + '\n').encode('ascii')


def same(actual, expected, reason):
    require(canonical(actual) == canonical(expected), reason)


def pin(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None
            and value != '0' * 64, 'independently supplied nonplaceholder digest required')


def parsed_document(raw):
    """Decode a byte-pinned source document, not a canonical envelope record."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('nonfinite JSON constant')
    require(type(raw) is bytes, 'literal record bytes required')
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    require(type(value) is dict, 'object document required')
    return value


def parsed(raw):
    value = parsed_document(raw)
    require(encoded(value) == raw, 'canonical object record required')
    return value


def physical_root(value):
    path = Path(value).absolute()
    require('..' not in path.parts and path.resolve() == path and path.is_dir(), 'physical directory required')
    return path


class Snapshots:
    """Retain bytes and read-time identities; refuse redirected/hardlinked input."""
    def __init__(self):
        self.files = {}

    def read(self, path, limit=16 * 1024**2):
        path = Path(path).absolute()
        require('..' not in path.parts and path.resolve() == path, 'physical record path required')
        try:
            before = path.lstat()
            require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                    and 0 < before.st_size <= limit, 'bounded singly linked regular record required')
            raw = path.read_bytes()
            after = path.lstat()
        except OSError as error:
            raise ValueError('missing/unreadable record: ' + str(path)) from error
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(identity(before) == identity(after) and len(raw) == before.st_size, 'record changed during read')
        if path in self.files:
            require(self.files[path] == (raw, identity(after)), 'record drift')
        self.files[path] = raw, identity(after)
        return raw

    def check(self):
        for path, (raw, _) in tuple(self.files.items()):
            require(self.read(path, len(raw)) == raw, 'record drift after audit')


def inventory(root):
    names = set()
    allowed_directories = {str(parent) for name in SOURCE_PATHS | EXTRAS | {MANIFEST}
                           for parent in PurePosixPath(name).parents if str(parent) != '.'}
    for base, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            path = Path(base) / name
            require(not path.is_symlink() and path.resolve() == path, 'redirected bundle directory')
            require(str(path.relative_to(root)) in allowed_directories, 'extra source-bundle directory')
        for name in files:
            path = Path(base) / name
            require(not path.is_symlink(), 'redirected bundle member')
            names.add(str(path.relative_to(root)))
        require(len(names) <= 43, 'extra source-bundle member')
    require(names == SOURCE_PATHS | EXTRAS | {MANIFEST}, 'exact 43-file source bundle required')
    return names


def validate_bundle(root, snapshots):
    members = inventory(root)
    raw = snapshots.read(root / MANIFEST)
    require(sha(raw) == MANIFEST_SHA, 'fixed source manifest differs')
    manifest = parsed(raw)
    require(manifest.get('schema') == 'mc4-numerical-qualified-source-bundle-v2', 'bundle schema')
    files = manifest.get('filesSha256')
    require(type(files) is dict and set(files) == members - {MANIFEST}, 'manifest membership')
    for name, digest in files.items():
        pin(digest)
        require(sha(snapshots.read(root / name)) == digest, 'bundle member differs: ' + name)
    binding = manifest['sourceBinding']
    require(type(binding) is dict and set(binding) == {'commit', 'filesSha256', 'sourceSetSha256'}
            and binding['commit'] == SOURCE_COMMIT, 'qualified source binding')
    require(type(binding['filesSha256']) is dict and set(binding['filesSha256']) == SOURCE_PATHS,
            'exact 37-source binding')
    same(binding['filesSha256'], {name: files[name] for name in SOURCE_PATHS}, 'source binding/files differ')
    require(binding['sourceSetSha256'] == sha(canonical(binding['filesSha256']).encode()), 'source set hash')
    for field in ('nativeExecutionAuthorized', 'authorAllocationCreated'):
        require(manifest.get(field) is False, 'bundle data cannot confer authority')
    require(type(manifest.get('newObjectiveCalls')) is int and manifest['newObjectiveCalls'] == 0,
            'bundle objective-free count')
    qualification_raw = snapshots.read(root / 'qualification/receipt.json')
    require(sha(qualification_raw) == QUALIFICATION_SHA == manifest['sourceQualificationSha256'],
            'fixed qualification receipt differs')
    qualification = parsed(qualification_raw)
    same(qualification.get('sourceBinding'), binding, 'qualification/source binding differs')
    require(qualification.get('schema') == 'mc4-numerical-executor-source-qualification-v2'
            and qualification.get('status') == 'passed' and qualification.get('nativeExecutionAuthorized') is False,
            'passed source-only qualification required')
    for field in ('newObjectiveCalls', 'trainingTransitions'):
        require(type(qualification.get(field)) is int and qualification[field] == 0, 'qualification literal zero')
    roster = []
    for name in TEST_NAMES:
        for cls in ast.parse(snapshots.read(root / 'tests' / name)).body:
            if isinstance(cls, ast.ClassDef):
                roster.extend(f'{Path(name).stem}.{cls.name}.{f.name}' for f in cls.body
                              if isinstance(f, ast.FunctionDef) and f.name.startswith('test_'))
    require(len(roster) == len(set(roster)) == 359, 'complete actual source roster required')
    report = qualification.get('testReport')
    require(type(report) is dict and type(report.get('testsRun')) is int and report['testsRun'] == 359,
            'literal qualification count')
    for checks in (qualification.get('sourceExpectedChecks'), report.get('discoveredChecks'), report.get('executedChecks')):
        require(type(checks) is list and all(type(x) is str for x in checks)
                and len(checks) == len(set(checks)) and sorted(checks) == sorted(roster), 'complete qualification rosters')
    for field in ('failures', 'errors', 'skips', 'expectedFailures', 'unexpectedSuccesses'):
        require(type(report.get(field)) is int and report[field] == 0, 'unsuccessful qualification')
    outputs = qualification.get('outputsSha256')
    require(type(outputs) is dict and set(outputs) == {'tests.stdout.log', 'tests.stderr.log'}, 'qualification log membership')
    for name, digest in outputs.items():
        require(digest == files['qualification/' + name], 'qualification log pin differs')
    return manifest, qualification_raw


def load_helper(root, name, snapshots, binding):
    require(name in ('plan', 'archive'), 'only read-only helpers may load')
    path = root / (HERE + name + '.py')
    raw = snapshots.read(path)
    require(sha(raw) == binding['filesSha256'][HERE + name + '.py'], 'helper source pin')
    module = ModuleType('_mc4_read_only_' + name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    require(snapshots.read(path) == raw, 'helper load drift')
    return module


def literals(root, name, snapshots):
    tree = ast.parse(snapshots.read(root / 'research/measurement_portability_v1/shared_claim.py'))
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    require(len(matches) == 1, 'pinned claim constants required')
    return ast.literal_eval(matches[0])


@contextmanager
def no_bytecode():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def read_measurement(source_root, measurement_root, *, trusted_reservation_sha256=None):
    # Do this before filesystem inspection or any qualified helper load.
    pin(trusted_reservation_sha256)
    with no_bytecode():
        return _read(source_root, measurement_root, trusted_reservation_sha256)


def _read(source_root, measurement_root, trusted_digest):
    source, target = physical_root(source_root), physical_root(measurement_root)
    require(source != target and source not in target.parents and target not in source.parents,
            'separate nonoverlapping source and measurement roots required')
    snapshots = Snapshots()
    manifest, qualification_raw = validate_bundle(source, snapshots)
    binding = manifest['sourceBinding']
    require(snapshots.read(target / MANIFEST) == snapshots.read(source / MANIFEST), 'artifact manifest copy differs')
    for name in EXTRAS | SOURCE_PATHS:
        path = target / name
        if os.path.lexists(path):
            require(snapshots.read(path) == snapshots.read(source / name), 'copied qualified member differs: ' + name)
    require(snapshots.read(target / 'qualification/receipt.json') == qualification_raw, 'artifact qualification copy differs')
    envelope = target / 'host-handoff/envelope'
    require(envelope.resolve() == envelope and envelope.is_dir()
            and {p.name for p in envelope.iterdir()} == set(RECORDS), 'exact physical five-file envelope required')
    raws = {name: snapshots.read(envelope / name) for name in RECORDS}
    require(sha(raws['reservation.json']) == trusted_digest, 'externally trusted reservation differs')
    ticket, request, qualification, permission, claim = [parsed(raws[n]) for n in RECORDS]
    require(set(ticket) == {'schema', 'status', 'requestSha256', 'permissionSha256', 'qualificationSha256',
        'claim', 'binding', 'targetRequest', 'sourceQualification', 'humanPermission', 'authorityWitness',
        'baselineWitness', 'terminalWitness', 'inputPlan', 'acceptancePolicy', 'policySha256',
        'numericalAcceptancePolicySha256', 'newObjectiveCalls', 'nativeMeasurementCompleted', 'publicReleaseAuthorized'},
        'exact reservation record membership required')
    require(raws['request.json'] == snapshots.read(target / 'host-handoff/request.json'), 'request copy differs')
    require(raws['qualification.json'] == qualification_raw, 'envelope source qualification bytes differ')
    require(ticket.get('schema') == 'mc4-numerical-native532-author-reservation-v2'
            and ticket.get('status') == 'reserved', 'actual unexecuted author ticket required')
    for field, value in (('newObjectiveCalls', 0), ('nativeMeasurementCompleted', False), ('publicReleaseAuthorized', False)):
        require(type(ticket.get(field)) is type(value) and ticket[field] == value, 'literal reservation flags')
    for name, hash_field, data_field in (('request.json', 'requestSha256', 'targetRequest'),
                                        ('qualification.json', 'qualificationSha256', 'sourceQualification'),
                                        ('permission.json', 'permissionSha256', 'humanPermission')):
        require(ticket.get(hash_field) == sha(raws[name]), 'ticket raw witness hash differs')
        same(ticket.get(data_field), parsed(raws[name]), 'ticket embedded witness differs')
    require(set(request) == {'schema', 'hostId', 'jobId', 'targetOutputRelative', 'sourceBinding', 'preflight', 'backendBinding'}
            and request['schema'] == 'mc4-numerical-native532-target-request-v2'
            and request['targetOutputRelative'] == OUTPUT, 'recorded request scope differs')
    same(request['sourceBinding'], binding, 'recorded target source binding differs')
    for name in ('hostId', 'jobId'):
        require(type(request[name]) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}', request[name]), 'recorded target ID')
    prior_raw = snapshots.read(source / 'research/measurement_release_v1/authorization.json')
    authority_raw = snapshots.read(source / (HERE + 'authorization.json'))
    # These are original, hash-pinned source documents (pretty-printed UTF-8),
    # unlike the canonical ASCII/newline envelope and permanent claim records.
    prior, authority = parsed_document(prior_raw), parsed_document(authority_raw)
    witness = dict(priorAuthoritySha256=sha(prior_raw), numericalAuthoritySha256=sha(authority_raw),
                   priorAuthority=prior, numericalAuthority=authority)
    same(ticket.get('authorityWitness'), witness, 'pinned authority witness differs')
    expected_permission = dict(schema='mc4-numerical-native532-target-permission-v2', approved=True,
        authorizationId=authority['authorizationId'], permissionEvidence=authority['permissionEvidence'],
        requestSha256=sha(raws['request.json']), priorAuthoritySha256=sha(prior_raw),
        numericalAuthoritySha256=sha(authority_raw), externalExecutionAllowed=True,
        checkpointStateAccessAllowed=True, maximumFreshCalls=532, retriesAllowed=False, targetOutputRelative=OUTPUT)
    same(permission, expected_permission, 'recorded derived permission differs')
    planner = load_helper(source, 'plan', snapshots, binding)
    pins = {name: binding['filesSha256'][HERE + name] for name in ('contract.py', 'geometry.py', 'preflight.py')}
    rows, policy, plan_report = planner.build_plan(snapshots.read(source / FIXTURE), encoded(request['preflight']),
                                                 snapshots.read(source / GEOMETRY), pins)
    require(len(rows) == len(policy) == 532, 'literal complete derived input plan')
    same(ticket.get('inputPlan'), plan_report, 'reserved input plan differs')
    same(ticket.get('acceptancePolicy'), list(policy), 'reserved acceptance policy differs')
    policy_sha = sha(json.dumps(policy, separators=(',', ':'), allow_nan=False).encode())
    require(ticket.get('policySha256') == ticket.get('numericalAcceptancePolicySha256')
            == plan_report['policySha256'] == policy_sha, 'all numerical policy hashes must match')
    env = request['preflight']['environmentBinding']
    require(env.get('system') == 'Linux' and env.get('machine') == 'x86_64', 'recorded Linux host required')
    baseline, terminal = ticket['baselineWitness'], ticket['terminalWitness']
    baseline_constants = literals(source, '_BASELINE', snapshots)
    require(type(baseline) is dict and type(terminal) is dict and terminal.get('status') == 'terminal-and-exited', 'recorded baseline/terminal witness')
    for key, value in dict(auditedAttempted=9120, auditedCompleted=9120, qualificationCeiling=10000,
                           available=880, newObjectiveCalls=0).items():
        require(type(baseline.get(key)) is int and baseline[key] == value, 'literal baseline counts')
    for key in ('backendReceiptSha256', 'pulseReceiptSha256', 'backendLedgerSha256', 'backendComparisonsSha256',
                'pulseChargesSha256', 'pulsePayloadInventorySha256', 'backendEventChainSha256', 'pulseEventChainSha256'):
        require(baseline[key] == baseline_constants[key], 'recorded closed baseline pin differs')
    shared = dict(literals(source, '_FIXED', snapshots), jobId=request['jobId'], hostId=request['hostId'],
        sourceCommit=binding['commit'], jobSha256=sha(raws['request.json']),
        hostSha256=sha(canonical(dict(hostId=request['hostId'], environment=env)).encode()),
        sourceSetSha256=binding['sourceSetSha256'], fixturePointsSha256=sha(b''.join(r['point'] for r in rows[:32])),
        environmentSha256=sha(canonical(env).encode()), outputSha256=sha(OUTPUT.encode()),
        checkpointAccessPermissionWitnessSha256=sha(raws['permission.json']), executionPermissionWitnessSha256=sha(raws['permission.json']),
        sourceQualificationWitnessSha256=sha(qualification_raw), targetPreflightSha256=sha(canonical(request['preflight']).encode()),
        baselineAuditWitnessSha256=sha(canonical(baseline).encode()), terminalProductionWitnessSha256=sha(canonical(terminal).encode()),
        pointBatchesSha256={arm: sha(b''.join(r['point'] for r in rows[32 + j*100:132 + j*100])) for j, arm in enumerate(ARMS)},
        baseline=baseline_constants)
    same(ticket.get('binding'), shared, 'original shared reservation binding differs')
    require(raws['claim.json'] == encoded(shared) and type(ticket['claim']) is dict
            and ticket['claim'].get('claimSha256') == sha(raws['claim.json']), 'copied claim bytes/hash differ')
    same(claim, shared, 'claim record differs')
    ledger_binding = dict(sharedClaimSha256=sha(raws['claim.json']), hostJobSha256=sha(raws['request.json']),
        preflightSha256=shared['targetPreflightSha256'], sourceSetSha256=binding['sourceSetSha256'],
        environmentSha256=shared['environmentSha256'], hostId=request['hostId'], jobId=request['jobId'],
        baselineAttempted=9120, ceiling=10000, reserved=532)
    output = target / OUTPUT
    require(snapshots.read(output / 'admission.json') == raws['reservation.json'], 'archived admission bytes differ')
    require(not os.path.lexists(output / 'failure.json'), 'failed artifact cannot be reported complete')
    receipt_raw = snapshots.read(output / 'receipt.json')
    receipt = parsed(receipt_raw)
    require(receipt.get('schema') == 'mc4-numerical-native532-host-execution-v2' and receipt.get('status') == 'completed'
            and receipt.get('trustedReservationSha256') == trusted_digest, 'recorded completed host receipt required')
    for field, value in (('newObjectiveCalls', 532), ('trainingTransitions', 0), ('nativeMeasurementCompleted', True), ('independentTeamReplication', False)):
        require(type(receipt.get(field)) is type(value) and receipt[field] == value, 'literal recorded completion flags/counts')
    for actual, expected in ((receipt.get('sourceBinding'), binding), (receipt.get('binding'), ledger_binding),
                              (receipt.get('inputPlan'), plan_report)):
        same(actual, expected, 'host receipt binding/plan differs')
    native = receipt['nativeTransport']
    require(type(native) is dict and native.get('fixtureNumericalPassed') is True, 'recorded native fixture completion')
    for field in ('nativeCallbacks', 'completed'):
        require(type(native.get(field)) is int and native[field] == 532, 'literal recorded native count')
    ledger_path, marker_path, spool_path = output / 'ledger.sqlite3', output.parent / 'native532_v2.claim', output / 'raw-capture.jsonl'
    for path in (ledger_path, marker_path, spool_path):
        snapshots.read(path, 8 * 1024**2)
    auditor = load_helper(source, 'archive', snapshots, binding)
    joined = auditor.audit_capture(ledger_path, marker_path, spool_path, rows, ledger_binding, policy)
    same(receipt.get('archiveAudit'), joined, 'recorded host/archive audit differs')
    require(joined.get('rawCaptureJoined') is True and type(joined.get('recordedCalls')) is int
            and joined['recordedCalls'] == 532, 'complete raw join required')
    for field, value in (('fixtureCalls', 32), ('routingCalls', 500), ('newObjectiveCalls', 0), ('trainingTransitions', 0)):
        require(type(joined.get(field)) is int and joined[field] == value, 'literal raw-audit counts')
    for field in ('nativeHostQualified', 'nativeExecutionVerified', 'publicAccessVerified', 'authorizationVerified'):
        require(joined.get(field) is False, 'archive consistency cannot confer authority')
    same(joined.get('binding'), ledger_binding, 'raw-audit binding differs')
    require(joined.get('policySha256') == policy_sha, 'raw-audit policy differs')
    effects = joined['effects']
    require(len(effects['removalContrasts']) == 4 and len(effects['controlContrasts']) == 2, 'all six effects required')
    returned = {}
    for line in snapshots.read(spool_path, 8 * 1024**2).splitlines():
        event = parsed(line + b'\n')['body']
        if event['kind'] == 'returned':
            index = event['index']
            require(type(index) is int and index not in returned, 'duplicate raw return')
            returned[index] = event
    require(set(returned) == set(range(532)), 'all raw returns required')
    comparisons = []
    for index, row in enumerate(rows):
        raw = bytes.fromhex(returned[index]['valueHex'])
        require(len(raw) == 8, 'raw scalar width')
        actual, expected = struct.unpack('<d', raw)[0], struct.unpack('<d', row['expected'])[0]
        require(math.isfinite(actual) and math.isfinite(expected) and math.isfinite(actual - expected), 'finite raw comparison')
        entry = dict(index=index, label=row['label'], actualRawHex=raw.hex(), expectedRawHex=row['expected'].hex(),
                     returnType=returned[index]['returnType'], returnMetadata=returned[index]['extra'], actual=actual,
                     cachedExpected=expected, rawDelta=actual - expected, absoluteRawDelta=abs(actual - expected),
                     valueRadius=(1e-9 + 1e-12 * abs(expected)) if index < 32 else policy[index])
        if index >= 32:
            arm_index, particle = divmod(index - 32, 100)
            arm = effects['arms'][ARMS[arm_index]]
            entry.update(coordinateRadius=request['preflight']['batches'][arm_index]['coordinateErrorRadius'][particle],
                         coordinateErrorObserved=request['preflight']['batches'][arm_index]['coordinateErrorObserved'][particle],
                         actualTie=arm['actualTieMask'][particle], expectedTie=arm['expectedTieMask'][particle],
                         strictDecisionChanged=arm['strictPbestMaskFlipMask'][particle], tieClassificationChanged=arm['tieFlipMask'][particle])
        comparisons.append(entry)
    snapshots.check()
    require(inventory(source) == SOURCE_PATHS | EXTRAS | {MANIFEST}, 'bundle membership drift')
    return dict(schema='mc4-relocated-measurement-consistency-reader-v2', status='archive-consistent',
        sourceManifestSha256=MANIFEST_SHA, sourceQualificationSha256=QUALIFICATION_SHA,
        trustedReservationSha256=trusted_digest, recordedHostReceiptSha256=sha(receipt_raw),
        recordedHostedExecution=dict(hostId=request['hostId'], jobId=request['jobId'], nativeCallbacks=532,
                                     sourceCommit=binding['commit'], nativeMeasurementCompleted=True),
        archiveAudit=joined, rawComparisons=comparisons, effects=effects, newObjectiveCalls=0, trainingTransitions=0,
        independentTeamReplication=False, nativeExecutionVerified=False, authorizationVerified=False,
        scope='Supplied-record consistency and recorded hosted execution only; no independent native provenance or admission verified.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--measurement-root', required=True)
    parser.add_argument('--trusted-reservation-sha256', required=True)
    args = parser.parse_args(argv)
    result = read_measurement(args.source_root, args.measurement_root,
                              trusted_reservation_sha256=args.trusted_reservation_sha256)
    print(canonical(result))


if __name__ == '__main__':
    main()
