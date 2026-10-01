"""Verify immutable public statistical readers without a research checkout."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import urllib.request
import zipfile

BASE = 'https://github.com/redeem123/PWA-component-evidence/releases/download/'
WHEEL_SHA = 'fd83c01228a688733f1ded5201c678f0c53ecc1006ffbc404db9f7a899ac6249'
COMMON = {'MANIFEST.json','README.md','outcomes.json','reader.py','verification.json'}
PACKAGES = (
    dict(name='component',asset='component-evidence-v2/portable_component_evidence_v2.zip',
         sha256='33e18188ff0da8937125cdfa07153f213214873edf4f3c6a6f7d12fca0a99754',
         members=COMMON|{'legacy_core.py'},
         counts={'sameState.csv':28,'pulse.csv':36,'scalePrimary.csv':20,'scaleSecondary.csv':24,'cleanSac.csv':28},
         expected=dict(status='passed',contrasts=136,bootstrapIntervals=136,crossedIntervals=14,
                       sameStateSnapshots=6960,cleanSacSnapshots=10440,holmFamilies=5,tableRowsExported=136)),
    dict(name='terminal',asset='terminal-credit-evidence-v1/portable_terminal_credit_evidence_v1.zip',
         sha256='6948c591c791a570f1abe399537f90ed736556880083860a5942d7dee893e983',members=COMMON,
         counts={'terminal_labels.csv':1160,'terminal_label_strata.csv':10,'terminal_screen_categories.csv':4,
                 'terminal_screen_strata.csv':10,'terminal_screen_functions.csv':29,'terminal_screen_overall.csv':1,
                 'held_out_decisions.csv':290,'terminal_credit_concordance.csv':9},
         expected=dict(status='passed',labels=1160,menus=290,heldOutCategories=4,failedCategoryChecks=4)),
    dict(name='native',asset='native-search-evidence-v1/portable_native_search_evidence_v1.zip',
         sha256='ee3349ee41904a5444dc4c42a75fa0752ccc0b3e8b6f461f4e017f90736ae823',members=COMMON,
         counts={'endpoints.csv':5220,'comparisons.csv':6,'functions.csv':174,'actors.csv':18,'aggregate_curves.csv':2400},
         expected=dict(schema='pwa-native-search-verification-v1',endpoints=5220,pairedTriplets=1740,
                       contrasts=6,holmFamilies=2,functionEffects=174,actorStrata=18,
                       aggregateCurves=6,aggregateCurvePoints=2400,publicRawTrajectoryReplay=False)),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unpack(data, spec, directory):
    require(len(data) <= 8*1024*1024 and digest(data) == spec['sha256'], 'Archive identity or size differs')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        require(len(entries) == len(spec['members']) and {e.filename for e in entries} == spec['members'],
                'Archive member grid differs')
        require(sum(e.file_size for e in entries) <= 64*1024*1024 and
                all(not e.is_dir() and stat.S_IFMT(e.external_attr >> 16) in (0,stat.S_IFREG) for e in entries),
                'Unsafe archive member or expanded size')
        require(not directory.exists() and directory.resolve() == directory, 'Fresh unredirected package required')
        directory.mkdir()
        for entry in entries:
            with (directory/entry.filename).open('xb') as stream:
                stream.write(archive.read(entry))


def exact_fields(summary, expected):
    require(type(summary) is dict, 'Reader must return a JSON object')
    for name,value in expected.items():
        require(name in summary and type(summary[name]) is type(value) and summary[name] == value,
                'Reader summary differs at '+name)
    for name in ('objectiveCalls','trainingTransitions'):
        require(type(summary.get(name)) is int and summary[name] == 0, 'Reader executed scientific work')
    require(summary.get('independentScientificReplication') is False, 'Scientific replication claim differs')
    if 'scientificFits' in summary:
        require(type(summary['scientificFits']) is int and summary['scientificFits'] == 0, 'Unexpected fitting')


def verify(spec, directory, python):
    require(not directory.exists() and directory.resolve() == directory, 'Fresh unredirected output required')
    with urllib.request.urlopen(BASE+spec['asset'],timeout=60) as response:
        data = response.read(8*1024*1024+1)
    require(digest(data) == spec['sha256'] and len(data) <= 8*1024*1024, 'Public archive identity differs')
    directory.mkdir()
    (directory/'download.zip').write_bytes(data)
    unpack(data,spec,directory/'package')
    command = [str(python),'-I',str(directory/'package/reader.py'),'--tables-output',str(directory/'tables')]
    result = subprocess.run(command,cwd=directory,capture_output=True,text=True,timeout=240)
    (directory/'reader_stdout.txt').write_text(result.stdout)
    (directory/'reader_stderr.txt').write_text(result.stderr)
    require(result.returncode == 0, 'Reader failed with exit '+str(result.returncode)+': '+result.stderr)
    summary = json.loads(result.stdout)
    exact_fields(summary,spec['expected'])
    paths = list((directory/'tables').iterdir())
    require({p.name for p in paths} == set(spec['counts']) and
            all(stat.S_ISREG(p.lstat().st_mode) for p in paths), 'CSV membership differs')
    tables = {}
    for path in sorted(paths):
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames), 'CSV header differs')
            records = list(reader)
        require(len(records) == spec['counts'][path.name] and all(None not in row and None not in row.values() for row in records),
                'CSV row membership differs')
        tables[path.name] = dict(rows=len(records),sha256=digest(path.read_bytes()))
    return dict(url=BASE+spec['asset'],archiveSha256=digest(data),archiveBytes=len(data),
                authenticatedRequest=False,readerExitCode=0,readerSummary=summary,tables=tables)


def main(output):
    import numpy as np
    require(sys.version_info[:2] == (3,12) and np.__version__ == '2.2.6' and sys.prefix != sys.base_prefix,
            'Fresh recorded Python 3.12 / NumPy 2.2.6 environment required')
    require(os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('RUNNER_ENVIRONMENT') == 'github-hosted'
            and os.environ.get('GITHUB_REPOSITORY') == 'redeem123/PWA-component-evidence'
            and platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Expected public hosted Linux job required')
    require(not output.exists() and output.resolve() == output, 'Fresh unredirected receipt output required')
    output.mkdir(parents=True)
    results = {spec['name']:verify(spec,output/spec['name'],Path(sys.executable)) for spec in PACKAGES}
    receipt = dict(schema='pwa-public-hosted-reader-verification-v1',status='passed',
        verificationHost='GitHub-hosted Ubuntu 24.04 x86_64',authorHost=False,freshEnvironment=True,
        python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        numpyWheelSha256=WHEEL_SHA,helperSha256=digest(Path(__file__).read_bytes()),
        publicCommit=os.environ['GITHUB_SHA'],runId=os.environ['GITHUB_RUN_ID'],
        runAttempt=os.environ['GITHUB_RUN_ATTEMPT'],
        runUrl='https://github.com/redeem123/PWA-component-evidence/actions/runs/'+os.environ['GITHUB_RUN_ID'],
        packages=results,csvTables=sum(len(v['tables']) for v in results.values()),
        authenticatedDownloads=False,privateCheckout=False,benchmarkRuntime=False,
        objectiveCalls=0,trainingTransitions=0,scientificFits=0,independentScientificReplication=False)
    (output/'verification.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('PWA_PUBLIC_READER_RECEIPT '+json.dumps(receipt,sort_keys=True),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    main(args.output.absolute())
