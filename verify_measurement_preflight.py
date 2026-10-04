"""Download the public archive anonymously and inspect or preflight it once.

The preflight child blocks COCO imports. This helper never admits native calls.
All logs and failed results remain in one new caller-selected physical folder.
"""
import argparse
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

URL='https://github.com/redeem123/PWA-component-evidence/releases/download/supplied-state-measurement-v1/PWA-supplied-state-measurement-v1.zip'
ARCHIVE_SHA='e430e700897a4e9cfe5fd9f4aebb9990312894455a48e44aa6e0f1777aa12f16'
ARCHIVE_BYTES=14756914
MANIFEST_SHA='8a74a20a41836e485d8b51346527b45d7f7def22ae18c7b0dc41e85f3e5e4988'
CHILD='''import importlib.abc,runpy,sys
class NoNative(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]=='cocoex':raise RuntimeError('Native import prohibited in objective-free preflight')
sys.meta_path.insert(0,NoNative())
entry=sys.argv[1];sys.argv=sys.argv[1:]
runpy.run_path(entry,run_name='__main__')
'''


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode()


def parsed(raw):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key');out[k]=v
        return out
    def invalid(item):raise ValueError('nonfinite JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)


def members(raw):
    require(len(raw)==ARCHIVE_BYTES and sha(raw)==ARCHIVE_SHA,'public archive digest/size differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos=archive.infolist();names=[p.filename for p in infos]
        require(len(names)==len(set(names)) and len(names)==245,'exact unique archive membership')
        require(sum(p.file_size for p in infos)<=32*1024**2,'bounded expanded archive')
        payload={}
        for p in infos:
            path=Path(p.filename)
            require(path.parts and not path.is_absolute() and '..' not in path.parts and str(path)==p.filename,
                    'unsafe archive name')
            require(not p.is_dir() and stat.S_IFMT(p.external_attr>>16) in (0,stat.S_IFREG),
                    'special/redirected archive entry')
            require(0<p.file_size<=32*1024**2,'bounded nonempty archive entry')
            payload[p.filename]=archive.read(p)
    manifest_raw=payload['independent-package.json']
    require(sha(manifest_raw)==MANIFEST_SHA,'trusted public manifest differs')
    manifest=parsed(manifest_raw)
    expected=manifest['filesSha256']
    require(set(payload)==set(expected)|{'independent-package.json'},'manifest/archive membership differs')
    for p,h in expected.items():require(sha(payload[p])==h,'public member digest differs: '+p)
    return payload


def write(path,raw):
    require(path.resolve()==path,'physical output file')
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())


def run(output,*,preflight=False):
    output=Path(output).absolute()
    require(output.resolve()==output and output.parent.is_dir() and not os.path.lexists(output),
            'new physical retrieval output required')
    output.mkdir(exist_ok=False)
    receipt=dict(schema='public-measurement-retrieval-v1',status='failed',url=URL,
        archiveSha256=ARCHIVE_SHA,packageManifestSha256=MANIFEST_SHA,
        authenticationUsed=False,newObjectiveCalls=0,nativeExecutionAuthorized=False,
        preflightRequested=preflight,python=platform.python_version(),system=platform.system(),
        machine=platform.machine(),independentTeamReplication=False)
    try:
        with urllib.request.urlopen(URL,timeout=120) as response:raw=response.read(ARCHIVE_BYTES+1)
        write(output/'download.zip',raw)
        payload=members(raw)
        package=output/'package';package.mkdir()
        for p,raw in payload.items():
            path=package/p;path.parent.mkdir(parents=True,exist_ok=True);write(path,raw)
        operation='preflight' if preflight else 'inspect'
        command=[sys.executable,'-I','-B','-c',CHILD,str(package/'replay.py'),operation]
        if preflight:command+=['--trusted-package-sha256',MANIFEST_SHA]
        env=dict(os.environ)
        for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONINSPECT','PYTHONOPTIMIZE'):
            env.pop(key,None)
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
                    'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT'):env[key]='1'
        with (output/'replay.stdout.log').open('xb') as stdout,(output/'replay.stderr.log').open('xb') as stderr:
            try:
                done=subprocess.run(command,cwd=package,env=env,stdin=subprocess.DEVNULL,
                    stdout=stdout,stderr=stderr,timeout=900)
            finally:
                stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
        receipt['childReturnCode']=done.returncode
        require(done.returncode==0,'public '+operation+' failed; retained stderr explains the gate')
        result=parsed((output/'replay.stdout.log').read_bytes())
        require(type(result.get('newObjectiveCalls')) is int and result['newObjectiveCalls']==0,
                'retrieval/preflight consumed objective calls')
        if preflight:
            require(result['schema']=='independent-measurement-preflight-v1'
                    and result['nativeExecutionAuthorized'] is False,'unexpected preflight result')
        else:
            require(result['packageManifestSha256']==MANIFEST_SHA and result['members']==244
                    and result['distributionStatus']=='authorized-public-release','public inspection differs')
        require({p:sha((package/p).read_bytes()) for p in payload}=={p:sha(raw) for p,raw in payload.items()},
                'public package changed during child')
        receipt.update(status='preflight-passed' if preflight else 'inspection-passed',
                       payloadMembers=244,publicAccessVerified=True)
    except BaseException as error:
        receipt.update(error=repr(error),failedOutputPreserved=True)
        raise
    finally:
        receipt['outputsSha256']={p.name:sha(p.read_bytes()) for p in output.iterdir() if p.is_file()}
        write(output/'receipt.json',encoded(receipt))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    print(encoded(run(args.output,preflight=args.preflight)).decode(),end='')
