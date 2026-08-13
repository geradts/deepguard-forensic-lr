#!/usr/bin/env python3
"""Prepare a DF40 dataset workspace on Google Drive.

This script deliberately does not guess a third-party download URL. DF40 data are
large and distributed under the upstream project's terms. It creates the expected
Drive layout, checks free space, optionally verifies an existing manifest, and
provides a safe hand-off for an official download/archive supplied by the user.
"""
from pathlib import Path
import argparse, hashlib, shutil, json, datetime

EXPECTED_GB = 110

def sha256(path, chunk=8*1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--drive-root', default='/content/drive/MyDrive/DeepGuard')
    ap.add_argument('--archive', default=None, help='Officially obtained DF40 archive/file to stage')
    ap.add_argument('--expected-sha256', default=None)
    args=ap.parse_args()
    root=Path(args.drive_root); ds=root/'datasets'/'DF40'
    ds.mkdir(parents=True,exist_ok=True)
    usage=shutil.disk_usage(root.anchor if root.anchor else '/content/drive')
    report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset_dir':str(ds),'free_bytes':usage.free,'expected_working_gb':EXPECTED_GB}
    if usage.free < EXPECTED_GB*1024**3:
        raise RuntimeError(f'Insufficient free space: {usage.free/1024**3:.1f} GB available; reserve at least {EXPECTED_GB} GB.')
    if args.archive:
        src=Path(args.archive)
        if not src.exists(): raise FileNotFoundError(src)
        digest=sha256(src); report['archive']=str(src); report['sha256']=digest
        if args.expected_sha256 and digest.lower()!=args.expected_sha256.lower():
            raise RuntimeError('SHA-256 mismatch; refusing to stage archive.')
        print('Archive verified:', digest)
        print('Stage/extract it with the official DF40 instructions; no unverified URL is used here.')
    else:
        print('DF40 workspace ready:', ds)
        print('No download URL was guessed. Obtain DF40 through its official distribution and point --archive at it.')
    (root/'logs'/'df40_download_manifest.json').parent.mkdir(parents=True,exist_ok=True)
    (root/'logs'/'df40_download_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
