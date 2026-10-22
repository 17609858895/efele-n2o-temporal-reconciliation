"""Check every published scientific/source file against the public manifest."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent

def main():
    entries=json.loads((ROOT/'provenance/public_manifest.json').read_text(encoding='utf-8'))
    failed=[]
    for rel,expected in entries.items():
        p=ROOT/rel
        actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        if actual != expected: failed.append(rel)
    if failed: raise RuntimeError('Manifest mismatch: '+', '.join(failed))
    print(f'PASS: {len(entries)} published files match SHA-256 manifest.')
    return len(entries)

if __name__=='__main__':main()
