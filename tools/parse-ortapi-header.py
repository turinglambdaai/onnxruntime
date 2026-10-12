#!/usr/bin/env python3
"""Emit the Racket OrtApi name->index module from onnxruntime_c_api.h.

Usage:
    python3 tools/parse-ortapi-header.py /path/to/onnxruntime_c_api.h \
        > onnxruntime/private/api-index.rkt

Parses `struct OrtApi { ... }` from the C API header and records the
ordered function names (the table is append-only across versions) plus
each function's introduction version from the "\\since Version X.Y."
doc annotations (annotations start at 1.11; unannotated entries predate
it). Declaration forms handled: ORT_API2_STATUS(Name, ...),
ORT_API2(Name, ...), Ret(ORT_API_CALL* Name)(...), ORT_CLASS_RELEASE(X).
Python 3 stdlib only; build-time tool, not a package dependency.
"""
import json
import re
import sys
from collections import Counter


def main():
    header = sys.argv[1] if len(sys.argv) > 1 else \
        '/opt/homebrew/include/onnxruntime/onnxruntime_c_api.h'
    if len(sys.argv) > 2:
        print('usage: parse-ortapi-header.py [header-path] (prints Racket to stdout)',
              file=sys.stderr)
        sys.exit(2)

    lines = open(header).read().splitlines()

    start = next((i for i, l in enumerate(lines)
                  if re.match(r'^struct OrtApi \{', l)), None)
    if start is None:
        sys.exit('struct OrtApi not found')
    end = next((j for j in range(start + 1, len(lines)) if lines[j] == '};'), None)
    if end is None:
        sys.exit('closing brace of struct OrtApi not found')

    pat_status = re.compile(r'\bORT_API2_STATUS\(\s*(\w+)\s*,')
    pat_plain = re.compile(r'\bORT_API2\(\s*(\w+)\s*,')
    pat_direct = re.compile(r'\(ORT_API_CALL\s*\*\s*(\w+)\)')
    pat_release = re.compile(r'\bORT_CLASS_RELEASE\((\w+)\)')
    pat_since = re.compile(r'\\since Version (\d+)\.(\d+)')

    entries = []
    pending_since = None
    pending_seen = False
    for l in lines[start + 1:end]:
        m = pat_since.search(l)
        if m:
            if not pending_seen:
                pending_since = (int(m.group(1)), int(m.group(2)))
                pending_seen = True
            continue
        for pat, kind in ((pat_status, 'status'), (pat_plain, 'plain'),
                          (pat_release, 'release'), (pat_direct, 'direct')):
            m = pat.search(l)
            if m:
                name = m.group(1)
                if kind == 'release':
                    name = 'Release' + name
                entries.append({'name': name, 'kind': kind,
                                'since': pending_since, 'index': len(entries)})
                pending_since = None
                pending_seen = False
                break

    names = [e['name'] for e in entries]
    dupes = [n for n, c in Counter(names).items() if c > 1]
    if dupes:
        sys.exit('duplicate entries: %s' % dupes)

    # -- report what we saw on stderr so redirection stays clean
    print('parsed %d OrtApi entries from %s' % (len(entries), header),
          file=sys.stderr)
    print('since-version histogram: %s' % dict(sorted(
        Counter(str(e['since']) for e in entries).items())), file=sys.stderr)

    # -- emit Racket
    out = []
    w = out.append
    w('#lang racket/base')
    w('')
    w(';; GENERATED FILE -- do not edit by hand.')
    w(';; Generated from onnxruntime_c_api.h by tools/parse-ortapi-header.py.')
    w(';; Source header: %s' % header)
    w(';; (onnxruntime 1.31.0, ORT_API_VERSION 31, shipped by the Homebrew')
    w(';;  onnxruntime formula at generation time).')
    w(';;')
    w(';; The OrtApi function table is append-only across versions: the function')
    w(';; at index i in API version 31 sits at the same index in every older API')
    w(';; version that has it at all. This module records the name -> index map')
    w(';; for all %d entries of struct OrtApi, plus per-function introduction' % len(entries))
    w(';; versions as annotated with "\\since Version X.Y." in the header')
    w(';; (annotations start at 1.11; unannotated entries therefore predate 1.11).')
    w('')
    w('(provide ort-api-count')
    w('         ort-api-index)')
    w('')
    w(';; Number of function pointers in OrtApi as of header version 31.')
    w('(define ort-api-count %d)' % len(entries))
    w('')
    w(';; An association list symbol -> 0-based function index.')
    w('(define ort-api-table')
    for i in range(0, len(entries), 3):
        chunk = entries[i:i + 3]
        line = ' '.join('(%s . %d)' % (e['name'], e['index']) for e in chunk)
        w(("  '( " if i == 0 else '    ') + line)
    w('))')
    w('')
    w(';; (ort-api-index symbol) -> exact-nonneg-integer?')
    w(';; Index of the named function inside struct OrtApi.')
    w('(define (ort-api-index name)')
    w('  (cond')
    w('    [(assq name ort-api-table) => cdr]')
    w("    [else (raise-argument-error 'ort-api-index \"known OrtApi function name\" name)]))")
    w('')
    sys.stdout.write('\n'.join(out))


if __name__ == '__main__':
    main()
