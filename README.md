# onnxruntime

Racket FFI bindings to the [ONNX Runtime](https://onnxruntime.ai/) C API
for CPU inference of ONNX models — the runtime leg of
*train in Python once, export `.onnx`, infer anywhere*.

**Version 0.1 scope:** float32 tensors, CPU execution provider, models
loaded from a file path or from bytes.

## Install

```sh
raco pkg install onnxruntime
```

You also need the native ONNX Runtime library where the loader can find it:

- **macOS (Homebrew):** `brew install onnxruntime`. The loader probes
  `/opt/homebrew/opt/onnxruntime/lib` (Apple silicon) and
  `/usr/local/opt/onnxruntime/lib` (Intel) automatically; you can also
  export `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib`.
- **Linux:** extract an official release tarball and put its `lib/`
  directory on `LD_LIBRARY_PATH`, or install `libonnxruntime.so` into a
  standard location.
- **Explicit override:** set `ONNXRUNTIME_LIB` to the library file (or a
  directory containing it); it is tried first.

## Version negotiation

The ORT C API is a versioned function table. This package asks for API
version 31 (from the onnxruntime 1.31.0 header its function index was
generated from) and walks down to 10 until the loaded runtime answers.
`(onnx-api-version)` reports the negotiated version; older runtimes are
supported down to API version 10 (onnxruntime 1.10.x). While probing
downward, ORT prints one warning line per unavailable version to stderr.

## Quick start

```racket
(require onnxruntime racket/flonum)

(onnx-version)                              ; "1.31.0"
(define s (open-onnx-session "add.onnx"))
(session-input-names s)                     ; '("X")
(session-input-shape s "X")                 ; '(2 2)

(define x (make-ort-tensor '(2 2) (flvector 1.0 1.0 1.0 1.0)))
(results-ref (onnx-run s (list (cons "X" x))) "Z")
;; (ort-tensor '(2 2) (flvector 2.0 3.0 4.0 5.0))
```

See `onnxruntime/scribblings/onnxruntime.scrbl` (or the docs installed
with the package) for the full reference.

## Development

- Run tests (macOS):
  `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib raco test -p onnxruntime`
- Build docs:
  `raco scribble --dest /tmp/onnxruntime-docs onnxruntime/scribblings/onnxruntime.scrbl`
- Regenerate the API index after a header upgrade:
  `python3 tools/parse-ortapi-header.py /path/to/onnxruntime_c_api.h \
     > onnxruntime/private/api-index.rkt`
  (Python 3, stdlib only — a build-time tool, not a package dependency.)
- Regenerate the tiny test models in `test/data/` (optional; they are
  committed) with the Python `onnx` package:
  `python3 tools/gen-fixtures.py test/data`

## License

MIT — see [LICENSE](LICENSE). ONNX Runtime itself is separately licensed
by Microsoft under the MIT license.
