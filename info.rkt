#lang info

(define collection 'multi)
(define version "0.1")
;; rackunit-lib is a run-time dependency: the package ships its test suite
;; and the catalog's dependency check scans shipped modules in run mode.
(define deps '("base" "rackunit-lib"))
(define build-deps '("scribble-lib" "racket-doc" "rackunit-lib"))
(define pkg-desc "FFI bindings to the ONNX Runtime C API for CPU inference of ONNX models")
(define pkg-authors '("turinglambdaai"))
(define license 'MIT)
;; The catalog build farm runs `raco test --package`, executing every module
;; in the package as a test.  The modules that load the ONNX Runtime native
;; library at require time cannot run on machines without it (the farm), so
;; they are omitted here; the CI workflow and
;; `raco test onnxruntime/test/onnxruntime-test.rkt` exercise the full suite
;; through the probe entry point.
(define test-omit-paths
  '("onnxruntime/test/onnxruntime-tests.rkt" ; loaded conditionally by the probe
    "onnxruntime/scribblings"                ; documentation, not tests
    "onnxruntime/main.rkt"
    "onnxruntime/private/ffi.rkt"
    "onnxruntime/private/session.rkt"))
