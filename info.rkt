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
(define scribblings '(("onnxruntime/scribblings/onnxruntime.scrbl" ())))
;; "onnxruntime-tests.rkt" is loaded conditionally by "onnxruntime-test.rkt"
;; (native library probe); the scribblings module is not a test.
(define test-omit-paths '("onnxruntime/test/onnxruntime-tests.rkt"
                          "onnxruntime/scribblings"))
