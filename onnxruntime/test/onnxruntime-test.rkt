#lang racket/base

;; Test entry point.  The real suite lives in "onnxruntime-tests.rkt" and
;; is loaded only when the ONNX Runtime native library is available:
;; machines without it (e.g. the package catalog's build farm) skip
;; cleanly instead of reporting failures.

(define lib-available?
  (with-handlers ([exn:fail? (λ (_) #f)])
    (dynamic-require 'onnxruntime/main #f)
    #t))

(unless lib-available?
  (printf "onnxruntime: ONNX Runtime native library not available; skipping tests\n"))

(when lib-available?
  (dynamic-require "onnxruntime-tests.rkt" #f))
