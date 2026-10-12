#lang info

(define collection 'multi)

(define version "0.1.0")

(define deps '("base"))
(define build-deps '("scribble-lib" "racket-doc" "rackunit-lib"))

(define pkg-desc "FFI bindings to the ONNX Runtime C API for CPU inference of ONNX models")
(define pkg-authors '("turinglambdaai"))

(define license 'MIT)

(define scribblings '(("onnxruntime/scribblings/onnxruntime.scrbl" ())))
