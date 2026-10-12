#lang racket/base

;; Tests for the onnxruntime package. These exercise the real native
;; library: on macOS run with
;;   DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib raco test -p onnxruntime
;; (the loader also probes /opt/homebrew/opt/onnxruntime/lib as a fallback).

(require rackunit
         racket/file
         racket/flonum
         racket/runtime-path
         racket/string
         onnxruntime)

(define-runtime-path fixture-dir "../../test/data")
(define (fixture name) (build-path fixture-dir name))

;; racket/flonum in Racket 9.3 has no flvector->list; build via in-flvector
(define (flvector->list* v) (for/list ([x (in-flvector v)]) x))

;; expected fixture results (exact in float32)
(define add-input (make-ort-tensor '(2 2) (flvector 1.0 1.0 1.0 1.0)))
(define add-expected '(2.0 3.0 4.0 5.0))

(define gemm-input (make-ort-tensor '(1 3) (flvector 1.0 2.0 3.0)))
(define gemm-expected '(32.0 48.0))

(test-case "version and negotiation"
  (define v (onnx-version))
  (check-true (string? v))
  (check-true (string-prefix? v "1.") (format "version ~a should start with 1." v))
  (define api (onnx-api-version))
  (check-true (and (integer? api) (exact? api) (>= api 0)))
  ;; the runtime must negotiate a version within [floor, header version]
  (check-true (<= 10 api 31))
  ;; when the runtime is current (>= 1.31), negotiation must pick 31;
  ;; older runtimes legitimately negotiate lower (exercised on Linux CI)
  (when (string>=? v "1.31")
    (check-equal? api 31)))

(test-case "make-ort-tensor validation"
  (check-exn exn:fail? (lambda () (make-ort-tensor '(2 2) (flvector 1.0))))
  (check-exn exn:fail? (lambda () (make-ort-tensor '(2 -1) (flvector 1.0 2.0))))
  (check-exn exn:fail? (lambda () (make-ort-tensor '(2 2) #(1.0 2.0 3.0 4.0)))))

(test-case "add.onnx from path"
  (define s (open-onnx-session (fixture "add.onnx")))
  (check-equal? (session-input-names s) '("X"))
  (check-equal? (session-output-names s) '("Z"))
  (check-equal? (session-input-shape s "X") '(2 2))
  (check-equal? (session-output-shape s "Z") '(2 2))
  (define results (onnx-run s (list (cons "X" add-input))))
  (check-equal? (length results) 1)
  (check-equal? (car (car results)) "Z")
  (check-equal? (ort-tensor-shape (cdr (car results))) '(2 2))
  (check-equal? (flvector->list* (ort-tensor-data (cdr (car results))))
                add-expected))

(test-case "gemm.onnx from path"
  (define s (open-onnx-session (fixture "gemm.onnx")))
  (check-equal? (session-input-names s) '("A"))
  (check-equal? (session-output-names s) '("Y"))
  (check-equal? (session-input-shape s "A") '(1 3))
  (check-equal? (session-output-shape s "Y") '(1 2))
  (define results (onnx-run s (list (cons "A" gemm-input))))
  (check-equal? (car (car results)) "Y")
  (check-equal? (ort-tensor-shape (cdr (car results))) '(1 2))
  (check-equal? (flvector->list* (ort-tensor-data (cdr (car results))))
                gemm-expected))

(test-case "session from bytes equals session from path"
  (define model (file->bytes (fixture "add.onnx")))
  (define from-path (open-onnx-session (fixture "add.onnx")))
  (define from-bytes (open-onnx-session model))
  (check-equal? (session-input-names from-bytes) (session-input-names from-path))
  (define rp (onnx-run from-path (list (cons "X" add-input))))
  (define rb (onnx-run from-bytes (list (cons "X" add-input))))
  (check-equal? (flvector->list* (ort-tensor-data (cdr (car rb))))
                (flvector->list* (ort-tensor-data (cdr (car rp))))))

(test-case "errors surface as exn:fail with ORT messages"
  ;; nonexistent file
  (check-exn exn:fail?
             (lambda () (open-onnx-session (build-path fixture-dir "nope.onnx"))))
  ;; garbage bytes
  (check-exn exn:fail?
             (lambda ()
               (open-onnx-session #"this is definitely not a protobuf model")))
  ;; unknown input name
  (define s (open-onnx-session (fixture "add.onnx")))
  (check-exn exn:fail?
             (lambda () (onnx-run s (list (cons "NOT-X" add-input)))))
  ;; unknown shape-query name
  (check-exn exn:fail? (lambda () (session-input-shape s "NOT-X")))
  ;; missing inputs entirely
  (check-exn exn:fail? (lambda () (onnx-run s '()))))

(test-case "symbolic dimensions are reported as 'dynamic"
  (define s (open-onnx-session (fixture "add-dynamic.onnx")))
  (check-equal? (session-input-shape s "X") '(dynamic 2))
  (check-equal? (session-output-shape s "Z") '(dynamic 2))
  ;; the model still runs: [1,2] input broadcasts against the [1,2] initializer
  (define r1 (onnx-run s (list (cons "X" (make-ort-tensor '(1 2) (flvector 0.5 0.5))))))
  (check-equal? (ort-tensor-shape (cdr (car r1))) '(1 2))
  (check-equal? (flvector->list* (ort-tensor-data (cdr (car r1))))
                '(1.5 2.5))
  ;; and a [2,2] input works too: N is bound per-run
  (define r2
    (onnx-run s (list (cons "X" (make-ort-tensor '(2 2)
                                                 (flvector 0.5 0.5 0.5 0.5))))))
  (check-equal? (ort-tensor-shape (cdr (car r2))) '(2 2))
  (check-equal? (flvector->list* (ort-tensor-data (cdr (car r2))))
                '(1.5 2.5 1.5 2.5)))

(test-case "optimization-level keyword is accepted"
  (define s (open-onnx-session (fixture "add.onnx") #:optimization-level 'basic))
  (define r (onnx-run s (list (cons "X" add-input))))
  (check-equal? (flvector->list* (ort-tensor-data (cdr (car r)))) add-expected))

(test-case "repeated runs are stable"
  (define s (open-onnx-session (fixture "add.onnx")))
  (for ([_ (in-range 5)])
    (define r (onnx-run s (list (cons "X" add-input))))
    (check-equal? (flvector->list* (ort-tensor-data (cdr (car r)))) add-expected)))
