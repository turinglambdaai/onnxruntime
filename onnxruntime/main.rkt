#lang racket/base

;; onnxruntime: Racket FFI bindings to the ONNX Runtime C API.
;;
;; The runtime leg of "train in Python once, export .onnx, infer anywhere".
;; v0.1 scope: float32 tensors, CPU execution provider, models from file or
;; bytes.

(require racket/contract
         racket/flonum
         "private/session.rkt"
         (only-in "private/ffi.rkt"
                  [ort-runtime-version onnx-version/raw]
                  [negotiated-api-version negotiated-api-version/raw]))

(provide
 (contract-out
  ;; smoke API: ONNX Runtime library version, e.g. "1.31.0"
  (onnx-version (-> string?))
  ;; the OrtApi table version negotiated with the loaded runtime
  ;; (<= 31; equals the header version when the runtime is at least as new)
  (onnx-api-version (-> exact-nonneg-integer?))

  ;; float32 tensor value: shape product must match the flvector length
  (struct ort-tensor ((shape (listof exact-nonneg-integer?))
                      (data flvector?)))
  (make-ort-tensor (-> (listof exact-nonneg-integer?) flvector? ort-tensor?))

  ;; opaque handle holding the native session, env and negotiated version
  (onnx-session? (-> any/c boolean?))

  ;; src is either a path to a .onnx file or the model bytes themselves;
  ;; #:optimization-level maps to ORT's graph optimization levels
  (open-onnx-session
   (->* ((or/c path-string? bytes?))
        (#:optimization-level (or/c 'disable 'basic 'extended 'all))
        onnx-session?))

  (session-input-names (-> onnx-session? (listof string?)))
  (session-output-names (-> onnx-session? (listof string?)))

  ;; model metadata shape: symbolic dims come back as 'dynamic
  (session-input-shape
   (-> onnx-session? string?
       (listof (or/c exact-positive-integer? 'dynamic))))
  (session-output-shape
   (-> onnx-session? string?
       (listof (or/c exact-positive-integer? 'dynamic))))

  ;; run: inputs must cover every model input; outputs come back in model
  ;; order as (name . ort-tensor) pairs
  (onnx-run
   (-> onnx-session?
       (listof (cons/c string? ort-tensor?))
       (listof (cons/c string? ort-tensor?))))

  ;; look up one output tensor by name in an onnx-run result
  (results-ref
   (-> (listof (cons/c string? ort-tensor?)) string? ort-tensor?))))

;; (onnx-version) -> string?
(define (onnx-version) onnx-version/raw)

;; (onnx-api-version) -> exact-nonneg-integer?
(define (onnx-api-version) negotiated-api-version/raw)
