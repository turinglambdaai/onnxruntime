#lang racket/base

;; Session management and the public value types for onnxruntime.
;; (Contracts are applied in onnxruntime/main.rkt.)

(require racket/list
         racket/flonum
         "ffi.rkt")

(provide (all-defined-out))

;; Racket 9.3 does not ship `natural?` or `exact-nonneg-integer?`; define
;; the latter locally so it can be shared with the public contracts.
(define (exact-nonneg-integer? v)
  (and (integer? v) (exact? v) (not (negative? v))))

;; ------------------------------------------------------------- value types

;; shape: (listof exact-nonneg-integer?), data: flvector of float32 values
;; (v0.1: float32 only). shape product must equal the flvector length.
(struct ort-tensor (shape data) #:transparent)

(define (make-ort-tensor shape data)
  (unless (and (list? shape) (andmap exact-nonneg-integer? shape))
    (raise-argument-error 'make-ort-tensor
                          "(listof exact-nonneg-integer?)" shape))
  (unless (flvector? data)
    (raise-argument-error 'make-ort-tensor "flvector?" data))
  (define expected (for/product ([d (in-list shape)]) d))
  (unless (= expected (flvector-length data))
    (raise-argument-error
     'make-ort-tensor
     (format "flvector of length ~a (shape ~a)" expected shape)
     (flvector-length data)))
  (ort-tensor shape data))

;; Opaque session handle. env/session are native objects owned here; the
;; default allocator is a process-global owned by ORT. Sessions live until
;; the Racket process exits (v0.1 has no close API).
(struct onnx-session (api-version
                      env
                      session
                      allocator
                      input-names
                      output-names))

;; ----------------------------------------------------------- open a session

(define (open-onnx-session src #:optimization-level [level 'all])
  (unless (memq level '(disable basic extended all))
    (raise-argument-error 'open-onnx-session
                          "(or/c 'disable 'basic 'extended 'all)" level))
  (unless (or (path-string? src) (bytes? src))
    (raise-argument-error 'open-onnx-session
                          "(or/c path-string? bytes?)" src))
  (define env (ort-create-env ort-logging-fatal "onnxruntime-racket"))
  (define session
    (let ([options #f] [sess #f])
      (dynamic-wind
        void
        (lambda ()
          (set! options (ort-create-session-options))
          (ort-set-session-graph-optimization-level
           options (optimization-level-code level))
          (set! sess
                (if (bytes? src)
                    (ort-create-session-from-array
                     env src (bytes-length src) options)
                    (ort-create-session env src options))))
        (lambda ()
          ;; options may be released as soon as the session exists; if
          ;; creation failed, give the env back too
          (when options (ort-release-session-options options))
          (unless sess (ort-release-env env))))
      sess))
  (define allocator (ort-get-allocator-with-default-options))
  (define input-names
    (query-session-names session allocator
                         ort-session-get-input-count
                         ort-session-get-input-name))
  (define output-names
    (query-session-names session allocator
                         ort-session-get-output-count
                         ort-session-get-output-name))
  (onnx-session negotiated-api-version env session allocator
                input-names output-names))

;; Query input/output names through the session's default allocator; each
;; C string is copied and returned to the allocator.
(define (query-session-names session allocator count-fn name-fn)
  (define n (count-fn session))
  (for/list ([i (in-range n)])
    (take-allocator-string allocator (name-fn session i allocator))))

;; ------------------------------------------------------------------ shapes

(define (session-input-names s) (onnx-session-input-names s))
(define (session-output-names s) (onnx-session-output-names s))

(define (session-input-shape s name)
  (session-shape s name (session-input-names s)
                 ort-session-get-input-type-info))

(define (session-output-shape s name)
  (session-shape s name (session-output-names s)
                 ort-session-get-output-type-info))

;; Model-metadata shape: concrete dims as exact integers, symbolic dims as
;; 'dynamic. The OrtTypeInfo is released after reading; the tensor info
;; view of it is borrowed.
(define (session-shape s name known-names get-type-info)
  (define idx (index-of known-names name))
  (unless idx
    (raise (exn:fail (format "onnxruntime: no session input/output named ~s (have: ~a)"
                             name known-names)
                     (current-continuation-marks))))
  (define type-info (get-type-info (onnx-session-session s) idx))
  (dynamic-wind
    void
    (lambda ()
      (tensor-info->shape (ort-cast-type-info-to-tensor-info type-info)))
    (lambda ()
      (ort-release-type-info type-info))))

;; -------------------------------------------------------------------- run

;; Look up the tensor bound to name in an onnx-run result.
(define (results-ref results name)
  (define hit (assoc name results))
  (unless hit
    (raise (exn:fail (format "onnxruntime: no output named ~s in result" name)
                     (current-continuation-marks))))
  (cdr hit))

;; inputs: (listof (cons/c string? ort-tensor?)) -- every model input must
;; be provided on each run. Returns outputs in model order.
(define (onnx-run s inputs)
  (unless (and (list? inputs)
               (andmap (lambda (e) (and (pair? e) (string? (car e))
                                        (ort-tensor? (cdr e))))
                       inputs))
    (raise-argument-error 'onnx-run
                          "(listof (cons/c string? ort-tensor?))" inputs))
  (define in-names (map car inputs))
  (define known-inputs (session-input-names s))
  (for ([n (in-list in-names)])
    (unless (member n known-inputs)
      (raise (exn:fail
              (format "onnxruntime: unknown input name ~s (session inputs: ~a)"
                      n known-inputs)
              (current-continuation-marks)))))
  (define dup (check-duplicates in-names string=?))
  (when dup
    (raise (exn:fail (format "onnxruntime: duplicate input name ~s" dup)
                     (current-continuation-marks))))
  (when (null? in-names)
    (raise (exn:fail "onnxruntime: no inputs given" (current-continuation-marks))))

  (define session* (onnx-session-session s))
  (define out-names (session-output-names s))

  ;; fresh CPU memory info per run, released at the end
  (define mem-info (ort-create-cpu-memory-info 1 0)) ; arena allocator, default mem type

  ;; input tensor data: byte strings that ORT wraps (not copies), so they
  ;; must stay referenced until Run has finished and the values are released
  (define keep-alive '())
  (define values-to-release '())
  (define (keep! b) (set! keep-alive (cons b keep-alive)) b)

  (define input-values
    (for/list ([e (in-list inputs)])
      (define t (cdr e))
      (define shape (ort-tensor-shape t))
      (define data-bytes (keep! (flvector->f32-bytes (ort-tensor-data t))))
      (define value
        (ort-create-tensor-with-data mem-info
                                     data-bytes (bytes-length data-bytes)
                                     shape (length shape)
                                     onnx-tensor-element-float))
      (set! values-to-release (cons value values-to-release))
      value))

  ;; zeroed => every entry starts as NULL: ORT allocates fresh OrtValues
  (define outs-buf (make-bytes (* 8 (length out-names))))

  (dynamic-wind
    void
    (lambda ()
      (ort-run session* #f
               in-names input-values (length in-names)
               out-names (length out-names)
               outs-buf)
      (define raw-outputs (ort-run-outputs outs-buf (length out-names)))
      ;; read each output then hand the OrtValue straight back to ORT
      (for/list ([value (in-list raw-outputs)] [name (in-list out-names)])
        (define-values (shape floats) (ort-value->shape-and-floats value))
        (ort-release-value value)
        (cons name (ort-tensor shape floats))))
    (lambda ()
      (for ([v (in-list values-to-release)])
        (ort-release-value v))
      ;; drop the data byte strings only after the values are released
      (set! keep-alive '())
      (ort-release-memory-info mem-info))))
