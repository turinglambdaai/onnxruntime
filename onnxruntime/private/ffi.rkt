#lang racket/base

;; Low-level FFI layer for the ONNX Runtime C API.
;;
;; The ORT C API is a versioned function table: OrtGetApiBase() returns a
;; two-field OrtApiBase struct whose GetApi(uint32_t) member yields an
;; OrtApi* -- one large struct of function pointers in a fixed, append-only
;; order (see private/api-index.rkt). We fetch individual function pointers
;; by index and cast each one we need to a typed _fun prototype.
;;
;; ptr-ref offset semantics (verified experimentally against libonnxruntime
;; 1.31.0 on darwin-arm64): for a value of ctype T, offset k in
;; (ptr-ref p T k) means byte offset k * sizeof(T), i.e. k counts ELEMENTS
;; of T, not bytes.

(require racket/format
         racket/flonum
         ffi/unsafe
         ffi/unsafe/define
         "api-index.rkt")

(provide (all-defined-out))

;; ---------------------------------------------------------------- loading

;; Ordered load candidates: ONNXRUNTIME_LIB (file or directory) first, then
;; the plain soname (system search: DYLD_/LD_LIBRARY_PATH, ldconfig), then
;; well-known install locations so plain `racket` runs work even when
;; worker processes lose the environment variable.
(define (ort-library-candidates)
  (define env (getenv "ONNXRUNTIME_LIB"))
  (define explicit
    (cond [(not env) #f]
          [(directory-exists? env)
           (path->string
            (build-path env
                        (case (system-type 'os)
                          [(macosx) "libonnxruntime.dylib"]
                          [else "libonnxruntime.so"])))]
          [else env]))
  (append (if explicit (list explicit) '())
          (list "libonnxruntime")
          (case (system-type 'os)
            [(macosx)
             (list "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib"
                   "/usr/local/opt/onnxruntime/lib/libonnxruntime.dylib")]
            [else
             (list "/usr/lib/libonnxruntime.so"
                   "/usr/local/lib/libonnxruntime.so")])))

(define (load-ort-lib)
  (let loop ([candidates (ort-library-candidates)] [last-exn #f])
    (cond
      [(null? candidates)
       (raise
        (exn:fail
         (string-append
          "onnxruntime: cannot load the ONNX Runtime native library.\n"
          (if last-exn (format "last error: ~a\n" (exn-message last-exn)) "")
          "Install ONNX Runtime (`brew install onnxruntime` on macOS, or extract\n"
          "an official onnxruntime-linux-*.tgz release on Linux), then either set\n"
          "ONNXRUNTIME_LIB to the full library path, or add its directory to\n"
          "DYLD_FALLBACK_LIBRARY_PATH (macOS) / LD_LIBRARY_PATH (Linux).")
         (current-continuation-marks)))]
      [else
       (with-handlers ([exn:fail? (lambda (e) (loop (cdr candidates) e))])
         (ffi-lib (car candidates)))])))

(define ort-lib (load-ort-lib))

(define-ffi-definer define-ort ort-lib)

;; ------------------------------------------------------------- OrtApiBase

;; struct OrtApiBase { const OrtApi* (*GetApi)(uint32_t);
;;                     const char* (*GetVersionString)(void); }
(define-cstruct _ortapibase
  ([get-api _fpointer]
   [get-version-string _fpointer]))

(define-ort ort-get-api-base
  (_fun -> _ortapibase-pointer)
  #:c-id OrtGetApiBase)

(define ort-apibase (ort-get-api-base))

;; Runtime version string, e.g. "1.31.0". OrtApiBase::GetVersionString is
;; present in every ORT release, so it works regardless of negotiation.
(define ort-runtime-version
  ((function-ptr (ortapibase-get-version-string ort-apibase)
                 (_fun -> _string/utf-8))))

;; -------------------------------------------------- API version negotiation

;; Highest version the generated index knows about (ORT_API_VERSION 31).
(define ort-max-api-version 31)

;; Negotiation floor. Every function this package binds is unannotated in
;; the header, i.e. it predates the first "\since Version 1.11."
;; annotations, so API version 10 is guaranteed to contain them all.
(define ort-min-api-version 10)

;; Negotiated OrtApi version (set below). Requesting an unavailable version
;; makes GetApi return NULL; ORT logs one warning line per failed probe on
;; stderr, which is expected while negotiating downward on older runtimes.
(define negotiated-api-version #f)

(define ort-api
  (let ([get-api (function-ptr (ortapibase-get-api ort-apibase)
                               (_fun _uint32 -> _pointer))])
    (let loop ([v ort-max-api-version])
      (cond
        [(< v ort-min-api-version)
         (raise (exn:fail
                 (format "onnxruntime: no supported OrtApi version found (probed ~a down to ~a); \
runtime ~a is too old for this package"
                         ort-max-api-version ort-min-api-version ort-runtime-version)
                 (current-continuation-marks)))]
        [else
         (define api (get-api v))
         (cond
           [api (set! negotiated-api-version v) api]
           [else (loop (sub1 v))])]))))

;; Fetch a function pointer by index inside the OrtApi struct.
;; ptr-ref offsets count elements of _pointer (verified; see file header).
(define (ort-api-ptr name)
  (ptr-ref ort-api _pointer (ort-api-index name)))

;; Cast a table entry to a typed prototype.
(define (ort-fn name type)
  (function-ptr (ort-api-ptr name) type))

;; ---------------------------------------------------- status & raw bindings

;; OrtStatus* plumbing. ReleaseStatus is a void-returning class release;
;; error messages are borrowed (not freed).
(define ort-release-status (ort-fn 'ReleaseStatus (_fun _pointer -> _void)))
(define ort-get-error-code (ort-fn 'GetErrorCode (_fun _pointer -> _int)))
(define ort-get-error-message
  (ort-fn 'GetErrorMessage (_fun _pointer -> _string/utf-8)))

;; (check-status! status) -- raise exn:fail carrying the ORT error message
;; when status is non-null; release the status either way. Every
;; status-returning prototype below funnels through here.
(define (check-status! status)
  (when status
    (define message (ort-get-error-message status))
    (define code (ort-get-error-code status))
    (ort-release-status status)
    (raise (exn:fail (format "onnxruntime: ~a (code ~a)" message code)
                     (current-continuation-marks)))))

;; Idiom used for status-returning calls with one pointer out-param:
;;   (_fun args ... (out : (_ptr o _pointer)) -> (st : _pointer)
;;         (begin (check-status! st) out))

;; --- environment ---
(define ort-create-env
  (ort-fn 'CreateEnv
          (_fun _int _string/utf-8
                (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

;; --- session ---
(define ort-create-session
  (ort-fn 'CreateSession
          (_fun _pointer _path _pointer
                (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

;; The model byte string is copied by ORT during the call, so the FFI's
;; call-duration _bytes pointer is sufficient here.
(define ort-create-session-from-array
  (ort-fn 'CreateSessionFromArray
          (_fun _pointer _bytes _size _pointer
                (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

(define ort-create-session-options
  (ort-fn 'CreateSessionOptions
          (_fun (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

;; GraphOptimizationLevel: disable=0 basic=1 extended=2 layout=3 all=99.
(define ort-set-session-graph-optimization-level
  (ort-fn 'SetSessionGraphOptimizationLevel
          (_fun _pointer _int
                -> (st : _pointer) -> (check-status! st))))

;; --- memory info ---
(define ort-create-memory-info
  (ort-fn 'CreateMemoryInfo
          (_fun _string/utf-8 _int _int _int
                (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

(define ort-create-cpu-memory-info
  (ort-fn 'CreateCpuMemoryInfo
          (_fun _int _int
                (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

;; --- allocator (the default allocator is a global owned by ORT: never release it) ---
(define ort-get-allocator-with-default-options
  (ort-fn 'GetAllocatorWithDefaultOptions
          (_fun (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

(define ort-allocator-free
  (ort-fn 'AllocatorFree
          (_fun _pointer _pointer
                -> (st : _pointer) -> (check-status! st))))

;; Copy a null-terminated allocator-owned char* into a Racket string and
;; free the C copy.
(define (take-allocator-string allocator cstr)
  (begin0 (if cstr (cast cstr _pointer _string/utf-8) "")
    (when cstr (ort-allocator-free allocator cstr))))

;; --- session metadata ---
(define ort-session-get-input-count
  (ort-fn 'SessionGetInputCount
          (_fun _pointer (out : (_ptr o _size))
                -> (st : _pointer) -> (begin (check-status! st) out))))

(define ort-session-get-output-count
  (ort-fn 'SessionGetOutputCount
          (_fun _pointer (out : (_ptr o _size))
                -> (st : _pointer) -> (begin (check-status! st) out))))

(define ort-session-get-input-name
  (ort-fn 'SessionGetInputName
          (_fun _pointer _size _pointer
                (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

(define ort-session-get-output-name
  (ort-fn 'SessionGetOutputName
          (_fun _pointer _size _pointer
                (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

(define ort-session-get-input-type-info
  (ort-fn 'SessionGetInputTypeInfo
          (_fun _pointer _size
                (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

(define ort-session-get-output-type-info
  (ort-fn 'SessionGetOutputTypeInfo
          (_fun _pointer _size
                (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

;; The returned OrtTensorTypeAndShapeInfo is owned by the OrtTypeInfo and
;; must NOT be released separately (it stays valid until the type info is
;; released).
(define ort-cast-type-info-to-tensor-info
  (ort-fn 'CastTypeInfoToTensorInfo
          (_fun _pointer
                (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

;; --- tensor type & shape ---
(define ort-get-tensor-type-and-shape
  (ort-fn 'GetTensorTypeAndShape
          (_fun _pointer
                (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

(define ort-get-dimensions-count
  (ort-fn 'GetDimensionsCount
          (_fun _pointer (out : (_ptr o _size))
                -> (st : _pointer) -> (begin (check-status! st) out))))

;; GetDimensions(info, int64_t* values, size_t values_count): fills a
;; caller-provided byte buffer (8 bytes per dim, host byte order).
(define ort-get-dimensions-raw
  (ort-fn 'GetDimensions
          (_fun _pointer _bytes _size
                -> (st : _pointer) -> (check-status! st))))

(define ort-get-tensor-element-type
  (ort-fn 'GetTensorElementType
          (_fun _pointer (out : (_ptr o _int))
                -> (st : _pointer) -> (begin (check-status! st) out))))

;; --- tensor values ---
;; p_data is passed as a byte string: the FFI anchors it for the call, and
;; the OrtValue wraps (does not copy) the buffer, so the caller must keep
;; the bytes object referenced until the OrtValue is released. GC'd Racket
;; blocks never move, so a held reference is sufficient.
(define ort-create-tensor-with-data
  (ort-fn 'CreateTensorWithDataAsOrtValue
          (_fun _pointer _bytes _size
                (shape : (_list i _int64)) _size
                _int
                (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

(define ort-get-tensor-mutable-data
  (ort-fn 'GetTensorMutableData
          (_fun _pointer
                (out : (_ptr o _pointer))
                -> (st : _pointer) -> (begin (check-status! st) out))))

;; Run(session, run_options, input_names, inputs, input_len, output_names,
;; output_names_len, outputs). The FFI converts the Racket lists to C
;; arrays anchored for the call duration. `outputs` is a caller-provided
;; zeroed byte buffer (8 bytes = one NULL pointer per output); ORT fills
;; each slot with a fresh OrtValue* (caller must release each with
;; ort-release-value). Reading slots back is ort-run-outputs below.
(define ort-run
  (ort-fn 'Run
          (_fun _pointer _pointer
                (input-names : (_list i _string/utf-8))
                (inputs : (_list i _pointer))
                _size
                (output-names : (_list i _string/utf-8))
                _size
                _bytes                       ; OrtValue** outputs, all NULL
                -> (st : _pointer) -> (check-status! st))))

;; Read the OrtValue* outputs that ORT placed into a zeroed buffer.
(define (ort-run-outputs outs-buf n)
  (define arr (cast outs-buf _bytes _pointer))
  (for/list ([i (in-range n)])
    (ptr-ref arr _pointer i)))

;; --- class releases (all void, no status) ---
(define ort-release-env (ort-fn 'ReleaseEnv (_fun _pointer -> _void)))
(define ort-release-memory-info
  (ort-fn 'ReleaseMemoryInfo (_fun _pointer -> _void)))
(define ort-release-session (ort-fn 'ReleaseSession (_fun _pointer -> _void)))
(define ort-release-value (ort-fn 'ReleaseValue (_fun _pointer -> _void)))
(define ort-release-type-info
  (ort-fn 'ReleaseTypeInfo (_fun _pointer -> _void)))
(define ort-release-tensor-type-and-shape-info
  (ort-fn 'ReleaseTensorTypeAndShapeInfo
          (_fun _pointer -> _void)))
(define ort-release-session-options
  (ort-fn 'ReleaseSessionOptions (_fun _pointer -> _void)))

;; ------------------------------------------------------------ C constants

;; ONNXTensorElementDataType
(define onnx-tensor-element-float 1)

;; OrtLoggingLevel: verbose=0 info=1 warning=2 error=3 fatal=4
(define ort-logging-fatal 4)

;; GraphOptimizationLevel codes
(define (optimization-level-code level)
  (case level
    [(disable) 0]
    [(basic) 1]
    [(extended) 2]
    [(all) 99]))

;; ------------------------------------------------- small C buffer helpers

;; All C buffers here are Racket byte strings: the FFI anchors bytes
;; arguments for the duration of each call, buffers that must outlive a
;; call (tensor data) are kept referenced by the caller until the owning
;; OrtValue is released, and nothing is ever explicitly freed (Racket FFI
;; malloc'd blocks must not be passed to C free -- freeing them aborts).

;; Encode an flvector as a float32 byte string in host byte order.
(define (flvector->f32-bytes v)
  (define n (flvector-length v))
  (define b (make-bytes (* 4 n)))
  (for ([i (in-range n)])
    (real->floating-point-bytes (flvector-ref v i) 4 (system-big-endian?) b (* 4 i)))
  b)

;; Read dims via GetDimensions into a byte buffer; negative (symbolic)
;; dimensions become 'dynamic, concrete dims stay exact integers.
(define (tensor-info->shape info)
  (define n (ort-get-dimensions-count info))
  (cond
    [(zero? n) '()]
    [else
     (define buf (make-bytes (* 8 n)))
     (ort-get-dimensions-raw info buf n)
     (for/list ([i (in-range n)])
       (define d (integer-bytes->integer buf #t (system-big-endian?) (* 8 i) (+ 8 (* 8 i))))
       (if (>= d 0) d 'dynamic))]))

;; ------------------------------------------------------------ shape query

(define (element-type-name code)
  (case code
    [(1) "float32"]
    [(10) "float64"]
    [(7) "int64"]
    [(6) "int32"]
    [else (~a "element-type-" code)]))

;; (values shape flvector) from a float32 OrtValue*; releases the temporary
;; tensor-type-and-shape info. Raises if the value is not float32.
(define (ort-value->shape-and-floats value)
  (define info (ort-get-tensor-type-and-shape value))
  (dynamic-wind
    void
    (lambda ()
      (define etype (ort-get-tensor-element-type info))
      (unless (= etype onnx-tensor-element-float)
        (raise (exn:fail
                 (format "onnxruntime: only float32 tensors are supported in v0.1, got ~a"
                         (element-type-name etype))
                 (current-continuation-marks))))
      (define shape (tensor-info->shape info))
      (define count (for/product ([d (in-list shape)]) d))
      (define data-ptr (ort-get-tensor-mutable-data value))
      (values shape
              (for/flvector #:length count ([i (in-range count)])
                (ptr-ref data-ptr _float i))))
    (lambda ()
      (ort-release-tensor-type-and-shape-info info))))
