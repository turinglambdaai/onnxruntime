#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     onnxruntime))

@title{onnxruntime: CPU inference of ONNX models in Racket}
@defmodule[onnxruntime]

This package provides Racket FFI bindings to the
@hyperlink["https://onnxruntime.ai/"]{ONNX Runtime} C API for CPU
inference of ONNX models. It is the runtime leg of the
``train in Python once, export @tt{.onnx}, infer anywhere'' workflow:
train with PyTorch/TensorFlow/scikit-learn, export the model, then load
and run it from Racket with a small, dependency-light API.

@bold{Version 0.1 scope.} float32 tensors only, CPU execution provider
only, models loaded from a file path or from bytes. Sequence, map, and
string tensors are out of scope; sessions cannot be closed explicitly
(native memory is reclaimed when the Racket process exits).

@section{Installation}

@itemlist[
 @item{@bold{Racket package:} @verbatim|{raco pkg install onnxruntime}|}

 @item{@bold{Native library:} the loader must find
  @tt{libonnxruntime} (@tt{.dylib} on macOS, @tt{.so} on Linux).

   On macOS with @hyperlink["https://brew.sh/"]{Homebrew}:

   @verbatim|{brew install onnxruntime}|

   The dylibs land in @tt{/opt/homebrew/opt/onnxruntime/lib/} (Apple
   silicon), which the loader probes automatically as a fallback. You
   can also export
   @tt{DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib}
   to make the system dynamic linker find it directly.

   On Linux, download and extract an official release tarball, e.g.

   @verbatim|{
ORT_VERSION=1.19.2
curl -fLO "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
tar -xzf onnxruntime-linux-x64-${ORT_VERSION}.tgz
export LD_LIBRARY_PATH="$PWD/onnxruntime-linux-x64-${ORT_VERSION}/lib:$LD_LIBRARY_PATH"
}|}

 @item{@bold{Explicit override:} set @envvar{ONNXRUNTIME_LIB} to the full
  path of the library file (or a directory containing it) and it is tried
  first.}]

@section[#:tag "version-negotiation"]{API version negotiation}

The ORT C API is a @emph{versioned function table}: @tt{OrtGetApiBase()}
yields a @tt{GetApi(version)} accessor, and each API version's function
table is a fixed, append-only ordering of function pointers. At load
time this package asks for the newest table it knows (version 31, from
the onnxruntime 1.31.0 header its index was generated from) and walks
down to version 10 until the runtime answers with a non-null table.
Consequently:

@itemlist[
 @item{against onnxruntime >= 1.31 the negotiated version is 31;}
 @item{older runtimes work too, down to API version 10 (onnxruntime
  1.10.x), and @racket[(onnx-api-version)] reports what was negotiated;}
 @item{while negotiating downward on an older runtime, ORT prints one
  warning line per unavailable version to stderr -- that is expected.}]

@section{Quick start}

Given a model @tt{add.onnx} computing @tt{Z = X + [[1,2],[3,4]]} over a
float32 @tt{[2,2]} input @tt{X}:

@racketblock[
(require onnxruntime racket/flonum)

(define s (open-onnx-session "add.onnx"))
(session-input-names s)          ; '("X")
(session-output-names s)         ; '("Z")
(session-input-shape s "X")      ; '(2 2)

(define x (make-ort-tensor '(2 2) (flvector 1.0 1.0 1.0 1.0)))
(define results (onnx-run s (list (cons "X" x))))
(results-ref results "Z")
;; (ort-tensor '(2 2) (flvector 2.0 3.0 4.0 5.0))
]

@defproc[(results-ref [results (listof (cons/c string? ort-tensor?))]
                      [name string?])
         ort-tensor?]{

Convenience: look up the tensor bound to @racket[name] in an
@racket[onnx-run] result. Raises @racket[exn:fail] if absent.
}

@section{Reference}

@defproc[(onnx-version) string?]{

Returns the ONNX Runtime library version string, e.g.
@racketresult["1.31.0"].}

@defproc[(onnx-api-version) exact-nonneg-integer?]{

Returns the OrtApi table version negotiated at load time (at most 31;
see @secref["version-negotiation"] for what that means on older
runtimes).}

@defstruct*[ort-tensor ([shape (listof exact-nonneg-integer?)]
                        [data flvector?])]{

A float32 tensor: @racket[shape] in row-major order and @racket[data] as
a flat @racket[flvector] of exactly @racket[(product shape)] elements.}

@defproc[(make-ort-tensor [shape (listof exact-nonneg-integer?)]
                          [data flvector?])
         ort-tensor?]{

Constructor that validates the shape/data size agreement.}

@defproc[(open-onnx-session [src (or/c path-string? bytes?)]
                            [#:optimization-level
                             level (or/c 'disable 'basic 'extended 'all)
                             'all])
         onnx-session?]{

Opens a session from a model file path or from raw @tt{.onnx} bytes.
@racket[level] maps to ORT's graph optimization levels
(@tt{ORT_DISABLE_ALL}=0, @tt{ORT_ENABLE_BASIC}=1,
@tt{ORT_ENABLE_EXTENDED}=2, @tt{ORT_ENABLE_ALL}=99).}

@defproc[(onnx-session? [v any/c]) boolean?]{

Predicate for the opaque session handle.}

@defproc[(session-input-names [s onnx-session?]) (listof string?)]{

Model input names in declaration order.}

@defproc[(session-output-names [s onnx-session?]) (listof string?)]{

Model output names in declaration order.}

@defproc[(session-input-shape [s onnx-session?] [name string?])
         (listof (or/c exact-positive-integer? 'dynamic))]{

Model metadata shape for the named input. Symbolic dimensions come back
as @racket['dynamic].}

@defproc[(session-output-shape [s onnx-session?] [name string?])
         (listof (or/c exact-positive-integer? 'dynamic))]{

Model metadata shape for the named output. Symbolic dimensions come
back as @racket['dynamic].}

@defproc[(onnx-run [s onnx-session?]
                   [inputs (listof (cons/c string? ort-tensor?))])
         (listof (cons/c string? ort-tensor?))]{

Runs the model. @racket[inputs] must cover @emph{every} model input
(pass all of them, even unchanged ones); duplicate or unknown names
raise @racket[exn:fail]. Results come back in model output order. All
native scratch memory is released before the call returns.}

@section{Documented limits (v0.1)}

@itemlist[
 @item{float32 tensors only (other element types raise
  @racket[exn:fail]);}
 @item{CPU execution provider only;}
 @item{all model inputs must be passed on every run;}
 @item{no explicit session close: native session memory lives until
  process exit;}]

@section{License}

MIT. ONNX Runtime itself is separately licensed by Microsoft under the
MIT license.
