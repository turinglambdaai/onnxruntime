#lang info

;; Collection-level info: `raco setup` discovers documentation only from the
;; `scribblings` field of a collection's own info.rkt; for `collection 'multi`
;; packages a package-root declaration is never consulted when rendering.
(define collection "onnxruntime")
(define scribblings '(("scribblings/onnxruntime.scrbl" ())))
