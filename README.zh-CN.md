# onnxruntime

[English](README.md) · **中文**

[![CI](https://github.com/turinglambdaai/onnxruntime/actions/workflows/ci.yml/badge.svg)](https://github.com/turinglambdaai/onnxruntime/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[ONNX Runtime](https://onnxruntime.ai/) C API 的 Racket FFI 绑定，用于
ONNX 模型的 CPU 推理——*在 Python 中训练一次，导出 `.onnx`，随处推理*
这条路线上的运行时一环。

**版本 0.1 范围：** float32 张量、CPU execution provider、从文件路径或
字节加载模型。

## 安装

```sh
raco pkg install onnxruntime
```

还需要让加载器能找到原生 ONNX Runtime 库：

- **macOS（Homebrew）：** `brew install onnxruntime`。加载器会自动探测
  `/opt/homebrew/opt/onnxruntime/lib`（Apple silicon）与
  `/usr/local/opt/onnxruntime/lib`（Intel）；也可以导出
  `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib`。
- **Linux：** 解压官方 release 压缩包，把其中的 `lib/` 目录放进
  `LD_LIBRARY_PATH`，或将 `libonnxruntime.so` 安装到标准位置。
- **显式覆盖：** 将 `ONNXRUNTIME_LIB` 设为库文件（或包含它的目录）；
  它会最先被尝试。

## 版本协商

ORT C API 是一张带版本号的函数表。本包请求 API 版本 31（取自其函数
索引生成时所用的 onnxruntime 1.31.0 头文件），并逐级向下探测到 10，
直到加载的运行时给出应答为止。`(onnx-api-version)` 报告协商出的版本；
向下支持到 API 版本 10（onnxruntime 1.10.x）的旧运行时。向下探测
期间，ORT 会为每个不可用的版本向 stderr 打印一行警告。

## 快速上手

```racket
(require onnxruntime racket/flonum)

(onnx-version)                              ; "1.31.0"
(define s (open-onnx-session "add.onnx"))
(session-input-names s)                     ; '("X")
(session-input-shape s "X")                 ; '(2 2)

(define x (make-ort-tensor '(2 2) (flvector 1.0 1.0 1.0 1.0)))
(results-ref (onnx-run s (list (cons "X" x))) "Z")
;; (ort-tensor '(2 2) (flvector 2.0 3.0 4.0 5.0))
```

完整参考见 `onnxruntime/scribblings/onnxruntime.scrbl`（或随包安装的
文档）。

## 开发

- 运行测试（macOS）：
  `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib raco test -p onnxruntime`
- 构建文档：
  `raco scribble --dest /tmp/onnxruntime-docs onnxruntime/scribblings/onnxruntime.scrbl`
- 头文件升级后重新生成 API 索引：
  `python3 tools/parse-ortapi-header.py /path/to/onnxruntime_c_api.h \
     > onnxruntime/private/api-index.rkt`
  （仅需 Python 3 标准库——是构建期工具，不是包依赖。）
- 重新生成 `test/data/` 中的小型测试模型（可选；它们已提交在仓库中），
  使用 Python `onnx` 包：
  `python3 tools/gen-fixtures.py test/data`

## 许可证

MIT —— 见 [LICENSE](LICENSE)。ONNX Runtime 本身由 Microsoft 以 MIT
许可证单独授权。
