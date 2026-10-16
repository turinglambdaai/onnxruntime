# AGENTS.md

ONNX Runtime C API 的 Racket FFI 绑定（MIT，已上架 Racket catalog）。

## 结构

Multi-collection 包：collection 即 `onnxruntime/` 子目录，scribblings 声明在 `onnxruntime/info.rkt`（collection 级）；`test/`、`tools/` 在包内。

## 常用命令

```bash
brew install onnxruntime        # 原生依赖（CI 同款）
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib   # macOS 必需
raco pkg install --auto --link "$(pwd)"
raco test onnxruntime/test/onnxruntime-test.rkt
```

## 验证与发版

- CI（`ci.yml`）：brew 装 onnxruntime → 设库搜索路径 → 安装包 → 显式路径跑测试。
- 发版 = tag `v<版本>` + CHANGELOG 同步；catalog 侧为手动提交（pkgd）。
