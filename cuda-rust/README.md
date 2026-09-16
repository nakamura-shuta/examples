# CUDA Rust を実際に試したときの検証メモ

NVIDIA Technical Blog の [Introducing CUDA Rust: Two Tracks for Writing GPU Kernels](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/) を読んで、
cuda-oxide（SIMT）と cutile-rs（Tile）の両方を AWS の NVIDIA L4（g6.xlarge）と Apple Silicon の Linux（OrbStack）で動かしたときのメモです。
解説記事の付録にあたる内容で、セットアップでつまずいた点、GPU が無い環境でどこまでできるか、同梱サンプルの実行結果、プロファイラの数値を置いています。

記事本体：（公開後にリンクを追加）

検証したバージョンは cuda-oxide `cargo-oxide` v0.2.1（コミット 26754ae5）、cutile-rs 0.3.1、CUDA Toolkit 13.2 / 13.3、2026年9月時点のものです。
どちらのプロジェクトも early-stage なので、現在の仕様とは違っている可能性があります。

## 目次

- [A. セットアップの補足](#a-セットアップの補足)
- [B. GPU なし環境（OrbStack）での検証の詳細](#b-gpu-なし環境orbstackでの検証の詳細)
- [C. リポジトリ同梱サンプルを動かす](#c-リポジトリ同梱サンプルを動かす)
- [D. 自作 softmax の補足](#d-自作-softmax-の補足)
- [E. 測定環境](#e-測定環境)
- [F. 論文の性能数値](#f-論文の性能数値)
- [記事で自作したカーネル](#記事で自作したカーネル)

## A. セットアップの補足

### A-1. 公式ブログと book で要件が違う

ブログは「Linux、compute capability 8.0 以上の GPU、CUDA Toolkit 12.x 以降、clang と libclang、固定 nightly。システム LLVM は optional」としています。
一方リポジトリの `cuda-oxide-book/getting-started/installation.md` の Prerequisites はもっと厳しく、
CUDA Toolkit 13.0 以上、LLVM 21 以上（NVPTX バックエンド必須）、Clang 21 以上、ドライバ 580 以降、Ubuntu 24.04 でテスト済み、となっています。
LLVM 21 以上が要る理由も書かれています。

> Why LLVM 21? We emit TMA / tcgen05 / WGMMA intrinsics that `llc` from LLVM 20 and earlier can't handle.

実際に `cargo oxide doctor` を通すと、`llc` は `/usr/lib/llvm-21` ではなく rustup の `llvm-tools` から解決されました。

```
llc (LLVM)... ✓ LLVM version 23.1.0-rust-1.100.0-nightly
clang / libclang resource dir... ✓ /usr/lib/llvm-21/lib/clang/21
```

`rustup component add llvm-tools` を入れておけば rustc 同梱の LLVM（この nightly では 23）が使われるので、システム LLVM は optional です。
book の表は、システム LLVM を使う場合の下限と読むのがよさそうです。clang / libclang のほうは必須です。
x86_64 の g6.xlarge でも `llc` は同じく rustup 側から解決されたので、環境依存ではありません。

### A-2. ドライバと Toolkit の組み合わせの罠

> [!WARNING]
> book にこの図があります。
>
> ```text
> LLVM llc ───────────────► PTX 8.7 ──► driver 580 ✓
> CUDA 13.3 nvJitLink ───► PTX 9.3 ──► driver 580 ✗ error 222
> ```
>
> CUDA 13.x のマイナーバージョン互換はドライバ 580 から始まりますが、ドライバ 580 は CUDA 13.3 が生成する PTX 9.3 を JIT できません。
> 「新しい Toolkit を入れれば安心」ではないということです。
> DLAMI のドライバは 595.91.07 なので 580 の条件は満たしますが、Toolkit を update するときは注意してください。

### A-3. llvm.sh と update-alternatives

`apt.llvm.org` の `llvm.sh 21` が何も指定しないときに入れるのは `clang-21` / `lld-21` / `lldb-21` / `clangd-21` の4つだけで、
`llc` を提供する `llvm-21` はその中に含まれません。
cuda-oxide は標準では `llc` で LLVM IR を PTX に変換するので、これが無いと `cargo oxide doctor` が `No working llc` でエラーになります。
手元では `libclang-21-dev` の依存として入りましたが、book のとおり `sudo apt install llvm-21` と明示するほうが確実です。

もう1つが無印の `clang` です。`clang-21` を入れても `/usr/bin/clang` は Ubuntu 標準の 18 のままなので、bindgen が古い libclang を参照します。
book も、`cargo oxide doctor` が探すのはバージョン無しのエイリアスだとして、`update-alternatives` での対処を案内しています。

### A-4. curand.h

要件には `cuda.h` だけでなく `curand.h` も入っています。
`cuda-toolkit-13-3` メタパッケージだけでは入らないことがあるので、その場合は `sudo apt-get install -y libcurand-dev-13-3` を追加します。
DLAMI には最初から入っています（`/usr/local/cuda/include/curand.h`）。

### A-5. 再現環境（devcontainer / Nix）

cuda-oxide には devcontainer（CUDA 13.0 / LLVM 21 / Clang 21 / 固定 nightly）と Nix flake（CUDA 13 / LLVM 22 / Clang / 固定 nightly）があります。
devcontainer を使うなら、ホストに要るのは GPU・R580 以降のドライバ・NVIDIA Container Toolkit だけで、CUDA Toolkit のインストールは不要です。

```bash
npx -y @devcontainers/cli up --workspace-folder .
npx -y @devcontainers/cli exec --workspace-folder . cargo oxide doctor
```

cutile-rs にも Nix flake があり、CUDA 13.3 と Rust を含む dev shell が立ち上がります（`nix develop -c cargo run -p cutile-examples --example saxpy`）。

### A-6. cutile-rs の CUDA バージョン

README の要件は、compute capability `sm_80` 以上の GPU、CUDA 13.3 recommended、Rust 1.89 以上、Linux（Ubuntu 24.04 でテスト）です。
CUDA は「13.3 必須」ではありません。README は recommended、GitHub Release v0.3.1 の本文は `Requires CUDA 13.2+`、
CHANGELOG はホスト側クレートが 13.0 以上・Tile コンパイラの下限が 13.2 としています。

対応アーキテクチャは README に sm_ 番号で書かれています。

> - `sm_100+` is supported by CUDA 13.1+.
> - `sm_8x` support was added in CUDA 13.2.
> - CUDA 13.3 adds `sm_90` support, so CUDA 13.3 users now have `sm_80+` coverage.

L4 は sm_89 なので CUDA 13.2 でカバーされ、実際に DLAMI 同梱の 13.2 のまま動きました（DLAMI に 13.3 は入っていません）。
13.3 が必要な場合は、`cuda-keyring` を入れてから `sudo apt-get install -y cuda-toolkit-13-3` で追加し、`CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3` を指定します。
未設定のときは `/usr/local/cuda-13.3`、`/usr/local/cuda-13.2`、`/usr/local/cuda-13`、`/usr/local/cuda` の順に探す、と README にあります。

なお `cargo add cutile` だけで済む1クレート構成は 0.3.1 からで、それ以前は `cutile-compiler` も依存に足す必要がありました。

## B. GPU なし環境（OrbStack）での検証の詳細

### B-1. GPU 無しでのビルドは公式に想定

`gemm_views` の README には、GPU 不要のモードが明記されています。

```bash
cargo oxide run gemm_views     # GPU: correctness vs CPU reference, then benchmark
gemm_views --verify-ptx        # no GPU needed: structural PTX comparison
gemm_views --bench             # GPU: benchmark only
```

`--verify-ptx` は、生成された PTX に対して「コンパイル時だけに使う contract のマーカーが残っていないか」
「カーネルに `trap` 命令が含まれていないか」「safe 版と unsafe 版で条件分岐の数が一致するか」
「両者の global memory へのロード/ストア命令が同一か」を構造的に検証します。

### B-2. cargo oxide のサブコマンド

`new`（雛形生成。`--async` で tokio + cuda-async 版）、`build`、`run`、`doctor`（環境の検証）、
`setup`（codegen バックエンドのビルド）、`inspect`（生成された PTX の表示）、`pipeline`（パイプラインの詳細ダンプ）、
`sanitize`（NVIDIA Compute Sanitizer 下で実行）、`debug --tui`（cuda-gdb）、`clean` があります。

フラグも、`--arch sm_XX`、`--unchecked-indexing`（境界チェック除去）、`--materialize-cubin`、
`--emit-nvvm-ir`、`--lineinfo`（プロファイラ向け行番号）、`--device-debug`（nvcc の `-G` 相当）などがそろっています。
early alpha と言われている割に、必要なものは一通りそろっています。

### B-3. `gemm_views --verify-ptx` が通らない

その `--verify-ptx` がエラーになりました。

```
Error: "naive guard branches differ: safe=11, raw=10"
```

safe 版と unsafe 版で条件分岐の数が1つ食い違う、という検証エラーです。
`--arch sm_89` や `--arch sm_120` を明示しても同じだったので、ターゲットアーキテクチャには依存しません。

原因はおそらく LLVM のバージョンです。README にはこうあります。

> These numbers depend on the pipeline disabling llc's late branch folding: LLVM 23 started rewriting loop branches into a single negated conditional, which ptxas's SASS unroller does not recognize, and the naive kernels lose about a quarter of their throughput.

codegen 側は対策していて、`crates/cuda-oxide-codegen/src/ptx.rs` は `llc` の全呼び出しに `-disable-branch-fold` と `-disable-block-placement` を渡しています。
同じファイルには `Observed with llc-22 (CI's floor pin); llc-23 emits its debug sections differently` というコメントもあり、
LLVM 23 で挙動が変わっていることは認識されているようです。
手元の `llc` は A-1 のとおり rustup 同梱の 23.1.0-rust-1.100.0-nightly なので、対策が十分に働かない組み合わせに当たったのだと思われます。
`-disable-branch-fold` のコメントには、恒久的な回避策ではなく上流待ちであることも書かれています。

> Permanent until ptxas learns to unroll both layouts or upstream adds an opt-out for the NVPTX folding; an internal NVBug and an LLVM issue are to be filed to track both ends.

> [!NOTE]
> このエラーは aarch64 固有ではありません。x86_64 の AWS 環境（NVIDIA L4）でも `safe=11, raw=10` まで同一でした。
> ホストのアーキテクチャにも GPU の有無にも依存しない、再現性のある問題です。
> つまり、プロジェクトが自分で pin している toolchain（`nightly-2026-08-28` 付属の LLVM 23 の `llc`）では、自前の構造検証が通らない状態です。
> CI の floor は llc-22 と書かれているので、CI では検出されない組み合わせかもしれません。

### B-4. GPU 無しで `cargo oxide run`

記事の Try 1-1 のとおり、ビルドとリンクはできますが、実行するとエラーで止まります。book にこの挙動の説明があります。

> The host runtime does not link `libcuda` at build time. The shared `cuda-bindings` crate loads it at the first driver call, so a binary starts without a driver and fails on that call with `CUDA_ERROR_NOT_INITIALIZED`; the error message names the library files the loader tried.

GPU パススルーによる回避もできません（理由は記事の Environment に記述）。CUDA の device emulation mode もすでに廃止されています。

cutile-rs 側も同じです。Tile IR の JIT コンパイラ `tileiras` は CUDA Toolkit に同梱されていて（パッケージ名 `cuda-tileiras-13-3`）ビルドは通りますが、
カーネルを最初に必要になった時点で JIT する設計なので、実行には GPU が要ります。

### B-5. doctor の差分

g6.xlarge で `cargo oxide doctor` を実行すると、OrbStack と比べて変わるのは1行だけでした。

```diff
- NVIDIA driver / GPU... - no NVIDIA driver detected
+ NVIDIA driver / GPU... ✓ NVIDIA L4 (compute capability 8.9, driver 595.91.07)
```

CUDA は DLAMI 同梱の 13.2（V13.2.51）がそのまま使われ、cuda-oxide の要件（13.0+）を満たしています。
## C. リポジトリ同梱サンプルを動かす

記事の softmax とは別に、両リポジトリに入っているサンプルも L4 で動かしました。

### C-1. cuda-oxide：`gemm_views` でsafe は unsafe と同じマシン語になる？

`gemm_views` は、行列積 `C = alpha * A * B + beta * C` を計算するサンプルです（`m` / `n` / `k` は実行時に指定）。
README は、safe なカーネルが手書きの unsafe 版と同じ機械語になることを、計算結果・PTX の構造・ベンチマークの3つで示すとしています。

> This example shows safe kernels reaching the same machine code as hand-written unsafe ones, and proves it three ways: results, PTX structure, and a benchmark.

違いは境界チェックの回数です。

```
ordinary indexing:  check, load, check, load, fma, ...     per iteration
views:              whole-row check ONCE, whole-column check ONCE, then
                    load, load, fma, advance, branch       no checks left
```

`a[i]` のように普通に書くと、ループを1周するたびに境界チェックが入ります。
`gemm_views` では、`MatrixView32::row(row, k)` と
`MatrixView32::col(col, k)` で行と列が範囲内かを最初に1回だけ確かめ、
`zip_exact` で両者の長さが同じことも1回だけ確かめます。
そのため内積のループには、終了判定の比較しか残りません。

実行方法は3通りです。

```bash
cd ~/cuda-oxide
cargo oxide run gemm_views                    # CPU 参照実装と比較 → ベンチマーク
cargo oxide run gemm_views -- --verify-ptx    # GPU 不要。PTX の構造比較
cargo oxide run gemm_views -- --bench         # ベンチマークのみ
```

L4 での結果です（1024×1024×1024、5回平均）。

```
naive safe vs raw: max diff = 0.000e0
tiled safe vs raw: max diff = 0.000e0

gemm_views bench: 1024x1024x1024, 5 timed runs
  naive views (safe)      1.160 ms     1851.3 GFLOPS
  naive raw (unsafe)      1.160 ms     1850.7 GFLOPS
  tiled views (safe)      0.898 ms     2391.4 GFLOPS
  tiled raw (unsafe)      0.902 ms     2381.1 GFLOPS

SUCCESS: view-based SGEMM matched raw twins and CPU reference
```

safe 版と unsafe 版の計算結果は、ビット単位で一致（`max diff = 0.000e0`）。
速度も naive は 1.160 ms で同じ、
tiled は safe 版がわずかに速い程度（2391.4 対 2381.1 GFLOPS）です。
※誤差の範囲

この条件では速度の低下は見られませんでした。
README にある RTX 5090 の値と並べると以下。

| | RTX 5090（README） | L4（今回） |
|---|---:|---:|
| naive views (safe) | 7201 GFLOPS | 1851.3 GFLOPS |
| tiled views (safe) | 9370 GFLOPS | 2391.4 GFLOPS |

L4（最大 72 W の推論向け GPU）の速さは RTX 5090 の約4分の1ですが、
shared memory を使った tiled 版が naive 版より速くなる割合はほぼ同じ。
比較のため、ループのたびに境界チェックが入る`gemm` サンプルも L4 で動かしました。

```bash
cargo oxide run gemm
```

```
=== Unified GEMM Example (Naive Implementation) ===
Performance: 2.512 ms, 855.00 GFLOPS
```

855 GFLOPS で、`gemm_views` の naive views（1851.3 GFLOPS）の半分以下です。
README によると RTX 5090 でもこの `gemm` は約 2940 GFLOPS で、
naive views はその2.4倍でした。
GPU が違っても、境界チェックを安全に外すと2倍以上速くなる、という結果は同じです。

`#[launch_contract]` の `requires` の動きも確かめられます。
このサンプルは、`k` を2倍にした不正な起動をわざと1回混ぜています。
この起動は GPU に届く前に CPU 側で拒否され、
（どの条件がどう反故されたかが）エラーに出ます。

```
contract rejected oversized-K launch on the CPU: sgemm_naive_views:
  size requirement `a.len() >= m * k` violated:
  left-hand side is 8192, right-hand side is 16384
```

> [!WARNING]
> README によると、この数値は cuda-oxide が `llc` の後段にある分岐の畳み込み
> （branch folding）を無効にしていることが前提です。
> LLVM 23 はループの分岐を1つの否定条件に書き換えるようになり、
> `ptxas` がそれをうまく展開できないため、
> naive カーネルのスループットが約4分の1下がるようです。
> ※ `crates/cuda-oxide-codegen/src/ptx.rs` の `DISABLE_BRANCH_FOLD`参照
>
> また、16×16 ブロックの倍数でないサイズ（端数のタイル）には、
> 今の段階では意図的に対応していません。

これは `gemm_views` の2種類のカーネルを
1024×1024×1024 で測った範囲の話で、
安全性のコストがゼロだという意味ではありません。
論文にある「unsafe Rust との差 0.3% 以内」も、
B200 で M=N=K=8192 の GEMM を測った値です。

### C-2. その他のサンプル

- `sharedmem`（cuda-oxide）：`SharedArray` を使って、ブロック内のスレッド同士で値を受け渡すサンプル。`sync_threads()` のあと、隣のスレッドが書いた値を正しく読めた。自作 SIMT softmax の reduction と同じ仕組みで、shared memory へのアクセスには `unsafe` が必要
- `pinned_overlap`（cuda-oxide）：ホスト側のメモリを pinned にし、転送とカーネルを3つのストリームで重ねるサンプル。256 MiB の GPU→ホスト転送は、pageable の 1.90 GB/s に対して pinned は 13.22 GB/s、パイプライン化で 1.22倍速くなった。
- `softmax`（cutile-rs）：同梱の Tile 版 softmax 。記事の自作 Tile 版とほぼ同じコード
- `gemm`（cutile-rs）：`mma` を回すだけの GEMM 。正しく計算でき、Tile IR ではタイル同士の積和 `mmaf` 1命令が内側のループになっていた
- `interop`（cutile-rs）：手書き PTX のカーネルと Tile カーネルを、同じストリームでつなぐサンプル。`3 * (n + n)` が全要素で一致。既存の CUDA 資産（PTX）と一緒に使える例

`pinned_overlap` の値は、README にある RTX 4090 での参照値より低めでした。
（pinned 22〜25 GB/s、パイプライン化で 1.76倍）
RTX 4090 は L4 と同じ Ada 世代ですが、SM 数やメモリ帯域、電力枠は大きく違います。
なお、ここで測っているのは主に転送なので、
GPU の違いよりホスト側の転送環境の差が原因だと思われます（未検証）。
また pageable 側は毎回転送先を確保し、pinned 側は確保済みのバッファを使い回しているので、
差にはメモリ確保のコストも含まれます。

cutile-rs 同梱の softmax ベンチ（`cargo bench -p cutile-benchmarks -- softmax`）は、
N=2048 までと N=4096 以降で傾向が変わります。
ただし N=4096 以降はタイル幅 BN が行の長さ N より短く、
行全体ではなく BN 要素ごとの softmax を計算しています。
ワーキングセットが L2（48 MB）を超える点とも重なるので、
この変化をキャッシュ容量だけでは説明できません。
また、このベンチは `unsafe fn` と `unchecked_accesses=true` を使っており、
記事の自作版とは条件が違うので注意。

### C-3. 2つのトラックをつなぐ：`cutile_inter_kernel`

cuda-oxide の examples にある `cutile_inter_kernel` は、
2つのトラックのカーネルを1本の処理につなぐサンプルです。

```text
input ──[cutile-rs Tile kernel: row_softmax]──► softmax_out
      ──[cuda-oxide SIMT kernel: threshold_scale_f32]──► gated_out
```

cutile-rs の Tile カーネルで行ごとの softmax を計算し、
その結果を cuda-oxide の SIMT カーネルが受け取って、
閾値より小さい値を 0 にし、残りを定数倍します。
※これは Tile でも書けるので、「SIMT でしか書けない処理」の例ではない

別々のコンパイラが作った2つのカーネルを、1つのストリームでつなぐ例です。
接続するには cutile-rs の `DeviceOp::then` を使い、
2つの起動を同じ CUDA ストリームに順番にセットします。
tensorはそのまま `CUdeviceptr` として SIMT 側に渡すので、
SIMT 側のカーネルはrawポインタを受け取る `unsafe fn` です。
※README も、これは将来の形ではなく、今使えるつなぎ方と記述

> This is not the future intra-kernel interop path. It is the interop that works today: separate kernels, shared stream, shared device memory.

```bash
cargo oxide run cutile_inter_kernel
```

```
PASS: cutile-rs Tile softmax -> cuda-oxide SIMT threshold/scale passed
first row input:    [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
first row softmax:  [0.0121, 0.0200, 0.0329, 0.0542, 0.0894, 0.1474, 0.2431, 0.4008]
first row output:   [0.0,    0.0,    0.0,    0.0,   0.3577, 0.5898, 0.9724, 1.6032]
```

Tile カーネルが入力を確率に変え、SIMT カーネルが 0.08 未満を 0 にして
残りを4倍しています（0.0894 × 4 = 0.3577）。

> [!NOTE]
> このサンプルを動かすには、CUDA Toolkit 13.1 以上の `nvcc` と `tileiras` が必要です。
> また cutile-rs の CUDA バインディングのビルドで `CUDA_TOOLKIT_PATH` を参照するので、
> Toolkit が `/usr/local/cuda` 以外にある場合は指定してください。

## D. 自作 softmax の補足

### D-1. `warp::reduce_max_f32` の PTX と NaN

`warp::reduce_max_f32` は単純な `max.f32` 1命令にはなりません。

```
shfl.sync.bfly.b32  %r14, %r2, 16, 31, -1;
setp.nan.f32        %p9,  %r2, %r2;        ← NaN かどうか調べて
setp.ge.f32         %p10, %r14, %r2;
selp.f32            %r15, %r14, %r2, %p10;
selp.f32            %r16, %r14, %r15, %p9;  ← %r2 が NaN なら、もう片方を選ぶ
```

Rust の `f32::max` の意味が、PTX まで保たれています。
Rust の `f32::max` は「片方だけが NaN なら、もう片方（数値のほう）を返す」と定義されており、
生成された PTX も `%r2` が NaN のときもう一方の `%r14` を選ぶ形になっています。

一方、自作した `if m > row_max { row_max = m; }` は
`setp.gt.f32` + `selp.f32` の2命令だけで、NaN の考慮がありません。
ライブラリ関数と自前の比較で、正確さと命令数のトレードオフが可視化されている形です。

### D-2. L2 を温めた状態のプロファイル

`ncu` は何も指定しないとキャッシュをフラッシュしてから測るので、
連続実行時（L2 が温まった状態）も見ておきます。
`--cache-control none` を付け、50回目の起動を測ります。

| | 実行時間 | L2 ヒット率 | DRAM 転送 |
|---|---:|---:|---:|
| SIMT | 15.14 µs | 100.5% | 218 KB |
| Tile | 13.28 µs | 67.1% | 3.04 MB |

SIMT 版は L2 のヒット率が高く DRAM にほぼ触れてないのにTileより遅い。
少なくとも DRAM 帯域が原因ではなさそうで、命令数の差が原因だという仮説が立ちます。
ただし L2 帯域やロード命令の依存関係といった他の要因を、
この2つの指標だけで除外できるわけではありません。

L2 ヒット率が 100.5% と 100% を超えているのは、
Nsight Compute が短いカーネルやリプレイの都合でこうした値を出すことがあるためです。

### D-3. v2 と v3 のプロファイル比較

プロファイラで両者を比較した結果は以下。
※`--cache-control none`、50回目の起動を計測

| 指標 | v2（ストライド・unsafe） | v3（連続・安全 API） |
|---|---:|---:|
| 実行時間 | 8.86 µs | 9.89 µs |
| L1/TEX のグローバルロード セクタ数 | 131,072 | 262,144 |
| L1/TEX のグローバルストア セクタ数 | 131,072 | 262,144 |
| 実行命令数 | 2,064,384 | 2,211,840 |
| レジスタ/スレッド | 19 | 19 |
| achieved occupancy | 83.32% | 83.58% |

ncu のメトリクス名は順に `gpu__time_duration.sum`、`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`、
同 `_st.sum`、`smsp__inst_executed.sum` です。

L1/TEX で数えたセクタアクセス数がちょうど2倍になっている。
全要素 1,048,576 を1セクタ32バイト（f32 で8個）で割ると 131,072 なので、
v2 は1命令ごとのアクセスがセクタにぴったり収まっています。

v3 が2倍になるのは配置のせいです。`LinearTiles<N>` は
スレッド `t` に連続領域を割り当てるので、1つのロード命令だけを見ると、
ワープ内のアクセスがストライド2になります（レーン0が要素0、レーン1が要素2、レーン2が要素4……）。
そのため、1命令あたりでは、読み込んだセクタの半分しか使われません。
※記事で説明した coalescing の話

これは1命令あたりの利用効率の話で、次の命令が残り半分の要素を読むため、
カーネル全体でデータの半分を捨てているわけではありませんし、
DRAM の転送量が2倍になるという意味でもありません
※ L1/TEX のセクタアクセス数と DRAM の転送量は別の指標

まとめると、セクタアクセス数の2倍は配置によるアクセス効率の差を強く裏付けていますが、
API・配置・命令数（+7%）が同時に変わっているので、15%の差のすべてを配置に帰属させることはできません。
配置の影響だけを確かめるには、配置以外の条件を揃えた比較が必要です。
(`tile_2d32` のように配置を選べる API もあるので、それを使ってみるとか）

### D-4. タイル形状の目安

`cutile-book` の [performance.md](https://nvlabs.github.io/cutile-rs/main/) には、
タイルの大きさについて以下のように書かれています。

> Tile size controls how much work each tile block performs. Larger tiles improve data reuse and reduce launch overhead per element, but they also consume more registers and can reduce occupancy.

同じページには、ワークロード別の目安も書いてます。
reduction については「レジスタ圧を過度に上げない軸サイズ」とだけ書かれていて、
具体値は示されていません。

| Workload | Starting point |
|---|---|
| Elementwise 1D | `[128]`, `[256]`, `[512]` |
| Elementwise 2D | `[16, 16]`, `[32, 32]`, `[64, 16]` |
| GEMM | Tile shapes compatible with Tensor Core MMA dimensions |
| Reductions | Axis sizes that avoid excessive register pressure |

## E. 測定環境

| 項目 | 値 |
|---|---|
| GPU | NVIDIA L4（Ada Lovelace / AD104、compute capability 8.9、24 GB GDDR6、メモリ帯域 300 GB/s、L2 48 MB、最大消費電力 72 W） |
| インスタンス | AWS g6.xlarge（us-east-1） |
| NVIDIA ドライバ | 595.91.07 |
| CUDA Toolkit | 13.2（DLAMI 同梱、V13.2.51） |
| AMI | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) |
| cuda-oxide | `cargo-oxide` v0.2.1（コミット 26754ae5） |
| cutile-rs | 0.3.1 |
| Rust | stable 1.98.1 / nightly-2026-04-03 および nightly-2026-08-28 |
| 自作ベンチの試行回数 | ウォームアップ1回のあと200回 |
| GPU クロック | 既定（cutile 同梱ベンチのみ `setclock.sh` で 1350 MHz に固定） |

## F. 論文の性能数値

本文の 5.1 節はもう少し細かく、M=N=K=8192 の GEMM で safe な Rust が
2.07 PFlop/s（B200 の dense f16 ピーク比92%、cuBLAS 比96.4%）、
unsafe Rust との差は 0.3% 以内、cuTile Python は 2.04 PFlop/s（cuBLAS 比94.9%）としています。
要素ごとの加算では N=2^28 で safe / unsafe Rust ともに 7.02 TB/s、
cuTile Python が 7.01 TB/s で、ピークの 7.68 TB/s に近いところまで出ています。

注意点として、この論文の測定は cuTile Rust 0.2.0 に対するものです（[再現アーティファクトの README](https://github.com/NVlabs/cutile-rs/blob/main/cutile-benchmarks/paper/README.md) に明記）。
本記事執筆時点の最新は 0.3.1 なので、「最新版でこの数字」と読むのは正確ではありません。
また論文自身も、一部の行列サイズで cuBLAS との性能差が残っていることを限界として挙げています。
（Grout はモデルの GEMM で cuBLAS にフォールバック）

## 記事で自作したカーネル

記事の Try 2 で書いた row-wise softmax のソースです。
そのままでは動かないので、`cargo oxide`（SIMT 版）または stable Rust（Tile 版）の環境を用意してください。手順は A を参照。

| ディレクトリ | 内容 | L4 での測定値（batched） |
|---|---|---|
| `kernels/softmax_simt` | SIMT v1。1行を1ブロック、1要素を1スレッド | 14.2 µs |
| `kernels/softmax_simt2` | SIMT v2。128スレッド/ブロック、1スレッド2要素（ストライド配置、書き出しは `unsafe`） | 8.2 µs |
| `kernels/softmax_simt3` | SIMT v3。v2 と同じ形で、書き出しを `LinearTiles` の安全な API に変更 | 9.4 µs |
| `kernels/softmax_tile` | Tile 版（`BM=1, BN=256`） | 10.0 µs |

いずれも 4,096 行 × 256 列、ウォームアップ1回のあと200回の平均です。
v2 / v3 / Tile 版は入力を `x[i] = i * 0.01` に揃えてあります。

v1 だけは、ここに置いてあるのが古いリビジョンです。カーネル本体は記事に載せたものと同じですが、
ホスト側が「入力の作り方が違う」「batched と per-iteration を分けて測っていない」状態です。
記事の 14.2 µs は、この計測部分を v2 / v3 と同じ形に直してから測った値です。

```bash
# SIMT 版（cuda-oxide。固定 nightly が必要）
cd kernels/softmax_simt && cargo oxide run

# Tile 版（cutile-rs。stable Rust）
cd kernels/softmax_tile && cargo run --release
```
