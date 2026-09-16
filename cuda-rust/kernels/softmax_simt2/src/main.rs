//! row-wise softmax を SIMT モデル（cuda-oxide）で書く。
//!
//! 設計：1行を1ブロックが担当し、1要素を1スレッドが担当する。
//!   grid  = (ROWS, 1, 1)
//!   block = (COLS, 1, 1)   ← COLS がそのまま1行の長さ
//! これで thread::index_1d() が「行 * COLS + 列」というフラット添字になり、
//! 入力・出力の配列添字にそのまま使える。
//!
//! softmax は行ごとに
//!   1. 最大値を求める        （reduction）
//!   2. 最大値を引いて exp    （elementwise）
//!   3. 総和を求める          （reduction）
//!   4. 総和で割る            （elementwise）
//! の4段。SIMT では 1 と 3 の reduction を自分で組む必要がある。

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{kernel, launch_bounds, thread, warp, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use std::time::Instant;

const COLS: usize = 256; // 1行の長さ
const TPB: usize = 128; // 1ブロックのスレッド数（1スレッドが COLS/TPB 要素を担当）
const PER: usize = COLS / TPB; // 1スレッドが担当する要素数 = 2
const ROWS: usize = 4096; // 行数 = ブロック数
const WARPS: usize = TPB / 32; // ブロック内の warp 数 = 4

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(128)]
    pub fn softmax_rows(x: &[f32], mut y: DisjointSlice<f32>) {
        static mut REDUCE: SharedArray<f32, WARPS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;   // ブロック内のスレッド番号（0..128）
        let row = thread::blockIdx_x() as usize;    // 担当する行
        let base = row * COLS + tid;                // 自分が担当する先頭要素
        let lane = warp::lane_id();
        let wid = warp::warp_id() as usize;

        // 1スレッドが PER 個（=2個）の要素を TPB 間隔で担当する。
        // 隣接スレッドが隣接要素を読むので coalescing は保たれる。
        let mut v = [0.0f32; PER];
        for k in 0..PER {
            v[k] = x[base + k * TPB];
        }

        // ---- 1. 行の最大値 ----
        let mut local = v[0];
        for k in 1..PER {
            if v[k] > local { local = v[k]; }
        }
        let wmax = warp::reduce_max_f32(local);
        if lane == 0 {
            unsafe { REDUCE[wid] = wmax };
        }
        thread::sync_threads();

        let mut row_max = unsafe { REDUCE[0] };
        for i in 1..WARPS {
            let m = unsafe { REDUCE[i] };
            if m > row_max { row_max = m; }
        }
        thread::sync_threads();

        // ---- 2. exp（担当分すべて）----
        let mut e = [0.0f32; PER];
        let mut local_sum = 0.0f32;
        for k in 0..PER {
            e[k] = (v[k] - row_max).exp();
            local_sum += e[k];
        }

        // ---- 3. 行の総和 ----
        let wsum = warp::reduce_sum_f32(local_sum);
        if lane == 0 {
            unsafe { REDUCE[wid] = wsum };
        }
        thread::sync_threads();

        let mut row_sum = 0.0f32;
        for i in 0..WARPS {
            row_sum += unsafe { REDUCE[i] };
        }

        // ---- 4. 書き出し ----
        // 1スレッドが複数要素を書くので、DisjointSlice の
        // 「index_1d() が指す1要素」という枠から外れる。
        for k in 0..PER {
            let idx = base + k * TPB;
            unsafe { *y.get_unchecked_mut(idx) = e[k] / row_sum };
        }
    }
}

/// CPU 側の参照実装（正しさの確認用）
fn softmax_cpu(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for r in 0..ROWS {
        let row = &x[r * COLS..(r + 1) * COLS];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
        let s: f32 = exps.iter().sum();
        for c in 0..COLS {
            out[r * COLS + c] = exps[c] / s;
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let n = ROWS * COLS;
    // Tile 版と厳密に同じ入力にする（あちらは api::linspace で同じ列を作る）
    let x_host: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    let x_dev = DeviceBuffer::from_host(&stream, &x_host)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (TPB as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // ---- 正しさの確認 ----
    // SAFETY: launch 形状はカーネルの想定と一致し、バッファは全アクセスを覆う。
    unsafe { module.softmax_rows(stream.as_ref(), cfg, &x_dev, &mut y_dev) }?;
    let y_gpu = y_dev.to_host_vec(&stream)?;
    let y_cpu = softmax_cpu(&x_host);

    let max_err = y_gpu
        .iter()
        .zip(y_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // 各行の総和が 1.0 になっているかも確認する
    let mut worst_sum_err = 0.0f32;
    for r in 0..ROWS {
        let s: f32 = y_gpu[r * COLS..(r + 1) * COLS].iter().sum();
        worst_sum_err = worst_sum_err.max((s - 1.0).abs());
    }

    println!("SIMT softmax (2要素/スレッド)  rows={ROWS} cols={COLS} tpb={TPB}");
    println!("  max |gpu - cpu|      = {max_err:.3e}");
    println!("  max |rowsum - 1.0|   = {worst_sum_err:.3e}");
    if max_err > 1e-6 || worst_sum_err > 1e-5 {
        eprintln!("FAILED: 精度が許容範囲外");
        std::process::exit(1);
    }

    // ---- 計測 ----
    // 読み書きするバイト数（入力を1回読み、出力を1回書く）
    let bytes = 2.0 * n as f64 * 4.0;
    const ITERS: u32 = 200;

    // ウォームアップ
    // SAFETY: 上と同じ launch。
    unsafe { module.softmax_rows(stream.as_ref(), cfg, &x_dev, &mut y_dev) }?;
    stream.synchronize()?;

    // (a) 200回まとめて投げて最後に1回だけ同期する（カーネル本体の速さ）
    let t0 = Instant::now();
    for _ in 0..ITERS {
        // SAFETY: 上と同じ launch。
        unsafe { module.softmax_rows(stream.as_ref(), cfg, &x_dev, &mut y_dev) }?;
    }
    stream.synchronize()?;
    let batched = t0.elapsed().as_secs_f64() / ITERS as f64;

    // (b) 毎回同期する（起動と同期のオーバーヘッド込み。Tile 版と比較する用）
    let t1 = Instant::now();
    for _ in 0..ITERS {
        // SAFETY: 上と同じ launch。
        unsafe { module.softmax_rows(stream.as_ref(), cfg, &x_dev, &mut y_dev) }?;
        stream.synchronize()?;
    }
    let per_sync = t1.elapsed().as_secs_f64() / ITERS as f64;

    println!("  batched (1 sync)     = {:.1} us  ({:.1} GB/s)", batched * 1e6, bytes / batched / 1e9);
    println!("  per-iteration sync   = {:.1} us  ({:.1} GB/s)", per_sync * 1e6, bytes / per_sync / 1e9);
    println!("PASSED");
    Ok(())
}
