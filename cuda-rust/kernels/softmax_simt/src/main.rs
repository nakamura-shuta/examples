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

const COLS: usize = 256; // 1行の長さ = ブロックのスレッド数
const ROWS: usize = 4096; // 行数 = ブロック数
const WARPS: usize = COLS / 32; // ブロック内の warp 数 = 8

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    pub fn softmax_rows(x: &[f32], mut y: DisjointSlice<f32>) {
        // ブロック内 reduction の中継地点。warp ごとに1スロット使う。
        // shared memory は SIMT トラックでは現状 unsafe が必要。
        static mut REDUCE: SharedArray<f32, WARPS> = SharedArray::UNINIT;

        let gid = thread::index_1d(); // 行 * COLS + 列
        let lane = warp::lane_id(); // warp 内での位置（0..32）
        let wid = warp::warp_id() as usize; // ブロック内での warp 番号（0..8）

        let v = x[gid.get()];

        // ---- 1. 行の最大値 ----
        // まず warp 内32レーンを1命令群で集約し、その代表値だけを
        // shared memory に集めてから、ブロック全体の最大値にする。
        let wmax = warp::reduce_max_f32(v);
        if lane == 0 {
            unsafe { REDUCE[wid] = wmax };
        }
        thread::sync_threads(); // 全 warp の書き込み完了を待つ

        let mut row_max = unsafe { REDUCE[0] };
        for i in 1..WARPS {
            let m = unsafe { REDUCE[i] };
            if m > row_max {
                row_max = m;
            }
        }
        thread::sync_threads(); // 全員が読み終わるまで上書きさせない

        // ---- 2. 最大値を引いて exp ----
        // 最大値を引くのは、exp のオーバーフローを避ける定番の安定化。
        let e = (v - row_max).exp();

        // ---- 3. 行の総和 ----
        let wsum = warp::reduce_sum_f32(e);
        if lane == 0 {
            unsafe { REDUCE[wid] = wsum };
        }
        thread::sync_threads();

        let mut row_sum = 0.0f32;
        for i in 0..WARPS {
            row_sum += unsafe { REDUCE[i] };
        }

        // ---- 4. 正規化して書き出し ----
        // 出力は DisjointSlice なので、各スレッドは自分の1要素だけを触れる。
        if let Some(dst) = y.get_mut(gid) {
            *dst = e / row_sum;
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
    // 適当だが行ごとに分布が変わる入力
    let x_host: Vec<f32> = (0..n)
        .map(|i| {
            let r = (i / COLS) as f32;
            let c = (i % COLS) as f32;
            (c - 128.0) * 0.05 + (r % 7.0) * 0.3
        })
        .collect();

    let x_dev = DeviceBuffer::from_host(&stream, &x_host)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (COLS as u32, 1, 1),
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

    println!("SIMT softmax  rows={ROWS} cols={COLS}");
    println!("  max |gpu - cpu|      = {max_err:.3e}");
    println!("  max |rowsum - 1.0|   = {worst_sum_err:.3e}");
    if max_err > 1e-6 || worst_sum_err > 1e-5 {
        eprintln!("FAILED: 精度が許容範囲外");
        std::process::exit(1);
    }

    // ---- 計測 ----
    const ITERS: u32 = 200;
    // SAFETY: 上と同じ launch。
    unsafe { module.softmax_rows(stream.as_ref(), cfg, &x_dev, &mut y_dev) }?;
    stream.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        // SAFETY: 上と同じ launch。
        unsafe { module.softmax_rows(stream.as_ref(), cfg, &x_dev, &mut y_dev) }?;
    }
    stream.synchronize()?;
    let elapsed = t0.elapsed();

    let per_iter = elapsed.as_secs_f64() / ITERS as f64;
    // 読み書きするバイト数（入力を1回読み、出力を1回書く）
    let bytes = 2.0 * n as f64 * 4.0;
    println!("  time                 = {:.3} us", per_iter * 1e6);
    println!("  effective bandwidth  = {:.1} GB/s", bytes / per_iter / 1e9);
    println!("PASSED");
    Ok(())
}
