//! row-wise softmax（SIMT / cuda-oxide）v3
//!
//! v2 は「1スレッドが2要素」にして速くなったが、書き出しに unsafe を使った。
//! v3 では、安全な複数要素アクセス API（DisjointSlice<T, LinearTiles<N>> と
//! tile_thread32）だけで同じことを書き、unsafe を消せるか・性能はどうなるかを見る。
//!
//! 違いが1つある。v2 はストライド配置（スレッド t が t, t+TPB, ...）だったが、
//! LinearTiles<N> はスレッド t に t*N..t*N+N の連続領域を割り当てる。
//! つまりメモリアクセスのパターンが変わる。

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    kernel, launch_bounds, launch_contract, thread, warp, DisjointSlice, LinearTiles, SharedArray,
};
use cuda_host::cuda_module;
use std::time::Instant;

const COLS: usize = 256;
const TPB: usize = 128; // 1ブロックのスレッド数
const PER: usize = COLS / TPB; // 1スレッドが担当する要素数 = 2
const ROWS: usize = 4096;
const WARPS: usize = TPB / 32; // = 4

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
    pub fn softmax_rows(x: &[f32], mut y: DisjointSlice<f32, LinearTiles<PER>>) {
        static mut REDUCE: SharedArray<f32, WARPS> = SharedArray::UNINIT;

        let thread_index = thread::index_1d_u32(launch_context);
        let lane = warp::lane_id();
        let wid = warp::warp_id() as usize;

        // thread_index は tile_thread32 に消費される（所有権が移る）ので、
        // 入力側で使う先頭位置を先に取り出しておく。
        let base = thread_index.get() as usize * PER;

        // このスレッドが所有する PER 要素の連続領域。
        // 境界チェックは1回だけで、以降はチェック不要のビューが返る。
        let Some(mut run) = y.tile_thread32(thread_index) else {
            return;
        };

        // 入力は普通のスライスなので、同じ範囲を読む
        let mut v = [0.0f32; PER];
        for k in 0..PER {
            v[k] = x[base + k];
        }

        // ---- 1. 行の最大値 ----
        let mut local = v[0];
        for k in 1..PER {
            if v[k] > local {
                local = v[k];
            }
        }
        let wmax = warp::reduce_max_f32(local);
        if lane == 0 {
            unsafe { REDUCE[wid] = wmax };
        }
        thread::sync_threads();

        let mut row_max = unsafe { REDUCE[0] };
        for i in 1..WARPS {
            let m = unsafe { REDUCE[i] };
            if m > row_max {
                row_max = m;
            }
        }
        thread::sync_threads();

        // ---- 2. exp ----
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

        // ---- 4. 書き出し（ここが v2 との違い。unsafe なし）----
        // at_const は「このビューの何番目か」をコンパイル時に検査するので、
        // 実行時の境界チェックも unsafe も要らない。
        run.at_const::<0>().write(e[0] / row_sum);
        run.at_const::<1>().write(e[1] / row_sum);
    }
}

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
    let x_host: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    let x_dev = DeviceBuffer::from_host(&stream, &x_host)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;

    // SAFETY: このパッケージが kernels モジュール用の device bundle を所有している。
    let module = unsafe { kernels::load(&ctx)? };
    // launch_contract を宣言したので、起動には検証済みトークンが要る
    let prepared = module.prepare_softmax_rows(LaunchConfig1D::new(ROWS as u32, TPB as u32, 0))?;

    module.softmax_rows(&stream, &prepared, &x_dev, &mut y_dev)?;
    let y_gpu = y_dev.to_host_vec(&stream)?;
    let y_cpu = softmax_cpu(&x_host);

    let max_err = y_gpu
        .iter()
        .zip(y_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let all_finite = y_gpu.iter().all(|v| v.is_finite());

    let mut worst_sum_err = 0.0f32;
    for r in 0..ROWS {
        let s: f32 = y_gpu[r * COLS..(r + 1) * COLS].iter().sum();
        worst_sum_err = worst_sum_err.max((s - 1.0).abs());
    }

    println!("SIMT softmax v3 (安全API / 2要素連続)  rows={ROWS} cols={COLS} tpb={TPB}");
    println!("  all finite           = {all_finite}");
    println!("  max |gpu - cpu|      = {max_err:.3e}");
    println!("  max |rowsum - 1.0|   = {worst_sum_err:.3e}");
    if !all_finite || max_err > 1e-6 || worst_sum_err > 1e-5 {
        eprintln!("FAILED");
        std::process::exit(1);
    }

    let bytes = 2.0 * n as f64 * 4.0;
    const ITERS: u32 = 200;

    module.softmax_rows(&stream, &prepared, &x_dev, &mut y_dev)?;
    stream.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
            module.softmax_rows(&stream, &prepared, &x_dev, &mut y_dev)?;
    }
    stream.synchronize()?;
    let batched = t0.elapsed().as_secs_f64() / ITERS as f64;

    let t1 = Instant::now();
    for _ in 0..ITERS {
            module.softmax_rows(&stream, &prepared, &x_dev, &mut y_dev)?;
        stream.synchronize()?;
    }
    let per_sync = t1.elapsed().as_secs_f64() / ITERS as f64;

    println!(
        "  batched (1 sync)     = {:.1} us  ({:.1} GB/s)",
        batched * 1e6,
        bytes / batched / 1e9
    );
    println!(
        "  per-iteration sync   = {:.1} us  ({:.1} GB/s)",
        per_sync * 1e6,
        bytes / per_sync / 1e9
    );
    println!("PASSED");
    Ok(())
}
