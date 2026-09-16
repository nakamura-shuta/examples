//! row-wise softmax を Tile モデル（cutile-rs）で書く。
//!
//! SIMT 版（softmax_simt）とまったく同じ形・同じ入力で比較する。
//!   ROWS = 4096, COLS = 256
//!
//! Tile 側ではスレッドも shared memory も出てこない。
//! 「BM 行 × BN 列のタイル1枚に対して何をするか」だけを書き、
//! 実スレッドへの割り当てはコンパイラが決める。

use cutile::prelude::*;
use std::sync::Arc;
use std::time::Instant;

const ROWS: usize = 4096;
const COLS: usize = 256;
const BM: usize = 1; // タイル1枚が持つ行数（公式ベンチも 1 行/タイル）
const BN: usize = COLS; // 列は行まるごと（reduction が行内で閉じる必要がある）

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn softmax_rows<const BM: i32, const BN: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>, // 排他的な出力タイル
        x: &Tensor<f32, { [-1, -1] }>,     // 共有入力（形は起動時に決まる）
    ) {
        // このタイルに対応する入力を丸ごと読む
        let tx: Tile<f32, { [BM, BN] }> = x.load_like(y);

        // 1. 行ごとの最大値（軸1 = 列方向に畳む）
        let m: Tile<f32, { [BM] }> = reduce_max(tx, 1i32);
        let m: Tile<f32, { [BM, BN] }> = m.reshape(shape![BM, 1]).broadcast(y.shape());

        // 2. 最大値を引いて exp
        let num: Tile<f32, { [BM, BN] }> = exp(tx - m);

        // 3. 行ごとの総和
        let den: Tile<f32, { [BM] }> = reduce_sum(num, 1i32);
        let den: Tile<f32, { [BM, BN] }> = den.reshape(shape![BM, 1]).broadcast(y.shape());

        // 4. 正規化して書き戻す
        y.store(num / den);
    }
}

use kernels::softmax_rows;

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let n = ROWS * COLS;
    // SIMT 版の x[i] = i as f32 * 0.01 と同じ列を作る
    let stop = (n - 1) as f32 * 0.01;
    let x: Arc<Tensor<f32>> = api::linspace(0.0, stop, n)
        .sync_on(&stream)?
        .reshape(&[ROWS, COLS])
        .unwrap()
        .into();

    // ---- 正しさの確認（各行の総和が 1.0 か）----
    let y = api::zeros::<f32>(&[ROWS, COLS])
        .sync_on(&stream)?
        .partition([BM, BN]);
    let y_host: Vec<f32> = softmax_rows(y, x.clone())
        .first()
        .unpartition()
        .to_host_vec()
        .sync_on(&stream)?;

    let mut worst = 0.0f32;
    for r in 0..ROWS {
        let s: f32 = y_host[r * COLS..(r + 1) * COLS].iter().sum();
        worst = worst.max((s - 1.0).abs());
    }
    println!("Tile softmax  rows={ROWS} cols={COLS}  (BM={BM}, BN={BN})");
    println!("  max |rowsum - 1.0|   = {worst:.3e}");
    if worst > 1e-5 {
        eprintln!("FAILED: 行の総和が 1.0 になっていない");
        std::process::exit(1);
    }

    let bytes = 2.0 * n as f64 * 4.0;
    const ITERS: u32 = 200;

    // ウォームアップ（初回は Tile IR の JIT が走る）
    {
        let y = api::zeros::<f32>(&[ROWS, COLS])
            .sync_on(&stream)?
            .partition([BM, BN]);
        let _ = softmax_rows(y, x.clone()).first().sync_on(&stream)?;
    }

    // ---- (a) async_on で投げっぱなしにし、最後に1回だけ同期する ----
    // cutile-rs 公式ベンチ（cutile-benchmarks/benches/softmax.rs）と同じ測り方。
    // SIMT 版の batched に対応する。
    {
        let mut part = api::zeros::<f32>(&[ROWS, COLS])
            .sync_on(&stream)?
            .partition([BM, BN]);
        unsafe { stream.synchronize() }?;
        let t = Instant::now();
        for _ in 0..ITERS {
            let (out, _x) = unsafe { softmax_rows(part, x.clone()).async_on(&stream) }?;
            part = out;
        }
        unsafe { stream.synchronize() }?;
        let batched = t.elapsed().as_secs_f64() / ITERS as f64;
        println!(
            "  batched (async_on)   = {:.1} us  ({:.1} GB/s)",
            batched * 1e6,
            bytes / batched / 1e9
        );
    }

    // ---- (b) 毎回 sync_on でブロックする ----
    // ドキュメントが「やめておけ」と言っている使い方。差を見るために測る。
    {
        let mut part = Some(
            api::zeros::<f32>(&[ROWS, COLS])
                .sync_on(&stream)?
                .partition([BM, BN]),
        );
        let t = Instant::now();
        for _ in 0..ITERS {
            let out = softmax_rows(part.take().unwrap(), x.clone())
                .first()
                .sync_on(&stream)?;
            part = Some(out);
        }
        let per_sync = t.elapsed().as_secs_f64() / ITERS as f64;
        println!(
            "  per-iteration sync   = {:.1} us  ({:.1} GB/s)",
            per_sync * 1e6,
            bytes / per_sync / 1e9
        );
    }

    println!("PASSED");
    Ok(())
}
