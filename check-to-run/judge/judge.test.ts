import { test } from "node:test";
import assert from "node:assert/strict";

const target = process.env.TARGET;
if (!target) throw new Error("TARGET is required");
const { subtotal, applyDiscount, dueDate, installments, topItems } = await import(target);

test("subtotal", () => {
  assert.equal(subtotal([{ name: "x", unitPrice: 120, quantity: 3 }, { name: "y", unitPrice: 7, quantity: 11 }]), 437);
  assert.equal(subtotal([]), 0);
});

test("applyDiscount: 整数演算の結果と全件一致", () => {
  for (const p of [0, 5, 10, 15, 20, 25, 30, 33, 50, 70, 100]) {
    for (let a = 0; a <= 3000; a++) {
      assert.equal(applyDiscount(a, p), Math.floor((a * (100 - p)) / 100), `amount=${a} percent=${p}`);
    }
  }
});

test("dueDate: 2026〜2028 年の全日付", () => {
  const pad = (n: number) => String(n).padStart(2, "0");
  for (let y = 2026; y <= 2028; y++) {
    for (let m = 1; m <= 12; m++) {
      const days = new Date(Date.UTC(y, m, 0)).getUTCDate();
      const ny = m === 12 ? y + 1 : y;
      const nm = m === 12 ? 1 : m + 1;
      const last = new Date(Date.UTC(ny, nm, 0)).getUTCDate();
      for (let d = 1; d <= days; d++) {
        assert.equal(dueDate(`${y}-${pad(m)}-${pad(d)}`), `${ny}-${pad(nm)}-${pad(last)}`, `issued=${y}-${pad(m)}-${pad(d)}`);
      }
    }
  }
});

test("installments: 合計・切り捨て・余りは 1 回目", () => {
  for (let total = 0; total <= 500; total++) {
    for (let n = 1; n <= 12; n++) {
      const r = installments(total, n);
      assert.equal(r.length, n);
      assert.equal(r.reduce((s: number, v: number) => s + v, 0), total);
      const each = Math.floor(total / n);
      assert.equal(r[0], each + (total % n), `total=${total} n=${n}`);
      for (let i = 1; i < n; i++) assert.equal(r[i], each);
    }
  }
});

test("topItems: 順序・同額の安定性・引数を変更しない", () => {
  const items = [
    { name: "a", unitPrice: 10, quantity: 1 },
    { name: "b", unitPrice: 100, quantity: 3 },
    { name: "c", unitPrice: 150, quantity: 2 },
    { name: "d", unitPrice: 5, quantity: 2 },
    { name: "e", unitPrice: 300, quantity: 1 },
    { name: "f", unitPrice: 1000, quantity: 1 },
  ];
  const before = structuredClone(items);
  assert.deepEqual(topItems(items, 4).map((i: { name: string }) => i.name), ["f", "b", "c", "e"]);
  assert.deepEqual(items, before);
  assert.deepEqual(topItems(items, 0), []);
  assert.equal(topItems(items, 99).length, 6);
});
