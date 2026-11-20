import { test } from "node:test";
import assert from "node:assert/strict";
import { subtotal, applyDiscount, dueDate, installments, topItems, type Item } from "../src/billing.ts";

const items: Item[] = [
  { name: "A", unitPrice: 100, quantity: 2 },
  { name: "B", unitPrice: 500, quantity: 1 },
  { name: "C", unitPrice: 50, quantity: 4 },
];

test("subtotal: 単価×数量の合計", () => {
  assert.equal(subtotal(items), 900);
  assert.equal(subtotal([]), 0);
});

test("applyDiscount: 1 円未満は切り捨て", () => {
  assert.equal(applyDiscount(1000, 10), 900);
  assert.equal(applyDiscount(999, 10), 899);
  assert.equal(applyDiscount(90, 30), 63);
  assert.equal(applyDiscount(500, 0), 500);
  assert.equal(applyDiscount(500, 100), 0);
});

test("dueDate: 翌月の末日", () => {
  assert.equal(dueDate("2026-01-15"), "2026-02-28");
  assert.equal(dueDate("2026-07-31"), "2026-08-31");
  assert.equal(dueDate("2026-12-31"), "2027-01-31");
  assert.equal(dueDate("2028-01-31"), "2028-02-29");
});

test("installments: 余りは 1 回目、合計は total", () => {
  assert.deepEqual(installments(1000, 3), [334, 333, 333]);
  assert.deepEqual(installments(900, 3), [300, 300, 300]);
  assert.deepEqual(installments(5, 1), [5]);
});

test("topItems: 金額の大きい順、同額は元の順、引数は変更しない", () => {
  const before = structuredClone(items);
  assert.deepEqual(topItems(items, 2).map((i) => i.name), ["B", "A"]);
  assert.deepEqual(items, before);
});
