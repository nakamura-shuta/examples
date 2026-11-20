# 請求計算モジュールの仕様

金額はすべて円の整数で扱う。実装は `src/billing.ts` に置く。

## Item

```ts
type Item = { name: string; unitPrice: number; quantity: number };
```

## subtotal(items: Item[]): number

- 単価 × 数量の合計を返す。

## applyDiscount(amount: number, percent: number): number

- `amount` から `percent` % を割り引いた金額を返す。`percent` は 0〜100 の整数。
- 1 円未満は切り捨てる。

## dueDate(issued: string): string

- 支払期限は、請求日の翌月の末日とする。
- 引数と戻り値は `YYYY-MM-DD` 形式の文字列。
- 実行する環境のタイムゾーンによって結果が変わってはいけない。

## installments(total: number, n: number): number[]

- `total` を `n` 回の分割払いにする。`n` は 1 以上の整数。
- 各回の金額は `total / n` の切り捨てとし、余りは 1 回目に足す。
- 戻り値の合計は、必ず `total` と一致する。

## topItems(items: Item[], k: number): Item[]

- 金額（単価 × 数量）の大きい順に、先頭から `k` 件を返す。
- 同額の場合は、元の配列での順序を保つ。
- 引数の配列を変更してはいけない。
