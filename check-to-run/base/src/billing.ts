export type Item = { name: string; unitPrice: number; quantity: number };

export function subtotal(items: Item[]): number {
  return items.reduce((sum, item) => sum + item.unitPrice * item.quantity, 0);
}

export function applyDiscount(amount: number, percent: number): number {
  return Math.floor(amount * (1 - percent / 100));
}

export function dueDate(issued: string): string {
  const date = new Date(issued);
  date.setMonth(date.getMonth() + 2);
  date.setDate(0);
  return date.toISOString().slice(0, 10);
}

export function installments(total: number, n: number): number[] {
  const each = Math.floor(total / n);
  const result = new Array<number>(n).fill(each);
  result[n - 1] += total % n;
  return result;
}

export function topItems(items: Item[], k: number): Item[] {
  const amount = (item: Item) => item.unitPrice * item.quantity;
  return items.sort((a, b) => amount(b) - amount(a)).slice(0, k);
}
