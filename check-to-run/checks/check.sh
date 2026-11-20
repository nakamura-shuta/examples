#!/bin/bash
# 2 つのタイムゾーンでテストを実行する。どちらかが失敗したら exit 1。
set -e
cd "$(dirname "$0")/.."
TZ=Asia/Tokyo node --test "test/*.test.ts"
TZ=America/Los_Angeles node --test "test/*.test.ts"
