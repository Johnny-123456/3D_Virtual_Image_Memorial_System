#!/usr/bin/env bash
# 2.5D 怀念影像查看器 —— 启动脚本
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$DIR/build/memorial_viewer"

# 板上有图形会话时它属于 elf 用户，Xauthority 在 /run/user/1000/gdm/
if [ -z "${DISPLAY:-}" ]; then export DISPLAY=:0; fi
if [ -z "${XAUTHORITY:-}" ] && [ -f /run/user/1000/gdm/Xauthority ]; then
  export XAUTHORITY=/run/user/1000/gdm/Xauthority
fi

ASSET="${1:-$DIR/../assets/demo}"

if [ ! -x "$BIN" ]; then
  echo "未构建，先执行: cd $DIR && mkdir -p build && cd build && cmake .. && make -j4"
  exit 1
fi
if [ ! -f "$ASSET/meta.json" ]; then
  echo "资产不存在: $ASSET"
  echo "生成: python3 $DIR/../tools/make_asset.py --name $(basename "$ASSET")"
  exit 1
fi

exec "$BIN" --asset "$ASSET" "$@"
