"""
收盘后定时任务 —— 更新 30m 缓存 + 导出 live_signal.json

建议每天跑两次（玻璃 FG）：
  · 15:10  日盘收盘后（13:30–15:00）
  · 23:10  夜盘收盘后（21:00–23:00）

    python daily_job.py
    python daily_job.py --profile max
    python daily_job.py --skip-live    # 只更新缓存
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LMS = ROOT.parent
FG30 = LMS / "fg_30m"
LOG = ROOT / "artifacts" / "daily_job.log"


def log(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')}  {msg}"
    print(line)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="收盘后：30m 缓存 + 实盘信号")
    ap.add_argument("--profile", default="smart",
                    choices=["max", "smart", "auto", "regime", "balanced", "enhanced", "lms", "lms_dir", "lms_mom"])
    ap.add_argument("--skip-live", action="store_true", help="跳过 run.py --live")
    ap.add_argument("--symbol", default="", help="指定合约，默认当期主力")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from dominant import current_dominant

    sym = args.symbol.upper() or current_dominant()[0]
    log(f"=== daily_job 开始  symbol={sym}  profile={args.profile} ===")

    rc = subprocess.call(
        [sys.executable, str(FG30 / "refresh_cache.py"), "--symbol", sym],
        cwd=str(FG30),
    )
    if rc != 0:
        log(f"refresh_cache 失败 exit={rc}")
        return rc
    log("refresh_cache 完成")

    if not args.skip_live:
        rc = subprocess.call(
            [sys.executable, str(ROOT / "run.py"), "--live", "--profile", args.profile],
            cwd=str(ROOT),
        )
        if rc != 0:
            log(f"run.py --live 失败 exit={rc}")
            return rc
        sig = ROOT / "artifacts" / "live_signal.json"
        log(f"live_signal -> {sig}")

    log("=== daily_job 完成 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
