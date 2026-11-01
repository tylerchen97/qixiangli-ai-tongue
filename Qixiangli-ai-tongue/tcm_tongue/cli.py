import argparse
import json

import pandas as pd
import yaml

from .pipeline import DEFAULT_CFG, extract_folder, run_analysis, run_demo


def main():
    p = argparse.ArgumentParser(prog="tcm_tongue", description="舌象数字表型与多模态关联研究平台")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="合成数据端到端演示")
    d.add_argument("--out", default="outputs")
    d.add_argument("--subjects", type=int, default=60)
    d.add_argument("--visits", type=int, default=8)
    d.add_argument("--seed", type=int, default=42)

    e = sub.add_parser("extract", help="批量提取舌象数字表型")
    e.add_argument("--images", required=True)
    e.add_argument("--out", default="features.csv")

    a = sub.add_parser("analyze", help="对真实数据运行关联 / 监测 / 独立验证")
    a.add_argument("--features", required=True, help="extract 产出的 features.csv")
    a.add_argument("--meta", required=True, help="含 image_file、受试者、站点、指标的 CSV")
    a.add_argument("--config", default="configs/default.yaml")
    a.add_argument("--out", default="outputs")

    args = p.parse_args()
    if args.cmd == "demo":
        s = run_demo(args.out, args.subjects, args.visits, args.seed)
        print(json.dumps(s, ensure_ascii=False, indent=2, default=str)[:4000])
    elif args.cmd == "extract":
        extract_folder(args.images).to_csv(args.out, index=False)
        print(f"features saved -> {args.out}")
    else:
        cfg = {**DEFAULT_CFG, **(yaml.safe_load(open(args.config, encoding="utf-8")) or {})}
        df = pd.read_csv(args.meta).merge(pd.read_csv(args.features), on="image_file")
        run_analysis(df, cfg, args.out)
        print(f"results saved -> {args.out}")
