"""ragbench index | run | report"""

import argparse

from ragbench import config


def main():
    ap = argparse.ArgumentParser(prog="ragbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index", help="build shared OpenSearch indexes (and method-specific graphs with --method)")
    p.add_argument("--datasets", nargs="+", default=config.DATASETS)
    p.add_argument("--method", help="also run this method's prepare() step (lightrag, graph_r1)")
    p = sub.add_parser("run", help="answer + score questions")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--datasets", nargs="+", default=config.DATASETS)
    p.add_argument("-n", type=int, default=100, help="questions per dataset")
    p.add_argument("--workers", type=int, default=8)
    sub.add_parser("report", help="aggregate results/runs into tables and plots")
    args = ap.parse_args()

    if args.cmd == "index":
        from ragbench import index
        from ragbench.methods import METHODS
        for ds in args.datasets:
            index.build_base(ds)
            if args.method:
                print(ds, METHODS[args.method](ds).prepare())
    elif args.cmd == "run":
        from ragbench import llm, runner
        for method in args.methods:
            for ds in args.datasets:
                runner.run(method, ds, args.n, args.workers)
        print(f"real API spend so far: ${llm.spent_usd():.2f}")
    elif args.cmd == "report":
        from ragbench import report
        report.main()


if __name__ == "__main__":
    main()
