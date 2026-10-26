"""Command-line entry point: ``python -m agent_pinn.cli <command>``.

Commands
--------
manifest            freeze sha256 of the baseline artefacts
bootstrap           import outputs/runs/* into the state store (idempotent)
pareto              recompute and print the Pareto table
train-candidate     propose+train+post-process one structure
work                drain queued jobs in this process
jobs                list recent jobs
show-run            dump one run row (config/metrics/audits)
audit-run/audit-all Inspector (see agent_pinn.audit)
mine                Psi-NN structure mining (see agent_pinn.miner)
run-episode         Creator-Inspector loop (see agent_pinn.orchestrator)
serve               FastAPI dashboard
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from . import bootstrap as boot
from . import config, db, jobs, paths, repo, runner, scoring
from pinn_tg.structure import (legacy_structure, normalize_structure,
                               validate_structure, structure_hash)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_manifest(args) -> int:
    paths.ensure_dirs()
    entries = {}
    for run_dir in sorted(d for d in paths.RUNS.iterdir() if d.is_dir()):
        files = {}
        for name in ("config.json", "summary.json", "model.pt"):
            p = run_dir / name
            if p.is_file():
                files[name] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
        if files:
            entries[run_dir.name] = files
    sources = {}
    for p in sorted(paths.ROOT.joinpath("pinn_tg").glob("*.py")):
        sources[f"pinn_tg/{p.name}"] = _sha256(p)
    out = {"created_at": db.now(), "runs": entries, "sources": sources}
    dest = Path(args.out) if args.out else paths.AGENT_MANIFEST
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"manifest: {len(entries)} runs, {len(sources)} sources -> {dest}")
    return 0


def cmd_bootstrap(args) -> int:
    res = boot.bootstrap(args.runs_dir, args.db)
    print(json.dumps(res, indent=2))
    return 0


def cmd_pareto(args) -> int:
    conn = db.connect(args.db)
    rows = scoring.refresh(conn)
    top = sorted([r for r in rows if r["eligible"]],
                 key=lambda r: (r["front_rank"], -r["score"]))[:args.top]
    if not top:
        print("no eligible runs")
        return 1
    hdr = f"{'rank':>4} {'run':<26} {'N':>5} {'A':>6} {'C':>6} {'P':>6} " \
          f"{'score':>6} {'L_pde':>8} {'L_e':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in top:
        p = "  --  " if r["compliance"] is None else f"{r['compliance']:.2f}"
        n = r.get("param_count") or 0
        lp = r.get("L_pde")
        le = r.get("L_energy")
        print(f"{r['front_rank']:>4} {r['run_name']:<26} {n:>5} "
              f"{r['accuracy']:.3f} {r['complexity']:.3f} {p:>6} "
              f"{r['score']:.3f} "
              f"{'   --   ' if lp is None else f'{lp:8.3f}'} "
              f"{'    --   ' if le is None else f'{le:9.2e}'}")
    prov = sum(1 for r in rows if r.get("provisional"))
    print(f"\n{len(rows)} rows, {prov} provisional (not audited yet)")
    return 0


def _resolve_structure(args) -> dict:
    if args.from_run:
        from pinn_tg.checkpoint import load_run_net
        rd = paths.RUNS / args.from_run
        net, cfg, st = load_run_net(str(rd))
        return st
    if args.structure:
        raw = args.structure
        if os.path.exists(raw):
            cfg = json.loads(Path(raw).read_text(encoding="utf-8"))
        else:
            cfg = json.loads(raw)
        return normalize_structure(cfg)
    b = config.load_budgets()["train"]
    return legacy_structure(args.lam or b["lam"], args.m or b["m"],
                            args.width or b["width"], args.depth or b["depth"])


def _hyper(args) -> dict:
    b = dict(config.load_budgets()["train"])
    for k in ("beta", "lam", "m", "width", "depth", "adam", "lbfgs", "ny",
              "nz", "seed"):
        v = getattr(args, k, None)
        if v is not None:
            b[k] = v
    if getattr(args, "w_energy", None) is not None:
        b["w_energy"] = args.w_energy
    if getattr(args, "data", None):
        b["data"] = args.data
    if getattr(args, "asm_npz", None):
        b["asm_npz"] = args.asm_npz
    if getattr(args, "ro", None) is not None:
        b["ro"] = args.ro
    return b


def cmd_train_candidate(args) -> int:
    structure = _resolve_structure(args)
    errs = validate_structure(structure)
    if errs:
        print("invalid structure:", "; ".join(errs))
        return 2
    hyper = _hyper(args)
    if args.smoke:
        sb = config.load_budgets()["episode"]["smoke"]
        hyper.update({"adam": sb["adam"], "lbfgs": sb["lbfgs"],
                      "ny": sb["ny"], "nz": sb["nz"]})
    sh = structure_hash(structure)
    conn = db.connect(args.db)
    cid = repo.upsert_candidate(
        conn, structure=structure, hyper=hyper, structure_hash=sh,
        provenance={"source": "cli", "provider": "none",
                    "strategy": "manual"}, status="proposed")
    post_cfg = config.load_budgets()["post"]
    if args.smoke:
        post_cfg = {"ny": 65, "nz": 32, "n_lam": 2}
    jid = jobs.enqueue(conn, kind="train", run_name=args.name,
                       candidate_id=cid,
                       timeout_s=int(config.load_budgets()["episode"]
                                     ["train_timeout_s"]),
                       payload={"structure": structure, "hyper": hyper,
                                "post": not args.no_post,
                                "post_cfg": post_cfg})
    print(f"candidate {cid}  structure {sh}  job {jid}")
    if args.unsafe_async:
        print("queued; run `python -m agent_pinn.cli work` to execute")
        return 0
    res = jobs.run_now(conn, jid)
    print(json.dumps(res, indent=2, ensure_ascii=False)[:2000])
    return 0 if res and res.get("ok") else 1


def cmd_work(args) -> int:
    conn = db.connect(args.db)
    res = jobs.work(conn, max_jobs=args.max_jobs)
    print(json.dumps(res, indent=2, ensure_ascii=False)[:3000])
    return 0 if all(r.get("ok", True) for r in res) else 1


def cmd_jobs(args) -> int:
    conn = db.connect(args.db)
    for j in repo.list_jobs(conn, limit=args.top):
        print(f"{j['id']:<28} {j['kind']:<6} {str(j['run_name']):<26} "
              f"{j['state']:<10} {j.get('error') or ''}")
    return 0


def cmd_show_run(args) -> int:
    conn = db.connect(args.db)
    run = repo.get_run(conn, args.run_name)
    if not run:
        print("no such run")
        return 1
    print(json.dumps(run, indent=2, ensure_ascii=False)[:4000])
    audits = repo.audits_for(conn, run["id"])
    if audits:
        print("\naudits:")
        for a in audits:
            print(f"  {a['code']:<14} {a['kind']:<10} "
                  f"{'PASS' if a['passed'] else 'FAIL':<5} "
                  f"value={a['value']} thr={a['threshold']}")
    return 0


def cmd_audit_run(args) -> int:
    from .audit import audit_run
    conn = db.connect(args.db)
    res = audit_run(conn, args.run_name, db_path=args.db)
    print(json.dumps(res, indent=2, ensure_ascii=False)[:3000])
    return 0 if res.get("ok") else 1


def cmd_audit_all(args) -> int:
    from .audit import audit_all
    conn = db.connect(args.db)
    res = audit_all(conn, csv_path=args.csv)
    print(json.dumps(res, indent=2))
    return 0


def cmd_calibrate(args) -> int:
    from .audit.calibrate import calibrate
    conn = db.connect(args.db)
    res = calibrate(conn, out=args.out)
    print(json.dumps(res, indent=2))
    return 0


def cmd_mine(args) -> int:
    from .miner import mine
    res = mine(args.teacher, args.out, k_min=args.k_min, k_max=args.k_max,
               prune_frac=args.prune, distill_steps=args.distill_steps,
               distill_ny=args.distill_ny, distill_nz=args.distill_nz)
    print(json.dumps(res, indent=2, ensure_ascii=False)[:3000])
    return 0


def cmd_run_episode(args) -> int:
    from .orchestrator import Orchestrator
    from .llm import get_provider
    conn = db.connect(args.db)
    provider = get_provider(args.llm)
    budget = config.load_budgets()
    if args.max_trains:
        budget["episode"]["max_trains"] = args.max_trains
    if args.rounds:
        budget["episode"]["max_rounds"] = args.rounds
    if args.budget_min:
        budget["episode"]["max_wall_minutes"] = args.budget_min
    orch = Orchestrator(args.episode, db_path=args.db, provider=provider,
                        budget=budget, smoke=args.smoke, conn=conn)
    res = orch.run()
    print(json.dumps(res, indent=2, ensure_ascii=False)[:4000])
    return 0 if res.get("state") == "DONE" else 1


def cmd_serve(args) -> int:
    import uvicorn
    uvicorn.run("agent_pinn.web.app:app", host=args.host, port=args.port,
                workers=1, log_level="info")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="agent_pinn")
    ap.add_argument("--db", default=str(paths.AGENT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("manifest")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("bootstrap")
    p.add_argument("--runs-dir", default=None)
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("pareto")
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_pareto)

    p = sub.add_parser("train-candidate")
    p.add_argument("--name", required=True)
    p.add_argument("--structure", default="",
                   help="structure JSON or path to a JSON file")
    p.add_argument("--from-run", default="", help="copy structure from a run")
    for k in ("beta", "lam", "m", "width", "depth", "adam", "lbfgs", "ny",
              "nz", "seed"):
        p.add_argument(f"--{k}", type=float if k in ("beta", "lam") else int,
                       default=None)
    p.add_argument("--w-energy", dest="w_energy", type=float, default=None)
    p.add_argument("--data", default="")
    p.add_argument("--asm-npz", dest="asm_npz", default="")
    p.add_argument("--ro", type=float, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-post", action="store_true")
    p.add_argument("--unsafe-async", dest="unsafe_async", action="store_true")
    p.set_defaults(func=cmd_train_candidate)

    p = sub.add_parser("work")
    p.add_argument("--max-jobs", type=int, default=1)
    p.set_defaults(func=cmd_work)

    p = sub.add_parser("jobs")
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("show-run")
    p.add_argument("--run-name", required=True)
    p.set_defaults(func=cmd_show_run)

    p = sub.add_parser("audit-run")
    p.add_argument("--run-name", required=True)
    p.set_defaults(func=cmd_audit_run)

    p = sub.add_parser("audit-all")
    p.add_argument("--csv", default="")
    p.set_defaults(func=cmd_audit_all)

    p = sub.add_parser("calibrate-thresholds")
    p.add_argument("--out", default=str(paths.AUDIT_THRESHOLDS))
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("mine")
    p.add_argument("--teacher", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=4)
    p.add_argument("--prune", type=float, default=0.25)
    p.add_argument("--distill-steps", type=int, default=800)
    p.add_argument("--distill-ny", type=int, default=61,
                   help="coarse grid for the KD loop only; acceptance is "
                        "re-measured on the teacher grid")
    p.add_argument("--distill-nz", type=int, default=40)
    p.set_defaults(func=cmd_mine)

    p = sub.add_parser("run-episode")
    p.add_argument("--episode", required=True)
    p.add_argument("--rounds", type=int, default=0)
    p.add_argument("--max-trains", type=int, default=0)
    p.add_argument("--budget-min", type=float, default=0)
    p.add_argument("--llm", default="auto")
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(func=cmd_run_episode)

    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_serve)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths.ensure_dirs()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
