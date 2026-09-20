"""Command line interface.

    python -m guardrail.cli inspect "Ignore all previous instructions"
    python -m guardrail.cli inspect --json --fast-only "<|im_start|>system"
    python -m guardrail.cli scan            # interactive scanner, ctrl-d to exit

Exit status is 1 when the prompt is blocked, so the CLI composes with shell
pipelines and CI gates.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from guardrail.gateway import Gateway
from guardrail.models import GatewayConfig, InspectionResult, Verdict

_COLOUR = {Verdict.BLOCK: "\033[31m", Verdict.SANITIZE: "\033[33m", Verdict.ALLOW: "\033[32m"}
_RESET = "\033[0m"


def _render(result: InspectionResult, prompt: str, *, colour: bool, verbose: bool) -> str:
    tag = result.verdict.value
    if colour:
        tag = f"{_COLOUR[result.verdict]}{tag}{_RESET}"
    lines = [
        f"{tag}  threat={result.threat_detected.value}  confidence={result.confidence:.2f}",
        f"      decided by {result.path.value} in {result.latency_ms:.2f}ms"
        f" (fast {result.fast_path_ms:.2f}ms / semantic {result.semantic_path_ms:.2f}ms)",
    ]
    for reason in result.reasons:
        lines.append(f"      - {reason}")
    if result.semantic is not None and verbose:
        s = result.semantic
        lines.append(
            f"      vector: backend={s.backend} family={s.family} "
            f"score={s.score:.3f} threshold={s.threshold:.2f}"
        )
    if result.sanitized_prompt is not None and result.sanitized_prompt != prompt:
        lines.append(f"      sanitized -> {result.sanitized_prompt}")
    return "\n".join(lines)


def _config(args: argparse.Namespace) -> GatewayConfig:
    updates: dict[str, object] = {"fast_path_only": bool(args.fast_only)}
    if args.canary:
        updates["canary_token"] = str(args.canary)
    if args.threshold is not None:
        updates["semantic_threshold"] = float(args.threshold)
        updates["local_semantic_threshold"] = float(args.threshold)
    return GatewayConfig().model_copy(update=updates)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fast-only", action="store_true", help="skip the semantic vector gate")
    parser.add_argument("--threshold", type=float, default=None, help="override the semantic threshold")
    parser.add_argument("--canary", default=None, help="canary token to watch for")
    parser.add_argument("--verbose", action="store_true", help="show vector diagnostics")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="guardrail", description="dual-path prompt injection firewall")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = sub.add_parser("inspect", help="inspect a single prompt")
    inspect_cmd.add_argument("prompt", help="the untrusted user prompt")
    inspect_cmd.add_argument("--json", action="store_true", help="emit the InspectionResult as JSON")
    _add_common(inspect_cmd)

    scan_cmd = sub.add_parser("scan", help="interactive scanner (one prompt per line)")
    _add_common(scan_cmd)

    args = parser.parse_args(argv)
    gateway = Gateway(_config(args))

    if args.command == "inspect":
        result = gateway.inspect(args.prompt)
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print(_render(result, args.prompt, colour=sys.stdout.isatty(), verbose=args.verbose))
        return 1 if result.verdict is Verdict.BLOCK else 0

    print("guard-rail interactive scanner -- one prompt per line, ctrl-d to exit.")
    for line in sys.stdin:
        prompt = line.rstrip("\n")
        if not prompt.strip():
            continue
        print(_render(gateway.inspect(prompt), prompt, colour=sys.stdout.isatty(), verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
