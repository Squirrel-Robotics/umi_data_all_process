#!/usr/bin/env python3
"""Collect entries from matching parent directories into one target directory."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Operation:
    source: Path
    target: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将匹配父目录中的子项集中到一个目标目录；默认仅预览，"
            "只有添加 --execute 才会修改数据。"
        )
    )
    parser.add_argument("source", type=Path, help="待整理的源目录")
    parser.add_argument(
        "--target",
        type=Path,
        help="汇总目标目录；默认与 source 相同",
    )
    parser.add_argument(
        "--parent-pattern",
        action="append",
        required=True,
        metavar="GLOB",
        help="要展开的父目录匹配规则；可重复指定，例如 'collector_run_*'",
    )
    parser.add_argument(
        "--child-pattern",
        action="append",
        metavar="GLOB",
        help="父目录内待收集子项的匹配规则；默认 '*'，可重复指定",
    )
    parser.add_argument(
        "--entry-type",
        choices=("dirs", "files", "all"),
        default="dirs",
        help="收集目录、文件或全部；默认 dirs",
    )
    parser.add_argument(
        "--mode",
        choices=("move", "copy"),
        default="move",
        help="移动或复制；默认 move",
    )
    parser.add_argument(
        "--conflict",
        choices=("error", "skip", "rename"),
        default="error",
        help="同名处理：停止、跳过、自动改名；默认 error",
    )
    parser.add_argument(
        "--remove-empty-parents",
        action="store_true",
        help="移动完成后删除已经为空的匹配父目录",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际执行；不加此参数时只显示计划",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="逐条显示源路径和目标路径",
    )
    return parser.parse_args()


def lexical_exists(path: Path) -> bool:
    """Return True for normal paths and broken symbolic links."""
    return os.path.lexists(os.fspath(path))


def matches_type(path: Path, entry_type: str) -> bool:
    if path.is_symlink():
        return False
    if entry_type == "dirs":
        return path.is_dir()
    if entry_type == "files":
        return path.is_file()
    return path.is_dir() or path.is_file()


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: dict[str, Path] = {}
    for path in paths:
        result.setdefault(os.fspath(path.resolve()), path)
    return sorted(result.values(), key=lambda path: os.fspath(path))


def find_parents(source: Path, patterns: list[str], target: Path) -> list[Path]:
    matches = []
    for pattern in patterns:
        matches.extend(path for path in source.glob(pattern) if path.is_dir() and not path.is_symlink())
    return [path for path in unique_paths(matches) if path.resolve() != target]


def find_entries(
    parents: list[Path], patterns: list[str], entry_type: str, target: Path
) -> tuple[list[Path], list[Path]]:
    selected = []
    ignored = []
    for parent in parents:
        for pattern in patterns:
            for path in parent.glob(pattern):
                if path.resolve() == target:
                    ignored.append(path)
                elif matches_type(path, entry_type):
                    selected.append(path)
                else:
                    ignored.append(path)
    return unique_paths(selected), unique_paths(ignored)


def validate_no_nested_entries(entries: list[Path]) -> list[tuple[Path, Path]]:
    resolved = {path.resolve(): path for path in entries}
    nested = []
    for inner_resolved, inner in resolved.items():
        for ancestor in inner_resolved.parents:
            if ancestor in resolved:
                nested.append((resolved[ancestor], inner))
                break
    return nested


def renamed_target(target: Path, reserved: set[str], source_is_dir: bool) -> Path:
    if source_is_dir:
        base, suffix = target.name, ""
    else:
        base, suffix = target.stem, target.suffix
    number = 1
    while True:
        candidate = target.with_name(f"{base}__{number}{suffix}")
        key = os.fspath(candidate)
        if key not in reserved and not lexical_exists(candidate):
            return candidate
        number += 1


def build_plan(
    entries: list[Path], target_dir: Path, conflict: str
) -> tuple[list[Operation], list[Path], list[str]]:
    operations = []
    skipped = []
    errors = []
    reserved: set[str] = set()

    for source in entries:
        target = target_dir / source.name
        target_key = os.fspath(target)
        has_conflict = target_key in reserved or lexical_exists(target)

        if has_conflict and conflict == "error":
            errors.append(f"同名目标：{target}")
            continue
        if has_conflict and conflict == "skip":
            skipped.append(source)
            continue
        if has_conflict and conflict == "rename":
            target = renamed_target(target, reserved, source.is_dir())
            target_key = os.fspath(target)

        if source.resolve() == target.resolve():
            errors.append(f"源和目标相同：{source}")
            continue
        if source.resolve() in target_dir.parents or source.resolve() == target_dir:
            errors.append(f"目标目录位于待处理目录内部：{source} -> {target_dir}")
            continue

        reserved.add(target_key)
        operations.append(Operation(source, target))

    return operations, skipped, errors


def copy_entry(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, copy_function=shutil.copy2)
    else:
        shutil.copy2(source, target)


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    target = (args.target or source).expanduser().resolve()
    child_patterns = args.child_pattern or ["*"]

    if not source.is_dir():
        print(f"错误：源目录不存在或不是目录：{source}", file=sys.stderr)
        return 1
    if args.remove_empty_parents and args.mode != "move":
        print("错误：--remove-empty-parents 只能与 --mode move 一起使用", file=sys.stderr)
        return 1

    parents = find_parents(source, args.parent_pattern, target)
    if not parents:
        print("未找到符合 --parent-pattern 的父目录。")
        return 0

    entries, ignored = find_entries(parents, child_patterns, args.entry_type, target)
    nested = validate_no_nested_entries(entries)
    if nested:
        print("错误：匹配结果中同时包含父目录和它的子项，未执行任何操作：", file=sys.stderr)
        for outer, inner in nested:
            print(f"  {outer} 包含 {inner}", file=sys.stderr)
        return 2

    operations, skipped, errors = build_plan(entries, target, args.conflict)
    if errors:
        print("发现冲突或不安全路径，未执行任何操作：", file=sys.stderr)
        for message in errors:
            print(f"  {message}", file=sys.stderr)
        return 2

    state = "执行" if args.execute else "预览"
    print(
        f"{state}：父目录 {len(parents)} 个，计划{('移动' if args.mode == 'move' else '复制')} "
        f"{len(operations)} 项，冲突跳过 {len(skipped)} 项，类型/规则忽略 {len(ignored)} 项。"
    )
    if args.verbose:
        for operation in operations:
            print(f"  {operation.source} -> {operation.target}")

    if not args.execute:
        print("这是预览，没有修改数据；确认后添加 --execute 执行。")
        return 0

    target.mkdir(parents=True, exist_ok=True)
    completed = 0
    try:
        for operation in operations:
            if args.mode == "move":
                shutil.move(os.fspath(operation.source), os.fspath(operation.target))
            else:
                copy_entry(operation.source, operation.target)
            completed += 1
    except Exception as exc:
        print(f"执行中断：已完成 {completed}/{len(operations)} 项；错误：{exc}", file=sys.stderr)
        return 3

    removed = 0
    if args.remove_empty_parents:
        for parent in sorted(parents, key=lambda path: len(path.parts), reverse=True):
            try:
                parent.rmdir()
            except OSError:
                continue
            else:
                removed += 1

    print(f"完成：处理 {completed} 项，删除空父目录 {removed} 个。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
