#!/usr/bin/env python3
"""
Scan Test Smells - count the mechanically detectable test smells and ratchet them.

Companion to TEST-SMELLS.md (the full catalogue: Koskela *Effective Unit Testing* ch4-6,
Osherove *The Art of Unit Testing* ch8). This script only covers the smells an AST pass can
find reliably; the rest are judgment calls listed in the catalogue with `detection: judgment`.
Counts are *review prompts*, not verdicts - read each finding before acting on it.

Stdlib-only (ast + tokenize + tomllib). Same CLI shape as setup-ci-cd-pipeline's
check_skipped_tests.py: positional test dir, `--baseline` ratchet, `--update-baseline`,
`[tool.scan_test_smells]` in pyproject.toml. The ratchet (Core Rule 11) fails only when a
smell count goes UP versus the baseline, so existing debt is grandfathered but never grows.

Detectors (smell id -> what it flags):
    sleep              time.sleep / asyncio.sleep / bare sleep() inside a test (K 5.6)
    no-assert          test with no assert, pytest.raises/warns, assert*/verify*/check*/expect*
                       call, or pytest.fail (K 6.4 "tests that don't do anything")
    early-exit         `return` before the end of a test, pytest.skip(), self.skipTest() in a
                       body (K 6.7 conditional tests, 6.3 never-failing)
    platform-check     sys.platform / os.name / platform.system() etc. in a body (K 6.6)
    abs-path           /tmp, /var/tmp, /home/, /Users/ literal anywhere; a drive-letter literal
                       (C:\\...) only where it reaches open()/os.*/shutil.*/Path I/O (K 5.4)
    persistent-temp    NamedTemporaryFile(delete=False), mkdtemp, mkstemp (K 5.5)
    commented-test     real comment token holding `def test_...` (K 6.1)
    multi-mock-assert  assert_called*/assert_not_called/... on more than one distinct double in
                       one test (O 4.5 "one mock per test", O 8.2.7 overspecification)
    parametrize-no-ids parametrize with >=2 literal cases and no ids= / pytest.param(id=) (K 5.8)
    overprotective     `assert x is not None` immediately followed by an assert that uses x.attr
                       or x[...] (K 4.9)
    bitwise-assert     &, |, ^, ~, <<, >> inside an assert (K 4.3); set operands are skipped
    suboptimal-assert  `assert x == True/False/None` or `assert (a == b) is True`;
                       assertTrue/assertFalse/assertIs(<comparison>); assertEqual(x, True/False/None)
                       (PyNose "Suboptimal Assert": a wrapped comparison loses the operands, and
                       equality with True/False/None also passes for 1/0). assertIs(x, True/False/None)
                       is the strict form and is never flagged; the hint keeps a `!=`/Not negation

Suppress one finding with a trailing `# smell: ignore` or `# smell: ignore[sleep,abs-path]`
on the reported line. Prefer fixing; suppression is for deliberate, reviewed exceptions.

Exit codes: 0 ok (or report-only), 1 gate failed (baseline increased, or --strict with
findings), 2 usage / unreadable baseline.

Usage:
    python scan_test_smells.py tests/
    python scan_test_smells.py tests/ -v                       # list every finding
    python scan_test_smells.py tests/ --json
    python scan_test_smells.py tests/ --baseline .smell-baseline.json
    python scan_test_smells.py tests/ --baseline .smell-baseline.json --update-baseline
    python scan_test_smells.py tests/ --ignore parametrize-no-ids --exclude "golden/*"
    python scan_test_smells.py --list-smells

pyproject.toml:
    [tool.scan_test_smells]
    baseline = ".smell-baseline.json"   # used when --baseline is not given
    ignore   = ["parametrize-no-ids"]    # smell ids to skip entirely
    exclude  = ["golden/*", "conftest.py"]  # fnmatch globs relative to the scanned dir
"""

import argparse
import ast
import fnmatch
import io
import json
import re
import sys
import tokenize
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", ".mypy_cache"}

# smell id -> (one-line description, source section)
SMELLS = {
    "sleep": ("time.sleep/asyncio.sleep in a test", "Koskela 5.6 Sleeping snail"),
    "no-assert": ("test with no assertion", "Koskela 6.4 Shallow promises"),
    "early-exit": ("early return / pytest.skip() inside a test body", "Koskela 6.7 Conditional tests"),
    "platform-check": ("platform check inside a test body", "Koskela 6.6 Platform prejudice"),
    "abs-path": ("hardcoded absolute path literal", "Koskela 5.4 Crippling file path"),
    "persistent-temp": ("temp file/dir that outlives the test", "Koskela 5.5 Persistent temp files"),
    "commented-test": ("commented-out test definition", "Koskela 6.1 Commented-out tests"),
    "multi-mock-assert": ("assert_called* on >1 distinct double in one test", "Osherove 4.5 / 8.2.7"),
    "parametrize-no-ids": ("parametrize with >=2 cases and no ids", "Koskela 5.8 Parameterized mess"),
    "overprotective": ("`assert x is not None` guarding an assert on x.attr", "Koskela 4.9 Overprotective tests"),
    "bitwise-assert": ("bitwise operator inside an assert", "Koskela 4.3 Bitwise assertions"),
    "suboptimal-assert": (
        "assert compares to True/False/None, or assertTrue/False wraps a comparison",
        "PyNose (ASE 2021) Suboptimal Assert",
    ),
}

POSIX_ABS_PATH_RE = re.compile(r"^(?:/tmp(?:[/\\]|$)|/home/|/Users/|/var/tmp(?:[/\\]|$))")
DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
# Filesystem sinks for drive-letter literals (see t_abs_path).
FS_BUILTINS = {"open"}
FS_MODULES = {"shutil"}
FS_OS = {
    "open",
    "makedirs",
    "mkdir",
    "remove",
    "unlink",
    "rmdir",
    "removedirs",
    "rename",
    "renames",
    "replace",
    "listdir",
    "scandir",
    "walk",
    "stat",
    "lstat",
    "chdir",
    "chmod",
    "access",
    "startfile",
    "utime",
    "truncate",
    "link",
    "symlink",
}
FS_OSPATH = {
    "exists",
    "lexists",
    "isfile",
    "isdir",
    "islink",
    "ismount",
    "getsize",
    "getmtime",
    "getatime",
    "getctime",
    "samefile",
}
FS_PURE = {"copyfileobj", "get_terminal_size", "which"}
FS_PATH_METHODS = {
    "exists",
    "is_file",
    "is_dir",
    "open",
    "read_text",
    "read_bytes",
    "write_text",
    "write_bytes",
    "mkdir",
    "touch",
    "unlink",
    "rmdir",
    "iterdir",
    "glob",
    "rglob",
    "stat",
    "rename",
    "chmod",
}
PATH_BUILDERS = {
    "join",
    "Path",
    "PurePath",
    "WindowsPath",
    "PureWindowsPath",
    "PosixPath",
    "abspath",
    "normpath",
    "fspath",
    "str",
}
SUPPRESS_RE = re.compile(r"#\s*smell:\s*ignore(?:\[([^\]]*)\])?")
COMMENTED_TEST_RE = re.compile(r"^#+\s*(?:async\s+)?def\s+test_\w+")

ASSERT_CALL_PREFIXES = ("assert", "verify", "check", "expect")
RAISES_NAMES = {"pytest.raises", "pytest.warns", "pytest.deprecated_call", "raises", "warns", "pytest.fail"}
MOCK_ASSERT_NAMES = {
    "assert_called",
    "assert_called_once",
    "assert_called_with",
    "assert_called_once_with",
    "assert_any_call",
    "assert_has_calls",
    "assert_not_called",
    "assert_awaited",
    "assert_awaited_once",
    "assert_awaited_with",
    "assert_awaited_once_with",
    "assert_any_await",
    "assert_has_awaits",
    "assert_not_awaited",
}
PLATFORM_FUNCS = {"system", "platform", "machine", "uname", "release", "version", "win32_ver", "mac_ver"}
BITWISE_OPS = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)
NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def iter_test_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def dotted_name(node: ast.AST) -> str:
    """Best-effort dotted name: `pytest.mark.parametrize` / `time.sleep`; '' if not a plain chain."""
    if isinstance(node, ast.Call):
        node = node.func
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def root_name(node: ast.AST) -> str:
    """Leftmost Name id of an attribute/call/subscript chain, or ''."""
    while True:
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Subscript):
            node = node.value
        else:
            break
    return node.id if isinstance(node, ast.Name) else ""


def walk_same_scope(func):
    """Yield nodes inside `func` without descending into nested functions/classes/lambdas."""
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, NESTED_SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def docstring_ids(tree: ast.AST) -> set:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def find_test_functions(tree: ast.AST):
    """Functions named test* (module level or in classes), not nested inside another test."""
    found = []

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name.startswith("test") and not any(
                    dotted_name(d).rsplit(".", 1)[-1] == "fixture" for d in child.decorator_list
                ):
                    found.append(child)
                # helper functions are not descended into: their bodies are not tests
            elif isinstance(child, ast.ClassDef):
                visit(child)

    visit(tree)
    return found


# --------------------------------------------------------------------------- detectors
# Each detector yields (smell_id, lineno, detail). `func` detectors take a test function;
# `tree` detectors take the whole module.


def d_sleep(func):
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            name = dotted_name(node)
            if name in ("time.sleep", "asyncio.sleep", "sleep", "trio.sleep", "anyio.sleep"):
                yield "sleep", node.lineno, f"{name}() in test"


def d_no_assert(func):
    for node in ast.walk(func):
        if isinstance(node, ast.Assert):
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if dotted_name(item.context_expr) in RAISES_NAMES:
                    return
        if isinstance(node, ast.Call):
            name = dotted_name(node)
            if name in RAISES_NAMES:
                return
            leaf = name.rsplit(".", 1)[-1] if name else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
            if leaf.lower().lstrip("_").startswith(ASSERT_CALL_PREFIXES) and leaf not in ("check_output", "check_call"):
                return
        # `self.assertX(...)` where the receiver is not a plain name chain
        if isinstance(node, ast.Attribute) and node.attr.lower().lstrip("_").startswith(ASSERT_CALL_PREFIXES):
            return
    yield "no-assert", func.lineno, f"{func.name} has no assertion"


def d_early_exit(func):
    body_last = func.body[-1] if func.body else None
    for node in walk_same_scope(func):
        if isinstance(node, ast.Return) and node is not body_last:
            yield "early-exit", node.lineno, "return before the end of the test"
        elif isinstance(node, ast.Call):
            name = dotted_name(node)
            if name in ("pytest.skip", "skip") or name.endswith(".skipTest"):
                yield "early-exit", node.lineno, f"{name}() inside the body"


def d_platform(func):
    # body only: a skipif(sys.platform...) decorator is a visible gate (check_skipped_tests counts it)
    for node in (n for stmt in func.body for n in ast.walk(stmt)):
        if isinstance(node, ast.Attribute):
            name = dotted_name(node)
            if name in ("sys.platform", "os.name"):
                yield "platform-check", node.lineno, name
        if isinstance(node, ast.Call):
            name = dotted_name(node)
            head, _, leaf = name.rpartition(".")
            if head == "platform" and leaf in PLATFORM_FUNCS:
                yield "platform-check", node.lineno, f"{name}()"


def d_mock_asserts(func):
    roots = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Attribute) and node.attr in MOCK_ASSERT_NAMES:
            r = root_name(node.value)
            if r and r != "self":
                roots.setdefault(r, node.lineno)
    if len(roots) > 1:
        yield "multi-mock-assert", func.lineno, "asserted doubles: " + ", ".join(sorted(roots))


def _case_has_id(elt: ast.AST) -> bool:
    return (
        isinstance(elt, ast.Call)
        and dotted_name(elt).rsplit(".", 1)[-1] == "param"
        and any(kw.arg == "id" for kw in elt.keywords)
    )


def d_parametrize(func):
    for dec in func.decorator_list:
        if not isinstance(dec, ast.Call) or not dotted_name(dec).endswith("parametrize"):
            continue
        if any(kw.arg == "ids" for kw in dec.keywords):
            continue
        values = (
            dec.args[1] if len(dec.args) > 1 else next((kw.value for kw in dec.keywords if kw.arg == "argvalues"), None)
        )
        if not isinstance(values, (ast.List, ast.Tuple)):
            continue  # non-literal table: cannot verify statically, do not guess
        if len(values.elts) < 2:
            continue
        if all(_case_has_id(e) for e in values.elts):
            continue
        yield "parametrize-no-ids", dec.lineno, f"{len(values.elts)} cases without ids"


def _is_not_none_guard(stmt: ast.stmt) -> str:
    """Name guarded by `assert x is not None` / `assert x != None` / assertIsNotNone(x); else ''."""
    if isinstance(stmt, ast.Assert):
        t = stmt.test
        if (
            isinstance(t, ast.Compare)
            and len(t.ops) == 1
            and isinstance(t.left, ast.Name)
            and isinstance(t.ops[0], (ast.IsNot, ast.NotEq))
            and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value is None
        ):
            return t.left.id
    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr in ("assertIsNotNone", "assertNotNone")
            and call.args
            and isinstance(call.args[0], ast.Name)
        ):
            return call.args[0].id
    return ""


def _uses_deref(stmt: ast.stmt, name: str) -> bool:
    for node in ast.walk(stmt):
        if (
            isinstance(node, (ast.Attribute, ast.Subscript))
            and isinstance(node.value, ast.Name)
            and node.value.id == name
        ):
            return True
    return False


def d_overprotective(func):
    for node in ast.walk(func):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if not isinstance(stmts, list):
                continue
            for first, second in zip(stmts, stmts[1:], strict=False):
                name = _is_not_none_guard(first)
                if (
                    name
                    and (
                        isinstance(second, ast.Assert)
                        or (isinstance(second, ast.Expr) and isinstance(second.value, ast.Call))
                    )
                    and _uses_deref(second, name)
                ):
                    yield "overprotective", first.lineno, f"`{name} is not None` guards an assert on {name}"


def _has_set_operand(node: ast.BinOp) -> bool:
    for side in (node.left, node.right):
        if isinstance(side, (ast.Set, ast.SetComp)):
            return True
        if isinstance(side, ast.Call) and dotted_name(side) in ("set", "frozenset"):
            return True
    return False


def d_bitwise(func):
    for node in ast.walk(func):
        if not isinstance(node, ast.Assert):
            continue
        for sub in ast.walk(node.test):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, BITWISE_OPS) and not _has_set_operand(sub):
                yield "bitwise-assert", sub.lineno, f"operator {type(sub.op).__name__} in assert"
            elif isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.Invert):
                yield "bitwise-assert", sub.lineno, "operator Invert in assert"


def _is_singleton(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, bool))


# assertIs/assertIsNot(x, True/False/None) are the strict form, not a smell: assertTrue(x) would also pass for 1 or [0].
UNITTEST_EQ_ASSERTS = {"assertEqual", "assertNotEqual", "assertEquals", "assertNotEquals"}
# `is True` needs a real bool: numpy.True_ == True but is not True, so keep the equality for numpy values.
NUMPY_NOTE = "; for a numpy bool keep the equality and add `# smell: ignore[suboptimal-assert]`"


def _strict_hint(const: ast.Constant, negated: bool, unittest_name: str = "") -> str:
    """The identity check that replaces an equality against True/False/None, keeping its polarity."""
    v = const.value
    if unittest_name:
        if v is None:
            better = "assertIsNotNone(x)" if negated else "assertIsNone(x)"
        else:
            better = f"assertIsNot(x, {v!r})" if negated else f"assertIs(x, {v!r})"
        head = f"{unittest_name}(x, {v!r}): use {better}"
    else:
        head = f"`{'!=' if negated else '=='} {v!r}`: use `{'is not' if negated else 'is'} {v!r}`"
    return head + ("" if v is None else NUMPY_NOTE)


def d_suboptimal(func):
    for node in walk_same_scope(func):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
            t = node.test
            left, op, right = t.left, t.ops[0], t.comparators[0]
            if isinstance(op, (ast.Eq, ast.NotEq)) and (_is_singleton(left) or _is_singleton(right)):
                const = left if _is_singleton(left) else right
                yield "suboptimal-assert", node.lineno, _strict_hint(const, isinstance(op, ast.NotEq))
            elif (
                isinstance(op, (ast.Is, ast.IsNot))
                and isinstance(right, ast.Constant)
                and isinstance(right.value, bool)
                and isinstance(left, ast.Compare)
            ):
                yield "suboptimal-assert", node.lineno, f"comparison wrapped in `is {right.value!r}`"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
            name = node.func.attr
            if name in ("assertTrue", "assertFalse") and isinstance(node.args[0], ast.Compare):
                yield "suboptimal-assert", node.lineno, f"{name}(<comparison>): use assertEqual/assertIn/..."
            elif (
                name in ("assertIs", "assertIsNot")
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Compare)
                and _is_singleton(node.args[1])
            ):
                yield "suboptimal-assert", node.lineno, f"{name}(<comparison>, ...): use assertEqual/assertIn/..."
            elif (
                name in UNITTEST_EQ_ASSERTS
                and len(node.args) >= 2
                and (_is_singleton(node.args[0]) or _is_singleton(node.args[1]))
            ):
                const = node.args[1] if _is_singleton(node.args[1]) else node.args[0]
                yield "suboptimal-assert", node.lineno, _strict_hint(const, "Not" in name, name)


FUNC_DETECTORS = (
    d_sleep,
    d_no_assert,
    d_early_exit,
    d_platform,
    d_mock_asserts,
    d_parametrize,
    d_overprotective,
    d_bitwise,
    d_suboptimal,
)


def _is_fs_sink(call: ast.Call) -> bool:
    """True for a call that touches the filesystem: open(), os.*, shutil.*, Path(...).read_text()."""
    name = dotted_name(call)
    if name:
        head, _, leaf = name.rpartition(".")
        if not head:
            return leaf in FS_BUILTINS
        if head.split(".")[0] in FS_MODULES or head.endswith("path") and leaf in FS_OSPATH:
            return leaf not in FS_PURE
        if head == "os":
            return leaf in FS_OS
        if head in ("glob", "sqlite3", "io", "codecs"):
            return leaf in ("glob", "iglob", "connect", "open")
    func = call.func
    return isinstance(func, ast.Attribute) and func.attr in FS_PATH_METHODS


def _drive_literals(node: ast.AST):
    """Drive-letter literals reaching `node` through path-building only: joins, `/`, `+`, f-strings."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str) and DRIVE_PATH_RE.match(node.value):
            yield node
    elif isinstance(node, ast.Call):
        if dotted_name(node).rsplit(".", 1)[-1] in PATH_BUILDERS:
            for arg in [*node.args, *(kw.value for kw in node.keywords)]:
                yield from _drive_literals(arg)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
        yield from _drive_literals(node.left)
        yield from _drive_literals(node.right)
    elif isinstance(node, ast.JoinedStr):
        for value in node.values:
            yield from _drive_literals(value)
    elif isinstance(node, ast.FormattedValue):
        yield from _drive_literals(node.value)


def t_abs_path(tree):
    """POSIX temp/home literals are flagged anywhere. A drive-letter literal (`C:/TD/x.tox`) is
    flagged only where it reaches a filesystem call; elsewhere it is test data for path-handling
    code (a mocked `projectPath`, an expected string) and the test runs the same on any machine.
    A drive literal stored in a variable first is not traced."""
    skip = docstring_ids(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skip
            and POSIX_ABS_PATH_RE.match(node.value)
        ):
            yield "abs-path", node.lineno, repr(node.value[:40])
        elif isinstance(node, ast.Call) and _is_fs_sink(node):
            roots = [*node.args, *(kw.value for kw in node.keywords)]
            if isinstance(node.func, ast.Attribute):
                roots.append(node.func.value)  # Path("C:/x").read_text()
            for root in roots:
                for lit in _drive_literals(root):
                    yield "abs-path", lit.lineno, repr(lit.value[:40])


def t_temp(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        leaf = dotted_name(node).rsplit(".", 1)[-1]
        if leaf in ("mkdtemp", "mkstemp"):
            yield "persistent-temp", node.lineno, f"{leaf}() is never cleaned up automatically"
        elif leaf == "NamedTemporaryFile" and any(
            kw.arg == "delete" and isinstance(kw.value, ast.Constant) and kw.value.value is False
            for kw in node.keywords
        ):
            yield "persistent-temp", node.lineno, "NamedTemporaryFile(delete=False)"


def commented_tests(text: str):
    """(lineno, detail) for comment *tokens* that hold a test def; strings never match."""
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT and COMMENTED_TEST_RE.match(tok.string):
                yield tok.start[0], tok.string.strip()[:60]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return


# --------------------------------------------------------------------------- scan
def is_suppressed(line_text: str, smell: str) -> bool:
    m = SUPPRESS_RE.search(line_text)
    if not m:
        return False
    if m.group(1) is None:
        return True
    return smell in {s.strip() for s in m.group(1).split(",")}


def scan_file(path: Path, rel: str, ignore: set) -> tuple:
    """Return (findings, n_tests). findings: list of dicts."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(text, filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return [], 0
    lines = text.splitlines()
    raw = []  # (smell, lineno, detail, test_name)

    tests = find_test_functions(tree)
    for func in tests:
        for det in FUNC_DETECTORS:
            for smell, lineno, detail in det(func):
                raw.append((smell, lineno, detail, func.name))
    for smell, lineno, detail in t_abs_path(tree):
        raw.append((smell, lineno, detail, ""))
    for smell, lineno, detail in t_temp(tree):
        raw.append((smell, lineno, detail, ""))
    for lineno, detail in commented_tests(text):
        raw.append(("commented-test", lineno, detail, ""))

    findings = []
    seen = set()
    for smell, lineno, detail, test in raw:
        if smell in ignore:
            continue
        line_text = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        if is_suppressed(line_text, smell):
            continue
        key = (smell, lineno, detail)
        if key in seen:
            continue
        seen.add(key)
        findings.append({"smell": smell, "path": rel, "line": lineno, "test": test, "detail": detail})
    return findings, len(tests)


def load_config(root: Path) -> dict:
    """Read [tool.scan_test_smells] from pyproject.toml, if present. Never required."""
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return {}
    try:
        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data.get("tool", {}).get("scan_test_smells", {})


def compare_baseline(counts: dict, baseline_path: Path, update: bool) -> tuple:
    """Return (exit_code, status dict). Missing smell keys in the baseline count as 0."""
    status = {"path": str(baseline_path), "action": "compare", "increased": {}}
    payload = {"counts": counts, "total": sum(counts.values())}
    if update or not baseline_path.exists():
        baseline_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        status["action"] = "updated" if update else "created"
        return 0, status
    try:
        base = json.loads(baseline_path.read_text(encoding="utf-8"))["counts"]
    except (json.JSONDecodeError, KeyError, OSError, TypeError) as exc:
        status["action"] = "error"
        status["error"] = str(exc)
        return 2, status
    for smell, n in counts.items():
        if n > base.get(smell, 0):
            status["increased"][smell] = [base.get(smell, 0), n]
    status["improved"] = {s: [b, counts.get(s, 0)] for s, b in base.items() if counts.get(s, 0) < b}
    return (1 if status["increased"] else 0), status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test_dir", nargs="?", default="tests", help="Directory to scan (default: tests)")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline JSON; fail only if any smell count increased (ratchet mode)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write the current counts to --baseline instead of gating on them",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="SMELL",
        help="Smell id to skip (repeatable; also [tool.scan_test_smells].ignore)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="fnmatch glob (relative to the scan dir) to skip (repeatable)",
    )
    parser.add_argument("--strict", action="store_true", help="Exit 1 if there is any finding (no baseline)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text")
    parser.add_argument("-v", "--verbose", action="store_true", help="List every finding (file:line)")
    parser.add_argument("--list-smells", action="store_true", help="Print the detector ids and exit")
    args = parser.parse_args()

    if args.list_smells:
        for sid, (desc, src) in SMELLS.items():
            print(f"{sid:20s} {desc}  [{src}]")
        return 0

    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        print(f"error: not a directory: {test_dir}", file=sys.stderr)
        return 2

    config = load_config(test_dir.parent if test_dir.name in ("tests", "test") else test_dir)
    ignore = set(args.ignore) | set(config.get("ignore", []))
    unknown = ignore - set(SMELLS)
    if unknown:
        print(f"error: unknown smell id(s): {', '.join(sorted(unknown))} (see --list-smells)", file=sys.stderr)
        return 2
    exclude = list(args.exclude) + list(config.get("exclude", []))
    baseline = args.baseline
    if baseline is None and config.get("baseline"):
        baseline = Path(config["baseline"])

    findings, n_tests = [], 0
    for path in iter_test_files(test_dir):
        rel = path.relative_to(test_dir).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in exclude):
            continue
        f, n = scan_file(path, rel, ignore)
        findings.extend(f)
        n_tests += n

    counts = {sid: 0 for sid in SMELLS if sid not in ignore}
    for f in findings:
        counts[f["smell"]] += 1
    total = sum(counts.values())

    code, status = 0, None
    if baseline is not None:
        code, status = compare_baseline(counts, baseline, args.update_baseline)
    elif args.strict and total:
        code = 1

    if args.json:
        print(
            json.dumps(
                {
                    "dir": str(test_dir),
                    "tests": n_tests,
                    "counts": counts,
                    "total": total,
                    "findings": findings,
                    "baseline": status,
                    "exit_code": code,
                },
                indent=2,
            )
        )
        return code

    print(f"Test-smell scan: {test_dir}")
    print("=" * 70)
    print(f"Test functions scanned : {n_tests}")
    for sid, n in counts.items():
        print(f"{sid:20s}   : {n:4d}   {SMELLS[sid][1]}")
    print(f"{'TOTAL':20s}   : {total:4d}")
    if args.verbose and findings:
        print()
        for f in sorted(findings, key=lambda x: (x["path"], x["line"])):
            where = f" ({f['test']})" if f["test"] else ""
            print(f"  {f['path']}:{f['line']}: {f['smell']}{where}: {f['detail']}")
    elif findings:
        by_file = {}
        for f in findings:
            by_file[f["path"]] = by_file.get(f["path"], 0) + 1
        print("\nTop files (use -v for every finding):")
        for name, n in sorted(by_file.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {name}: {n}")

    if status is not None:
        print()
        if status["action"] in ("created", "updated"):
            print(
                f"Baseline {status['action']}: {status['path']} (total={total})"
                + ("  Nothing to compare against on this run." if status["action"] == "created" else "")
            )
        elif status["action"] == "error":
            print(f"error: could not read baseline {status['path']}: {status['error']}", file=sys.stderr)
        elif status["increased"]:
            for smell, (was, now) in sorted(status["increased"].items()):
                print(f"FAIL: {smell} increased {was} -> {now}")
            print("Fix the new findings, or rerun with --update-baseline if the increase was reviewed.")
        else:
            print("OK: no smell count increased.")
            if status.get("improved"):
                print(
                    "Counts dropped for: "
                    + ", ".join(sorted(status["improved"]))
                    + " - run --update-baseline to lock in the improvement."
                )
    elif args.strict and total:
        print("\nFAIL (--strict): findings present.")
    return code


if __name__ == "__main__":
    sys.exit(main())
