"""
Parse board-grep expressions into parameterized SQLite queries.
Supports '&' (AND), '|' (OR), and grouped expressions '()'.
"""

from __future__ import annotations

from typing import Any, List, Tuple

_SPECIAL = "|&()"


class QueryError(ValueError):
    pass


def _tokenize(s: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    buf: List[str] = []

    def flush() -> None:
        w = "".join(buf).strip()
        buf.clear()
        if w:
            out.append(("WORD", w))

    for ch in s or "":
        if ch in _SPECIAL:
            flush()
            out.append((ch, ch))
        else:
            buf.append(ch)
    flush()
    return out


def parse(expr: str) -> Any:
    toks = _tokenize(expr)
    if not toks:
        raise QueryError("empty expression")
    pos = 0
    depth = 0

    def peek() -> str:
        return toks[pos][0] if pos < len(toks) else ""

    def parse_or() -> Any:
        nonlocal pos
        parts = [parse_and()]
        while peek() == "|":
            pos += 1
            parts.append(parse_and())
        return parts[0] if len(parts) == 1 else ("or", parts)

    def parse_and() -> Any:
        nonlocal pos
        parts = [parse_factor()]
        while peek() == "&":
            pos += 1
            parts.append(parse_factor())
        return parts[0] if len(parts) == 1 else ("and", parts)

    def parse_factor() -> Any:
        nonlocal pos, depth
        k = peek()
        if k == "(":
            pos += 1
            depth += 1
            if peek() == ")":
                raise QueryError("empty group '()'")
            inner = parse_or()
            if peek() != ")":
                raise QueryError("unbalanced '('")
            pos += 1
            depth -= 1
            return inner
        if k == "WORD":
            w = toks[pos][1]
            pos += 1
            return ("term", w)
        if k == ")":
            if depth > 0:
                raise QueryError("expected a term before ')'")
            raise QueryError("unexpected ')'")
        if k in ("&", "|"):
            raise QueryError(f"'{k}' needs a term on both sides")
        raise QueryError("expected a search term")

    node = parse_or()
    if pos != len(toks):
        raise QueryError("syntax error in query")
    return node


# Alias
parse_query = parse



def compile_sqlite_sql(node: Any) -> Tuple[str, List[Any]]:
    """Compile parsed AST to SQLite WHERE fragment and parameters with '?' placeholder."""
    if node[0] == "term":
        # Escape % and _ for LIKE
        escaped = node[1].replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return "content LIKE ? ESCAPE '\\'", [f"%{escaped}%"]

    joiner = " AND " if node[0] == "and" else " OR "
    frags: List[str] = []
    params: List[Any] = []
    for child in node[1]:
        f, p = compile_sqlite_sql(child)
        frags.append(f)
        params.extend(p)
    return "(" + joiner.join(frags) + ")", params
