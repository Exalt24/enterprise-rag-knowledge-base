"""Mutation check: break the guarded code one way at a time and require a real test failure each time.

A test that stays green while the thing it guards is broken is worth nothing, so each mutant below is a small change
to the source, found by walking the syntax tree (never by line number or text), applied to a COPY of the code in a
temporary directory (the working tree is never touched), and then the eval tests are run against the copy. A mutant is
CAUGHT only when pytest exits with code 1 (a test failed). Exit code 0 means the mutant SURVIVED (a weak test); any
other code means the mutant did not run (a collection error), which is not proof of anything and counts as a failure
of the check. A mutant whose target cannot be found also fails the check: it means the code moved and this file is stale.

    python eval/mutation_check.py                 run every mutant
    python eval/mutation_check.py --list          list them
    python eval/mutation_check.py --only bm25-branch-removed
    python eval/mutation_check.py --skip-mcp      leave out the MCP server mutants (they need requirements-mcp.txt)
    python eval/mutation_check.py --only-ingest   only the ingestion, source view and Hatchet task mutants
    python eval/mutation_check.py --skip-ingest   leave those out (the CI job that runs them is separate)
    python eval/mutation_check.py --skip-hatchet  leave out the hatchet-* mutants (they need requirements-hatchet.txt)

The MCP server mutants break mcp_server/server.py and run only the server tests; each must fail a named test.
The workbook loader mutants (xlsx-*) break app/services/document_parser.py and run only tests/test_document_parser_xlsx.py.
The ingestion mutants (guard-*, pipeline-*, run-store-*, identity-*, store-*, view-*, route-*) break the archive guard, the
archive rules, the status files, the chunk lookup and the two archive routes, and run only the tests that guard them. The
hatchet-* mutants break the Hatchet task definitions and run tests/test_hatchet_tasks.py (they need requirements-hatchet.txt).
The documentation mutants (readme-*, docs-*) edit a number or a date in the README or docs/evaluation.md and run only the
tests that read them: a count written by hand and left stale must fail tests/test_eval_readme_counts.py, and a claim that
stopped being true (the design notes saying there is no CI test run) must fail tests/test_eval_docs_not_stale.py.
"""
import argparse
import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

BACKEND = Path(__file__).resolve().parents[1]
ADVANCED = "app/services/advanced_retrieval.py"
HARNESS = "eval/harness.py"
PARSER = "app/services/document_parser.py"

# Cheapest first would be test_hybrid_ranking, but the eval tests run first (the generated class tests before the older
# ones) so the report shows which of the NEW tests catch each mutant; the stubbed ranking test only gets a turn when they all stayed green.
INVOCATIONS = [
    [
        "tests/test_eval_baseline_changes.py",
        "tests/test_eval_docs_not_stale.py",
        "tests/test_eval_fallback_cause.py",
        "tests/test_eval_readme_claims.py",
        "tests/test_eval_readme_counts.py",
        "tests/test_eval_labels.py",
        "tests/test_eval_fallback_counter.py",
        "tests/test_eval_harness.py",
        "tests/test_eval_baseline.py",
        "tests/test_eval_questions.py",
    ],
    ["tests/test_hybrid_ranking.py"],
]
# The MCP server mutants run only the server tests (they need the SDK: pip install -r requirements-mcp.txt).
MCP_SERVER = "mcp_server/server.py"
MCP_RUNS = (("tests/test_mcp_server.py", "tests/test_mcp_server_classes.py", "tests/test_mcp_server_stdio.py"),)
# The documentation mutants edit TEXT (the README, docs/evaluation.md), not code, and run only the tests that read the docs.
DOC_RUNS = (("tests/test_eval_readme_counts.py", "tests/test_eval_readme_claims.py", "tests/test_eval_docs_not_stale.py"),)
# The workbook loader mutants break document_parser.py and run only the loader tests.
XLSX_RUNS = (("tests/test_document_parser_xlsx.py",),)
README = "../README.md"  # relative to backend/, so the same path works in the real tree and in the temporary copy
EVALUATION_DOC = "../docs/evaluation.md"
DESIGN_NOTES = "../docs/SYSTEM-KNOWLEDGE.md"


class TargetNotFound(Exception):
    pass


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise TargetNotFound(f"function {name} found {len(found)} times")
    return found[0]


def _only(nodes: list, what: str):
    if len(nodes) != 1:
        raise TargetNotFound(f"{what}: expected exactly one match, found {len(nodes)}")
    return nodes[0]


def _names(node: ast.AST) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


# ---------------------------------------------------------------- the mutations (each edits the tree in place)


def bm25_branch_removed(tree):
    """hybrid_search gets no BM25 results at all."""
    fn = _function(tree, "hybrid_search")
    assign = _only(
        [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "bm25_results" for t in n.targets)],
        "assignment of bm25_results",
    )
    assign.value = ast.List(elts=[], ctx=ast.Load())


def bm25_bonus_sign_flipped(tree):
    """The keyword bonus for a chunk found by both searches is subtracted instead of added."""
    fn = _function(tree, "hybrid_search")
    aug = _only(
        [n for n in ast.walk(fn) if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add) and "bm25_weight" in _names(n.value)],
        "bonus augmented assignment",
    )
    aug.op = ast.Sub()


def ranking_inverted_again(tree):
    """The original bug: the cosine similarity is turned into 1 / (1 + score), so the worst chunks rank first."""
    fn = _function(tree, "hybrid_search")
    mult = _only(
        [n for n in ast.walk(fn) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) and _names(n) == {"vector_sim", "vector_weight"}],
        "vector_sim * vector_weight",
    )
    inverted = ast.BinOp(
        left=ast.Constant(1),
        op=ast.Div(),
        right=ast.BinOp(left=ast.Constant(1), op=ast.Add(), right=ast.Name(id="vector_sim", ctx=ast.Load())),
    )
    ast.copy_location(inverted, mult)
    _replace(tree, mult, inverted)


def fallback_counter_never_increments(tree):
    """The fallback branch swallows the error and stops counting it."""
    fn = _function(tree, "hybrid_search")
    handler = _only([n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)], "except handler of hybrid_search")
    aug = _only(
        [n for n in handler.body if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Attribute) and n.target.attr == "hybrid_fallback_count"],
        "counter increment",
    )
    handler.body[handler.body.index(aug)] = ast.Pass()


def threshold_comparison_flipped(tree):
    """A question is declined when its best score is ABOVE the threshold instead of below."""
    fn = _function(tree, "abstains")
    cmp_ = _only([n for n in ast.walk(fn) if isinstance(n, ast.Compare)], "comparison in abstains")
    cmp_.ops = [ast.Gt()]


def threshold_boundary_loosened(tree):
    """A score exactly on the threshold is declined (< becomes <=)."""
    fn = _function(tree, "abstains")
    cmp_ = _only([n for n in ast.walk(fn) if isinstance(n, ast.Compare)], "comparison in abstains")
    cmp_.ops = [ast.LtE()]


def must_contain_matching_negated(tree):
    """A chunk counts as relevant when it does NOT hold the phrase."""
    fn = _function(tree, "phrase_in_text")
    cmp_ = _only([n for n in ast.walk(fn) if isinstance(n, ast.Compare) and isinstance(n.ops[0], ast.In)], "phrase in text")
    cmp_.ops = [ast.NotIn()]


def must_contain_empty_phrase_matches(tree):
    """An empty phrase (an unanswerable label) is allowed to match every chunk."""
    fn = _function(tree, "phrase_in_text")
    ret = _only([n for n in ast.walk(fn) if isinstance(n, ast.Return)], "return in phrase_in_text")
    boolop = ret.value
    if not (isinstance(boolop, ast.BoolOp) and isinstance(boolop.op, ast.And)):
        raise TargetNotFound("phrase_in_text no longer has the shape `bool(phrase) and phrase in text`")
    ret.value = boolop.values[-1]


def baseline_tolerance_ignored(tree):
    """A pinned number can fall by any amount without the check noticing."""
    fn = _function(tree, "compare_to_baseline")
    cmp_ = _only(
        [n for n in ast.walk(fn) if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == "fall" and isinstance(n.ops[0], ast.Gt)],
        "fall > tolerance",
    )
    cmp_.comparators = [ast.Constant(1000)]


def baseline_rise_flagged(tree):
    """A number that ROSE is judged like a fall (the size of the change is taken without its sign), so an improvement fails."""
    fn = _function(tree, "compare_to_baseline")
    assign = _only(
        [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "fall" for t in n.targets)],
        "assignment of fall",
    )
    assign.value = ast.Call(func=ast.Name(id="abs", ctx=ast.Load()), args=[assign.value], keywords=[])


def baseline_question_hash_not_compared(tree):
    """An edit to questions.jsonl or calibration.jsonl is no longer noticed, so a label fix reads as a retrieval change."""
    fn = _function(tree, "compare_to_baseline")
    cond = _only(
        [n for n in ast.walk(fn) if isinstance(n, ast.If) and any(isinstance(s, ast.Name) and s.id == "key" for s in ast.walk(n.test))],
        "question hash comparison",
    )
    cond.test = ast.Constant(False)


def baseline_fallen_back_method_still_judged(tree):
    """A method that fell back to vector-only is judged against the baseline anyway, so one fault prints several lines."""
    fn = _function(tree, "compare_to_baseline")
    cond = _only(
        [n for n in ast.walk(fn) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.ops[0], ast.In)
         and isinstance(n.test.comparators[0], ast.Name) and n.test.comparators[0].id == "fell_back"],
        "method in fell_back",
    )
    cond.test = ast.Constant(False)


def fallback_cause_never_kept(tree):
    """The fallback branch counts the failure and drops its cause, so the CI log has a count and no reason."""
    fn = _function(tree, "hybrid_search")
    handler = _only([n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)], "except handler of hybrid_search")
    cond = _only(
        [n for n in handler.body if isinstance(n, ast.If) and "hybrid_fallback_cause" in ast.dump(n.test)],
        "cause assignment guard",
    )
    handler.body[handler.body.index(cond)] = ast.Pass()


def fallback_cause_overwritten_by_later_faults(tree):
    """The cause keeps the LAST exception instead of the first, so a later, different fault hides the one that started it."""
    fn = _function(tree, "hybrid_search")
    handler = _only([n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)], "except handler of hybrid_search")
    cond = _only(
        [n for n in handler.body if isinstance(n, ast.If) and "hybrid_fallback_cause" in ast.dump(n.test)],
        "cause assignment guard",
    )
    handler.body[handler.body.index(cond) : handler.body.index(cond) + 1] = cond.body


def readme_table_drops_recall(tree):
    """The generated README table loses its recall columns, so a multi-document hit looks like everything was found."""
    assign = _only(
        [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TABLE_METRICS" for t in n.targets)],
        "TABLE_METRICS",
    )
    assign.value.elts = [e for e in assign.value.elts if not (isinstance(e.elts[0], ast.Constant) and str(e.elts[0].value).startswith("recall"))]


def bootstrap_upper_percentile_moved(tree):
    """The upper bound of the 95% interval is read from the 95th percentile instead of the 97.5th."""
    fn = _function(tree, "percentile_bounds")
    const = _only([n for n in ast.walk(fn) if isinstance(n, ast.Constant) and n.value == 0.975], "the 0.975 index")
    const.value = 0.95


def bootstrap_lower_percentile_moved(tree):
    """The lower bound of the 95% interval is read from the 5th percentile instead of the 2.5th."""
    fn = _function(tree, "percentile_bounds")
    const = _only([n for n in ast.walk(fn) if isinstance(n, ast.Constant) and n.value == 0.025], "the 0.025 index")
    const.value = 0.05


def eval_chunking_follows_the_environment(tree):
    """The eval takes its chunk size and overlap from the app settings again, so CHUNK_SIZE in the environment moves it."""
    fn = _function(tree, "chunk_corpus")
    call = _only(
        [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "chunk_documents"],
        "TextChunker.chunk_documents call",
    )
    call.keywords = []


def label_validation_skipped(tree):
    """Labels are no longer checked when the questions load, so a wrong one dies later inside the metrics."""
    fn = _function(tree, "load_questions")
    cond = _only([n for n in ast.walk(fn) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "validate"], "if validate")
    cond.test = ast.Constant(False)


def baseline_chunk_hash_not_compared(tree):
    """A change in the chunk texts is no longer noticed."""
    fn = _function(tree, "compare_to_baseline")
    cond = _only(
        [n for n in ast.walk(fn) if isinstance(n, ast.If) and any(isinstance(s, ast.Constant) and s.value == "chunk_hash" for s in ast.walk(n.test))],
        "chunk hash comparison",
    )
    cond.test = ast.Constant(False)


# ---------------------------------------------------------------- the MCP server mutants (target mcp_server/server.py)


def _splice(tree: ast.AST, old: ast.AST, new_nodes: list) -> None:
    """Replace one statement by several (or none) in whichever body list holds it."""
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if isinstance(value, list) and any(item is old for item in value):
                i = next(i for i, item in enumerate(value) if item is old)
                value[i : i + 1] = new_nodes
                return
    raise TargetNotFound("could not splice the statement")


def _redirect_withs(root: ast.AST) -> list:
    """Every `with contextlib.redirect_stdout(...)` statement under root."""
    found = []
    for n in ast.walk(root):
        if isinstance(n, ast.With) and any(
            isinstance(item.context_expr, ast.Call) and getattr(item.context_expr.func, "attr", "") == "redirect_stdout" for item in n.items
        ):
            found.append(n)
    return found


def mcp_import_time_stdout_redirect_removed(tree):
    """The services are imported with stdout still pointing at the protocol channel, so their print calls corrupt it."""
    top = _only([n for n in tree.body if isinstance(n, ast.With) and n in _redirect_withs(tree)], "module level redirect around the service imports")
    _splice(tree, top, top.body)


def mcp_tool_call_stdout_redirect_removed(tree):
    """Tool calls run with stdout pointing at the protocol channel, so the services' progress lines go to it."""
    fn = _function(tree, "_stdout_to_stderr")
    inner = _only(_redirect_withs(fn), "redirect inside the tool wrapper")
    _splice(tree, inner, inner.body)


def mcp_citation_id_built_from_the_wrong_field(tree):
    """The chunk number in a citation id comes from total_chunks, so an id names a chunk that is not the one retrieved."""
    fn = _function(tree, "_chunk_id")
    const = _only([n for n in ast.walk(fn) if isinstance(n, ast.Constant) and n.value == "chunk_index"], "the chunk_index lookup")
    const.value = "total_chunks"


def mcp_failed_generation_returned_as_an_answer(tree):
    """A generation that failed (model_used "none") is returned as a normal answer, error text and all."""
    fn = _function(tree, "ask_question")
    cond = _only(
        [n for n in ast.walk(fn) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and any(isinstance(c, ast.Constant) and c.value == "none" for c in n.test.comparators)],
        'the model_used == "none" check',
    )
    _splice(tree, cond, [])


def mcp_write_tool_registered(tree):
    """A tool that deletes a document is registered next to the read-only ones."""
    extra = ast.parse(
        "@mcp.tool()\n"
        "def delete_document(file_name: str) -> dict:\n"
        "    return file_management.delete_document(file_name)\n"
    ).body[0]
    main = _function(tree, "main")
    _splice(tree, main, [extra, main])


def mcp_read_only_hint_dropped(tree):
    """search_documents is annotated as not read-only, so a client would ask for confirmation or refuse it."""
    annotations = sorted(
        (n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "ToolAnnotations"), key=lambda n: n.lineno
    )
    if not annotations:
        raise TargetNotFound("no ToolAnnotations call")
    for k in annotations[0].keywords:
        if k.arg == "read_only_hint":
            k.value = ast.Constant(False)


def mcp_limit_bound_removed(tree):
    """A limit of 0 or 1000 is no longer refused."""
    fn = _function(tree, "search_documents")
    cond = _only([n for n in ast.walk(fn) if isinstance(n, ast.If) and "MAX_LIMIT" in ast.dump(n.test)], "limit bound check")
    _splice(tree, cond, [])


def mcp_query_length_cap_removed(tree):
    """A query of any length is accepted."""
    fn = _function(tree, "_text_argument")
    cond = _only([n for n in ast.walk(fn) if isinstance(n, ast.If) and "MAX_QUERY_CHARS" in ast.dump(n.test)], "length check")
    _splice(tree, cond, [])


def mcp_document_filter_ignored(tree):
    """The document argument is accepted and then not applied, so hits come from every document."""
    fn = _function(tree, "search_documents")
    comp = _only(
        [n for n in ast.walk(fn) if isinstance(n, ast.ListComp) and n.generators[0].ifs and "document" in ast.dump(n.generators[0].ifs[0])],
        "filtering comprehension",
    )
    comp.generators[0].ifs = [ast.Constant(True)]


# ---------------------------------------------------------------- the documentation mutants (each edits the TEXT of a doc)


def _bump_first(text: str, pattern: str) -> str:
    """Add one to the number held by group 1 of the first match of `pattern`."""
    found = re.search(pattern, text)
    if not found:
        raise TargetNotFound(f"no sentence matches {pattern!r}: the doc was reworded, update the mutant")
    return text[: found.start(1)] + str(int(found.group(1)) + 1) + text[found.end(1) :]


def readme_chunk_count_stale(text: str) -> str:
    """The README still gives the chunk count from before a corpus edit (one fewer than the run produces)."""
    return _bump_first(text, r"They give (\d+) chunks")


def readme_question_total_edited_by_one(text: str) -> str:
    """The README's question total is off by one."""
    return _bump_first(text, r"The (\d+) questions in")


def readme_category_count_edited_by_one(text: str) -> str:
    """The README's count of unanswerable questions is off by one."""
    return _bump_first(text, r"and (\d+) unanswerable\.")


def readme_n_column_edited_by_one(text: str) -> str:
    """The n cell of the first Table 1 row is off by one."""
    return _bump_first(text, r"\| vector \| direct \| (\d+) \|")


def readme_real_model_date_removed(text: str) -> str:
    """Table 2 (the real-model run CI cannot repeat) loses the date of its run."""
    edited, n = re.subn(r"(\*\*Table 2, real model \(local run) on \d{4}-\d{2}-\d{2}\)", r"\1)", text, count=1)
    if not n:
        raise TargetNotFound("the Table 2 lead-in no longer carries a run date to remove")
    return edited


def evaluation_doc_calibration_count_edited_by_one(text: str) -> str:
    """docs/evaluation.md gives the calibration set size off by one."""
    return _bump_first(text, r"(\d+) calibration questions")


def design_notes_claim_there_is_no_ci(text: str) -> str:
    """The design notes' Testing section says again that there is no CI test run, which CI has long since contradicted."""
    marker = "\n## What I would change"
    if text.count(marker) != 1:
        raise TargetNotFound("docs/SYSTEM-KNOWLEDGE.md no longer has exactly one '## What I would change' heading to anchor on")
    return text.replace(marker, "\nThere is no CI test run.\n" + marker, 1)


def _replace(tree: ast.AST, old: ast.AST, new: ast.AST) -> None:
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if value is old:
                setattr(parent, field, new)
                return
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if item is old:
                        value[i] = new
                        return
    raise TargetNotFound("could not replace node")


def _xlsx_calls(fn: ast.AST, name: str) -> list:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == name]


def _if_testing(fn: ast.AST, predicate) -> ast.If:
    return _only([n for n in ast.walk(fn) if isinstance(n, ast.If) and predicate(n.test)], "the guarded if statement")


def xlsx_formulas_read_as_text(tree):
    """The workbook is opened without data_only, so a formula comes back as its text and not as the value Excel saved."""
    fn = _function(tree, "_parse_xlsx")
    kw = _only([k for c in _xlsx_calls(fn, "load_workbook") for k in c.keywords if k.arg == "data_only"], "data_only argument")
    kw.value = ast.Constant(False)


def xlsx_password_signature_ignored(tree):
    """A password protected workbook (the OLE container) is no longer recognised and falls through to the generic error."""
    fn = _function(tree, "_xlsx_check_container")
    cond = _if_testing(fn, lambda t: isinstance(t, ast.Compare) and "_OLE_MAGIC" in ast.unparse(t))
    cond.test = ast.Constant(False)


def xlsx_encrypted_entries_ignored(tree):
    """A zip whose entries carry the encrypted flag is no longer refused as password protected."""
    fn = _function(tree, "_xlsx_check_container")
    cond = _if_testing(fn, lambda t: "flag_bits" in ast.unparse(t))
    _splice(tree, cond, [])


def xlsx_unpacked_size_limit_removed(tree):
    """A workbook that unpacks to a huge size is opened anyway."""
    fn = _function(tree, "_xlsx_check_container")
    cond = _if_testing(fn, lambda t: "file_size" in ast.unparse(t))
    _splice(tree, cond, [])


def xlsx_row_limit_removed(tree):
    """No limit on the rows read from a workbook."""
    fn = _function(tree, "_parse_xlsx")
    cond = _if_testing(fn, lambda t: isinstance(t, ast.Compare) and "rows_read" in ast.unparse(t))
    _splice(tree, cond, [])


def xlsx_row_limit_boundary_loosened(tree):
    """A workbook with exactly the limit of rows is refused (> becomes >=)."""
    fn = _function(tree, "_parse_xlsx")
    cond = _if_testing(fn, lambda t: isinstance(t, ast.Compare) and "rows_read" in ast.unparse(t))
    cond.test.ops = [ast.GtE()]


def xlsx_column_limit_removed(tree):
    """A sheet wider than the column limit is cut to the limit without saying so."""
    fn = _function(tree, "_parse_xlsx")
    cond = _if_testing(fn, lambda t: isinstance(t, ast.BoolOp) and "XLSX_MAX_COLUMNS" in ast.unparse(t))
    _splice(tree, cond, [])


def xlsx_column_overflow_never_read(tree):
    """The row is read only up to the column limit, so a cell beyond it is never seen and the sheet is never refused."""
    fn = _function(tree, "_parse_xlsx")
    kw = _only([k for c in _xlsx_calls(fn, "iter_rows") for k in c.keywords if k.arg == "max_col"], "max_col argument")
    kw.value = kw.value.left  # cls.XLSX_MAX_COLUMNS + 1  ->  cls.XLSX_MAX_COLUMNS


def xlsx_sheet_limit_removed(tree):
    """No limit on the number of sheets."""
    fn = _function(tree, "_parse_xlsx")
    cond = _if_testing(fn, lambda t: isinstance(t, ast.Compare) and "XLSX_MAX_SHEETS" in ast.unparse(t))
    _splice(tree, cond, [])


def xlsx_hidden_sheets_read(tree):
    """Hidden sheets are read like the visible ones."""
    fn = _function(tree, "_parse_xlsx")
    comp = _only([n for n in ast.walk(fn) if isinstance(n, ast.ListComp) and "sheet_state" in ast.unparse(n)], "the visible sheet filter")
    comp.generators[0].ifs = []


def xlsx_row_numbers_start_at_zero(tree):
    """Row numbers in the cell range are one too low."""
    fn = _function(tree, "_parse_xlsx")
    call = _only([c for c in _xlsx_calls(fn, "enumerate") if any(k.arg == "start" for k in c.keywords)], "enumerate over the rows")
    _only([k for k in call.keywords if k.arg == "start"], "start argument").value = ast.Constant(0)


def xlsx_location_without_the_sheet(tree):
    """The citation location is the bare cell range, with no sheet name."""
    fn = _function(tree, "_xlsx_block")
    entry = _only([(k, v) for d in ast.walk(fn) if isinstance(d, ast.Dict) for k, v in zip(d.keys, d.values) if isinstance(k, ast.Constant) and k.value == "location"], "the location entry")
    for d in ast.walk(fn):
        if isinstance(d, ast.Dict) and any(v is entry[1] for v in d.values):
            d.values[d.values.index(entry[1])] = ast.Name(id="cell_range", ctx=ast.Load())


def xlsx_block_row_cap_removed(tree):
    """A block may hold any number of rows."""
    fn = _function(tree, "_parse_xlsx")
    cmp_ = _only([n for n in ast.walk(fn) if isinstance(n, ast.Compare) and "XLSX_BLOCK_ROWS" in ast.unparse(n)], "the block row comparison")
    _replace(tree, cmp_, ast.Constant(False))


def xlsx_block_char_cap_removed(tree):
    """A block may grow past the chunker's size."""
    fn = _function(tree, "_parse_xlsx")
    cmp_ = _only([n for n in ast.walk(fn) if isinstance(n, ast.Compare) and "XLSX_BLOCK_CHARS" in ast.unparse(n)], "the block size comparison")
    _replace(tree, cmp_, ast.Constant(False))


def xlsx_block_preamble_not_counted(tree):
    """The sheet, columns and range lines are left out of the block size, so a block comes out over the chunker's size."""
    fn = _function(tree, "_xlsx_preamble_chars")
    ret = _only([n for n in ast.walk(fn) if isinstance(n, ast.Return)], "return")
    ret.value = ast.Constant(0)


def xlsx_empty_workbook_accepted(tree):
    """A workbook with no text in it returns no documents instead of being refused."""
    fn = _function(tree, "_parse_xlsx")
    cond = _if_testing(fn, lambda t: isinstance(t, ast.UnaryOp) and "documents" in ast.unparse(t))
    _splice(tree, cond, [])


def xlsx_header_row_not_detected(tree):
    """The first row is never taken as the column headers."""
    fn = _function(tree, "_parse_xlsx")
    call = _only(_xlsx_calls(fn, "all"), "the all() header test")
    _replace(tree, call, ast.Constant(False))


def xlsx_blank_rows_not_skipped(tree):
    """A row with nothing in it is treated like a row with data."""
    fn = _function(tree, "_parse_xlsx")
    cond = _if_testing(fn, lambda t: isinstance(t, ast.UnaryOp) and "filled" in ast.unparse(t))
    _splice(tree, cond, [])


def xlsx_long_cells_not_cut(tree):
    """A cell of any length goes into the text whole."""
    fn = _function(tree, "_xlsx_cell_text")
    cond = _if_testing(fn, lambda t: isinstance(t, ast.Compare) and "XLSX_CELL_CHARS" in ast.unparse(t))
    _splice(tree, cond, [])


def xlsx_extension_not_supported(tree):
    """The parser stops listing .xlsx as a supported format."""
    assign = _only(
        [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "SUPPORTED_EXTENSIONS" for t in n.targets)],
        "SUPPORTED_EXTENSIONS",
    )
    assign.value.elts = [e for e in assign.value.elts if not (isinstance(e, ast.Constant) and e.value == ".xlsx")]


# ---------------------------------------------------------------- the ingestion and source view mutants
# Built from a few shapes (drop an `if`, force its test, flip a comparison, drop a keyword) so each guard is one short
# line below. Every mutant is found by walking the syntax tree, like the ones above, and fails the check when its target
# is not exactly one place in the source.


def _any_function(tree: ast.AST, name: str):
    found = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(found) != 1:
        raise TargetNotFound(f"function {name} found {len(found)} times")
    return found[0]


def _if_with(fn: ast.AST, needle: str) -> ast.If:
    return _only([n for n in ast.walk(fn) if isinstance(n, ast.If) and needle in ast.unparse(n.test)], f"if statement testing {needle!r}")


def _documented(doc: str, apply: Callable) -> Callable:
    apply.__doc__ = doc
    return apply


def drop_if(function: str, needle: str, doc: str) -> Callable:
    """Remove the `if` whose test mentions `needle`, body and all."""

    def apply(tree):
        _splice(tree, _if_with(_any_function(tree, function), needle), [])

    return _documented(doc, apply)


def force_if(function: str, needle: str, value: bool, doc: str) -> Callable:
    """Make the test of the `if` that mentions `needle` always True or always False."""

    def apply(tree):
        _if_with(_any_function(tree, function), needle).test = ast.Constant(value)

    return _documented(doc, apply)


def flip_compare(function: str, needle: str, new_op: ast.cmpop, doc: str) -> Callable:
    """Change the comparison operator of the one comparison that mentions `needle`."""

    def apply(tree):
        fn = _any_function(tree, function)
        cmp_ = _only([n for n in ast.walk(fn) if isinstance(n, ast.Compare) and needle in ast.unparse(n)], f"comparison with {needle!r}")
        cmp_.ops = [new_op]

    return _documented(doc, apply)


def drop_keyword(function: str, keyword: str, doc: str, call: Optional[str] = None) -> Callable:
    """Remove one keyword argument, from the decorator or call that carries it (the one named `call`, if given)."""

    def apply(tree):
        fn = _any_function(tree, function)
        holders = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and any(k.arg == keyword for k in n.keywords)
            and (call is None or call in ast.unparse(n.func))
        ]
        holder = _only(holders, f"call with keyword {keyword}")
        holder.keywords = [k for k in holder.keywords if k.arg != keyword]

    return _documented(doc, apply)


def set_keyword(function: str, keyword: str, value, doc: str) -> Callable:
    """Give one keyword argument of a decorator or call another constant value."""

    def apply(tree):
        fn = _any_function(tree, function)
        holder = _only([n for n in ast.walk(fn) if isinstance(n, ast.Call) and any(k.arg == keyword for k in n.keywords)], f"call with keyword {keyword}")
        _only([k for k in holder.keywords if k.arg == keyword], keyword).value = ast.Constant(value)

    return _documented(doc, apply)


def drop_statement(function: str, needle: str, doc: str) -> Callable:
    """Remove the one statement (not an `if`) whose source mentions `needle`."""

    def apply(tree):
        fn = _any_function(tree, function)
        stmt = _only([n for n in ast.walk(fn) if isinstance(n, (ast.Expr, ast.For)) and needle in ast.unparse(n)], f"statement with {needle!r}")
        _splice(tree, stmt, [])

    return _documented(doc, apply)


def unwrap_try(function: str, doc: str) -> Callable:
    """Replace the one try block of the function by its body, so the error it caught now escapes."""

    def apply(tree):
        fn = _any_function(tree, function)
        block = _only([n for n in ast.walk(fn) if isinstance(n, ast.Try)], "try block")
        _splice(tree, block, block.body)

    return _documented(doc, apply)


def bare_reraise(function: str, doc: str) -> Callable:
    """Turn the `raise NonRetryableException(...)` of the function into a bare `raise`, so the original exception goes on."""

    def apply(tree):
        fn = _any_function(tree, function)
        raise_ = _only([n for n in ast.walk(fn) if isinstance(n, ast.Raise) and "NonRetryableException" in ast.unparse(n)], "raise NonRetryableException")
        raise_.exc, raise_.cause = None, None

    return _documented(doc, apply)


def guard_drive_letter_not_recognised(tree):
    """A name like C:/windows/x.txt is no longer taken for an absolute path."""
    fn = _function(tree, "checked_member_path")
    cond = _if_with(fn, "startswith")
    cond.test = cond.test.values[0]


def guard_damaged_zip_not_refused(tree):
    """A file that is not a zip raises the library's own error instead of refusing the upload."""
    fn = _function(tree, "plan_archive")
    block = _only([n for n in ast.walk(fn) if isinstance(n, ast.Try)], "try block")
    _splice(tree, block, block.body)


def guard_junk_entries_counted(tree):
    """Mac and Windows leftovers are read as files."""
    fn = _function(tree, "_is_junk")
    _only([n for n in ast.walk(fn) if isinstance(n, ast.Return)], "return").value = ast.Constant(False)


def guard_directories_counted_as_files(tree):
    """Folder entries count as files, so they fill the file cap."""
    fn = _function(tree, "plan_archive")
    comp = _only([n for n in ast.walk(fn) if isinstance(n, ast.ListComp) and "is_dir" in ast.unparse(n)], "the file entry filter")
    comp.generators[0].ifs = []


def pipeline_unit_conversion_dropped(tree):
    """The total size cap is read as a number of bytes when the setting is in megabytes."""
    fn = _function(tree, "limits_from_settings")
    call = _only([n for n in ast.walk(fn) if isinstance(n, ast.Call) and any(k.arg == "max_total_bytes" for k in n.keywords)], "ArchiveLimits call")
    _only([k for k in call.keywords if k.arg == "max_total_bytes"], "max_total_bytes").value = ast.parse("settings.archive_max_total_mb", mode="eval").body


def pipeline_reject_run_does_nothing(tree):
    """A refused archive is not marked as rejected."""
    _function(tree, "reject_run").body = [ast.Pass()]


def pipeline_bad_file_error_not_wrapped(tree):
    """The parse stage raises the loader's own error for a bad file instead of BadFile."""
    fn = _function(tree, "parse_stage")
    handler = _only([n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)], "except handler")
    raise_ = _only([n for n in handler.body if isinstance(n, ast.Raise)], "raise")
    raise_.exc, raise_.cause = None, None


def pipeline_source_path_leaks(tree):
    """The chunk metadata keeps the loader's source path, which is the unpack folder, not the archive."""
    fn = _function(tree, "chunk_stage")
    loop = _only([n for n in ast.walk(fn) if isinstance(n, ast.For) and "source" in ast.unparse(n)], "source loop")
    _splice(tree, loop, [])


def pipeline_file_name_not_the_archive_path(tree):
    """Chunks keep the loader's file name, which is the hash the file was unpacked under."""
    fn = _function(tree, "chunk_stage")
    call = _only([n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "annotate_chunks"], "annotate_chunks call")
    call.keywords = [k for k in call.keywords if k.arg != "file_name"]


def pipeline_point_ids_random(tree):
    """Chunks are written without their derived ids, so a retry adds copies."""
    fn = _function(tree, "upsert_stage")
    call = _only([n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_documents"], "add_documents call")
    call.keywords = []


def pipeline_work_folders_kept(tree):
    """The unpacked files and stage files stay on disk after the run."""
    fn = _function(tree, "finish_run")
    for call in [n for n in ast.walk(fn) if isinstance(n, ast.Expr) and "rmtree" in ast.unparse(n)]:
        _splice(tree, call, [])


def run_store_pattern_allows_separators(tree):
    """The run id pattern accepts slashes and dots, so an id can name another folder."""
    assign = _only([n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "RUN_ID" for t in n.targets)], "RUN_ID")
    pattern = assign.value.args[0]
    pattern.value = pattern.value.replace("[0-9A-Za-z-]", "[0-9A-Za-z/.-]")


def run_store_rows_keyed_by_content(tree):
    """A row is stored under the file's content hash, so two paths with the same content share one row."""
    fn = _function(tree, "add_row")
    assign = _only([n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "key" for t in n.targets)], "key assignment")
    assign.value = ast.parse('item.get("doc_id") or "none"', mode="eval").body


def run_store_counts_the_wrong_rows(tree):
    """The counts per status count the rows that do NOT have the status."""
    fn = _function(tree, "load")
    cmp_ = _only([n for n in ast.walk(fn) if isinstance(n, ast.Compare) and "status" in ast.unparse(n)], "status comparison")
    cmp_.ops = [ast.NotEq()]


def identity_position_always_zero(tree):
    """Every chunk of a document is numbered 0."""
    fn = _function(tree, "annotate_chunks")
    assign = _only([n for n in ast.walk(fn) if isinstance(n, ast.Assign) and "chunk_seq" in ast.unparse(n)], "chunk_seq assignment")
    assign.value = ast.Constant(0)


def identity_total_always_zero(tree):
    """The number of chunks a document produced is not recorded."""
    fn = _function(tree, "annotate_chunks")
    assign = _only([n for n in ast.walk(fn) if isinstance(n, ast.Assign) and "doc_chunks_total" in ast.unparse(n)], "doc_chunks_total assignment")
    assign.value = ast.Constant(0)


def identity_point_id_ignores_position(tree):
    """Every chunk of a document gets the same point id, so each overwrites the last."""
    fn = _function(tree, "chunk_point_id")
    call = _only([n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "uuid5"], "uuid5 call")
    call.args[1] = ast.Name(id="doc_id", ctx=ast.Load())


def identity_location_shows_chunk_number_for_a_pdf(tree):
    """A PDF chunk is located by chunk number, not by page."""
    _splice(tree, _if_with(_function(tree, "display_location"), "page"), [])


def identity_chunk_number_starts_at_zero(tree):
    """The chunk number a reader sees is one lower than it should be."""
    fn = _function(tree, "display_location")
    plus = _only([n for n in ast.walk(fn) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add) and "seq" in ast.unparse(n)], "seq + 1")
    _replace(tree, plus, ast.Name(id="seq", ctx=ast.Load()))


def store_partial_file_counts_as_indexed(tree):
    """A file with any stored chunk counts as indexed."""
    fn = _function(tree, "is_indexed")
    _only([n for n in ast.walk(fn) if isinstance(n, ast.Return)], "return").value = ast.parse("found > 0", mode="eval").body


def store_window_ignores_the_document(tree):
    """The chunk lookup filters on the position only, so another document's chunks come back."""
    fn = _function(tree, "_doc_filter")
    assign = _only([n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "must" for t in n.targets)], "must assignment")
    assign.value = ast.List(elts=[], ctx=ast.Load())


def store_window_one_chunk_too_wide(tree):
    """The window reaches one chunk further after the cited one than the context asks for."""
    fn = _function(tree, "get_chunk_window")
    assign = _only([n for n in ast.walk(fn) if isinstance(n, ast.Assign) and "low, high = (" in ast.unparse(n)], "low, high assignment")
    high = assign.value.elts[1]
    assign.value.elts[1] = ast.BinOp(left=high, op=ast.Add(), right=ast.Constant(1))


def store_window_returned_for_a_missing_chunk(tree):
    """The neighbours of a chunk that is not there are returned anyway."""
    fn = _function(tree, "get_chunk_window")
    ret = _only([n for n in ast.walk(fn) if isinstance(n, ast.Return) and isinstance(n.value, ast.IfExp)], "conditional return")
    ret.value = ret.value.body


def store_ids_ignored(tree):
    """add_documents accepts point ids and then lets the library invent new ones."""
    fn = _function(tree, "add_documents")
    call = _only([n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_documents" and n.keywords], "inner add_documents call")
    call.keywords = []


def view_neighbours_swapped(tree):
    """The chunks after the cited one are listed as the chunks before it."""
    fn = _function(tree, "build_chunk_view")
    cmp_ = _only([n for n in ast.walk(fn) if isinstance(n, ast.Compare) and isinstance(n.ops[0], ast.Lt)], "before comparison")
    cmp_.ops = [ast.Gt()]


def hatchet_upsert_attempt_count_off_by_one(tree):
    """The upsert task thinks the attempt before the last one is the last, and records a failure Hatchet is about to retry."""
    fn = _function(tree, "embed_and_upsert")
    kw = _only([k for n in ast.walk(fn) if isinstance(n, ast.Call) for k in n.keywords if k.arg == "max_attempts"], "max_attempts keyword")
    kw.value = kw.value.left


GUARD = "app/workflows/archive_guard.py"
PIPELINE = "app/workflows/pipeline.py"
RUN_STORE = "app/workflows/run_store.py"
IDENTITY = "app/services/chunk_identity.py"
VECTOR_STORE = "app/services/vector_store.py"
SOURCE_VIEW = "app/services/source_view.py"
ROUTES = "app/api/routes.py"
INGEST = "app/workflows/ingest.py"
GUARD_RUNS = (("tests/test_archive_guard.py",),)
PIPELINE_RUNS = (("tests/test_archive_pipeline.py",),)
STORE_RUNS = (("tests/test_archive_pipeline.py", "tests/test_source_view.py"),)
API_RUNS = (("tests/test_archive_api.py", "tests/test_source_view.py"),)
HATCHET_RUNS = (("tests/test_hatchet_tasks.py",),)


def route_oversize_upload_kept(tree):
    """An upload over the size cap is refused but its partial file stays in the spool."""
    cond = _if_with(_any_function(tree, "ingest_archive_upload"), "written > limit")
    cond.body = [st for st in cond.body if not (isinstance(st, ast.Expr) and "unlink" in ast.unparse(st))]


def route_failed_start_keeps_the_upload(tree):
    """When the workflow cannot be started the saved zip stays in the spool."""
    handler = _only([n for n in ast.walk(_any_function(tree, "ingest_archive_upload")) if isinstance(n, ast.ExceptHandler)], "except handler")
    handler.body = [st for st in handler.body if not (isinstance(st, ast.Expr) and "unlink" in ast.unparse(st))]


@dataclass(frozen=True)
class Mutant:
    name: str
    file: str
    apply: Callable[[ast.AST], None]
    invocations: Optional[tuple] = None  # pytest invocations to run; None means the evaluation's INVOCATIONS
    text: bool = False  # True: `apply` takes the file's text and returns the edited text (a doc), not a syntax tree
    group: str = "core"  # core: the evaluation, the MCP server, the workbook loader and the docs; ingest: archive ingestion and the source view; hatchet: the task definitions (needs hatchet-sdk)

    @property
    def runs(self) -> List[List[str]]:
        return [list(i) for i in self.invocations] if self.invocations else INVOCATIONS


MUTANTS: List[Mutant] = [
    Mutant("bm25-branch-removed", ADVANCED, bm25_branch_removed),
    Mutant("bm25-bonus-sign-flipped", ADVANCED, bm25_bonus_sign_flipped),
    Mutant("ranking-inverted-again", ADVANCED, ranking_inverted_again),
    Mutant("fallback-counter-never-increments", ADVANCED, fallback_counter_never_increments),
    Mutant("unanswerable-threshold-comparison-flipped", HARNESS, threshold_comparison_flipped),
    Mutant("unanswerable-threshold-boundary-loosened", HARNESS, threshold_boundary_loosened),
    Mutant("must-contain-matching-negated", HARNESS, must_contain_matching_negated),
    Mutant("must-contain-empty-phrase-matches", HARNESS, must_contain_empty_phrase_matches),
    Mutant("baseline-tolerance-ignored", HARNESS, baseline_tolerance_ignored),
    Mutant("baseline-chunk-hash-not-compared", HARNESS, baseline_chunk_hash_not_compared),
    Mutant("baseline-rise-flagged-as-a-fall", HARNESS, baseline_rise_flagged),
    Mutant("baseline-question-hash-not-compared", HARNESS, baseline_question_hash_not_compared),
    Mutant("baseline-fallen-back-method-still-judged", HARNESS, baseline_fallen_back_method_still_judged),
    Mutant("fallback-cause-never-kept", ADVANCED, fallback_cause_never_kept),
    Mutant("fallback-cause-overwritten-by-later-faults", ADVANCED, fallback_cause_overwritten_by_later_faults),
    Mutant("readme-table-drops-recall", HARNESS, readme_table_drops_recall),
    Mutant("bootstrap-upper-percentile-moved", HARNESS, bootstrap_upper_percentile_moved),
    Mutant("bootstrap-lower-percentile-moved", HARNESS, bootstrap_lower_percentile_moved),
    Mutant("eval-chunking-follows-the-environment", HARNESS, eval_chunking_follows_the_environment),
    Mutant("label-validation-skipped", HARNESS, label_validation_skipped),
    Mutant("mcp-stdout-redirect-removed-at-import", MCP_SERVER, mcp_import_time_stdout_redirect_removed, MCP_RUNS),
    Mutant("mcp-stdout-redirect-removed-in-tool-calls", MCP_SERVER, mcp_tool_call_stdout_redirect_removed, MCP_RUNS),
    Mutant("mcp-citation-id-from-the-wrong-field", MCP_SERVER, mcp_citation_id_built_from_the_wrong_field, MCP_RUNS),
    Mutant("mcp-failed-generation-returned-as-an-answer", MCP_SERVER, mcp_failed_generation_returned_as_an_answer, MCP_RUNS),
    Mutant("mcp-write-tool-registered", MCP_SERVER, mcp_write_tool_registered, MCP_RUNS),
    Mutant("mcp-read-only-hint-dropped", MCP_SERVER, mcp_read_only_hint_dropped, MCP_RUNS),
    Mutant("mcp-limit-bound-removed", MCP_SERVER, mcp_limit_bound_removed, MCP_RUNS),
    Mutant("mcp-query-length-cap-removed", MCP_SERVER, mcp_query_length_cap_removed, MCP_RUNS),
    Mutant("mcp-document-filter-ignored", MCP_SERVER, mcp_document_filter_ignored, MCP_RUNS),
    Mutant("readme-chunk-count-stale", README, readme_chunk_count_stale, DOC_RUNS, text=True),
    Mutant("readme-question-total-edited-by-one", README, readme_question_total_edited_by_one, DOC_RUNS, text=True),
    Mutant("readme-category-count-edited-by-one", README, readme_category_count_edited_by_one, DOC_RUNS, text=True),
    Mutant("readme-n-column-edited-by-one", README, readme_n_column_edited_by_one, DOC_RUNS, text=True),
    Mutant("readme-real-model-date-removed", README, readme_real_model_date_removed, DOC_RUNS, text=True),
    Mutant("docs-calibration-count-edited-by-one", EVALUATION_DOC, evaluation_doc_calibration_count_edited_by_one, DOC_RUNS, text=True),
    Mutant("docs-design-notes-claim-there-is-no-ci", DESIGN_NOTES, design_notes_claim_there_is_no_ci, DOC_RUNS, text=True),
    Mutant("xlsx-formulas-read-as-text", PARSER, xlsx_formulas_read_as_text, XLSX_RUNS),
    Mutant("xlsx-password-signature-ignored", PARSER, xlsx_password_signature_ignored, XLSX_RUNS),
    Mutant("xlsx-encrypted-entries-ignored", PARSER, xlsx_encrypted_entries_ignored, XLSX_RUNS),
    Mutant("xlsx-unpacked-size-limit-removed", PARSER, xlsx_unpacked_size_limit_removed, XLSX_RUNS),
    Mutant("xlsx-row-limit-removed", PARSER, xlsx_row_limit_removed, XLSX_RUNS),
    Mutant("xlsx-row-limit-boundary-loosened", PARSER, xlsx_row_limit_boundary_loosened, XLSX_RUNS),
    Mutant("xlsx-column-limit-removed", PARSER, xlsx_column_limit_removed, XLSX_RUNS),
    Mutant("xlsx-column-overflow-never-read", PARSER, xlsx_column_overflow_never_read, XLSX_RUNS),
    Mutant("xlsx-sheet-limit-removed", PARSER, xlsx_sheet_limit_removed, XLSX_RUNS),
    Mutant("xlsx-hidden-sheets-read", PARSER, xlsx_hidden_sheets_read, XLSX_RUNS),
    Mutant("xlsx-row-numbers-start-at-zero", PARSER, xlsx_row_numbers_start_at_zero, XLSX_RUNS),
    Mutant("xlsx-location-without-the-sheet", PARSER, xlsx_location_without_the_sheet, XLSX_RUNS),
    Mutant("xlsx-block-row-cap-removed", PARSER, xlsx_block_row_cap_removed, XLSX_RUNS),
    Mutant("xlsx-block-char-cap-removed", PARSER, xlsx_block_char_cap_removed, XLSX_RUNS),
    Mutant("xlsx-block-preamble-not-counted", PARSER, xlsx_block_preamble_not_counted, XLSX_RUNS),
    Mutant("xlsx-empty-workbook-accepted", PARSER, xlsx_empty_workbook_accepted, XLSX_RUNS),
    Mutant("xlsx-header-row-not-detected", PARSER, xlsx_header_row_not_detected, XLSX_RUNS),
    Mutant("xlsx-blank-rows-not-skipped", PARSER, xlsx_blank_rows_not_skipped, XLSX_RUNS),
    Mutant("xlsx-long-cells-not-cut", PARSER, xlsx_long_cells_not_cut, XLSX_RUNS),
    Mutant("xlsx-extension-not-supported", PARSER, xlsx_extension_not_supported, XLSX_RUNS),
]


INGEST_MUTANTS: List[Mutant] = [
    # the archive guard (app/workflows/archive_guard.py): each rule that refuses an archive or sets an entry aside
    Mutant("guard-parent-folder-climb-allowed", GUARD, drop_if("checked_member_path", "'..' in parts", "An entry name with .. in it is accepted (zip slip)."), GUARD_RUNS, group="ingest"),
    Mutant("guard-absolute-path-allowed", GUARD, drop_if("checked_member_path", "startswith", "An entry name that starts with / or a drive letter is accepted."), GUARD_RUNS, group="ingest"),
    Mutant("guard-drive-letter-not-recognised", GUARD, guard_drive_letter_not_recognised, GUARD_RUNS, group="ingest"),
    Mutant("guard-file-count-cap-removed", GUARD, drop_if("plan_archive", "max_files", "An archive with any number of files is accepted."), GUARD_RUNS, group="ingest"),
    Mutant("guard-file-count-boundary-loosened", GUARD, flip_compare("plan_archive", "max_files", ast.GtE(), "An archive with exactly the cap of files is refused."), GUARD_RUNS, group="ingest"),
    Mutant("guard-total-size-cap-removed", GUARD, drop_if("plan_archive", "max_total_bytes", "An archive that unpacks to any size is accepted (zip bomb)."), GUARD_RUNS, group="ingest"),
    Mutant("guard-total-size-boundary-loosened", GUARD, flip_compare("plan_archive", "max_total_bytes", ast.GtE(), "An archive that unpacks to exactly the cap is refused."), GUARD_RUNS, group="ingest"),
    Mutant("guard-per-file-cap-removed", GUARD, drop_if("_entry_problem", "max_file_bytes", "A single file of any size is read."), GUARD_RUNS, group="ingest"),
    Mutant("guard-per-file-boundary-loosened", GUARD, flip_compare("_entry_problem", "max_file_bytes", ast.GtE(), "A file of exactly the per file cap is set aside."), GUARD_RUNS, group="ingest"),
    Mutant("guard-extension-filter-removed", GUARD, drop_if("_entry_problem", "allowed_extensions", "Files with any extension are read."), GUARD_RUNS, group="ingest"),
    Mutant("guard-nested-archive-not-named", GUARD, drop_if("_entry_problem", "ARCHIVE_EXTENSIONS", "A nested archive is set aside as an unknown extension and not as a nested archive."), GUARD_RUNS, group="ingest"),
    Mutant("guard-encrypted-entry-read", GUARD, drop_if("_entry_problem", "flag_bits", "A password protected entry is read."), GUARD_RUNS, group="ingest"),
    Mutant("guard-damaged-zip-not-refused", GUARD, guard_damaged_zip_not_refused, GUARD_RUNS, group="ingest"),
    Mutant("guard-empty-archive-accepted", GUARD, drop_if("plan_archive", "not members", "An archive with no readable file is accepted."), GUARD_RUNS, group="ingest"),
    Mutant("guard-junk-entries-read-as-files", GUARD, guard_junk_entries_counted, GUARD_RUNS, group="ingest"),
    Mutant("guard-folders-count-as-files", GUARD, guard_directories_counted_as_files, GUARD_RUNS, group="ingest"),
    Mutant("guard-set-aside-entry-unpacked", GUARD, drop_if("unpack", "error_code", "An entry that was set aside is unpacked anyway."), GUARD_RUNS, group="ingest"),
    Mutant("guard-damaged-entry-escapes", GUARD, unwrap_try("unpack", "A damaged entry raises out of the unpack and stops the others."), GUARD_RUNS, group="ingest"),
    # the archive rules (app/workflows/pipeline.py)
    Mutant("pipeline-copy-in-the-archive-not-detected", PIPELINE, force_if("prepare_run", "doc_id in first_seen", False, "A second copy of a file inside one archive is indexed again."), PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-indexed-file-not-skipped", PIPELINE, force_if("prepare_run", "index.is_indexed", False, "A file that is already indexed is processed again (a rerun does the work again)."), PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-set-aside-entry-not-recorded", PIPELINE, force_if("prepare_run", "member.error_code is not None", False, "An entry the guard set aside gets no row."), PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-damaged-entry-not-recorded", PIPELINE, force_if("prepare_run", "path is None", False, "An entry that fails to unpack gets no row."), PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-bad-file-error-not-wrapped", PIPELINE, pipeline_bad_file_error_not_wrapped, PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-file-with-no-text-accepted", PIPELINE, drop_if("parse_stage", "not documents", "A file the loader reads as no text goes on to the next stage."), PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-file-with-no-chunks-accepted", PIPELINE, drop_if("chunk_stage", "not chunks", "A file that gives no chunks goes on to the upsert."), PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-unpack-path-leaks-into-metadata", PIPELINE, pipeline_source_path_leaks, PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-file-name-is-the-hash", PIPELINE, pipeline_file_name_not_the_archive_path, PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-chunks-written-without-derived-ids", PIPELINE, pipeline_point_ids_random, PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-last-failure-not-recorded", PIPELINE, force_if("run_upsert", "attempt >= max_attempts", False, "The failure of the last attempt gets no row."), PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-every-failure-recorded", PIPELINE, force_if("run_upsert", "attempt >= max_attempts", True, "A failure that will be retried already gets a failed row."), PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-last-attempt-boundary-moved", PIPELINE, flip_compare("run_upsert", "attempt >= max_attempts", ast.Gt(), "The failure of the last attempt is recorded one attempt too late."), PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-success-not-recorded", PIPELINE, drop_statement("run_upsert", "record_success", "A file that was stored gets no ingested row."), PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-silent-task-not-failed", PIPELINE, force_if("finish_run", "job.file not in reported", False, "A file whose task reported nothing gets no row."), PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-refused-archive-not-marked", PIPELINE, pipeline_reject_run_does_nothing, PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-unpacked-files-kept", PIPELINE, pipeline_work_folders_kept, PIPELINE_RUNS, group="ingest"),
    Mutant("pipeline-size-setting-read-as-bytes", PIPELINE, pipeline_unit_conversion_dropped, PIPELINE_RUNS, group="ingest"),
    # the status files (app/workflows/run_store.py)
    Mutant("run-store-run-id-not-checked", RUN_STORE, drop_if("checked_run_id", "RUN_ID", "Any string is taken for a run id."), (("tests/test_archive_pipeline.py", "tests/test_archive_api.py"),), group="ingest"),
    Mutant("run-store-run-id-pattern-allows-separators", RUN_STORE, run_store_pattern_allows_separators, (("tests/test_archive_pipeline.py", "tests/test_archive_api.py"),), group="ingest"),
    Mutant("run-store-submit-overwrites-the-worker-state", RUN_STORE, force_if("start", "keep_existing", False, "A late submit resets a run the worker already moved on."), PIPELINE_RUNS, group="ingest"),
    Mutant("run-store-rows-keyed-by-content", RUN_STORE, run_store_rows_keyed_by_content, PIPELINE_RUNS, group="ingest"),
    Mutant("run-store-counts-the-wrong-rows", RUN_STORE, run_store_counts_the_wrong_rows, PIPELINE_RUNS, group="ingest"),
    Mutant("run-store-unknown-status-accepted", RUN_STORE, drop_if("row", "STATUSES", "A row can carry any status word."), PIPELINE_RUNS, group="ingest"),
    # chunk identity and the lookup (app/services/chunk_identity.py, vector_store.py, source_view.py)
    Mutant("identity-position-always-zero", IDENTITY, identity_position_always_zero, STORE_RUNS, group="ingest"),
    Mutant("identity-chunk-total-always-zero", IDENTITY, identity_total_always_zero, STORE_RUNS, group="ingest"),
    Mutant("identity-point-id-ignores-position", IDENTITY, identity_point_id_ignores_position, STORE_RUNS, group="ingest"),
    Mutant("identity-pdf-located-by-chunk-number", IDENTITY, identity_location_shows_chunk_number_for_a_pdf, STORE_RUNS, group="ingest"),
    Mutant("identity-chunk-number-starts-at-zero", IDENTITY, identity_chunk_number_starts_at_zero, STORE_RUNS, group="ingest"),
    Mutant("identity-file-renamed-when-not-asked", IDENTITY, force_if("annotate_chunks", "file_name is not None", True, "Chunks are renamed even when no name is given."), STORE_RUNS, group="ingest"),
    Mutant("store-partial-file-counts-as-indexed", VECTOR_STORE, store_partial_file_counts_as_indexed, STORE_RUNS, group="ingest"),
    Mutant("store-complete-file-not-counted-as-indexed", VECTOR_STORE, flip_compare("is_indexed", "found >= total", ast.Gt(), "A file whose chunks are all stored does not count as indexed."), STORE_RUNS, group="ingest"),
    Mutant("store-window-ignores-the-document", VECTOR_STORE, store_window_ignores_the_document, STORE_RUNS, group="ingest"),
    Mutant("store-window-one-chunk-too-wide", VECTOR_STORE, store_window_one_chunk_too_wide, STORE_RUNS, group="ingest"),
    Mutant("store-window-kept-for-a-missing-chunk", VECTOR_STORE, store_window_returned_for_a_missing_chunk, STORE_RUNS, group="ingest"),
    Mutant("store-point-ids-ignored", VECTOR_STORE, store_ids_ignored, STORE_RUNS, group="ingest"),
    Mutant("store-point-id-count-not-checked", VECTOR_STORE, drop_if("add_documents", "len(ids) != len(documents)", "A list of ids of the wrong length is accepted."), STORE_RUNS, group="ingest"),
    Mutant("view-neighbours-swapped", SOURCE_VIEW, view_neighbours_swapped, STORE_RUNS, group="ingest"),
    # the routes (app/api/routes.py)
    Mutant("route-context-bound-removed", ROUTES, drop_keyword("get_document_chunk", "le", "A context of any size is accepted."), API_RUNS, group="ingest"),
    Mutant("route-missing-chunk-not-a-404", ROUTES, force_if("get_document_chunk", "view is None", False, "A chunk that is not there is not answered with 404."), API_RUNS, group="ingest"),
    Mutant("route-bad-run-id-is-an-error", ROUTES, unwrap_try("get_archive_run", "A run id that is not one raises instead of answering 404."), API_RUNS, group="ingest"),
    Mutant("route-upload-size-cap-removed", ROUTES, drop_if("ingest_archive_upload", "written > limit", "An archive upload of any size is accepted."), API_RUNS, group="ingest"),
    Mutant("route-oversize-upload-kept", ROUTES, route_oversize_upload_kept, API_RUNS, group="ingest"),
    Mutant("route-only-a-zip-name-not-checked", ROUTES, drop_if("ingest_archive_upload", "endswith", "Any file name is accepted as an archive."), API_RUNS, group="ingest"),
    Mutant("route-missing-token-not-reported", ROUTES, drop_if("ingest_archive_upload", "is_configured", "An upload is accepted when Hatchet is not set up."), API_RUNS, group="ingest"),
    Mutant("route-failed-start-keeps-the-upload", ROUTES, route_failed_start_keeps_the_upload, API_RUNS, group="ingest"),
    # the Hatchet task definitions (app/workflows/ingest.py); these need hatchet-sdk (requirements-hatchet.txt)
    Mutant("hatchet-parse-retried", INGEST, set_keyword("parse", "retries", 3, "The parse task is retried when it fails."), HATCHET_RUNS, group="hatchet"),
    Mutant("hatchet-chunk-retried", INGEST, set_keyword("chunk", "retries", 3, "The chunk task is retried when it fails."), HATCHET_RUNS, group="hatchet"),
    Mutant("hatchet-upsert-not-retried", INGEST, drop_keyword("embed_and_upsert", "retries", "The embed and upsert task is never retried."), HATCHET_RUNS, group="hatchet"),
    Mutant("hatchet-upsert-backoff-removed", INGEST, drop_keyword("embed_and_upsert", "backoff_factor", "The retries of the upsert task wait no longer between attempts."), HATCHET_RUNS, group="hatchet"),
    Mutant("hatchet-bad-file-retried", INGEST, bare_reraise("parse", "A bad file raises its own error instead of the one Hatchet never retries."), HATCHET_RUNS, group="hatchet"),
    Mutant("hatchet-refused-archive-retried", INGEST, bare_reraise("validate", "A refused archive raises its own error instead of the one Hatchet never retries."), HATCHET_RUNS, group="hatchet"),
    Mutant("hatchet-last-attempt-off-by-one", INGEST, hatchet_upsert_attempt_count_off_by_one, HATCHET_RUNS, group="hatchet"),
    Mutant("hatchet-fan-out-not-after-validate", INGEST, drop_keyword("fan_out", "parents", "The fan out task is not made to wait for the archive check."), HATCHET_RUNS, group="hatchet"),
    Mutant("hatchet-chunk-not-after-parse", INGEST, drop_keyword("chunk", "parents", "The chunk task is not made to wait for the parse task."), HATCHET_RUNS, group="hatchet"),
]
MUTANTS.extend(INGEST_MUTANTS)


def mutated_source(mutant: Mutant, root: Path = BACKEND) -> str:
    """Apply a mutant to the file under `root` and return the new source (raises TargetNotFound if the code moved)."""
    original = (root / mutant.file).read_text(encoding="utf-8")
    if mutant.text:
        changed = mutant.apply(original)
        if changed == original:
            raise TargetNotFound(f"{mutant.name} did not change the doc")
        return changed
    tree = ast.parse(original)
    mutant.apply(tree)
    ast.fix_missing_locations(tree)
    changed = ast.unparse(tree)
    if changed == ast.unparse(ast.parse(original)):
        raise TargetNotFound(f"{mutant.name} did not change the code")
    return changed


# ---------------------------------------------------------------- running


def _copy_tree(destination: Path) -> Path:
    target = destination / "backend"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "venv", "data")
    for part in ("app", "eval", "tests", "mcp_server"):
        shutil.copytree(BACKEND / part, target / part, ignore=ignore)
    shutil.copy(BACKEND / "pytest.ini", target / "pytest.ini")
    shutil.copy(BACKEND / "requirements-mcp.txt", target / "requirements-mcp.txt")  # a test checks the SDK pin against it
    # the README tests read the README, docs/ and the skill's scenario folder beside backend/
    for part in ("docs", "skills"):
        shutil.copytree(BACKEND.parent / part, destination / part, ignore=shutil.ignore_patterns("*.gif"))
    shutil.copy(BACKEND.parent / "README.md", destination / "README.md")
    # the upload formats test reads the upload box beside backend/ to keep its list in step with the parser's
    upload_box = Path("frontend") / "src" / "components" / "DocumentUpload.tsx"
    (destination / upload_box).parent.mkdir(parents=True)
    shutil.copy(BACKEND.parent / upload_box, destination / upload_box)
    return target


def _run_pytest(cwd: Path, targets: List[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ, RENDER="true", PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *targets]
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=540)


def _failed_tests(output: str) -> List[str]:
    return re.findall(r"^FAILED (\S+)", output, flags=re.M)


def run_control(chosen: Optional[List[Mutant]] = None) -> Optional[str]:
    """The unmutated copy must be green, otherwise 'caught' would mean nothing. Returns an error text or None.
    Only the invocations the chosen mutants will use are run, so the MCP tests are a control only when an MCP mutant is."""
    runs = [targets for m in (chosen or MUTANTS) for targets in m.runs]
    unique = [targets for i, targets in enumerate(runs) if targets not in runs[:i]]
    with tempfile.TemporaryDirectory() as tmp:
        cwd = _copy_tree(Path(tmp))
        for targets in unique:
            done = _run_pytest(cwd, targets)
            if done.returncode != 0:
                return f"control run is not green for {targets}:\n{done.stdout[-1500:]}\n{done.stderr[-500:]}"
    return None


def run_mutant(mutant: Mutant) -> dict:
    started = time.time()
    try:
        source = mutated_source(mutant)
    except TargetNotFound as exc:
        return {"name": mutant.name, "status": "TARGET NOT FOUND", "detail": str(exc), "seconds": 0.0}
    with tempfile.TemporaryDirectory() as tmp:
        cwd = _copy_tree(Path(tmp))
        (cwd / mutant.file).write_text(source, encoding="utf-8")
        for targets in mutant.runs:
            done = _run_pytest(cwd, targets)
            if done.returncode == 1:
                failed = _failed_tests(done.stdout)
                return {"name": mutant.name, "status": "CAUGHT", "detail": ", ".join(failed[:3]) or "failed", "seconds": time.time() - started}
            if done.returncode != 0:
                return {"name": mutant.name, "status": "DID NOT RUN", "detail": f"pytest exit {done.returncode}: {done.stdout[-600:]}{done.stderr[-300:]}", "seconds": time.time() - started}
    return {"name": mutant.name, "status": "SURVIVED", "detail": "every test stayed green", "seconds": time.time() - started}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--only", metavar="NAME")
    parser.add_argument("--skip-mcp", action="store_true", help="leave out the MCP server mutants (they need the SDK)")
    parser.add_argument("--only-ingest", action="store_true", help="only the ingestion, source view and Hatchet task mutants")
    parser.add_argument("--skip-ingest", action="store_true", help="leave out the ingestion, source view and Hatchet task mutants")
    parser.add_argument("--skip-hatchet", action="store_true", help="leave out the Hatchet task mutants (they need hatchet-sdk)")
    args = parser.parse_args(argv)

    chosen = [m for m in MUTANTS if not args.only or m.name == args.only]
    if args.skip_mcp:
        chosen = [m for m in chosen if m.file != MCP_SERVER]
    if args.only_ingest:
        chosen = [m for m in chosen if m.group in ("ingest", "hatchet")]
    if args.skip_ingest:
        chosen = [m for m in chosen if m.group == "core"]
    if args.skip_hatchet:
        chosen = [m for m in chosen if m.group != "hatchet"]
    if args.list:
        for m in MUTANTS:
            print(f"{m.name:<44} {m.file}  {m.apply.__doc__.strip()}")
        return 0
    if not chosen:
        print(f"no mutant named {args.only}")
        return 2
    if any(m.file == MCP_SERVER for m in chosen) and importlib.util.find_spec("mcp") is None:
        print("The MCP server mutants need the SDK: pip install -r requirements-mcp.txt (or pass --skip-mcp to leave them out).")
        return 2

    if any(m.group == "hatchet" for m in chosen) and importlib.util.find_spec("hatchet_sdk") is None:
        print("The Hatchet task mutants need the SDK: pip install -r requirements-hatchet.txt (or pass --skip-hatchet to leave them out).")
        return 2

    problem = run_control(chosen)
    if problem:
        print(problem)
        return 2
    print("control (unmutated copy): green")

    results = []
    for m in chosen:
        res = run_mutant(m)
        results.append(res)
        print(f"{res['status']:<17} {res['name']:<44} {res['seconds']:5.1f}s  {res['detail'][:150]}")
    caught = sum(1 for r in results if r["status"] == "CAUGHT")
    print(f"\n{caught} of {len(results)} mutants caught")
    return 0 if caught == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
