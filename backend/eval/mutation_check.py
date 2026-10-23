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

The MCP server mutants break mcp_server/server.py and run only the server tests; each must fail a named test.
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


@dataclass(frozen=True)
class Mutant:
    name: str
    file: str
    apply: Callable[[ast.AST], None]
    invocations: Optional[tuple] = None  # pytest invocations to run; None means the evaluation's INVOCATIONS
    text: bool = False  # True: `apply` takes the file's text and returns the edited text (a doc), not a syntax tree

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
]


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
    args = parser.parse_args(argv)

    chosen = [m for m in MUTANTS if not args.only or m.name == args.only]
    if args.skip_mcp:
        chosen = [m for m in chosen if m.file != MCP_SERVER]
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
