"""Every guard of app/workflows/archive_guard.py, driven with zips the test builds itself.

No Hatchet, no Qdrant, no model. Each guard has a case that must be refused or set aside and a case just inside the
limit that must pass, so both a removed guard and a guard that got too strict turn a test red. eval/mutation_check.py
breaks the guards one at a time and requires a failure from this file.
Run: pytest tests/test_archive_guard.py
"""
import os
import struct
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app.workflows.archive_guard import (  # noqa: E402
    ArchiveLimits,
    ArchiveRejected,
    checked_member_path,
    plan_archive,
    unpack,
)

ALLOWED = frozenset({".pdf", ".docx", ".xlsx", ".txt", ".md"})
LIMITS = ArchiveLimits(max_files=5, max_total_bytes=1000, max_file_bytes=400, allowed_extensions=ALLOWED)


def make_zip(tmp_path: Path, entries, name: str = "a.zip") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        for entry_name, data in entries:
            z.writestr(entry_name, data)
    return path


def rejected(tmp_path, entries, limits=LIMITS) -> ArchiveRejected:
    with pytest.raises(ArchiveRejected) as caught:
        plan_archive(make_zip(tmp_path, entries), limits)
    return caught.value


# ---------------------------------------------------------------- zip slip: the whole archive is refused


@pytest.mark.parametrize(
    "name",
    ["../escape.txt", "docs/../../escape.txt", "/etc/passwd.txt", "C:/windows/x.txt", "..\\escape.txt", "docs\\..\\..\\x.txt"],
)
def test_an_entry_that_leaves_the_unpack_folder_refuses_the_whole_archive(tmp_path, name):
    error = rejected(tmp_path, [("fine.txt", b"ok"), (name, b"bad")])
    assert error.code == "path_traversal"


def test_a_traversal_name_is_refused_before_any_file_is_unpacked(tmp_path):
    path = make_zip(tmp_path, [("fine.txt", b"ok"), ("../escape.txt", b"bad")])
    with pytest.raises(ArchiveRejected):
        plan_archive(path, LIMITS)
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("name,expected", [("a/b.txt", "a/b.txt"), ("a\\b.txt", "a/b.txt"), ("./a//b.txt", "a/b.txt"), ("dots..in..name.txt", "dots..in..name.txt")])
def test_a_safe_name_is_kept_with_forward_slashes(name, expected):
    assert checked_member_path(name) == expected


def test_unpacked_files_land_under_their_hash_whatever_the_entry_is_called(tmp_path):
    path = make_zip(tmp_path, [("deep/er/name.txt", b"hello")])
    members, _ = plan_archive(path, LIMITS)
    [(member, written, digest)] = list(unpack(path, members, tmp_path / "out"))
    assert written.parent == tmp_path / "out"
    assert written.name == f"{digest}.txt"
    assert written.read_bytes() == b"hello"


# ---------------------------------------------------------------- the caps


def test_more_files_than_the_cap_refuses_the_archive(tmp_path):
    error = rejected(tmp_path, [(f"f{i}.txt", b"x") for i in range(LIMITS.max_files + 1)])
    assert error.code == "too_many_files"


def test_exactly_the_cap_of_files_is_accepted(tmp_path):
    members, _ = plan_archive(make_zip(tmp_path, [(f"f{i}.txt", b"x") for i in range(LIMITS.max_files)]), LIMITS)
    assert len(members) == LIMITS.max_files


def test_directory_entries_do_not_count_against_the_file_cap(tmp_path):
    entries = [(f"d{i}/", b"") for i in range(10)] + [("d0/f.txt", b"x")]
    members, _ = plan_archive(make_zip(tmp_path, entries), LIMITS)
    assert [m.path for m in members] == ["d0/f.txt"]


def test_files_that_add_up_to_more_than_the_total_cap_refuse_the_archive(tmp_path):
    error = rejected(tmp_path, [("a.txt", b"x" * 400), ("b.txt", b"x" * 400), ("c.txt", b"x" * 201)])
    assert error.code == "archive_too_large"


def test_exactly_the_total_cap_is_accepted(tmp_path):
    members, _ = plan_archive(make_zip(tmp_path, [("a.txt", b"x" * 400), ("b.txt", b"x" * 400), ("c.txt", b"x" * 200)]), LIMITS)
    assert len(members) == 3


def test_a_zip_bomb_is_refused_from_its_file_table_without_unpacking(tmp_path):
    path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("zeros.txt", b"\x00" * 5_000_000)  # a few kilobytes on disk, five megabytes unpacked
    assert path.stat().st_size < 10_000
    with pytest.raises(ArchiveRejected) as caught:
        plan_archive(path, LIMITS)
    assert caught.value.code == "archive_too_large"


def test_an_entry_over_the_per_file_cap_is_set_aside_and_the_rest_go_on(tmp_path):
    members, _ = plan_archive(make_zip(tmp_path, [("big.txt", b"x" * 401), ("small.txt", b"x" * 400)]), LIMITS)
    by_path = {m.path: m for m in members}
    assert by_path["big.txt"].error_code == "file_too_large"
    assert by_path["small.txt"].error_code is None


# ---------------------------------------------------------------- what an entry may be


def test_an_extension_the_loaders_do_not_read_is_set_aside(tmp_path):
    members, _ = plan_archive(make_zip(tmp_path, [("photo.png", b"x"), ("note.TXT", b"x"), ("noext", b"x")]), LIMITS)
    by_path = {m.path: m for m in members}
    assert by_path["photo.png"].error_code == "extension_not_allowed"
    assert by_path["noext"].error_code == "extension_not_allowed"
    assert by_path["note.TXT"].error_code is None, "extensions are compared in lower case"


@pytest.mark.parametrize("name", ["inner.zip", "inner.7z", "inner.tar.gz", "INNER.ZIP"])
def test_a_nested_archive_is_set_aside_as_nested_not_as_an_unknown_extension(tmp_path, name):
    members, _ = plan_archive(make_zip(tmp_path, [(name, b"PK")]), LIMITS)
    assert members[0].error_code == "nested_archive"


def test_a_password_protected_entry_is_set_aside(tmp_path):
    path = make_zip(tmp_path, [("secret.txt", b"x"), ("open.txt", b"x")])
    data = bytearray(path.read_bytes())
    # set the "encrypted" bit (bit 0) of the general purpose flags in the central directory record for secret.txt
    central = data.index(b"PK\x01\x02")
    (flags,) = struct.unpack_from("<H", data, central + 8)
    struct.pack_into("<H", data, central + 8, flags | 0x1)
    path.write_bytes(bytes(data))
    members, _ = plan_archive(path, LIMITS)
    by_path = {m.path: m for m in members}
    assert by_path["secret.txt"].error_code == "password_protected"
    assert by_path["open.txt"].error_code is None


def test_a_file_that_is_not_a_zip_refuses_the_upload(tmp_path):
    path = tmp_path / "fake.zip"
    path.write_bytes(b"this is not a zip file at all")
    with pytest.raises(ArchiveRejected) as caught:
        plan_archive(path, LIMITS)
    assert caught.value.code == "not_a_zip"


def test_an_archive_with_nothing_to_read_is_refused(tmp_path):
    assert rejected(tmp_path, [("__MACOSX/._a.txt", b"x"), ("d/", b"")]).code == "empty_archive"


def test_mac_and_windows_leftovers_are_ignored_and_counted(tmp_path):
    members, ignored = plan_archive(
        make_zip(tmp_path, [("a.txt", b"x"), ("__MACOSX/._a.txt", b"x"), ("d/.DS_Store", b"x"), ("d/Thumbs.db", b"x"), ("._b.txt", b"x")]), LIMITS
    )
    assert [m.path for m in members] == ["a.txt"]
    assert ignored == 4


def test_an_entry_whose_data_is_damaged_is_reported_by_unpack_and_the_next_one_still_unpacks(tmp_path):
    path = make_zip(tmp_path, [("bad.txt", b"hello hello hello"), ("good.txt", b"fine")])
    data = bytearray(path.read_bytes())
    start = data.index(b"hello hello hello")
    data[start] ^= 0xFF  # flip a byte of the stored data: the CRC no longer matches
    path.write_bytes(bytes(data))
    members, _ = plan_archive(path, LIMITS)
    results = {m.path: (p, o) for m, p, o in unpack(path, members, tmp_path / "out")}
    assert results["bad.txt"][0] is None and "bad.txt" in results["bad.txt"][1]
    assert results["good.txt"][0] is not None


def test_a_set_aside_member_is_never_unpacked(tmp_path):
    path = make_zip(tmp_path, [("photo.png", b"x"), ("a.txt", b"x")])
    members, _ = plan_archive(path, LIMITS)
    unpacked = [m.path for m, p, _ in unpack(path, members, tmp_path / "out") if p is not None]
    assert unpacked == ["a.txt"]
    assert len(list((tmp_path / "out").iterdir())) == 1


def test_a_path_that_cannot_be_opened_refuses_the_archive_with_its_own_code(tmp_path):
    with pytest.raises(ArchiveRejected) as caught:
        plan_archive(tmp_path / "missing.zip", LIMITS)
    assert caught.value.code == "unreadable_archive" and "missing.zip" in caught.value.reason
