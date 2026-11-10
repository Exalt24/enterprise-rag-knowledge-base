"""
Archive guard: what a zip must satisfy before any of it is unpacked.

A data room arrives as one zip from someone else, so the zip is treated as hostile. Everything here reads the zip's own
file table (names, declared sizes, flags) and writes nothing. Two kinds of problem are told apart:

- The whole archive is refused (ArchiveRejected): it is not a zip, an entry name tries to leave the unpack folder (zip
  slip), it holds more files than the cap, or its entries add up to more than the size cap once unpacked (zip bomb).
  Nothing is unpacked.
- One entry is set aside (Member.error_code): it is bigger than the per file cap, has an extension the loaders do not
  read, is itself an archive, or is password protected. The entry is recorded with its reason and the rest go on.

Names never touch the file system: unpacked files are stored under their SHA-256 (unpack), so even a name the
checks missed could not choose where a file lands.
"""
import hashlib
import re
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

ARCHIVE_EXTENSIONS = frozenset({".zip", ".7z", ".rar", ".tar", ".gz", ".tgz", ".bz2", ".xz"})
_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")
_JUNK_PARTS = ("__MACOSX",)
_READ_BLOCK = 1024 * 1024


class ArchiveRejected(Exception):
    """The whole archive is refused. `code` is stable and meant for tests and the status endpoint."""

    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code = code
        self.reason = reason


@dataclass(frozen=True)
class ArchiveLimits:
    max_files: int
    max_total_bytes: int
    max_file_bytes: int
    allowed_extensions: frozenset


@dataclass(frozen=True)
class Member:
    """One file entry of the archive. error_code is None when it is fit to be read."""

    path: str  # the name inside the archive, checked and with forward slashes
    size: int  # uncompressed size as the zip declares it
    error_code: Optional[str] = None
    error_reason: Optional[str] = None
    info: Optional[zipfile.ZipInfo] = field(default=None, compare=False, repr=False)  # the entry, to open it later


def checked_member_path(name: str) -> str:
    """The entry name as a safe relative path with forward slashes, or ArchiveRejected("path_traversal")."""
    unified = name.replace("\\", "/")
    if unified.startswith("/") or _DRIVE_LETTER.match(unified):
        raise ArchiveRejected("path_traversal", f"entry name {name!r} is an absolute path")
    parts = [part for part in unified.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise ArchiveRejected("path_traversal", f"entry name {name!r} climbs out of the folder it is unpacked into")
    return "/".join(parts)


def _is_junk(path: str) -> bool:
    parts = path.split("/")
    return any(part in _JUNK_PARTS for part in parts) or parts[-1].startswith("._") or parts[-1] in (".DS_Store", "Thumbs.db")


def _entry_problem(path: str, info: zipfile.ZipInfo, limits: ArchiveLimits) -> Optional[Tuple[str, str]]:
    suffix = Path(path).suffix.lower()
    if suffix in ARCHIVE_EXTENSIONS:
        return "nested_archive", f"{path} is an archive inside the archive; nested archives are not opened"
    if suffix not in limits.allowed_extensions:
        allowed = ", ".join(sorted(limits.allowed_extensions))
        return "extension_not_allowed", f"{path} has an extension the loaders do not read (allowed: {allowed})"
    if info.flag_bits & 0x1:
        return "password_protected", f"{path} is stored encrypted inside the archive"
    if info.file_size > limits.max_file_bytes:
        return "file_too_large", f"{path} unpacks to {info.file_size} bytes; the per file cap is {limits.max_file_bytes}"
    return None


def plan_archive(zip_path, limits: ArchiveLimits) -> Tuple[List[Member], int]:
    """Check the archive and list its file entries. Returns (members, ignored) where ignored counts junk entries.

    Raises ArchiveRejected when the archive as a whole must be refused. Nothing is unpacked.
    """
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as error:
        raise ArchiveRejected("not_a_zip", f"the upload is not a readable zip archive: {error}")
    except OSError as error:
        raise ArchiveRejected("unreadable_archive", f"the archive could not be opened: {error}")
    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) > limits.max_files:
            raise ArchiveRejected("too_many_files", f"the archive holds {len(infos)} files; the cap is {limits.max_files}")
        declared = sum(info.file_size for info in infos)
        if declared > limits.max_total_bytes:
            raise ArchiveRejected(
                "archive_too_large",
                f"the files add up to {declared} bytes unpacked; the cap is {limits.max_total_bytes}",
            )
        members: List[Member] = []
        ignored = 0
        for info in infos:
            path = checked_member_path(info.filename)  # a bad name anywhere refuses the whole archive
            if _is_junk(path):
                ignored += 1
                continue
            problem = _entry_problem(path, info, limits)
            members.append(Member(path, info.file_size, *(problem or (None, None)), info=info))
    if not members:
        raise ArchiveRejected("empty_archive", "the archive holds no files to read")
    return members, ignored


def unpack(zip_path, members: List[Member], dest_dir: Path) -> Iterator[Tuple[Member, Optional[Path], Optional[str]]]:
    """Unpack the fit members one at a time into dest_dir, each under its SHA-256.

    Yields (member, path, sha256) for a file written and (member, None, reason) for an entry that turned out to be
    damaged. Members that already carry an error_code are not touched. The bytes go to a temporary name first and are
    renamed once the digest is known, so an entry's own name never reaches the file system. zipfile reads no more than
    the size an entry declares and checks the CRC at the end, so an entry whose real size differs from its header fails
    here instead of growing.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    temp = dest_dir / ".part"
    with zipfile.ZipFile(zip_path) as archive:
        for member in members:
            if member.error_code is not None:
                continue
            digest = hashlib.sha256()
            try:
                with archive.open(member.info) as source, open(temp, "wb") as target:
                    for block in iter(lambda: source.read(_READ_BLOCK), b""):
                        digest.update(block)
                        target.write(block)
            except (zipfile.BadZipFile, RuntimeError, EOFError, zlib.error, NotImplementedError) as error:
                yield member, None, f"{member.path} could not be unpacked: {error}"
                continue
            final = dest_dir / f"{digest.hexdigest()}{Path(member.path).suffix.lower()}"
            temp.replace(final)
            yield member, final, digest.hexdigest()
    temp.unlink(missing_ok=True)
