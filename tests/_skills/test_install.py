# test_install.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import re

import pytest

from pathlib import Path

from ledidi._skills.install import SKILL_NAME
from ledidi._skills.install import _bundled_skill_dir
from ledidi._skills.install import _default_dest
from ledidi._skills.install import install_skill
from ledidi._skills.install import main


def test_bundled_skill_dir_exists():
	src = _bundled_skill_dir()
	assert src.is_dir()
	assert (src / "SKILL.md").is_file()
	assert (src / "references").is_dir()


def test_bundled_skill_frontmatter():
	text = (_bundled_skill_dir() / "SKILL.md").read_text()
	assert text.startswith("---\n")
	assert "name: ledidi" in text
	assert "description:" in text


# Reference files that ledidi's skill points at by name but does not own: they
# ship with tangermeme's skill, which the text names explicitly at every one of
# these mentions. They use the same `references/x.md` form, so the resolution
# check below has to exempt them, and the exemption is asserted disjoint from
# ledidi's own files so it can never hide one going missing.
TANGERMEME_REFERENCES = frozenset([
	"annotate.md",
	"comparing-models.md",
	"deep_lift_shap.md",
	"design.md",
	"io-loci.md",
	"model-wrapping.md",
	"plot.md",
])


def _skill_documents():
	"""Return every Markdown file that ships in the skill, as (name, text)."""

	src = _bundled_skill_dir()
	paths = [src / "SKILL.md"] + sorted((src / "references").glob("*.md"))
	return [(path.name, path.read_text()) for path in paths]


def test_every_reference_is_linked_from_skill():
	src = _bundled_skill_dir()
	text = (src / "SKILL.md").read_text()

	for path in sorted((src / "references").glob("*.md")):
		assert "`references/{}`".format(path.name) in text, path.name


def test_no_markdown_links_to_reference_files():
	"""Cross-references are backticked paths, never Markdown links.

	Nothing that reads a skill renders Markdown, so a link spends twice the
	characters on the same path. A link would also slip past the checks above
	and below, since neither of those looks inside link syntax.
	"""

	found = []
	for name, text in _skill_documents():
		for link in re.findall(r'\[[^\]]*\]\([^)]*\.md\)', text):
			found.append("{}: {}".format(name, link))

	assert found == [], "use `references/x.md`, not a link: {}".format(found)


def test_cross_references_are_complete_paths_that_resolve():
	"""Every backticked `*.md` is a skill-root path naming a real file.

	A bare `masks.md` does not say which directory it lives in, so an agent
	following the pointer has to search for the target. Mentions are written
	relative to the skill root, `references/masks.md`, and must name a file
	that exists -- or one of TANGERMEME_REFERENCES, which ledidi's skill
	deliberately points at in tangermeme's.
	"""

	existing = {path.name
		for path in (_bundled_skill_dir() / "references").glob("*.md")}

	assert existing.isdisjoint(TANGERMEME_REFERENCES)

	for name, text in _skill_documents():
		for target in re.findall(r'`([A-Za-z0-9_/.-]*\.md)`', text):
			assert target == "SKILL.md" or target.startswith("references/"), (
				"{} names `{}` by bare filename; write the complete "
				"skill-root-relative path".format(name, target))

			basename = target.split("/")[-1]
			assert basename in existing or basename in TANGERMEME_REFERENCES, (
				"{} points at `{}`, which is not a file in the skill".format(
					name, target))


def test_default_dest():
	assert _default_dest() == Path.home() / ".claude" / "skills" / SKILL_NAME


def test_install_skill(tmp_path):
	dest = install_skill(dest=tmp_path / "ledidi")

	assert dest == tmp_path / "ledidi"
	assert (dest / "SKILL.md").is_file()
	assert (dest / "references" / "objective.md").is_file()


def test_install_skill_default_dest(tmp_path, monkeypatch):
	monkeypatch.setattr(Path, "home", lambda: tmp_path)

	dest = install_skill()

	assert dest == tmp_path / ".claude" / "skills" / "ledidi"
	assert (dest / "SKILL.md").is_file()


def test_install_skill_existing_raises(tmp_path):
	dest = tmp_path / "ledidi"
	dest.mkdir()

	with pytest.raises(FileExistsError):
		install_skill(dest=dest)


def test_install_skill_force_overwrites(tmp_path):
	dest = install_skill(dest=tmp_path / "ledidi")
	(dest / "stale.md").write_text("stale")

	install_skill(dest=dest, force=True)

	assert not (dest / "stale.md").exists()
	assert (dest / "SKILL.md").is_file()


def test_install_skill_missing_source_raises(tmp_path, monkeypatch):
	monkeypatch.setattr("ledidi._skills.install._bundled_skill_dir",
		lambda: tmp_path / "absent")

	with pytest.raises(FileNotFoundError):
		install_skill(dest=tmp_path / "ledidi")


def test_main_installs(tmp_path, capsys):
	code = main(["--dest", str(tmp_path / "ledidi")])

	assert code == 0
	assert (tmp_path / "ledidi" / "SKILL.md").is_file()

	out = capsys.readouterr().out
	assert "Installed ledidi skill" in out
	assert "tangermeme-install-skills" in out


def test_main_print_path(tmp_path, capsys):
	code = main(["--print-path"])

	assert code == 0
	assert capsys.readouterr().out.strip() == str(_bundled_skill_dir())


def test_main_force(tmp_path):
	assert main(["--dest", str(tmp_path / "ledidi")]) == 0
	assert main(["--dest", str(tmp_path / "ledidi"), "--force"]) == 0


def test_main_existing_returns_error(tmp_path, capsys):
	dest = tmp_path / "ledidi"
	dest.mkdir()

	code = main(["--dest", str(dest)])

	assert code == 1
	assert "already exists" in capsys.readouterr().err
