# test_install_skills.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

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


def test_every_reference_is_linked_from_skill():
	src = _bundled_skill_dir()
	text = (src / "SKILL.md").read_text()

	for path in sorted((src / "references").glob("*.md")):
		assert "references/{}".format(path.name) in text, path.name


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
