from pathlib import Path

from unreal_agent.tool import Skill, StaticTranslators, discover_skills, new_registry


def write(path: Path, contents: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(0o600)


def test_discover_skills(tmp_path: Path) -> None:
    directory = tmp_path / "skills"
    write(
        directory / "alpha" / "SKILL.md",
        "---\nname: alpha\ndescription: Handle alpha tasks.\n---\n\nAlpha instructions.\n",
    )
    write(
        directory / "beta" / "SKILL.md",
        "---\nname: beta\ndescription: Handle beta tasks.\nmetadata:\n  ignored: true\n---\n\n"
        "Beta instructions.\n",
    )
    skills, errors = discover_skills(str(directory))
    assert errors == []
    assert skills == [
        Skill(
            name="alpha",
            description="Handle alpha tasks.",
            path=str(directory / "alpha" / "SKILL.md"),
        ),
        Skill(
            name="beta", description="Handle beta tasks.", path=str(directory / "beta" / "SKILL.md")
        ),
    ]


def test_discover_skills_allows_missing_directory(tmp_path: Path) -> None:
    assert discover_skills(str(tmp_path / "missing")) == ([], [])


def test_discover_skills_continues_after_invalid_frontmatter(tmp_path: Path) -> None:
    directory = tmp_path / "skills"
    invalid_path = directory / "invalid" / "SKILL.md"
    valid_path = directory / "valid" / "SKILL.md"
    write(invalid_path, "name: invalid\n")
    write(valid_path, "---\nname: valid\ndescription: Valid skill.\n---\n")
    skills, errors = discover_skills(str(directory))
    assert skills == [Skill(name="valid", description="Valid skill.", path=str(valid_path))]
    assert len(errors) == 1
    assert "missing opening YAML frontmatter delimiter" in str(errors[0])


def test_discover_skills_rejects_invalid_metadata_and_duplicate_names(tmp_path: Path) -> None:
    for name, contents in (
        ("alpha", "---\nname: review\ndescription: Review code.\n---\n"),
        ("duplicate", "---\nname: review\ndescription: Duplicate.\n---\n"),
        ("missing-name", "---\ndescription: Missing name.\n---\n"),
        ("missing-description", "---\nname: missing-description\n---\n"),
    ):
        write(tmp_path / name / "SKILL.md", contents)
    skills, errors = discover_skills(str(tmp_path))
    assert skills == [
        Skill(name="review", description="Review code.", path=str(tmp_path / "alpha" / "SKILL.md"))
    ]
    assert len(errors) == 3
    registry = new_registry(StaticTranslators())
    for skill in skills:
        registry.register_skill(skill)
