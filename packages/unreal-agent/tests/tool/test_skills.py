from pathlib import Path

import pytest
from unreal_agent.tool import Registry, Skill, StaticTranslators, discover_skills


def write(path: Path, contents: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(contents.encode())
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


@pytest.mark.parametrize(
    ("metadata", "want_description"),
    [
        pytest.param(
            "name: \"review\"\ndescription: 'Review code.'\n", "Review code.", id="quoted"
        ),
        pytest.param(
            "name: review\ndescription: >-\n  Review code\n  carefully.\n",
            "Review code carefully.",
            id="folded",
        ),
        pytest.param(
            "name: review\ndescription: |\n  Review code.\n  Check tests.\n",
            "Review code.\nCheck tests.",
            id="literal",
        ),
        pytest.param(
            "name: review # stable name\ndescription: Review code. # instructions\n",
            "Review code.",
            id="comments",
        ),
        pytest.param(
            'name: review\ndescription: "Review: code # carefully."\n',
            "Review: code # carefully.",
            id="quoted punctuation",
        ),
        pytest.param(
            'name: "review"\r\ndescription: >-\r\n  Review code\r\n  carefully.\r\n',
            "Review code carefully.",
            id="CRLF",
        ),
        pytest.param(
            "name: review\ndescription: 1.10\nversion: [1, 2]\n",
            "1.10",
            id="non-string scalar keeps source text",
        ),
    ],
)
def test_discover_skills_yaml_metadata(
    tmp_path: Path, metadata: str, want_description: str
) -> None:
    path = tmp_path / "review" / "SKILL.md"
    write(path, "---\n" + metadata + "---\nBody: [not YAML\n")
    skills, errors = discover_skills(str(tmp_path))
    assert errors == []
    assert skills == [Skill(name="review", description=want_description, path=str(path))]


@pytest.mark.parametrize(
    ("contents", "want_error"),
    [
        pytest.param(
            "name: invalid\n",
            "missing opening YAML frontmatter delimiter",
            id="missing opening delimiter",
        ),
        pytest.param(
            "---\nname: invalid\n",
            "missing closing YAML frontmatter delimiter",
            id="missing closing delimiter",
        ),
        pytest.param(
            "---\nname: invalid\ndescription: [unfinished\n---\n",
            "decode YAML frontmatter",
            id="invalid YAML",
        ),
        pytest.param(
            "---\nname: [invalid]\ndescription: Invalid skill.\n---\n",
            "decode YAML frontmatter: yaml: unmarshal errors:\n"
            "  line 1: cannot unmarshal !!seq into string",
            id="invalid name type",
        ),
        pytest.param(
            "---\nname: invalid\ndescription: {nested: value}\n---\n",
            "decode YAML frontmatter: yaml: unmarshal errors:\n"
            "  line 2: cannot unmarshal !!map into string",
            id="invalid description type",
        ),
        pytest.param(
            "---\nname: invalid\nname: duplicate\ndescription: Invalid skill.\n---\n",
            "decode YAML frontmatter: yaml: unmarshal errors:\n"
            '  line 2: mapping key "name" already defined at line 1',
            id="duplicate key",
        ),
        pytest.param(
            "---\njust a sentence\n---\n",
            "decode YAML frontmatter: yaml: unmarshal errors:\n"
            "  line 1: cannot unmarshal !!str `just a ...` into tool.skillFrontmatter",
            id="scalar document",
        ),
    ],
)
def test_discover_skills_continues_after_invalid_frontmatter(
    tmp_path: Path, contents: str, want_error: str
) -> None:
    directory = tmp_path / "skills"
    invalid_path = directory / "invalid" / "SKILL.md"
    valid_path = directory / "valid" / "SKILL.md"
    write(invalid_path, contents)
    write(valid_path, "---\nname: valid\ndescription: Valid skill.\n---\n")
    skills, errors = discover_skills(str(directory))
    assert skills == [Skill(name="valid", description="Valid skill.", path=str(valid_path))]
    assert len(errors) == 1
    assert str(invalid_path) in str(errors[0])
    assert want_error in str(errors[0])


def test_discover_skills_rejects_invalid_metadata_and_duplicate_names(tmp_path: Path) -> None:
    for name, contents in (
        ("alpha", "---\nname: review\ndescription: Review code.\n---\n"),
        ("duplicate", '---\nname: "review"\ndescription: Duplicate.\n---\n'),
        ("missing-name", "---\ndescription: Missing name.\n---\n"),
        ("missing-description", "---\nname: missing-description\n---\n"),
        ("empty-name", "---\nname: ' '\ndescription: Empty name.\n---\n"),
        ("empty-description", "---\nname: empty-description\ndescription: ''\n---\n"),
        ("null-name", "---\nname: ~\ndescription: Null name.\n---\n"),
    ):
        write(tmp_path / name / "SKILL.md", contents)
    skills, errors = discover_skills(str(tmp_path))
    assert skills == [
        Skill(name="review", description="Review code.", path=str(tmp_path / "alpha" / "SKILL.md"))
    ]
    assert len(errors) == 6
    registry = Registry(StaticTranslators())
    for skill in skills:
        registry.register_skill(skill)
