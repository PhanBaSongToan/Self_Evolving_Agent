import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANNED = {"torch", "torchvision", "tensorflow", "keras", "transformers", "sentence_transformers", "sklearn.neural_network"}


def test_production_imports_and_dependencies_exclude_deep_learning():
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any(name == banned or name.startswith(banned + ".") for name in imported for banned in BANNED), path
    for path in (ROOT / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"\bMLPClassifier\b|\bembed(?:ding)?\s*\(", source), path
    dependency_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert not any(token in dependency_text for token in ("torch", "tensorflow", "keras", "transformers", "sentence-transformers"))
