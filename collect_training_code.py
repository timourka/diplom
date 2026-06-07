from pathlib import Path
from datetime import datetime
import argparse
import fnmatch

EXCLUDE_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",

    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",

    "data",
    "datasets",
    "dataset",
    "uploads",
    "extracted",
    "labels",
    "images",
    "videos",
    "frames",

    "runs",
    "weights",
    "models",
    "checkpoints",
    "artifacts",
    "outputs",
    "results",
    "logs",

    "tmp",
    "temp",
    "build",
    "dist",
}

CODE_EXTENSIONS = {
    ".py",
    ".ipynb",
}

CONFIG_EXTENSIONS = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
}

PROJECT_EXTENSIONS = {
    ".txt",   # requirements.txt
    ".md",    # README, если нужен
    ".dockerfile",
}

ALLOWED_ROOT_FILES = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".gitignore",
    ".dockerignore",
    "README.md",
}

EXCLUDE_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",

    "debug.log",
    "train.log",
    "last.pt",
    "best.pt",
}

EXCLUDE_FILE_PATTERNS = {
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.tflite",
    "*.engine",
    "*.weights",

    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.webp",
    "*.bmp",
    "*.gif",

    "*.mp4",
    "*.mov",
    "*.avi",
    "*.mkv",

    "*.zip",
    "*.rar",
    "*.7z",
    "*.tar",
    "*.gz",

    "*.db",
    "*.sqlite",
    "*.sqlite3",

    "*.log",
    "*.cache",
    "*.tmp",

    "*.csv",
    "*.tsv",

    # YOLO-разметка кадров, обычно это не код
    "frame_*.txt",
}

GENERATED_FILE_PATTERNS = {
    "*_pb2.py",
    "*_pb2_grpc.py",
}


def to_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def has_excluded_parent(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return any(part in EXCLUDE_DIR_NAMES for part in parts)


def matches_any(name: str, patterns: set[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def is_allowed_file(path: Path, root: Path, include_configs: bool, include_docs: bool) -> bool:
    relative = path.relative_to(root)
    name = path.name
    suffix = path.suffix.lower()

    if len(relative.parts) == 1 and name in ALLOWED_ROOT_FILES:
        if name == "README.md":
            return include_docs
        return True

    if suffix in CODE_EXTENSIONS:
        return True

    if include_configs and suffix in CONFIG_EXTENSIONS:
        return True

    if include_docs and suffix == ".md":
        return True

    # requirements-*.txt оставляем, но обычные .txt не берём
    if name.startswith("requirements") and suffix == ".txt":
        return True

    return False


def should_skip_file(path: Path, root: Path, include_generated: bool) -> bool:
    name = path.name

    if has_excluded_parent(path, root):
        return True

    if name in EXCLUDE_FILE_NAMES:
        return True

    if matches_any(name, EXCLUDE_FILE_PATTERNS):
        return True

    if not include_generated and matches_any(name, GENERATED_FILE_PATTERNS):
        return True

    return False


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return path.read_bytes().decode("utf-8", errors="replace")


def collect_files(root: Path, include_configs: bool, include_docs: bool, include_generated: bool) -> list[Path]:
    files = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if should_skip_file(path, root, include_generated):
            continue

        if is_allowed_file(path, root, include_configs, include_docs):
            files.append(path)

    return sorted(files, key=lambda p: to_posix(p.relative_to(root)).lower())


def main():
    parser = argparse.ArgumentParser(
        description="Собирает код сервиса обучения из training_service без датасетов, весов моделей и артефактов обучения."
    )

    parser.add_argument(
        "folder",
        nargs="?",
        default="training_service",
        help="Папка сервиса обучения. По умолчанию: training_service",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="training_service_code_dump.txt",
        help="Файл результата. По умолчанию: training_service_code_dump.txt",
    )

    parser.add_argument(
        "--no-configs",
        action="store_true",
        help="Не включать json/yaml/toml/ini/cfg конфиги.",
    )

    parser.add_argument(
        "--include-docs",
        action="store_true",
        help="Включить README и markdown-документацию.",
    )

    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Включить сгенерированные *_pb2.py и *_pb2_grpc.py.",
    )

    args = parser.parse_args()

    root = Path(args.folder).resolve()
    output = Path(args.output).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Папка не найдена: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Это не папка: {root}")

    files = collect_files(
        root=root,
        include_configs=not args.no_configs,
        include_docs=args.include_docs,
        include_generated=args.include_generated,
    )

    with output.open("w", encoding="utf-8", newline="\n") as out:
        out.write("# Сборка кода сервиса обучения\n\n")
        out.write(f"Папка: {root}\n")
        out.write(f"Дата сборки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Количество файлов: {len(files)}\n\n")

        out.write("## Включённые файлы\n\n")
        out.write("```text\n")
        for file in files:
            out.write(to_posix(file.relative_to(root)) + "\n")
        out.write("```\n\n")

        out.write("## Содержимое файлов\n\n")

        for file in files:
            relative = to_posix(file.relative_to(root))

            out.write("\n")
            out.write("=" * 100)
            out.write("\n")
            out.write(f"Файл: {relative}\n")
            out.write("=" * 100)
            out.write("\n\n")

            try:
                content = read_text_file(file)
            except Exception as exc:
                out.write(f"[Не удалось прочитать файл: {exc}]\n")
                continue

            out.write(content)

            if not content.endswith("\n"):
                out.write("\n")

    print(f"Готово. Собрано файлов: {len(files)}")
    print(f"Результат сохранён в: {output}")


if __name__ == "__main__":
    main()