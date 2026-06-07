from pathlib import Path
from datetime import datetime
import argparse
import fnmatch

EXCLUDE_DIR_NAMES = {
    ".git",
    ".vs",
    ".vscode",
    ".idea",

    "bin",
    "obj",
    "packages",
    "node_modules",
    "TestResults",
    "coverage",
    ".nuget",

    # твой мусор/данные/выгрузки
    "uploads",
    "datasets",
    "dataset",
    "data",
    "extracted",
    "labels",
    "images",
    "runs",
    "weights",
    "logs",
    "temp",
    "tmp",
    "publish",
    "wwwroot",
}

CODE_EXTENSIONS = {
    ".cs",
    ".csproj",
    ".sln",
    ".props",
    ".targets",
    ".cshtml",
    ".razor",
}

CONFIG_EXTENSIONS = {
    ".json",
    ".xml",
    ".config",
    ".yml",
    ".yaml",
}

EXCLUDE_FILE_NAMES = {
    "project.assets.json",
    "project.nuget.cache",
    "packages.lock.json",

    # секреты/локальные настройки
    "appsettings.Development.json",
    "appsettings.Local.json",
    "secrets.json",
    ".env",
}

EXCLUDE_FILE_PATTERNS = {
    "*.dll",
    "*.exe",
    "*.pdb",
    "*.cache",
    "*.user",
    "*.suo",
    "*.nupkg",
    "*.snupkg",
    "*.pfx",
    "*.cer",
    "*.key",
    "*.pem",
    "*.log",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.bak",
    "*.zip",
    "*.rar",
    "*.7z",

    # данные и картинки не нужны
    "*.txt",
    "*.md",
    "*.csv",
    "*.tsv",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.ico",
    "*.mp4",
    "*.mov",
    "*.avi",
    "*.pt",
    "*.onnx",
    "*.tflite",
}

GENERATED_FILE_PATTERNS = {
    "*.g.cs",
    "*.g.i.cs",
    "*.AssemblyInfo.cs",
    "*.AssemblyAttributes.cs",
    "*.GlobalUsings.g.cs",
}


def to_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def has_excluded_parent(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return any(part in EXCLUDE_DIR_NAMES for part in parts)


def matches_any(name: str, patterns: set[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def is_allowed_file(path: Path, include_configs: bool) -> bool:
    suffix = path.suffix.lower()

    if suffix in CODE_EXTENSIONS:
        return True

    if include_configs and suffix in CONFIG_EXTENSIONS:
        return True

    return False


def should_skip_file(path: Path, root: Path, include_configs: bool, include_generated: bool) -> bool:
    name = path.name

    if has_excluded_parent(path, root):
        return True

    if name in EXCLUDE_FILE_NAMES:
        return True

    if matches_any(name, EXCLUDE_FILE_PATTERNS):
        return True

    if not include_generated and matches_any(name, GENERATED_FILE_PATTERNS):
        return True

    if not is_allowed_file(path, include_configs):
        return True

    return False


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return path.read_bytes().decode("utf-8", errors="replace")


def collect_files(root: Path, include_configs: bool, include_generated: bool) -> list[Path]:
    files = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if should_skip_file(path, root, include_configs, include_generated):
            continue

        files.append(path)

    return sorted(files, key=lambda p: to_posix(p.relative_to(root)).lower())


def main():
    parser = argparse.ArgumentParser(
        description="Собирает только исходный код .NET-решения без uploads/datasets/bin/obj и прочего мусора."
    )

    parser.add_argument(
        "folder",
        nargs="?",
        default="netSolution",
        help="Папка .NET-решения. По умолчанию: netSolution",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="net_solution_code_dump.txt",
        help="Файл результата. По умолчанию: net_solution_code_dump.txt",
    )

    parser.add_argument(
        "--no-configs",
        action="store_true",
        help="Не включать json/xml/yaml/config файлы, оставить только .cs/.csproj/.sln/.razor/.cshtml.",
    )

    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Включить сгенерированные .g.cs, AssemblyInfo.cs и т.п.",
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
        include_generated=args.include_generated,
    )

    with output.open("w", encoding="utf-8", newline="\n") as out:
        out.write("# Сборка кода .NET-решения\n\n")
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