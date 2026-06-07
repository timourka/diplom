from pathlib import Path
from datetime import datetime
import argparse

EXCLUDE_DIR_NAMES = {
    ".git",
    ".dart_tool",
    ".idea",
    ".vscode",
    "build",
    ".gradle",
    ".cxx",
    ".externalNativeBuild",
    "DerivedData",
    "Pods",
    ".symlinks",
    "ephemeral",
    "generated",
    "intermediates",
    "outputs",
    "tmp",
    "temp",
    "__pycache__",
}

EXCLUDE_FILE_NAMES = {
    "pubspec.lock",
    ".metadata",
    "GeneratedPluginRegistrant.java",
    "GeneratedPluginRegistrant.kt",
    "GeneratedPluginRegistrant.swift",
    "GeneratedPluginRegistrant.m",
    "GeneratedPluginRegistrant.h",
    "firebase_options.dart",
}

GENERATED_DART_SUFFIXES = (
    ".g.dart",
    ".freezed.dart",
    ".gr.dart",
    ".gen.dart",
    ".mocks.dart",
)

ROOT_FILES = {
    "pubspec.yaml",
    "analysis_options.yaml",
    "README.md",
}

TEXT_CODE_EXTENSIONS = {
    ".dart",
    ".kt",
    ".kts",
    ".java",
    ".swift",
    ".m",
    ".mm",
    ".h",
    ".gradle",
    ".xml",
    ".yaml",
    ".yml",
    ".properties",
    ".plist",
    ".pbxproj",
    ".entitlements",
    ".arb",
}

SECRET_FILE_NAMES = {
    ".env",
    "google-services.json",
    "GoogleService-Info.plist",
}


def to_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def has_excluded_parent(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return any(part in EXCLUDE_DIR_NAMES for part in relative.parts)


def is_generated_dart(path: Path) -> bool:
    name = path.name
    return any(name.endswith(suffix) for suffix in GENERATED_DART_SUFFIXES)


def is_under(parts: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(parts) >= len(prefix) and parts[:len(prefix)] == prefix


def is_allowed_flutter_source(path: Path, root: Path, include_platform: bool, include_assets_text: bool) -> bool:
    relative = path.relative_to(root)
    parts = relative.parts
    name = path.name
    suffix = path.suffix.lower()

    if name in ROOT_FILES:
        return True

    if suffix not in TEXT_CODE_EXTENSIONS:
        return False

    # Основной код Flutter-приложения
    if is_under(parts, ("lib",)):
        return True

    if is_under(parts, ("test",)):
        return True

    if is_under(parts, ("integration_test",)):
        return True

    if is_under(parts, ("tool",)):
        return True

    # Иногда в assets лежат json/yaml/arb, но по умолчанию не тащим ассеты
    if include_assets_text and is_under(parts, ("assets",)):
        return True

    if not include_platform:
        return False

    # Android: только ручной код и важные конфиги, без .cxx/build/intermediates
    if is_under(parts, ("android",)):
        android_allowed_exact = {
            ("android", "settings.gradle"),
            ("android", "settings.gradle.kts"),
            ("android", "build.gradle"),
            ("android", "build.gradle.kts"),
            ("android", "gradle.properties"),
            ("android", "app", "build.gradle"),
            ("android", "app", "build.gradle.kts"),
            ("android", "app", "src", "main", "AndroidManifest.xml"),
        }

        if parts in android_allowed_exact:
            return True

        if is_under(parts, ("android", "app", "src", "main", "kotlin")):
            return suffix in {".kt", ".kts"}

        if is_under(parts, ("android", "app", "src", "main", "java")):
            return suffix == ".java"

        if is_under(parts, ("android", "app", "src", "main", "res", "values")):
            return suffix == ".xml"

        if is_under(parts, ("android", "app", "src", "main", "res", "xml")):
            return suffix == ".xml"

        return False

    # iOS: только Runner и важные конфиги, без Pods/DerivedData
    if is_under(parts, ("ios",)):
        ios_allowed_exact = {
            ("ios", "Podfile"),
            ("ios", "Runner.xcodeproj", "project.pbxproj"),
            ("ios", "Runner", "Info.plist"),
            ("ios", "Runner", "Runner.entitlements"),
            ("ios", "Runner", "DebugProfile.entitlements"),
            ("ios", "Runner", "Release.entitlements"),
        }

        if parts in ios_allowed_exact:
            return True

        if is_under(parts, ("ios", "Runner")):
            return suffix in {".swift", ".m", ".mm", ".h", ".plist", ".entitlements"}

        return False

    return False


def should_skip_file(path: Path, root: Path, include_secrets: bool) -> bool:
    if has_excluded_parent(path, root):
        return True

    if path.name in EXCLUDE_FILE_NAMES:
        return True

    if not include_secrets and path.name in SECRET_FILE_NAMES:
        return True

    if path.suffix.lower() == ".dart" and is_generated_dart(path):
        return True

    return False


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            pass

    return path.read_bytes().decode("utf-8", errors="replace")


def collect_files(root: Path, include_platform: bool, include_assets_text: bool, include_secrets: bool) -> list[Path]:
    files = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if should_skip_file(path, root, include_secrets):
            continue

        if is_allowed_flutter_source(path, root, include_platform, include_assets_text):
            files.append(path)

    return sorted(files, key=lambda p: to_posix(p.relative_to(root)).lower())


def main():
    parser = argparse.ArgumentParser(
        description="Собирает только полезный код Flutter-приложения из mobile_app, без build/.cxx/.dart_tool."
    )

    parser.add_argument(
        "folder",
        nargs="?",
        default="mobile_app",
        help="Папка мобильного приложения. По умолчанию: mobile_app",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="mobile_app_code_dump.txt",
        help="Файл результата. По умолчанию: mobile_app_code_dump.txt",
    )

    parser.add_argument(
        "--no-platform",
        action="store_true",
        help="Не включать android/ios platform-код, собрать только Flutter-код из lib/test.",
    )

    parser.add_argument(
        "--include-assets-text",
        action="store_true",
        help="Включить текстовые файлы из assets.",
    )

    parser.add_argument(
        "--include-secrets",
        action="store_true",
        help="Включить потенциально чувствительные файлы: .env, google-services.json, GoogleService-Info.plist.",
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
        include_platform=not args.no_platform,
        include_assets_text=args.include_assets_text,
        include_secrets=args.include_secrets,
    )

    with output.open("w", encoding="utf-8", newline="\n") as out:
        out.write("# Сборка кода мобильного приложения\n\n")
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