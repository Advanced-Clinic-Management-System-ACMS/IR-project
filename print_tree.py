from pathlib import Path

IGNORE = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "venv",
    ".venv",
    "node_modules",
    ".DS_Store"
}


def build_tree(directory: Path, prefix: str = ""):
    lines = []

    entries = sorted(
        [e for e in directory.iterdir() if e.name not in IGNORE],
        key=lambda x: (x.is_file(), x.name.lower())
    )

    for index, entry in enumerate(entries):
        is_last = index == len(entries) - 1

        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + entry.name)

        if entry.is_dir():
            extension = "    " if is_last else "│   "
            lines.extend(build_tree(entry, prefix + extension))

    return lines


if __name__ == "__main__":
    root = Path(".").resolve()

    tree_lines = [root.name]
    tree_lines.extend(build_tree(root))

    output_file = "project_structure.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tree_lines))

    print(f"Project structure saved to '{output_file}'")