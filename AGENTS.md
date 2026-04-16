# Repository Guidelines

## Project Structure & Module Organization
This repository is currently data-centric.
- `README.md`: short project description for the Machine Vision course context.
- `Dataset_Projeto1/`: main dataset root.
- `Dataset_Projeto1/_Eucalipto_Escolhidos1/`: eucalyptus images (`Eucalipto1.jpg`, etc.).
- `Dataset_Projeto1/_Pinheiro_Escolhidos1/`: pine images (`Pinheiro1.jpg`, etc.).

Keep class-specific assets grouped in clearly named subfolders under `Dataset_Projeto1/`. Use descriptive, stable folder names and avoid mixing species/classes in one directory.

## Build, Test, and Development Commands
There is no build pipeline or automated test suite configured yet. Use lightweight validation before committing:
- `ls -R Dataset_Projeto1`: verify expected folder/file layout.
- `find Dataset_Projeto1 -type f | sort`: review tracked dataset files.
- `git status`: confirm only intended files changed.

If you add scripts later, place them in a dedicated `scripts/` directory and document usage in `README.md`.

## Coding Style & Naming Conventions
For new code or scripts:
- Use 4-space indentation and UTF-8 text files.
- Prefer snake_case for Python filenames/functions and descriptive variable names.
- Keep scripts small and single-purpose (for example, dataset validation or preprocessing).

For dataset assets:
- Preserve consistent naming patterns inside each class folder (for example, `Pinheiro1.jpg`, `Pinheiro2.jpg`).
- Avoid renaming existing files unless required; renames make history and diffs harder to audit.

## Testing Guidelines
No formal coverage target exists today. Minimum validation for contributions:
- Confirm files open correctly and are not corrupted.
- Confirm class folders contain only relevant images.
- For added code, include a simple reproducible check command in the PR description.

## Commit & Pull Request Guidelines
Current history uses short, imperative commit subjects (for example, `Initial commit`, `Add initial dataset images...`). Follow this pattern:
- Commit message format: imperative verb + concise scope.
- Keep commits focused (dataset update, script addition, or docs change).

For pull requests, include:
- What changed and why.
- A brief validation summary (commands run).
- Sample paths of added/changed files (for example, `Dataset_Projeto1/_Pinheiro_Escolhidos1/Pinheiro4.jpg`).
