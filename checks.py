from subprocess import run

if __name__ == "__main__":
    commands = [
        "uv run ruff check . --fix --select I",
        "uv run ruff format .",
        "uv run mypy --config-file pyproject.toml .",
    ]
    for command in commands:
        result = run(command, shell=True, capture_output=True, text=True)
        print(f"$ {command}")
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
        print(f"[exit {result.returncode}]")
