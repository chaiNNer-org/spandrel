from pathlib import Path


def main():
    github_file_log = Path("tests/_github_files.txt").resolve()

    if not github_file_log.exists():
        print("No _github_files.txt file found")
        return

    lines = list(filter(bool, github_file_log.read_text().splitlines()))
    print(f"Removing {len(lines)} files...")
    for line in lines:
        file = Path(line)
        if file.exists():
            file.unlink()
            print(f"Removed {file}")
        else:
            print(f"File not found: {file}")


if __name__ == "__main__":
    main()
