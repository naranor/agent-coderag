#!/usr/bin/env python3
import subprocess
import sys
import json

def run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def main():
    print("--- Starting Python Code Audit ---")
    
    # 1. Check if ruff is available
    _, _, code = run_command(["ruff", "--version"])
    if code != 0:
        print("Error: 'ruff' is not installed or not in PATH.")
        sys.exit(1)

    # 2. Run ruff check
    print("Running ruff check...")
    stdout, stderr, _ = run_command(["ruff", "check", ".", "--format", "json"])
    
    issues = []
    if stdout:
        try:
            issues = json.loads(stdout)
        except Exception:
            print("Error parsing ruff output.")
    
    if stderr:
        print(f"Ruff stderr: {stderr}")

    # 3. Categorize and summarize
    print(f"Total issues found: {len(issues)}")
    
    summary = {}
    for issue in issues:
        code = issue.get("code", "UNKNOWN")
        summary[code] = summary.get(code, 0) + 1
    
    if summary:
        print("\nIssue Summary:")
        for code, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
            print(f"- {code}: {count} occurrences")
    
    # 4. Detailed top issues (limit to 20 to avoid context overflow)
    if issues:
        print("\nTop 20 Issues:")
        for issue in issues[:20]:
            print(f"[{issue.get('code')}] {issue.get('filename')}:{issue.get('location', {}).get('row')}: {issue.get('message')}")
            
    print("\n--- Audit Complete ---")

if __name__ == "__main__":
    main()
