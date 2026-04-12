import os
import subprocess
import sys

def run_cmd(cmd, cwd=None):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def create_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def main(repo_path):
    if not os.path.exists(repo_path):
        os.makedirs(repo_path)
    
    # Initialize git if not already
    try:
        run_cmd("git status", cwd=repo_path)
    except subprocess.CalledProcessError:
        run_cmd("git init", cwd=repo_path)
    
    # --- MAIN BRANCH BASELINE ---
    # Use checkout -B to force create or reset main
    try:
        run_cmd("git checkout main", cwd=repo_path)
    except:
        run_cmd("git checkout -b main", cwd=repo_path)
    
    create_file(os.path.join(repo_path, "README.md"), "# Demo PR Testing Repository\nContains automated PRs for AI evaluation.")
    create_file(os.path.join(repo_path, "utils.py"), "def add(a, b):\n    return a + b\n")
    
    run_cmd("git add .", cwd=repo_path)
    try:
        run_cmd('git commit -m "Initial commit"', cwd=repo_path)
    except Exception:
        pass # might already be committed

    # --- PR 1: Off-by-one bug ---
    run_cmd("git checkout -B pr/off-by-one main", cwd=repo_path)
    pr1_code = """def chunk_list(lst, chunk_size):
    '''Splits a list into chunks. Contains an off-by-one logic bug.'''
    chunks = []
    # BUG: uses range(0, len(lst) - 1, chunk_size) instead of len(lst)
    for i in range(0, len(lst) - 1, chunk_size):
        chunks.append(lst[i:i + chunk_size])
    return chunks
"""
    pr1_test = """from chunker import chunk_list

def test_chunk_list():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk_list([], 2) == []
    print("All tests passed.")

if __name__ == "__main__":
    test_chunk_list()
"""
    create_file(os.path.join(repo_path, "chunker.py"), pr1_code)
    create_file(os.path.join(repo_path, "test_chunker.py"), pr1_test)
    run_cmd("git add .", cwd=repo_path)
    run_cmd('git commit -m "feat: add chunk_list utility with tests"', cwd=repo_path)

    # --- PR 2: Zero Division bug ---
    run_cmd("git checkout main", cwd=repo_path)
    run_cmd("git checkout -B pr/zero-division main", cwd=repo_path)
    pr2_code = """def calculate_average(numbers):
    '''Calculates average. Fails on empty lists.'''
    total = sum(numbers)
    # BUG: No check for len(numbers) == 0
    return total / len(numbers)
"""
    pr2_test = """from stats import calculate_average

def test_calculate_average():
    assert calculate_average([10, 20, 30]) == 20
    assert calculate_average([0, 0]) == 0
    assert calculate_average([]) == 0 # This will throw ZeroDivisionError
    print("All tests passed.")

if __name__ == "__main__":
    test_calculate_average()
"""
    create_file(os.path.join(repo_path, "stats.py"), pr2_code)
    create_file(os.path.join(repo_path, "test_stats.py"), pr2_test)
    run_cmd("git add .", cwd=repo_path)
    run_cmd('git commit -m "feat: add calculate_average math utility"', cwd=repo_path)

    # --- PR 3: Broken String Parser ---
    run_cmd("git checkout main", cwd=repo_path)
    run_cmd("git checkout -B pr/broken-parser main", cwd=repo_path)
    pr3_code = """def extract_hashtags(text):
    '''Extracts hashtags from text. Fails if text is None, and misses some logic.'''
    # BUG: text.split() will crash if text is None.
    words = text.split()
    return [w for w in words if w.startswith('#')]
"""
    pr3_test = """from parser import extract_hashtags

def test_extract_hashtags():
    assert extract_hashtags("hello #world") == ["#world"]
    assert extract_hashtags("no tags here") == []
    assert extract_hashtags(None) == [] # Will throw AttributeError
    print("All tests passed.")

if __name__ == "__main__":
    test_extract_hashtags()
"""
    create_file(os.path.join(repo_path, "parser.py"), pr3_code)
    create_file(os.path.join(repo_path, "test_parser.py"), pr3_test)
    run_cmd("git add .", cwd=repo_path)
    run_cmd('git commit -m "feat: add hashtag extractor"', cwd=repo_path)

    # --- PR 4: Infinite Recursion bug ---
    run_cmd("git checkout main", cwd=repo_path)
    run_cmd("git checkout -B pr/recursion-bug main", cwd=repo_path)
    pr4_code = """def factorial(n):
    '''Calculates factorial. BUG: No base case for negative numbers.'''
    if n == 0:
        return 1
    # BUG: If n < 0, this will recurse forever.
    return n * factorial(n - 1)
"""
    pr4_test = """from math_utils import factorial
import pytest

def test_factorial():
    assert factorial(5) == 120
    assert factorial(0) == 1
    with pytest.raises(RecursionError): # This tests the failure
        factorial(-1)
    print("All tests passed.")

if __name__ == "__main__":
    test_factorial()
"""
    create_file(os.path.join(repo_path, "math_utils.py"), pr4_code)
    create_file(os.path.join(repo_path, "test_math_utils.py"), pr4_test)
    run_cmd("git add .", cwd=repo_path)
    run_cmd('git commit -m "feat: add factorial function"', cwd=repo_path)

    # --- PR 5: Resource Leak (File not closed) ---
    run_cmd("git checkout main", cwd=repo_path)
    run_cmd("git checkout -B pr/file-leak main", cwd=repo_path)
    pr5_code = """def count_lines(filepath):
    '''Reads a file and counts lines. BUG: Does not close the file handle.'''
    # BUG: File is opened but never closed. Should use 'with open(...)'.
    f = open(filepath, 'r')
    lines = f.readlines()
    return len(lines)
"""
    pr5_test = """from file_utils import count_lines
import os

def test_count_lines():
    with open("temp.txt", "w") as f:
        f.write("line1\\nline2\\nline3")
    assert count_lines("temp.txt") == 3
    os.remove("temp.txt")
    print("All tests passed.")

if __name__ == "__main__":
    test_count_lines()
"""
    create_file(os.path.join(repo_path, "file_utils.py"), pr5_code)
    create_file(os.path.join(repo_path, "test_file_utils.py"), pr5_test)
    run_cmd("git add .", cwd=repo_path)
    run_cmd('git commit -m "feat: add line counter utility"', cwd=repo_path)

    # --- PR 6: Key Error bug ---
    run_cmd("git checkout main", cwd=repo_path)
    run_cmd("git checkout -B pr/key-error main", cwd=repo_path)
    pr6_code = """def get_user_email(users, user_id):
    '''Returns user email based on ID. BUG: Crashes if ID is missing.'''
    # BUG: Accessing users[user_id] directly throws KeyError if not found.
    return users[user_id]['email']
"""
    pr6_test = """from user_utils import get_user_email

def test_get_user_email():
    db = {"1": {"email": "alice@example.com"}, "2": {"email": "bob@example.com"}}
    assert get_user_email(db, "1") == "alice@example.com"
    # This will throw KeyError
    try:
        get_user_email(db, "3")
    except KeyError:
        print("Caught expected KeyError")
        return
    assert False, "Should have raised KeyError"

if __name__ == "__main__":
    test_get_user_email()
"""
    create_file(os.path.join(repo_path, "user_utils.py"), pr6_code)
    create_file(os.path.join(repo_path, "test_user_utils.py"), pr6_test)
    run_cmd("git add .", cwd=repo_path)
    run_cmd('git commit -m "feat: add user email lookup"', cwd=repo_path)


    run_cmd("git checkout main", cwd=repo_path)
    print("\nDummy Repo branches generated successfully in:", repo_path)
    print("Next steps:")
    print("1. cd", repo_path)
    print("2. git remote add origin <your-github-repo-url>")
    print("3. git push --all origin")
    print("4. Go to GitHub and open PRs for 'pr/off-by-one', 'pr/zero-division', and 'pr/broken-parser'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_pr_repo.py <target_directory>")
        sys.exit(1)
    main(sys.argv[1])
