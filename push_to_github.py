import os
import requests
import subprocess
import tempfile
import shutil

# CONFIG
TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_OWNER = "hemang1404"
REPO_NAME = "dummy-bench"
# We'll use the token in the headers instead of the URL to be safer
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

def run_cmd(cmd, cwd=None, env=None):
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=env)

def create_pull_request(branch, title, body):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls"
    headers = {
        "Authorization": f"token {TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "title": title,
        "head": branch,
        "base": "main",
        "body": body
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 201:
        print(f"Created PR for {branch}: {resp.json().get('html_url')}")
        return resp.json().get("number")
    else:
        print(f"Failed to create PR for {branch}: {resp.text}")
        return None

def main():
    if not TOKEN:
        print("Error: GITHUB_TOKEN not found in environment.")
        return

    import stat
    def remove_readonly(func, path, _):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    tmp_dir = tempfile.mkdtemp()
    try:
        # Authenticate git
        # Configure git to use the token for this clone
        # We'll use the URL embedding for convenience in this script
        AUTH_URL = f"https://{TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git"
        
        run_cmd(f"git clone {AUTH_URL} .", cwd=tmp_dir)
        
        # Run the local generator to create branches
        generator_path = os.path.join(os.getcwd(), "generate_pr_repo.py")
        shutil.copy(generator_path, tmp_dir)
        run_cmd(f"python generate_pr_repo.py .", cwd=tmp_dir)
        
        # Push branches
        branches = ["pr/recursion-bug", "pr/file-leak", "pr/key-error"]
        for b in branches:
            try:
                run_cmd(f"git push origin {b}", cwd=tmp_dir)
                create_pull_request(b, f"Demo: {b.split('/')[-1].replace('-', ' ').title()}", "Automated demo PR for RuntimeTerror AI analysis.")
            except Exception as e:
                print(f"Skipping {b} (likely exists): {e}")

        print("\nAll new demo PRs pushed and created!")
    finally:
        shutil.rmtree(tmp_dir, onerror=remove_readonly)

if __name__ == "__main__":
    main()
