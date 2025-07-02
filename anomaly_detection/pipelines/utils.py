import git


def get_git_info():
    """Get git repository information"""
    try:
        repo = git.Repo(search_parent_directories=True)
        commit_id = repo.head.commit.hexsha

        # Get diff to last commit
        diff = repo.git.diff()

        # Get current branch
        branch = repo.active_branch.name

        return {"commit_id": commit_id, "branch": branch, "diff": diff}
    except git.InvalidGitRepositoryError:
        return {
            "commit_id": "Not a git repository",
            "branch": "N/A",
            "diff": "N/A",
        }