#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a trained model directory to Hugging Face Hub."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Local model directory containing settings.json and the model checkpoint.",
    )
    parser.add_argument(
        "--repo_id",
        default=None,
        help=(
            "Hugging Face repo id, for example ehodneland/my-model. "
            "Defaults to <authenticated-user>/<model-folder-name>."
        ),
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN or huggingface-cli login.",
    )
    parser.add_argument(
        "--hf_token_file",
        default=None,
        help="Optional file containing a Hugging Face token. Ignored if --hf_token is set.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Target branch/revision on Hugging Face. Defaults to main.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the repository as public. Default is private.",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help=(
            "Optional extra files or glob patterns relative to model_dir to upload. "
            "Example: --extra training_metrics*.csv README.md"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be uploaded without creating or uploading anything.",
    )
    return parser.parse_args()


def load_settings(model_dir):
    settings_path = os.path.join(model_dir, "settings.json")
    if not os.path.isfile(settings_path):
        raise FileNotFoundError(f"Could not find settings.json: {settings_path}")

    with open(settings_path) as f:
        return json.load(f)


def resolve_model_file(model_dir, meta):
    model_file = meta.get("model_file", "best_model.pth")
    model_path = model_file if os.path.isabs(model_file) else os.path.join(model_dir, model_file)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Could not find model checkpoint: {model_path}")

    path_in_repo = os.path.basename(model_file) if os.path.isabs(model_file) else model_file
    return os.path.normpath(model_path), path_in_repo.replace(os.sep, "/")


def nnunet_results_dir(meta):
    return os.path.join(
        meta["nnunet_results"],
        meta["nnunet_dataset_name"],
        f"{meta['nnunet_trainer']}__{meta['nnunet_plans']}__{meta['nnunet_configuration']}",
    )


def collect_nnunetv2_files(model_dir, meta):
    files = [(os.path.join(model_dir, "settings.json"), "settings.json")]
    results_dir = nnunet_results_dir(meta)
    fold_dir = os.path.join(results_dir, f"fold_{meta['nnunet_fold']}")

    for filename in ("dataset.json", "plans.json"):
        path = os.path.join(results_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Could not find nnU-Net file: {path}")
        files.append((path, os.path.relpath(path, model_dir).replace(os.sep, "/")))

    best_checkpoint = os.path.join(fold_dir, "checkpoint_best.pth")
    checkpoint_paths = (
        [best_checkpoint]
        if os.path.isfile(best_checkpoint)
        else sorted(glob.glob(os.path.join(fold_dir, "checkpoint*.pth")))
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"Could not find nnU-Net checkpoints in: {fold_dir}")
    for path in checkpoint_paths:
        files.append((path, os.path.relpath(path, model_dir).replace(os.sep, "/")))

    return files


def expand_extra_files(model_dir, patterns):
    files = []
    for pattern in patterns:
        matches = glob.glob(os.path.join(model_dir, pattern), recursive=True)
        if not matches:
            raise FileNotFoundError(f"No files matched --extra pattern: {pattern}")
        for path in matches:
            if os.path.isfile(path):
                rel = os.path.relpath(path, model_dir)
                files.append((os.path.normpath(path), rel.replace(os.sep, "/")))
    return files


def unique_uploads(files):
    seen = set()
    unique = []
    for local_path, path_in_repo in files:
        key = os.path.normpath(local_path)
        if key in seen:
            continue
        seen.add(key)
        unique.append((local_path, path_in_repo))
    return unique


def read_hf_token_file(token_file):
    if not token_file:
        return None

    token_file = os.path.abspath(os.path.expanduser(str(token_file)))
    if not os.path.isfile(token_file):
        raise FileNotFoundError(f"Could not find Hugging Face token file: {token_file}")

    with open(token_file) as f:
        for line in f:
            token = line.strip()
            if token:
                return token

    raise ValueError(f"Hugging Face token file is empty: {token_file}")


def require_hf_login(api):
    try:
        user = api.whoami()
    except Exception as exc:
        raise RuntimeError(
            "Could not authenticate with Hugging Face Hub.\n\n"
            "Run one of these first:\n"
            "  huggingface-cli login\n"
            "  hf auth login\n\n"
            "Or pass a write token explicitly:\n"
            "  python3 tools/upload_model_to_hf.py ... --hf_token hf_xxx\n\n"
            "You can also pass a token file:\n"
            "  python3 tools/upload_model_to_hf.py ... --hf_token_file ~/.hf_token\n\n"
            "The token must have write access. Browser login on huggingface.co is "
            "not enough for this Python process."
        ) from exc

    username = user.get("name") or user.get("fullname") or user.get("email")
    print(f"Authenticated HF user: {username}")
    return username


def default_repo_id(model_dir, username=None):
    repo_name = os.path.basename(model_dir.rstrip(os.sep))
    if username:
        return f"{username}/{repo_name}"
    return f"<authenticated-user>/{repo_name}"


def main():
    args = parse_args()

    model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    if not os.path.isdir(model_dir):
        raise NotADirectoryError(f"--model_dir is not a directory: {model_dir}")

    meta = load_settings(model_dir)
    is_nnunetv2 = meta.get("model_architecture") == "nnUNetv2"
    if is_nnunetv2:
        base_files = collect_nnunetv2_files(model_dir, meta)
    else:
        model_path, model_path_in_repo = resolve_model_file(model_dir, meta)
        base_files = [
            (os.path.join(model_dir, "settings.json"), "settings.json"),
            (model_path, model_path_in_repo),
        ]

    files_to_upload = unique_uploads(
        [
            *base_files,
            *expand_extra_files(model_dir, args.extra),
        ]
    )

    repo_id_for_print = args.repo_id or default_repo_id(model_dir)
    print(f"Model directory : {model_dir}")
    print(f"HF repo         : {repo_id_for_print}")
    print(f"Visibility      : {'public' if args.public else 'private'}")
    print(f"Revision        : {args.revision}")
    print("\nFiles to upload:")
    for local_path, path_in_repo in files_to_upload:
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  {path_in_repo:30} <- {local_path} ({size_mb:.1f} MB)")

    if args.dry_run:
        print("\nDry run only. Nothing uploaded.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install it with: pip install huggingface_hub"
        ) from exc

    hf_token = args.hf_token or read_hf_token_file(args.hf_token_file)
    api = HfApi(token=hf_token)
    username = require_hf_login(api)
    repo_id = args.repo_id or default_repo_id(model_dir, username=username)

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=not args.public,
            exist_ok=True,
        )

        for local_path, path_in_repo in files_to_upload:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                revision=args.revision,
                commit_message=f"Upload {path_in_repo}",
            )
    except Exception as exc:
        print(f"\nUpload failed: {exc}", file=sys.stderr)
        print(
            "\nCheck that your token has write access and that repo_id starts with "
            "your Hugging Face username or an organization where you can create repos.",
            file=sys.stderr,
        )
        raise

    print(f"\nDone: https://huggingface.co/{repo_id}")
    print(f"Use it with: --model_dir {repo_id}")


if __name__ == "__main__":
    main()
