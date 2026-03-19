import os
import urllib.request

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "SMALify")

# Raw GitHub URLs for the SMAL model files from the smalst repo.
# Commit ba4b898 is the submodule pin used by SMALify.
_SMALST_BASE = (
    "https://github.com/silviazuffi/smalst/raw/ba4b898a23699b642f95d30a3a911bb463515893"
    "/smpl_models"
)

_SMAL_FILES = {
    os.path.join("SMALST", "smpl_models", "my_smpl_00781_4_all.pkl"):
        f"{_SMALST_BASE}/my_smpl_00781_4_all.pkl",
    os.path.join("SMALST", "smpl_models", "my_smpl_data_00781_4_all.pkl"):
        f"{_SMALST_BASE}/my_smpl_data_00781_4_all.pkl",
    os.path.join("SMALST", "smpl_models", "my_smpl_00781_4_all_template_w_tex_uv_001.pkl"):
        f"{_SMALST_BASE}/my_smpl_00781_4_all_template_w_tex_uv_001.pkl",
    os.path.join("SMALST", "smpl_models", "symIdx.pkl"):
        f"{_SMALST_BASE}/symIdx.pkl",
}

_PRIOR_BASE = (
    "https://github.com/lukasuz/SMALify/raw/main/data/priors"
)

_PRIOR_FILES = {
    os.path.join("priors", "walking_toy_symmetric_pose_prior_with_cov_35parts.pkl"):
        f"{_PRIOR_BASE}/walking_toy_symmetric_pose_prior_with_cov_35parts.pkl",
    os.path.join("priors", "unity_betas.npz"):
        f"{_PRIOR_BASE}/unity_betas.npz",
}


def download_data(dest_dir: str = _CACHE_DIR) -> None:
    """Download SMAL model files to *dest_dir* (default: ~/.cache/SMALify)."""
    all_files = {**_SMAL_FILES, **_PRIOR_FILES}
    for rel_path, url in all_files.items():
        dest = os.path.join(dest_dir, rel_path)
        if os.path.exists(dest):
            print(f"[SMALify] already exists, skipping: {rel_path}")
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"[SMALify] downloading {rel_path} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {rel_path} from {url}.\n"
                "You can also set SMALIFY_DATA_DIR to point at a local "
                "SMALify/data directory."
            ) from e
    print(f"[SMALify] data ready in {dest_dir}")


if __name__ == "__main__":
    download_data()
