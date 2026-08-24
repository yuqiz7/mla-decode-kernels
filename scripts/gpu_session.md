# GPU session checklist (Lambda H100)

Rules of thumb: if you will be away from the keyboard for more than 20 minutes,
Terminate first. Keep a single session to 2 hours or less.

## 1. Launch

1. Launch the instance. Base image **must be Lambda Stack 24.04** — not bare
   Ubuntu (bare Ubuntu is missing the driver/CUDA/torch stack recorded in
   `docs/g0/versions.txt`).
2. After the instance shows **Running**, wait 2–3 minutes before doing anything.

## 2. Preflight (before or right after first SSH)

3. Check both of the following. If either fails, **Terminate and launch a new
   instance — do not try to repair it**:
   - `nvidia-smi` produces output (GPU visible, driver loaded).
   - `cloud-init status` reports `done`.

## 3. Connect and enable profiling

4. Connect with VS Code Remote-SSH.
5. Write the NCU permission config and reboot:
   - `scripts/setup_env.sh` writes `/etc/modprobe.d/nvidia-profiling.conf`
     and prints `REBOOT REQUIRED` when a reboot is needed; then `sudo reboot`.
6. After the reboot, the driver may not be loaded during the first minute.
   Run `nvidia-smi` once first, then check
   `cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly` — it must be
   `0`.

## 4. Repository access

7. Deploy key: on GitHub, **delete the old deploy key and add the new one**
   (generated on this instance), with **write access** checked.
8. `git clone` this repository.
9. Git identity: `git config --global user.name yuqiz7` and
   `git config --global user.email yz5072@columbia.edu`.
   - **`git config user.email` must be yz5072@columbia.edu (never the gmail
     address — it belongs to a different GitHub account).**
   - **After the first commit, check `git log` shows no Co-Authored-By
     trailer.**

## 5. Tooling

10. Install Claude Code and log in.
11. Run `scripts/setup_env.sh` (add `--with-flashmla` if FlashMLA is needed).
12. Run `scripts/verify_env.sh` — proceed only on `VERIFY OK`.

## 6. Work and teardown

13. Do the work.
14. `git push` (work not pushed is lost at Terminate).
15. **Terminate** the instance.
