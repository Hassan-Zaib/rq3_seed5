# RQ3: seed 5

Hi, and thanks a lot for running these. This repository holds the exact RQ3 code from my DGX: SDAR attacking a ConvNeXt-Tiny split model on Tiny ImageNet, under five defenses (`none`, `q4`, `q2`, `noise0.01`, `noise0.05`) at four cuts (c1 to c4). You only run seed 5: 20 jobs, one per line of `jobs_seed5.txt`. Please don't change any setting in the scripts.

## What the code needs

- **Tiny ImageNet**, the original `tiny-imagenet-200` folder as unzipped. The code reads `wnids.txt`, `train/<wnid>/images/*.JPEG`, `val/val_annotations.txt` and `val/images/*.JPEG`. You link it in yourself (section 2).
- **The fixed evaluation images** in `fixed_probe/`: 20 Tiny ImageNet validation images, as `fixed_probe_images.npy`, `fixed_probe_images_norm.npy`, `fixed_probe_labels.npy` and `fixed_probe_indices.npy`, plus `fixed_probe_meta.json` and a preview, `fixed_probe_grid.png`. They are in the repository and must stay unchanged.
- **Pretrained weights:** only the AlexNet weights that LPIPS uses (section 3). The ConvNeXt model has no pretrained weights; every job trains it from scratch.

## 1. Environment

    conda env create -f env_rq3.yml
    conda activate rq3
    python -c "import torch; print(torch.cuda.is_available())"

The last command should print True.

## 2. Tiny ImageNet

If you don't have it yet:

    mkdir -p ~/datasets && cd ~/datasets
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip -q tiny-imagenet-200.zip
    cd -

Then link it into the repository folder:

    mkdir -p data
    ln -sfn ~/datasets/tiny-imagenet-200 data/tiny-imagenet-200

The code reads the validation images where the zip puts them, so you don't need to reorganise anything.

## 3. LPIPS weights

LPIPS needs the ImageNet AlexNet weights from torchvision. They are downloaded once, into `~/.cache/torch/hub/checkpoints/`:

    python -c "import lpips; lpips.LPIPS(net='alex')"

If your GPU nodes have no internet access, run this once on a machine that does and share the same home folder (or copy `~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`). Each job also loads LPIPS before it starts training, so if the weights are missing it stops right away.

## 4. First check

From the repository folder:

    bash check_seed5.sh

It trains nothing. It checks the syntax of every script, the job list and the fixed evaluation images. It also confirms that `scripts/run_one.sh` builds the same python command as my DGX seed-5 run. It should end with ALL CHECKS PASSED. If it doesn't, please send me the output.

## 5. Running

Each line of `jobs_seed5.txt` is one job (`DEFENSE CUT`). Please run each line as its own job, from the repository folder:

    bash scripts/run_one.sh none c1

For example, as job N of an array job:

    bash scripts/run_one.sh $(sed -n "${N}p" jobs_seed5.txt)

Each job must finish within 7 hours and needs about 27 GB of GPU memory. Run several jobs in parallel, on different GPUs or together on one GPU, only if the memory allows. To pick a GPU:

    CUDA_VISIBLE_DEVICES=1 bash scripts/run_one.sh q4 c2

Each job writes to `results_seed5/<defense>/<cut>/sdar/` and logs to `logs_seed5/<defense>_<cut>.log`. A job is done when `results_seed5/<defense>/<cut>/sdar/full_sdar_summary.json` exists. Starting a finished job again does nothing.

Each job saves a checkpoint every 2000 steps in `results_seed5/<defense>/<cut>/sdar/checkpoints/` and keeps the last two. When a job receives SIGTERM or SIGUSR1 (for example from the scheduler at a time limit), it finishes the step it is on, saves a checkpoint and stops. To resume a job that stopped, start the same command again. It continues from its last checkpoint with the same data order as an uninterrupted run, and the evaluation runs once at the end as usual.

`bash scripts/test_resume.sh` checks this on your GPU in a few short runs: it compares the original script, an uninterrupted run and a run stopped with SIGTERM and resumed, and prints PASS or FAIL. Its output goes to `test_resume/`, and it deletes its checkpoints at the end. Pass `--data-root DIR` if your Tiny ImageNet is not linked as `data/tiny-imagenet-200`. If your GPU runs out of memory, add `--batch-size 64`; that option is only for this test.

The script runs seed 5 only and refuses any other seed.

## 6. If something goes wrong

The job stops before training if the data link, the fixed evaluation images, the GPU or LPIPS is not right, and the log says which. If a job hits the time limit, start it again to resume it (section 5). If a job fails, please send me its log and I'll look at it. Please don't change any setting to make it fit.

## 7. Sending the results back

When all 20 jobs are done, run this from the repository folder and send me the zip:

    zip -r rq3_seed5_results.zip results_seed5 logs_seed5 -x '*/checkpoints/*'

The checkpoints are large and only needed for resuming, so they stay out of the zip. You can delete them once all jobs are done.

## Please don't change

The scripts, their settings (steps, batch size, learning rates and so on) and the files in `fixed_probe/`. They have to match my DGX runs exactly, otherwise the seeds can't be compared.

`dgx/run_seeds45_one_defense.sh` is the launcher I used on the DGX. It's only there as the reference for `check_seed5.sh`, so please don't run it.

Thanks again!

## DreamSim on finished jobs

`dreamsim_eval.py` computes DreamSim (lower = more similar) between originals and
reconstructions of every finished job, on your machine, so no images need to be
sent. The main pair is `originals.npy` against `reconstructions.npy` (the 1280
validation images that the SSIM, PSNR and LPIPS in `full_sdar_summary.json` use);
the 20 fixed probe images (`fixed_probe_originals.npy` against
`fixed_probe_reconstructions.npy`) are added as a second pair. A job counts as
finished when it has `full_sdar_summary.json`.

Install it in its own environment:

    conda create -n dreamsim python=3.10 -y
    conda activate dreamsim
    pip install dreamsim

The first real run downloads the DreamSim weights into `--cache-dir`. If your
GPU nodes have no internet, run it once on a login node first with the same
`--cache-dir`. From the repository folder, first check what it finds, then run it:

    python dreamsim_eval.py results_seed5 --dry-run
    python dreamsim_eval.py results_seed5 --cache-dir ~/dreamsim_models

It writes `dreamsim.json` next to each `full_sdar_summary.json` and one
`dreamsim_all.csv` here. Jobs that already have a `dreamsim.json` are skipped, so
you can run it again as more jobs finish; `--force` recomputes them.

What to send back (under 1 MB):

    zip rq3_dreamsim.zip dreamsim_all.csv results_seed5/*/*/sdar/dreamsim.json
