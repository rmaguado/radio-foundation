import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf

from dinov2.configs import dinov2_default_config


class Logs:
    def __init__(self, run_name, num_gpus=1):
        run_path = os.path.join("runs", run_name)
        cfg_path = os.path.join(run_path, "config.yaml")
        metrics_path = os.path.join(run_path, "training_metrics.json")

        default_cfg = OmegaConf.create(dinov2_default_config)
        cfg = OmegaConf.load(cfg_path)
        cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli())

        self.grad_accum_steps = cfg.train.grad_accum_steps
        self.official_epoch_len = cfg.train.OFFICIAL_EPOCH_LENGTH
        self.total_epochs = cfg.optim.epochs
        self.full_size_epochs = cfg.train.full_image.epochs
        self.full_size_grad_accum_steps = cfg.train.full_image.grad_accum_steps

        self.attr = {
            "iteration": [],
            "total_loss": [],
            "dino_local_crops_loss": [],
            "dino_global_crops_loss": [],
            "koleo_loss": [],
            "ibot_loss": [],
            "lr": [],
            "mom": [],
            "wd": [],
        }
        with open(metrics_path, "r") as f:
            lines = f.read().split("\n")
        json_lines = []
        for i, line in enumerate(lines):
            if line:
                json_lines.append(json.loads(line))

        for line in json_lines:
            if all(key in line for key in self.attr.keys()):
                for key in self.attr.keys():
                    self.attr[key].append(line[key])

        self.num_gpus = num_gpus
        self.batch_size_per_gpu = cfg.train.batch_size_per_gpu
        self.total_batch_size = (
            self.num_gpus * self.batch_size_per_gpu * self.grad_accum_steps
        )

        self.label = os.path.basename(run_path)

    @property
    def steps(self):
        total_iter = self.official_epoch_len * self.grad_accum_steps * self.total_epochs
        full_iter = (
            self.official_epoch_len
            * self.full_size_grad_accum_steps
            * self.full_size_epochs
        )
        cropped_iter = total_iter - full_iter

        last_iter = self.it[-1]
        if last_iter < cropped_iter:
            return [it / self.grad_accum_steps for it in self.it]
        else:
            full_size_index = [i for i, it in enumerate(self.it) if it > cropped_iter][
                0
            ]
            cropped_rescale = [
                it / self.grad_accum_steps for it in self.it[:full_size_index]
            ]
            full_rescale = [
                it / self.full_size_grad_accum_steps for it in self.it[full_size_index:]
            ]
            return cropped_rescale + full_rescale

    @property
    def epochs(self):
        return [step / self.official_epoch_len for step in self.steps]

    @property
    def it(self):
        return self.attr["iteration"]

    @property
    def total_loss(self):
        return self.attr["total_loss"]


def main(runs):
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    outdir = "runs/figures"
    outpath = os.path.join(outdir, f"{timestamp}.png")
    os.makedirs(outdir, exist_ok=True)

    logs = [Logs(run_name) for run_name in runs]

    sns.set(style="darkgrid", context="talk", palette="deep")
    plt.figure()

    for log in logs:
        plt.plot(log.epochs, log.total_loss, label=log.label)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Total Loss", fontsize=14)
    plt.legend(title="Runs", fontsize=12, title_fontsize='13')

    plt.savefig(outpath, dpi=300)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        runs = sys.argv[1:]
        main(runs)
    else:
        main(["default"])
