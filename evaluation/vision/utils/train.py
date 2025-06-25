from evaluation.utils.metrics import compute_metrics

from tqdm import tqdm
import torch
import numpy as np
import time


def evaluate(
    classifier_model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_eval_n: int = None,
    verbose: bool = False,
):
    classifier_model.eval()
    all_logits = []
    all_labels = []
    all_maps = {}

    if max_eval_n is None:
        total_eval_iter = len(val_loader)
    else:
        total_eval_iter = min(len(val_loader), max_eval_n)

    val_iter = iter(val_loader)
    if verbose:
        eval_iter_loop = tqdm(range(total_eval_iter))
    else:
        eval_iter_loop = range(total_eval_iter)

    for _ in eval_iter_loop:
        t0_data = time.time()
        map_ids, embeddings, mask, labels = next(val_iter)

        embeddings = embeddings.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            logits, attention_map = classifier_model(embeddings, mask=mask)

        all_logits.append(logits.cpu().flatten().numpy())
        all_labels.append(labels.float().cpu().flatten().numpy())
        attention_maps_cpu = attention_map.cpu().numpy()
        mask = mask.cpu().numpy()

        for i, map_id in enumerate(map_ids):
            all_maps[map_id] = attention_maps_cpu[i][~mask[i]]

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    metrics = compute_metrics(all_logits, all_labels)

    return dict(
        map_ids=list(all_maps.keys()),
        metrics=metrics,
        logits=all_logits,
        labels=all_labels,
        attention_maps=all_maps,
    )


def train(
    classifier_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    train_iters: int,
    accum_steps: int,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    verbose: bool = False,
) -> list:
    classifier_model.train()
    optimizer.zero_grad()

    train_iter = iter(train_loader)

    if verbose:
        train_iter_loop = tqdm(range(train_iters))
    else:
        train_iter_loop = range(train_iters)

    all_logits = []
    all_labels = []

    t0_train = time.time()
    data_times = []
    for iteration in train_iter_loop:
        try:
            t0_data = time.time()
            map_ids, embeddings, mask, labels = next(train_iter)
            data_times.append(time.time() - t0_data)

        except StopIteration:
            del train_iter
            train_iter = iter(train_loader)
            map_ids, embeddings, mask, labels = next(train_iter)

        logits, _ = classifier_model(embeddings.to(device), mask=mask.to(device))
        loss = criterion(logits.flatten(), labels.float().to(device))
        loss.backward()

        all_logits.append(logits.detach().cpu().flatten().numpy())
        all_labels.append(labels.float().cpu().flatten().numpy())

        torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), 1.0)

        if (iteration + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

    tf_train = time.time() - t0_train

    if verbose:
        print(f"Time per iteration: {tf_train / train_iters:.4f}")
        print(f"Mean data load time: {np.mean(data_times):.4f}")

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    metrics = compute_metrics(all_logits, all_labels)

    return dict(
        metrics=metrics,
        logits=all_logits,
        labels=all_labels,
    )


def save_classifier(path_to_save, model):
    os.makedirs(path_to_save, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path_to_save, "model.pth"))
