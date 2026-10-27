import time

import torch

from deepfake.metrics import compute_eer, llr_score


class EarlyStopper:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best = float('inf')

    def early_stop(self, validation_metric):
        if validation_metric < self.best:
            self.best = validation_metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def class_weights(y, device):
    num_bonafide = int(y.sum())
    num_spoof = int(len(y) - num_bonafide)
    total = num_bonafide + num_spoof

    return torch.tensor(
        [total / (2 * num_spoof), total / (2 * num_bonafide)],
        dtype=torch.float32
    ).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    total = 0
    all_labels = []
    all_scores = []

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        total += y.size(0)

        all_labels.append(y)
        all_scores.append(llr_score(logits).detach())

    epoch_loss = running_loss / total
    epoch_eer = compute_eer(torch.cat(all_labels).cpu().numpy(), torch.cat(all_scores).cpu().numpy())

    return epoch_loss, epoch_eer


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    total = 0
    all_labels = []
    all_scores = []

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X)
        loss = criterion(logits, y)
        running_loss += loss.item() * X.size(0)
        total += y.size(0)

        all_labels.append(y)
        all_scores.append(llr_score(logits))

    all_scores = torch.cat(all_scores).cpu().numpy()
    epoch_loss = running_loss / total
    eer = compute_eer(torch.cat(all_labels).cpu().numpy(), all_scores)

    return epoch_loss, eer, all_scores


def run_model(model, train_loader, dev_loader, criterion, optimizer, scheduler, device, epochs, patience=10):
    history = {
        "train_loss": [],
        "train_eer": [],
        "dev_loss": [],
        "dev_eer": [],
        "lr": [],
    }

    best_dev_eer = float("inf")
    best_model_state = None
    best_model_status = {}

    early_stopper = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        start = time.time()
        train_loss, train_eer = train_one_epoch(model, train_loader, criterion, optimizer, device)
        dev_loss, dev_eer, _ = evaluate(model, dev_loader, criterion, device)

        scheduler.step(dev_eer)

        history["train_loss"].append(train_loss)
        history["train_eer"].append(train_eer)
        history["dev_loss"].append(dev_loss)
        history["dev_eer"].append(dev_eer)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_model_status = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_eer": train_eer,
                "dev_loss": dev_loss,
                "best_dev_eer": best_dev_eer
            }

        stop = early_stopper.early_stop(dev_eer)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"LR: {history['lr'][-1]:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train EER: {train_eer:.4%} | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"Dev EER: {dev_eer:.4%} | "
            f"ES Counter: {early_stopper.counter}/{early_stopper.patience} | "
            f"Time: {time.time() - start:.0f}s",
            flush=True
        )

        if stop:
            print("Early Stopping...", flush=True)
            break

    return model, history, best_model_state, best_model_status


def evaluate_and_save(model, eval_loader, criterion, device, best_model_state, best_model_status, history):
    model.load_state_dict(best_model_state)
    _, eer, all_scores = evaluate(model, eval_loader, criterion, device)
    return {"best": best_model_status, "history": history, "eval_eer": eer, "eval_score": all_scores, "state_dict": best_model_state}