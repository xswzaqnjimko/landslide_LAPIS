"""
Training loops for SHRED and LAPIS temporal models.

Provides:
  - SHRED training: train_ensemble_shred, train_ensemble_frame_shred
    (these still accept a config object for SHRED-specific hyperparameters)
  - Temporal model training: train_forward_model, train_backward_model
    (these accept explicit keyword arguments — no config dependency)
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from functools import partial

from .shred import weighted_mse_loss, create_active_weights


# ============================================================================
# SHRED training (config-based — SHRED keys are consistent across projects)
# ============================================================================

@partial(jax.jit, static_argnums=(4,))
def train_step_shred(state, x, y, weights, hidden_size):
    def loss_fn(params):
        pred = state.apply_fn({"params": params}, x, train=True,
                              rngs={"dropout": jax.random.PRNGKey(0)})
        return weighted_mse_loss(pred, y, weights)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@partial(jax.jit)
def train_step_frame_shred(state, x, y, weights):
    def loss_fn(params):
        pred = state.apply_fn({"params": params}, x, train=True,
                              rngs={"dropout": jax.random.PRNGKey(0)})
        return weighted_mse_loss(pred, y, weights)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def train_ensemble_shred(model, dataset, active_mask, config, rng):
    """Train Seq2Seq SHRED on an ensemble of simulation sequences."""
    sample_x, _ = dataset[0]
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, sample_x[None, ...], train=False)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=config.LR_SHRED, weight_decay=config.WEIGHT_DECAY))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=tx)
    weights_jax = jnp.array(create_active_weights(active_mask, config.ACTIVE_WEIGHT),
                            dtype=jnp.float32)

    print(f"  Training SHRED for {config.EPOCHS_SHRED} epochs ...")
    for epoch in range(config.EPOCHS_SHRED):
        rng, brng = jax.random.split(rng)
        batches = dataset.get_batches(config.BATCH_SIZE, shuffle=True, rng=brng)
        eloss = 0.0
        for xb, yb in batches:
            state, loss = train_step_shred(state, xb, yb, weights_jax, config.SEQ2SEQ_HIDDEN)
            eloss += float(loss)
        eloss /= max(len(batches), 1)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{config.EPOCHS_SHRED}: loss = {eloss:.6f}")
    return state


def train_ensemble_frame_shred(model, dataset, active_mask, config, rng):
    """Train frame-by-frame SHRED on an ensemble of simulation grids."""
    sample_x = jnp.array(dataset.X_scaled[:1], dtype=jnp.float32)
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, sample_x, train=False)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=config.LR_SHRED, weight_decay=config.WEIGHT_DECAY))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=tx)
    weights_jax = jnp.array(create_active_weights(active_mask, config.ACTIVE_WEIGHT),
                            dtype=jnp.float32)

    print(f"  Training Frame SHRED for {config.EPOCHS_SHRED} epochs "
          f"({len(dataset)} samples, lags={config.LAGS}) ...")
    for epoch in range(config.EPOCHS_SHRED):
        rng, brng = jax.random.split(rng)
        batches = dataset.get_batches(config.BATCH_SIZE, shuffle=True, rng=brng)
        eloss = 0.0
        for xb, yb in batches:
            state, loss = train_step_frame_shred(state, xb, yb, weights_jax)
            eloss += float(loss)
        eloss /= max(len(batches), 1)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{config.EPOCHS_SHRED}: loss = {eloss:.6f}")
    return state


# ============================================================================
# Bucketed batch preparation (shared by forward and backward)
# ============================================================================

def _prepare_buckets(input_seqs, target_seqs, anchor_vecs, bucket_width=5):
    """Group variable-length (input, target, anchor) triples into length-buckets.

    This is a generalised bucketing helper used by both forward and backward
    temporal model training.  Sequences are grouped by target length into
    buckets of width ``bucket_width``.  Within each bucket, targets are
    padded to the bucket edge and a binary mask tracks real vs. padding frames.

    Args:
        input_seqs:  list of arrays, each (W_i, D) or (D,) — model input.
                     For windowed models these are (W, D); for terminal
                     models these are (D,).
        target_seqs: list of arrays, each (T_out_i, D) — supervision targets.
        anchor_vecs: list of arrays, each (D,) — anchor vector for each
                     sequence (e.g., z_init for backward, z_last_obs for
                     forward).
        bucket_width: int — sequences whose T_out falls within the same
                      ``bucket_width``-wide interval share one JIT compilation.

    Returns:
        list of bucket dicts, each containing:
            input_batch:  (B, ...) or (B, W, D) — model inputs
            target_batch: (B, T_bucket, D) — padded targets
            anchor_batch: (B, D)
            mask:         (B, T_bucket) — 1.0 for real, 0.0 for padding
            T_bucket:     int — padded length (static for JIT)
            n_real:       list[int] — original T_out per sequence
    """
    n = len(target_seqs)
    D = target_seqs[0].shape[-1]

    # Collect per-sequence metadata
    T_outs = [t.shape[0] for t in target_seqs]
    min_T, max_T = min(T_outs), max(T_outs)
    edges = list(range(min_T, max_T + bucket_width, bucket_width))

    # Assign to buckets
    bucket_map = {}  # edge -> list of indices
    for i, T_out in enumerate(T_outs):
        for edge in edges:
            if T_out <= edge:
                bucket_map.setdefault(edge, []).append(i)
                break
        else:
            bucket_map.setdefault(edges[-1], []).append(i)

    buckets = []
    for edge in sorted(bucket_map.keys()):
        indices = bucket_map[edge]
        T_bucket = edge

        inp_list, tgt_list, anc_list, mask_list, n_reals = [], [], [], [], []
        for idx in indices:
            T_out = T_outs[idx]
            n_reals.append(T_out)

            inp_list.append(input_seqs[idx])
            anc_list.append(anchor_vecs[idx])

            # Pad target to T_bucket
            tgt = target_seqs[idx]
            if T_out < T_bucket:
                pad_len = T_bucket - T_out
                padding = np.tile(tgt[-1:], (pad_len, 1))
                tgt_padded = np.concatenate([tgt, padding], axis=0)
                mask = np.concatenate([np.ones(T_out), np.zeros(pad_len)])
            else:
                tgt_padded = tgt[:T_bucket]
                mask = np.ones(T_bucket)

            tgt_list.append(tgt_padded)
            mask_list.append(mask)

        buckets.append({
            'input_batch':  jnp.array(np.stack(inp_list), dtype=jnp.float32),
            'target_batch': jnp.array(np.stack(tgt_list), dtype=jnp.float32),
            'anchor_batch': jnp.array(np.stack(anc_list), dtype=jnp.float32),
            'mask':         jnp.array(np.stack(mask_list), dtype=jnp.float32),
            'T_bucket':     int(T_bucket),
            'n_real':       n_reals,
        })

    return buckets


# ============================================================================
# Masked loss function (shared JIT kernel)
# ============================================================================

@partial(jax.jit, static_argnums=(6, 7, 8, 9))
def _train_step_temporal(state, input_batch, target_batch, anchor_batch,
                         rng_key, mask, T_out, lambda_recon, lambda_anchor, lambda_shape):
    """One gradient step for a temporal model (forward or backward).

    Works for both windowed input (B, W, D) and terminal input (B, D) —
    the model's ``__call__`` handles the dispatch.

    Loss components:
      - recon:  masked MSE between predicted and target latent trajectories
      - anchor: MSE between predicted first frame and anchor vector
      - shape:  masked temporal-difference matching + masked variance matching
    """
    def loss_fn(params):
        z_pred = state.apply_fn({"params": params}, input_batch, T_out,
                                train=True, rngs={"dropout": rng_key})

        # --- Masked reconstruction loss ---
        sq = (z_pred - target_batch) ** 2
        mask_3d = mask[:, :, None]
        n_real = jnp.maximum(jnp.sum(mask_3d) * z_pred.shape[-1], 1.0)
        l_recon = jnp.sum(sq * mask_3d) / n_real

        # --- Anchor loss ---
        l_anchor = jnp.mean((z_pred[:, 0, :] - anchor_batch) ** 2)

        # --- Masked shape loss (temporal derivatives) ---
        pd = z_pred[:, 1:, :] - z_pred[:, :-1, :]
        td = target_batch[:, 1:, :] - target_batch[:, :-1, :]
        diff_mask = (mask[:, 1:] * mask[:, :-1])[:, :, None]
        n_diff = jnp.maximum(jnp.sum(diff_mask) * z_pred.shape[-1], 1.0)
        l_shape = jnp.sum((pd - td) ** 2 * diff_mask) / n_diff

        # --- Masked variance loss ---
        mask_sum = jnp.maximum(jnp.sum(mask_3d, axis=1, keepdims=True), 1.0)
        pred_mean = jnp.sum(z_pred * mask_3d, axis=1, keepdims=True) / mask_sum
        tgt_mean = jnp.sum(target_batch * mask_3d, axis=1, keepdims=True) / mask_sum
        pred_var = jnp.sum(((z_pred - pred_mean) ** 2) * mask_3d, axis=1) / mask_sum[:, 0, :]
        tgt_var = jnp.sum(((target_batch - tgt_mean) ** 2) * mask_3d, axis=1) / mask_sum[:, 0, :]
        l_var = jnp.mean((pred_var - tgt_var) ** 2)

        total = (lambda_recon * l_recon +
                 lambda_anchor * l_anchor +
                 lambda_shape * (l_shape + 0.5 * l_var))
        return total, (l_recon, l_anchor, l_shape, l_var)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, aux


def _run_bucketed_training(model, buckets, rng, *,
                           lr, epochs, lambda_recon, lambda_anchor,
                           lambda_shape, weight_decay, label="temporal model"):
    """Generic bucketed training loop for any temporal model.

    Initialises the model using the largest bucket's T_bucket, builds an
    optimiser with cosine-decay LR, and iterates over shuffled buckets
    each epoch.
    """
    # Determine max T_out for model init (param shapes don't depend on T_out
    # for these architectures, but we need a concrete value for init)
    max_T_bucket = max(b['T_bucket'] for b in buckets)
    sample_input = buckets[0]['input_batch'][:1]
    D = buckets[0]['target_batch'].shape[-1]

    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, sample_input, max_T_bucket, train=False)

    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=epochs, alpha=0.01)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=tx)

    n_buckets = len(buckets)
    bucket_sizes = [len(b['n_real']) for b in buckets]
    bucket_Ts = [b['T_bucket'] for b in buckets]
    total_seqs = sum(bucket_sizes)

    print(f"    {total_seqs} sequences in {n_buckets} buckets "
          f"(T_bucket={bucket_Ts}, sizes={bucket_sizes})")
    print(f"    lr={lr}, epochs={epochs}, "
          f"lambda_recon={lambda_recon}, lambda_anchor={lambda_anchor}, "
          f"lambda_shape={lambda_shape}")

    lr_f = float(lambda_recon)
    la_f = float(lambda_anchor)
    ls_f = float(lambda_shape)

    best_loss, best_params = float("inf"), None

    for epoch in range(epochs):
        rng, perm_rng = jax.random.split(rng)
        bucket_order = jax.random.permutation(perm_rng, n_buckets)

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_anchor = 0.0
        epoch_shape = 0.0
        epoch_var = 0.0
        seqs_in_epoch = 0

        for bi in bucket_order:
            bi = int(bi)
            bkt = buckets[bi]
            rng, step_rng = jax.random.split(rng)

            state, loss, aux = _train_step_temporal(
                state, bkt['input_batch'], bkt['target_batch'],
                bkt['anchor_batch'], step_rng, bkt['mask'],
                bkt['T_bucket'], lr_f, la_f, ls_f)

            n_in = len(bkt['n_real'])
            epoch_loss += float(loss) * n_in
            l_r, l_a, l_s, l_v = aux
            epoch_recon += float(l_r) * n_in
            epoch_anchor += float(l_a) * n_in
            epoch_shape += float(l_s) * n_in
            epoch_var += float(l_v) * n_in
            seqs_in_epoch += n_in

        epoch_loss /= seqs_in_epoch
        epoch_recon /= seqs_in_epoch
        epoch_anchor /= seqs_in_epoch
        epoch_shape /= seqs_in_epoch
        epoch_var /= seqs_in_epoch

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_params = jax.tree.map(lambda x: x.copy(), state.params)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={epoch_loss:.6f}"
                  f"  (recon={epoch_recon:.6f}, anchor={epoch_anchor:.6f}, "
                  f"shape={epoch_shape:.6f}, var={epoch_var:.6f})")

    if best_params is not None:
        state = state.replace(params=best_params)
    print(f"    Best {label} loss: {best_loss:.6f}")
    return state


# ============================================================================
# Forward model training (explicit kwargs)
# ============================================================================

def train_forward_model(model, latent_trajectories, T_originals, obs_len, rng, *,
                        lr=1e-3, epochs=500, lambda_recon=2.0,
                        lambda_anchor=2.0, lambda_shape=30.0,
                        weight_decay=1e-5, bucket_width=5):
    """Train a forward temporal model (windowed observation).

    Prepares bucketed training data from latent trajectories: the first
    ``obs_len`` latent codes are the input window, the remainder are the
    target.  The anchor is the last observed latent (continuity at the
    boundary).

    All hyperparameters are explicit keyword arguments — no config object.
    """
    print("  Training forward temporal model (bucketed) ...")

    # Build per-sequence (input, target, anchor) triples
    input_seqs, target_seqs, anchor_vecs = [], [], []
    for traj, T_orig in zip(latent_trajectories, T_originals):
        traj = traj[:T_orig]
        W = min(obs_len, T_orig)
        inp = traj[:W]                 # (W, D)
        tgt = traj[obs_len:]           # (T_future, D) — may be empty for short seqs
        if len(tgt) == 0:
            continue
        input_seqs.append(inp)
        target_seqs.append(tgt)
        anchor_vecs.append(traj[W - 1])  # last observed latent

    buckets = _prepare_buckets(input_seqs, target_seqs, anchor_vecs,
                               bucket_width=bucket_width)

    return _run_bucketed_training(
        model, buckets, rng,
        lr=lr, epochs=epochs, lambda_recon=lambda_recon,
        lambda_anchor=lambda_anchor, lambda_shape=lambda_shape,
        weight_decay=weight_decay, label="forward model")


# ============================================================================
# Backward model training (explicit kwargs)
# ============================================================================

def train_backward_model(model, latent_trajectories, T_originals, obs_len, rng, *,
                         z_inits=None,
                         lr=1e-3, epochs=500, lambda_recon=2.0,
                         lambda_anchor=2.0, lambda_shape=30.0,
                         weight_decay=1e-5, bucket_width=5):
    """Train a backward temporal model.

    Auto-detects the model type:
      - BackwardFromTerminal (input is (B, D)): single terminal latent ->
        full preceding trajectory.  ``obs_len`` is ignored; input is the
        last latent code, target is everything before it, anchor is z_init.
      - BackwardFromWindow (input is (B, W, D)): terminal latent window ->
        preceding trajectory.  Input is the last ``obs_len`` latent codes,
        target is the preceding codes, anchor is z_init (first latent).

    All hyperparameters are explicit keyword arguments — no config object.
    """
    D = latent_trajectories[0].shape[1]
    max_T = max(T_originals)
    is_terminal = _is_terminal_model(model, D, max_T, rng)

    if is_terminal:
        print("  Training backward temporal model — terminal mode (bucketed) ...")
        z_init_list = z_inits or [t[0] for t in latent_trajectories]
        input_seqs, target_seqs, anchor_vecs = [], [], []
        for traj, T_orig, z_init in zip(latent_trajectories, T_originals, z_init_list):
            traj = traj[:T_orig]
            input_seqs.append(traj[-1])      # z_T, shape (D,)
            target_seqs.append(traj[:-1])     # {z_0, ..., z_{T-1}}, shape (T_orig-1, D)
            anchor_vecs.append(z_init)        # anchor: predicted t=0 should match z_init
    else:
        print("  Training backward temporal model — window mode (bucketed) ...")
        input_seqs, target_seqs, anchor_vecs = [], [], []
        for traj, T_orig in zip(latent_trajectories, T_originals):
            traj = traj[:T_orig]
            W = min(obs_len, T_orig)
            inp = traj[-W:]                   # terminal window, (W, D)
            tgt = traj[:T_orig - W]           # preceding frames, (T_prior, D)
            if len(tgt) == 0:
                continue
            input_seqs.append(inp)
            target_seqs.append(tgt)
            anchor_vecs.append(traj[0])       # anchor: predicted t=0 should match z_init

    buckets = _prepare_buckets(input_seqs, target_seqs, anchor_vecs,
                               bucket_width=bucket_width)

    return _run_bucketed_training(
        model, buckets, rng,
        lr=lr, epochs=epochs, lambda_recon=lambda_recon,
        lambda_anchor=lambda_anchor, lambda_shape=lambda_shape,
        weight_decay=weight_decay, label="backward model")


def _is_terminal_model(model, D, max_T, rng):
    """Check if model takes (B, D) input (terminal) vs (B, W, D) (window)."""
    try:
        rng_init = jax.random.split(rng)[0]
        model.init(rng_init, jnp.ones((1, D)), max_T - 1, train=False)
        return True
    except Exception:
        return False
