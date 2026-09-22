# Train_VAE_MPI_images_fixed.jl
# MPI data-parallel training for a Conv VAE on real geologic images (faults/folds)
# Improvements:
#   - deterministic validation (decode from μ, no latent sampling noise)
#   - separate validation metrics: recon / KL / total
#   - edge-aware reconstruction loss (better for faults/folds)
#   - best-checkpoint saving
#   - device-safe reparameterization (CPU/GPU-safe if extended later)
#   - clearer metric naming

import Random
import BSON
import MPI
import Flux
import Optimisers
import Functors
import Statistics: mean

using Flux
using Images, FileIO, ImageIO, ImageTransformations
using Functors: fmap

# -------------------------
# MPI init
# -------------------------
MPI.Init()
comm   = MPI.COMM_WORLD
rank   = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

# -----------------------------
# Config
# -----------------------------
const USE_GPU = false  # CPU-only in this script; leave false unless CUDA path is fully enabled
DATA_DIR   = "./image_train_plausible"

IMG_H, IMG_W, IMG_C = 256, 256, 1
BATCH_SIZE = 32
EPOCHS     = 3000
LR         = 1e-4
Z_DIM      = 128

# KL regularization (important for latent-space inversion later)
BETA_FINAL = 0.01f0      # increased from 0.001 for better latent regularization
WARMUP_E   = 100         # slower warmup can stabilize KL ramp

# Edge-aware reconstruction (helps faults/folds)
LAMBDA_EDGE = 0.10f0

SAVE_EVERY = 50
SAVE_PATH  = "vae_trained_finalmpi_plausible.bson"
BEST_PATH  = "vae_best_plausible.bson"

seed = 1234
Random.seed!(seed + rank)

# -------------------------
# Helpers
# -------------------------
function local_args(args::Vector{Int}, r::Int, np::Int)
    N = length(args)
    base, rem = divrem(N, np)
    counts = [i < rem ? base + 1 : base for i in 0:np-1]
    offs = cumsum([0; counts[1:end-1]])
    lo = offs[r+1] + 1
    hi = offs[r+1] + counts[r+1]
    return lo > hi ? Int[] : args[lo:hi]
end

function allreduce_grads!(grads, comm, nprocs)
    # grads is a model-structured tree from Flux.withgradient(model) do m ... end
    Functors.fmap(grads) do leaf
        leaf === nothing && return nothing
        buf = copy(leaf)
        MPI.Allreduce!(buf, leaf, MPI.SUM, comm)
        leaf ./= nprocs
        return leaf
    end
    return grads
end

function bcast_model!(model, comm, root::Int=0)
    for p in Flux.params(model)
        MPI.Bcast!(p, root, comm)
    end
    return model
end

# -----------------------------
# I/O helpers
# -----------------------------
function load_gray_resize(path::AbstractString; h=IMG_H, w=IMG_W)
    img_any = load(path)
    img_gray = ndims(img_any) == 2 ? Gray.(img_any) :
               Gray.(colorview(RGB, Images.channelview(img_any)))
    img_r = imresize(img_gray, (h, w))
    img_f = Float32.(img_r)              # [0,1]
    reshape(img_f, h, w, 1)              # (H,W,1)
end

function image_files(dir::AbstractString)
    exts = Set([".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp",".gif"])
    files = String[]
    for (root, _, names) in walkdir(dir)
        for name in names
            ext = lowercase(Base.Filesystem.splitext(name)[2])
            if ext in exts
                push!(files, joinpath(root, name))
            end
        end
    end
    sort!(files) # deterministic order on all ranks
    return files
end

function stack_batch_from_paths(paths::Vector{String})
    N = length(paths)
    xb = Array{Float32}(undef, IMG_H, IMG_W, IMG_C, N)
    failed = 0
    for (i, p) in enumerate(paths)
        try
            xb[:,:,:,i] = load_gray_resize(p)
        catch
            xb[:,:,:,i] .= 0f0
            failed += 1
        end
    end
    return xb, failed
end

# -----------------------------
# VAE model (Conv)
# IMPORTANT: use ::Any fields to avoid BSON convert issues later
# -----------------------------
encoder = Chain(
    Conv((4,4), 1=>32,   stride=2, pad=1), x -> relu.(x),
    Conv((4,4), 32=>64,  stride=2, pad=1), x -> relu.(x),
    Conv((4,4), 64=>128, stride=2, pad=1), x -> relu.(x),
    Conv((4,4), 128=>256,stride=2, pad=1), x -> relu.(x),
    x -> reshape(x, :, size(x,4))
)

flat_dim = 16 * 16 * 256
fc_mu     = Dense(flat_dim, Z_DIM)
fc_logvar = Dense(flat_dim, Z_DIM)
fc_dec    = Dense(Z_DIM, flat_dim)

decoder = Chain(
    x -> x,
    ConvTranspose((4,4), 256=>128, stride=2, pad=1), x -> relu.(x),
    ConvTranspose((4,4), 128=>64,  stride=2, pad=1), x -> relu.(x),
    ConvTranspose((4,4), 64=>32,   stride=2, pad=1), x -> relu.(x),
    ConvTranspose((4,4), 32=>1,    stride=2, pad=1),
    x -> sigmoid.(x)
)

mutable struct VAE
    encoder::Any
    fc_mu::Any
    fc_logvar::Any
    fc_dec::Any
    decoder::Any
end

Flux.@functor VAE
VAE(; encoder, fc_mu, fc_logvar, fc_dec, decoder) = VAE(encoder, fc_mu, fc_logvar, fc_dec, decoder)

function encode(m::VAE, x)
    h = m.encoder(x)
    μ = m.fc_mu(h)
    logvar = m.fc_logvar(h)
    return μ, logvar
end

# device-safe reparameterization (works on CPU arrays; also safer if later moved to GPU)
function reparameterize(μ, logvar)
    σ = exp.(0.5f0 .* logvar)
    ϵ = similar(σ)
    Random.randn!(ϵ)
    return μ .+ σ .* ϵ
end

function decode(m::VAE, z)
    h = m.fc_dec(z)
    h = reshape(h, 16, 16, 256, size(z,2))
    return m.decoder(h)
end

function forward(m::VAE, x)
    μ, logvar = encode(m, x)
    z = reparameterize(μ, logvar)
    recon = decode(m, z)
    return recon, μ, logvar
end

# deterministic forward for validation/visualization (no sampling noise)
function forward_deterministic(m::VAE, x)
    μ, logvar = encode(m, x)
    recon = decode(m, μ)
    return recon, μ, logvar
end

# -----------------------------
# Losses / metrics
# -----------------------------
mae(a,b) = mean(abs.(a .- b))
mse(a,b) = mean((a .- b).^2)

# KL per latent element averaged over batch
kld(μ, logvar) = -0.5f0 * mean(1 .+ logvar .- μ.^2 .- exp.(logvar))

β(epoch) = min(Float32(epoch) / Float32(WARMUP_E), 1f0) * BETA_FINAL

# Edge loss via finite differences (fault/fold-aware)
function edge_loss(a, b)
    # a,b shape: (H,W,C,N)

    dx_a = (@view a[2:end, :, :, :]) .- (@view a[1:end-1, :, :, :])
    dx_b = (@view b[2:end, :, :, :]) .- (@view b[1:end-1, :, :, :])

    dy_a = (@view a[:, 2:end, :, :]) .- (@view a[:, 1:end-1, :, :])
    dy_b = (@view b[:, 2:end, :, :]) .- (@view b[:, 1:end-1, :, :])

    return mean(abs.(dx_a .- dx_b)) + mean(abs.(dy_a .- dy_b))
end

function recon_loss(recon, x; λ_edge::Float32=LAMBDA_EDGE)
    rec_mae = mae(recon, x)
    rec_edge = edge_loss(recon, x)
    total = rec_mae + λ_edge * rec_edge
    return total, rec_mae, rec_edge
end

# stochastic train loss (with sampling)
function vae_loss_train(m::VAE, x, epoch::Int)
    recon, μ, logvar = forward(m, x)
    rec_total, rec_mae, rec_edge = recon_loss(recon, x)
    kld_l = kld(μ, logvar)
    total = rec_total + β(epoch) * kld_l
    return total, rec_total, rec_mae, rec_edge, kld_l
end

# deterministic val metrics (decode from μ)
function vae_metrics_val(m::VAE, x, epoch::Int)
    recon, μ, logvar = forward_deterministic(m, x)
    rec_total, rec_mae, rec_edge = recon_loss(recon, x)
    kld_l = kld(μ, logvar)
    total = rec_total + β(epoch) * kld_l
    return total, rec_total, rec_mae, rec_edge, kld_l
end

# -----------------------------
# Build file list + split (rank0 decides, broadcast indices)
# -----------------------------
files = image_files(DATA_DIR)
if rank == 0
    @info "Found $(length(files)) images in $(abspath(DATA_DIR))"
end
if isempty(files)
    rank == 0 && error("No images found in $(DATA_DIR).")
end

nfiles = length(files)
train_ratio = 0.8

train_idx = Int[]
val_idx   = Int[]
if rank == 0
    idx = collect(1:nfiles)
    Random.seed!(seed)   # same shuffle every run
    Random.shuffle!(idx)
    ntr = Int(floor(train_ratio * nfiles))
    train_idx = idx[1:ntr]
    val_idx   = idx[ntr+1:end]
end

# broadcast lengths first
lens = zeros(Int, 2)
if rank == 0
    lens[1] = length(train_idx)
    lens[2] = length(val_idx)
end
MPI.Bcast!(lens, 0, comm)

# allocate and broadcast arrays
if rank != 0
    train_idx = Vector{Int}(undef, lens[1])
    val_idx   = Vector{Int}(undef, lens[2])
end
MPI.Bcast!(train_idx, 0, comm)
MPI.Bcast!(val_idx,   0, comm)

# shard per-rank
train_local = local_args(train_idx, rank, nprocs)
val_local   = local_args(val_idx,   rank, nprocs)

if rank == 0
    @info "Split: train=$(length(train_idx)) val=$(length(val_idx)) | ranks=$nprocs"
end
@info "Rank $rank: train_local=$(length(train_local)) val_local=$(length(val_local))"

# -----------------------------
# Model + optimiser
# -----------------------------
model = VAE(; encoder, fc_mu, fc_logvar, fc_dec, decoder) |> Flux.f32
bcast_model!(model, comm, 0)

opt_state = Optimisers.setup(Optimisers.Adam(LR), model)

# compile sanity
x0 = zeros(Float32, IMG_H, IMG_W, IMG_C, 1)
_ = vae_loss_train(model, x0, 1)
_ = vae_metrics_val(model, x0, 1)

# -----------------------------
# Evaluation helper (deterministic validation)
# Returns global averages over provided local shard (allreduced inside)
# -----------------------------
function eval_val_metrics(model::VAE, files::Vector{String}, idxs::Vector{Int};
                          batch_size::Int=BATCH_SIZE, epoch::Int=1)
    local_total = 0.0
    local_rec   = 0.0
    local_mae   = 0.0
    local_edge  = 0.0
    local_kl    = 0.0
    local_n     = 0
    local_fail  = 0

    n = length(idxs)
    s = 1
    while s <= n
        e = min(s + batch_size - 1, n)
        batch_ids   = idxs[s:e]
        batch_paths = files[batch_ids]
        xb, failed  = stack_batch_from_paths(batch_paths)
        local_fail += failed

        L, RL, RMAE, REDGE, KL = vae_metrics_val(model, xb, epoch)
        bs = length(batch_ids)

        local_total += Float64(L)     * bs
        local_rec   += Float64(RL)    * bs
        local_mae   += Float64(RMAE)  * bs
        local_edge  += Float64(REDGE) * bs
        local_kl    += Float64(KL)    * bs
        local_n     += bs

        s = e + 1
    end

    global_total = MPI.Allreduce(local_total, MPI.SUM, comm)
    global_rec   = MPI.Allreduce(local_rec,   MPI.SUM, comm)
    global_mae   = MPI.Allreduce(local_mae,   MPI.SUM, comm)
    global_edge  = MPI.Allreduce(local_edge,  MPI.SUM, comm)
    global_kl    = MPI.Allreduce(local_kl,    MPI.SUM, comm)
    global_n     = MPI.Allreduce(local_n,     MPI.SUM, comm)
    global_fail  = MPI.Allreduce(local_fail,  MPI.SUM, comm)

    denom = max(global_n, 1)
    return (
        total = global_total / denom,
        rec   = global_rec   / denom,
        mae   = global_mae   / denom,
        edge  = global_edge  / denom,
        kl    = global_kl    / denom,
        n     = global_n,
        failed = global_fail
    )
end

# -----------------------------
# Training loop (MPI)
# -----------------------------
best_val_rec = Inf
best_epoch   = 0

for epoch in 1:EPOCHS
    MPI.Barrier(comm)
    t0 = MPI.Wtime()

    # local shuffle each epoch (only over local shard)
    Random.seed!(seed + 10_000 * epoch + rank)
    local_perm = copy(train_local)
    Random.shuffle!(local_perm)

    # local accumulators (train, stochastic)
    local_total = 0.0
    local_rec   = 0.0
    local_mae   = 0.0
    local_edge  = 0.0
    local_kl    = 0.0
    local_n     = 0
    local_fail  = 0

    nloc = length(local_perm)
    s = 1
    while s <= nloc
        e = min(s + BATCH_SIZE - 1, nloc)
        batch_ids   = local_perm[s:e]
        batch_paths = files[batch_ids]
        xb, failed  = stack_batch_from_paths(batch_paths)
        local_fail += failed

        # local batch grads
        loss_val, back = Flux.withgradient(model) do m
            L, _, _, _, _ = vae_loss_train(m, xb, epoch)
            return L
        end
        g = back[1]

        # gradient averaging across ranks
        allreduce_grads!(g, comm, nprocs)

        # apply synchronized update
        Optimisers.update!(opt_state, model, g)

        # log train metrics after update (stochastic train metrics)
        L, RL, RMAE, REDGE, KL = vae_loss_train(model, xb, epoch)
        bs = length(batch_ids)

        local_total += Float64(L)     * bs
        local_rec   += Float64(RL)    * bs
        local_mae   += Float64(RMAE)  * bs
        local_edge  += Float64(REDGE) * bs
        local_kl    += Float64(KL)    * bs
        local_n     += bs

        s = e + 1
    end

    MPI.Barrier(comm)
    t1 = MPI.Wtime()

    # global train averages
    global_total = MPI.Allreduce(local_total, MPI.SUM, comm)
    global_rec   = MPI.Allreduce(local_rec,   MPI.SUM, comm)
    global_mae   = MPI.Allreduce(local_mae,   MPI.SUM, comm)
    global_edge  = MPI.Allreduce(local_edge,  MPI.SUM, comm)
    global_kl    = MPI.Allreduce(local_kl,    MPI.SUM, comm)
    global_n     = MPI.Allreduce(local_n,     MPI.SUM, comm)
    global_fail  = MPI.Allreduce(local_fail,  MPI.SUM, comm)

    train_total = global_total / max(global_n, 1)
    train_rec   = global_rec   / max(global_n, 1)
    train_mae   = global_mae   / max(global_n, 1)
    train_edge  = global_edge  / max(global_n, 1)
    train_kl    = global_kl    / max(global_n, 1)

    # deterministic validation
    val = eval_val_metrics(model, files, val_local; batch_size=BATCH_SIZE, epoch=epoch)

    if rank == 0
        dt = t1 - t0
        println(
            "epoch: $epoch  time: $(round(dt, digits=3))  " *
            "train_total: $(train_total)  train_rec: $(train_rec)  train_mae: $(train_mae)  " *
            "train_edge: $(train_edge)  train_kl: $(train_kl)  " *
            "val_total: $(val.total)  val_rec: $(val.rec)  val_mae: $(val.mae)  val_edge: $(val.edge)  val_kl: $(val.kl)  " *
            "beta: $(β(epoch))  failed_imgs(train/val): $(global_fail)/$(val.failed)"
        )

        # Save best model based on deterministic validation reconstruction loss
        if val.rec < best_val_rec
            global best_val_rec = val.rec
            global best_epoch   = epoch
            BSON.@save BEST_PATH model epoch best_val_rec train_total train_rec train_mae train_edge train_kl val
            @info "New best checkpoint saved at epoch=$epoch with val_rec=$(best_val_rec)"
        end

        # periodic checkpoint
        if (epoch % SAVE_EVERY == 0) || (epoch == EPOCHS)
            BSON.@save "vae_checkpoint_epoch$(epoch).bson" model epoch train_total train_rec train_mae train_edge train_kl val
        end
    end
end

# Final save
if rank == 0
    BSON.@save SAVE_PATH model IMG_H IMG_W IMG_C Z_DIM BETA_FINAL WARMUP_E LR EPOCHS DATA_DIR LAMBDA_EDGE best_val_rec best_epoch
    @info "Saved final model to $SAVE_PATH"
    @info "Best checkpoint: $BEST_PATH (epoch=$best_epoch, val_rec=$best_val_rec)"
end

MPI.Barrier(comm)
MPI.Finalize()