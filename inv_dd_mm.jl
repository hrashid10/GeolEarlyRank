# physics_informed_inv_vae_latent_from_images_fixed.jl


using Pkg
for pkg in [
    "Flux","Zygote","Optimisers","Random","Statistics","CUDA","BSON","Printf",
    "DPFEHM","PyPlot","Images","FileIO","ImageIO","ImageTransformations"
]
    Base.find_package(pkg) === nothing && Pkg.add(pkg)
end

using Random
using Statistics: mean
using Flux
using Zygote
using Optimisers
using CUDA
using BSON
using Printf
using DPFEHM
using PyPlot
using Images, FileIO, ImageIO, ImageTransformations

# Config
const USE_GPU = false

LATENT_MODEL_PATH = "vae_best_mismatch.bson" 

# LATENT_MODEL_PATH = "vae_best.bson"        # or "vae_trained_finalmpi.bson"
DATA_DIR          = "./image_train_mismatch"  # directory containing true permeability images

# Dataset size
N_train   = 1550        # clipped to available data
N_val     =  50
batchsize =  32        # smaller batch often safer when simulator is in-loop
n_epochs  = 2000
lr        = 1e-4

# Optional subset per epoch (set <=0 to use all training data)
samples_per_epoch = 500  # e.g., 200 for faster epochs; 0 means full N_train

# Loss weights (tune these)

λ_perm   = 1.0f0
λ_latent = 1.0f0
λ_prior  = 1.0e-2

# Observation noise
use_noise_train = true
use_noise_val   = true
data_noise_relstd = 0.01f0  # relative noise std on BHP

# Log-permeability mapping from image [0,1] to log10(K)
const LOGK_MIN = -30.0f0
const LOGK_MAX =  -25.0f0

# DPFEHM / grid config
num_mon = 250
n       = 128
ns      = (n, n)
sidelength = 100.0
thickness  = 1.0
injrate    = 0.0
injection_node = 26

# Reproducibility
Random.seed!(12345)

# Save paths
CHECKPOINT_PREFIX = "inv_net_data_mismatch"
BEST_MODEL_PATH   = "$(CHECKPOINT_PREFIX)_best_mismatch.bson"
FINAL_MODEL_PATH  = "$(CHECKPOINT_PREFIX)_final_model_mismatch.bson"
FINAL_LOSS_PATH   = "$(CHECKPOINT_PREFIX)_final_losses_mismatch.bson"

dev(x) = (USE_GPU && CUDA.functional()) ? cu(x) : x

# ---------------------------------------------------------
# VAE definitions / loading (frozen)

mutable struct VAE
    encoder::Any
    fc_mu::Any
    fc_logvar::Any
    fc_dec::Any
    decoder::Any
end
Flux.@functor VAE

function encode(m::VAE, x)
    h = m.encoder(x)
    μ = m.fc_mu(h)
    logvar = m.fc_logvar(h)
    return μ, logvar
end

function decode(m::VAE, z)
    # FIXED: must match training script (256x256 with 4 downsamples => 16x16x256)
    h = m.fc_dec(z)
    h = reshape(h, 16, 16, 256, size(z,2))
    return m.decoder(h)
end

@info "Loading frozen VAE from $LATENT_MODEL_PATH"
vae = nothing
Z_DIM = 0
IMG_H = 0
IMG_W = 0
IMG_C = 0

IMG_H, IMG_W, IMG_C = 256, 256, 1
Z_DIM      = 128

# Load only known fields that exist in your saved VAE file
BSON.@load LATENT_MODEL_PATH model 
vae = model
@info "Loaded VAE metadata" IMG_H IMG_W IMG_C Z_DIM

if USE_GPU && CUDA.functional()
    vae = Flux.fmap(cu, vae)
    @info "Using GPU (VAE + inverse net)"
else
    @info "Using CPU"
end

function image_files(dir::AbstractString)
    exts = Set([".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp",".gif"])
    files = String[]
    for (root, _, names) in walkdir(dir)
        for name in names
            ext = lowercase(splitext(name)[2])
            if ext in exts
                push!(files, joinpath(root, name))
            end
        end
    end
    sort!(files)
    return files
end

function load_gray_resize(path::AbstractString; h::Int=IMG_H, w::Int=IMG_W)
    img_any = load(path)
    img_gray = ndims(img_any) == 2 ? Gray.(img_any) :
               Gray.(colorview(RGB, channelview(img_any)))
    img_r = imresize(img_gray, (h, w))
    img_f = Float32.(img_r)
    reshape(img_f, h, w, 1)  # (H,W,1)
end

# decode latent to image batch (IMG_H, IMG_W, 1, B)
function vae_decode(vae, z::AbstractMatrix{<:Real})
    zf = Float32.(z)
    h = vae.fc_dec(zf)
    h = reshape(h, 16, 16, 256, size(zf,2))   # FIXED
    return vae.decoder(h)
end

# Map image -> n×n log10(K)
function image_to_logK_grid(img_hw1::AbstractArray{<:Real,3}; n::Int)
    H, W, _ = size(img_hw1)
    ii = round.(Int, range(1, H; length=n))
    jj = round.(Int, range(1, W; length=n))
    g  = Float32.(img_hw1[ii, jj, 1])          # (n,n)
    g  = clamp.(g, 0f0, 1f0)
    return LOGK_MIN .+ g .* (LOGK_MAX - LOGK_MIN)
end

# ---------------------------------------------------------
# DPFEHM setup
# ---------------------------------------------------------
@info "Building DPFEHM grid + BCs..."
coords, neighbors, areasoverlengths, volumes =
    DPFEHM.regulargrid2d([-sidelength, -sidelength],
                         [ sidelength,  sidelength],
                         ns, thickness)

steadyhead = 0.0
dirichletnodes = Int[]
dirichleths = zeros(Float32, size(coords,2))

for i in 1:size(coords,2)
    if coords[1,i] == sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = Float32(steadyhead)
    end
end

for i in 1:size(coords,2)
    if coords[1,i] == -sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = 10.0f0
    end
end

ymax = maximum(coords[2,:])
ymin = minimum(coords[2,:])
for i in 1:size(coords,2)
    if coords[2,i] == ymax
        push!(dirichletnodes, i)
        dirichleths[i] = 0.5f0
    elseif coords[2,i] == ymin
        push!(dirichletnodes, i)
        dirichleths[i] = 0.0f0
    end
end

Random.seed!(1256879)
monitoring_nodes = sort!(randperm(n*n)[1:num_mon])
@info "Monitoring nodes selected" num_mon


function getQs(Qs, is, ncell)
    sum(Qs .* ((collect(1:ncell) .== i) for i in is))
end

logKs2Ks_neighbors(Ks, neighbors) =
    exp.(0.5f0 .* (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))

function solve_bhp_from_logK(logK_grid::AbstractMatrix{<:Real})
    ncell = size(coords,2)
    Q_vec = getQs([injrate], [injection_node], ncell)
    Ks_neighbors = logKs2Ks_neighbors(Float32.(logK_grid), neighbors)
    P_full = DPFEHM.groundwater_steadystate(
        Ks_neighbors, neighbors, areasoverlengths,
        dirichletnodes, dirichleths, Q_vec
    )
    return Float32.(P_full[monitoring_nodes])
end

@info "Loading true permeability images from $DATA_DIR"
files = image_files(DATA_DIR)
@info "Found $(length(files)) images"
@assert !isempty(files) "No images found in $DATA_DIR"

N_total = length(files)

if N_train + N_val > N_total
    @warn "Requested N_train+N_val > #images; adjusting." N_train N_val N_total
    N_train = max(1, min(N_train, N_total - 1))
    N_val   = N_total - N_train
end

perm_idx = randperm(N_total)
train_idx = perm_idx[1:N_train]
val_idx   = perm_idx[N_train+1 : N_train+N_val]

@info "Using N_train=$N_train, N_val=$N_val"

# Allocate
train_logK = Array{Float32}(undef, n,       n,       N_train)
train_bhp  = Array{Float32}(undef, num_mon,          N_train)
train_zmu  = Array{Float32}(undef, Z_DIM,            N_train)

val_logK   = Array{Float32}(undef, n,       n,       N_val)
val_bhp    = Array{Float32}(undef, num_mon,          N_val)
val_zmu    = Array{Float32}(undef, Z_DIM,            N_val)

@info "Precomputing training logK, BHP, and latent μ..."
for (k, idx) in enumerate(train_idx)
    img3   = load_gray_resize(files[idx]; h=IMG_H, w=IMG_W)  # (IMG_H,IMG_W,1)
    img4   = reshape(img3, IMG_H, IMG_W, 1, 1)               # (IMG_H,IMG_W,1,1)

    μ, _ = encode(vae, Float32.(img4))
    zμ = Array(Float32.(μ[:,1]))                             # move to CPU if needed

    logK = image_to_logK_grid(img3; n=n)                     # (n,n)
    bhp  = solve_bhp_from_logK(logK)                         # (num_mon)

    train_zmu[:,k]    .= zμ
    train_logK[:,:,k] .= logK
    train_bhp[:,k]    .= bhp
end

@info "Precomputing validation logK, BHP, and latent μ..."
for (k, idx) in enumerate(val_idx)
    img3   = load_gray_resize(files[idx]; h=IMG_H, w=IMG_W)
    img4   = reshape(img3, IMG_H, IMG_W, 1, 1)

    μ, _ = encode(vae, Float32.(img4))
    zμ = Array(Float32.(μ[:,1]))

    logK = image_to_logK_grid(img3; n=n)
    bhp  = solve_bhp_from_logK(logK)

    val_zmu[:,k]    .= zμ
    val_logK[:,:,k] .= logK
    val_bhp[:,k]    .= bhp
end

# ---------------------------------------------------------
# Noise helper for BHP observations
# ---------------------------------------------------------
function add_white_noise(y::AbstractArray{<:Real}; rel_std=0.01f0)
    yy = Float32.(y)
    μmag  = mean(abs, yy)
    σ  = rel_std * max(μmag, eps(Float32))
    return yy .+ σ .* randn(Float32, size(yy))
end

# ---------------------------------------------------------
# Latent -> logK_hat and BHP_hat (VAE only, no DDPM)
# ---------------------------------------------------------
# z_batch: (Z_DIM, B)
# function forward_logK_from_latent(z_batch::AbstractMatrix{<:Real})
#     B = size(z_batch, 2)
#     logK_list = map(1:B) do i
#         img4 = vae_decode(vae, z_batch[:, i:i])   # (IMG_H,IMG_W,1,1)
#         img3 = dropdims(img4; dims=4)             # (IMG_H,IMG_W,1)
#         image_to_logK_grid(img3; n=n)             # (n,n)
#     end
#     logK_hat = cat(logK_list...; dims=3)          # (n,n,B)
#     return Float32.(logK_hat)
# end

const RESAMPLE_II = round.(Int, range(1, IMG_H; length = n))
const RESAMPLE_JJ = round.(Int, range(1, IMG_W; length = n))

function vae_decode_batch(vae, z_batch)
    h = vae.fc_dec(z_batch)
    h = reshape(h, 16, 16, 256, size(z_batch, 2))
    return vae.decoder(h)   # (IMG_H, IMG_W, 1, B)
end

function forward_logK_from_latent(z_batch)
    img4 = vae_decode_batch(vae, z_batch)              # decode whole batch once
    g = img4[RESAMPLE_II, RESAMPLE_JJ, 1, :]           # (n, n, B)
    g = clamp.(g, 0f0, 1f0)
    return LOGK_MIN .+ g .* (LOGK_MAX - LOGK_MIN)
end

function bhp_from_logK_batch(logK_batch::Array{Float32,3})
    n1, n2, B = size(logK_batch)
    @assert n1 == n && n2 == n
    bhp_list = map(1:B) do i
        solve_bhp_from_logK(logK_batch[:,:,i])    # (num_mon)
    end
    bhp_hat = reduce(hcat, bhp_list)              # (num_mon,B)
    return Float32.(bhp_hat)
end

# ---------------------------------------------------------
# Inverse network & loss
# ---------------------------------------------------------
inverse_net = Chain(
    Dense(num_mon, 256, relu),
    Dense(256,    256, relu),
    Dense(256,    256, relu),
    Dense(256,    Z_DIM)
)

if USE_GPU && CUDA.functional()
    inverse_net = Flux.fmap(cu, inverse_net)
end

opt   = Optimisers.Adam(lr)
state = Optimisers.setup(opt, inverse_net)

mse(a, b) = mean((a .- b).^2)

"""
    loss_batch(bhp_obs_batch, bhp_target_batch, logK_true_batch, z_true_batch, model)

Inputs:
- bhp_obs_batch:     (num_mon, B)   noisy/observed pressure for inverse input
- bhp_target_batch:  (num_mon, B)   target pressure for physics loss (often clean true)
- logK_true_batch:   (n, n, B)
- z_true_batch:      (Z_DIM, B)     latent target from frozen VAE encoder mean μ
"""
function loss_batch(bhp_obs_batch::AbstractMatrix{Float32},
                    bhp_target_batch::AbstractMatrix{Float32},
                    logK_true_batch::Array{Float32,3},
                    z_true_batch::AbstractMatrix{Float32},
                    model)

    # 1) inverse: BHP_obs -> latent
    z_pred = Float32.(model(bhp_obs_batch))            # (Z_DIM,B)

    # 2) latent -> permeability via frozen VAE decoder
    logK_hat = forward_logK_from_latent(z_pred)        # (n,n,B)

    # 3) permeability -> pressure via differentiable simulator
    # bhp_hat  = bhp_from_logK_batch(logK_hat)           # (num_mon,B)

    # 4) losses
    # phys_loss   = mse(bhp_hat, bhp_target_batch)
    perm_loss   = mse(logK_hat, logK_true_batch)
    latent_loss = mse(z_pred, z_true_batch)
    prior_loss  = mean(sum(abs2, z_pred; dims=1))      # latent prior ||z||²

    total = λ_perm * perm_loss +
            λ_latent * latent_loss +
            λ_prior * prior_loss

    return total,  perm_loss, latent_loss, prior_loss
end


function eval_dataset_loss(bhp_data::Array{Float32,2},
                           logK_data::Array{Float32,3},
                           zmu_data::Array{Float32,2},
                           model;
                           batchsize::Int=20,
                           use_noise::Bool=false)

    N = size(bhp_data, 2)
    nb = cld(N, batchsize)

    total_sum  = 0f0
    perm_sum   = 0f0
    latent_sum = 0f0
    prior_sum  = 0f0
    count      = 0

    for b in 1:nb
        s = (b-1)*batchsize + 1
        e = min(b*batchsize, N)
        idxs = s:e
        B = length(idxs)

        bhp_true_batch  = Float32.(bhp_data[:, idxs])
        bhp_obs_batch   = use_noise ? add_white_noise(bhp_true_batch; rel_std=data_noise_relstd) : bhp_true_batch
        logK_true_batch = Float32.(logK_data[:, :, idxs])
        z_true_batch    = Float32.(zmu_data[:, idxs])

        ℓ, k, z, pr = loss_batch(bhp_obs_batch, bhp_true_batch, logK_true_batch, z_true_batch, model)

        total_sum  += Float32(ℓ) * B
        perm_sum   += Float32(k) * B
        latent_sum += Float32(z) * B
        prior_sum  += Float32(pr) * B
        count      += B
    end

    return (
        total  = total_sum  / max(count,1),
        perm   = perm_sum   / max(count,1),
        latent = latent_sum / max(count,1),
        prior  = prior_sum  / max(count,1)
    )
end


@info "Starting inverse training (VAE-only, no DDPM)..."

train_losses  = Float32[]
val_losses    = Float32[]

perm_losses   = Float32[]
latent_losses = Float32[]
prior_losses  = Float32[]


val_perm_losses   = Float32[]
val_latent_losses = Float32[]
val_prior_losses  = Float32[]

best_val = Inf32
best_epoch = 0

for epoch in 1:n_epochs
    # Choose training samples for this epoch
    if samples_per_epoch > 0
        M = min(samples_per_epoch, N_train)
        epoch_indices = randperm(N_train)[1:M]
    else
        M = N_train
        epoch_indices = randperm(N_train)
    end

    n_batches = cld(M, batchsize)

    epoch_total  = 0f0
    epoch_perm   = 0f0
    epoch_latent = 0f0
    epoch_prior  = 0f0
    epoch_count  = 0

    for b in 1:n_batches
        s = (b-1)*batchsize + 1
        e = min(b*batchsize, M)
        idxs = epoch_indices[s:e]
        B = length(idxs)

        # Gather batch
        bhp_true_batch  = Float32.(train_bhp[:, idxs])
        bhp_obs_batch   = use_noise_train ? add_white_noise(bhp_true_batch; rel_std=data_noise_relstd) : bhp_true_batch
        logK_true_batch = Float32.(train_logK[:, :, idxs])
        z_true_batch    = Float32.(train_zmu[:, idxs])

        if USE_GPU && CUDA.functional()
            bhp_true_batch  = cu(bhp_true_batch)
            bhp_obs_batch   = cu(bhp_obs_batch)
            logK_true_batch = cu(logK_true_batch)
            z_true_batch    = cu(z_true_batch)
        end

        # Gradient wrt inverse_net only
        L, back = Flux.withgradient(inverse_net) do m
            ℓ, _, _, _ = loss_batch(bhp_obs_batch, bhp_true_batch, logK_true_batch, z_true_batch, m)
            return ℓ
        end

        Optimisers.update!(state, inverse_net, back[1])

        # Recompute metrics for logging (expensive but clear)
        ℓ,  k, z, pr = loss_batch(bhp_obs_batch, bhp_true_batch, logK_true_batch, z_true_batch, inverse_net)

        scalar(x) = x isa Number ? Float32(x) : Float32(Array(x)[1])

        ℓf  = scalar(ℓ)
        kf  = scalar(k)
        zf  = scalar(z)
        prf = scalar(pr)

        epoch_total  += ℓf * B
        epoch_perm   += kf * B
        epoch_latent += zf * B
        epoch_prior  += prf * B
        epoch_count  += B
    end

    # Average train metrics
    train_total  = epoch_total  / max(epoch_count, 1)
    train_perm   = epoch_perm   / max(epoch_count, 1)
    train_latent = epoch_latent / max(epoch_count, 1)
    train_prior  = epoch_prior  / max(epoch_count, 1)

    push!(train_losses,  train_total)
    push!(perm_losses,   train_perm)
    push!(latent_losses, train_latent)
    push!(prior_losses,  train_prior)

    # Validation (clean + noisy optional)
    valm = eval_dataset_loss(
        val_bhp, val_logK, val_zmu, inverse_net;
        batchsize=batchsize, use_noise=use_noise_val
    )

    push!(val_losses,         Float32(valm.total))
    push!(val_perm_losses,    Float32(valm.perm))
    push!(val_latent_losses,  Float32(valm.latent))
    push!(val_prior_losses,   Float32(valm.prior))

    @printf("epoch %4d | train=%.6e ( perm=%.6e latent=%.6e prior=%.6e) | val=%.6e ( perm=%.6e latent=%.6e prior=%.6e)\n",
            epoch,
            train_total, train_perm, train_latent, train_prior,
            valm.total, valm.perm,  valm.latent,  valm.prior)

    # Save best model
    if Float32(valm.total) < best_val
        global best_val = Float32(valm.total)
        global best_epoch = epoch
        BSON.@save BEST_MODEL_PATH inverse_net epoch best_val best_epoch train_total train_perm train_latent train_prior
        @info "Saved best inverse model" best_epoch best_val
    end

    # Periodic checkpoint
    if epoch % 50 == 0
        ckpt_path = "$(CHECKPOINT_PREFIX)_checkpoint_epoch$(epoch).bson"
        BSON.@save ckpt_path inverse_net train_losses val_losses  perm_losses latent_losses prior_losses val_perm_losses val_latent_losses val_prior_losses 

    end
end

# Final save
BSON.@save FINAL_MODEL_PATH inverse_net best_val best_epoch 


BSON.@save FINAL_LOSS_PATH train_losses val_losses  perm_losses latent_losses prior_losses val_perm_losses val_latent_losses val_prior_losses

@info "Done. Saved final model/losses." FINAL_MODEL_PATH FINAL_LOSS_PATH BEST_MODEL_PATH

# # ---------------------------------------------------------
# # Optional quick plots
# # ---------------------------------------------------------
# figure(figsize=(9,5))
# plot(train_losses, label="train total")
# plot(val_losses, label="val total")
# yscale("log")
# xlabel("epoch")
# ylabel("loss")
# legend()
# tight_layout()
# savefig("inverse_training_total_loss.png", dpi=200)

# figure(figsize=(9,5))
# plot(phys_losses, label="train phys")
# plot(perm_losses, label="train perm")
# plot(latent_losses, label="train latent")
# plot(prior_losses, label="train prior")
# yscale("log")
# xlabel("epoch")
# ylabel("loss component")
# legend()
# tight_layout()
# savefig("inverse_training_components.png", dpi=200)