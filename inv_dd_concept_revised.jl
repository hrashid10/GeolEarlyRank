# train_inverse_new_concept_wipp_report_bc_fixed.jl
#
# Purpose
# -------
# Train a data-driven inverse model for the revised/new WIPP Culebra conceptual
# model using the same WIPP-scale grid and simplified report-style boundary
# conditions used in run_wipp_steady_state_head_match_report_bc_fixed.jl.
#
# The inverse model learns:
#
#     synthetic heads at WIPP observation wells  ->  VAE latent vector
#
# The frozen VAE decoder maps the inferred latent vector back to a generated
# log10 transmissivity field. The DPFEHM solver is used only to generate the
# synthetic training heads and for diagnostics. It is not differentiated through.
#
# Important fixes relative to the earlier script
# ----------------------------------------------
# 1. Uses report-style approximate WIPP BCs:
#      north/east/south = fixed head; west = natural no-flow.
# 2. Uses centered planar boundary-head fit and clamps boundary heads to the
#    observed-head range +/- BOUNDARY_HEAD_MARGIN_M.
# 3. Uses a new cache filename and stores a cache signature, so old all-boundary
#    caches are not silently reused.
# 4. Uses small absolute head noise, not 1% of mean head. For WIPP heads near
#    900 m, 1% is about 9 m and can swamp the training signal.
# 5. Reduces/removes latent prior regularization by default. A large prior can
#    push the inferred latent vector toward zero and decode average-looking maps.
# 6. Writes diagnostic CSVs for boundary heads and training-head variability.
#
# Required files in the run directory:
#   wipp_culebra_2000_steady_heads.csv
#   wipp_culebra_model_domain.csv
#   vae_best_wiip_new.bson       # or edit LATENT_MODEL_PATH
#   ./image_train_new_concept/*.png
#
# Run:
#   julia train_inverse_new_concept_wipp_report_bc_fixed.jl

using Pkg
for pkg in [
    "CSV", "DataFrames", "Flux", "Zygote", "Optimisers", "Random",
    "Statistics", "LinearAlgebra", "CUDA", "BSON", "Printf", "DPFEHM",
    "PyPlot", "Images", "FileIO", "ImageIO", "ImageTransformations"
]
    Base.find_package(pkg) === nothing && Pkg.add(pkg)
end

ENV["MPLBACKEND"] = "Agg"

using CSV
using DataFrames
using Flux
using Zygote
using Optimisers
using Random
using Statistics
using LinearAlgebra
using CUDA
using BSON
using Printf
using DPFEHM
using PyPlot
using Images, FileIO, ImageIO, ImageTransformations

# =============================================================================
# User configuration
# =============================================================================

const USE_GPU = false

# Revised/new conceptual model VAE and image ensemble.
const LATENT_MODEL_PATH = "vae_best_wiip_new_v2_l.bson"
const DATA_DIR          = "./image_new_concept_wipp2_l"

# WIPP observation and domain files.
const HEAD_CSV   = "wipp_culebra_2000_steady_heads_v2.csv"
const DOMAIN_CSV = "wipp_culebra_model_domain.csv"

# WIPP model grid: approximately 22.4 km by 30.7 km with 100 m cells.
const NX = 224
const NY = 307
const THICKNESS = 1.0

# Image/VAE resolution.
const IMG_H, IMG_W, IMG_C = 256, 256, 1
const Z_DIM = 128

# Generated image -> log10 transmissivity mapping.
# Dark pixels -> LOG10T_MIN, bright pixels -> LOG10T_MAX.
const LOG10T_MIN = -8.0f0
const LOG10T_MAX = -3.0f0

# Boundary condition setup matching the fixed WIPP test script.
# :report_approx = fixed heads on north/east/south, natural no-flow on west.
# :all_edges     = fixed heads on all edges, useful only as a debugging option.
const BOUNDARY_MODE = :report_approx

# Stable boundary-head trend. Avoid flexible RBF by default because it can
# extrapolate badly from sparse head data.
const BOUNDARY_HEAD_METHOD = :plane
const CLAMP_BOUNDARY_HEADS = true
const BOUNDARY_HEAD_MARGIN_M = 5.0

# Optional source/sink term. Keep false for the same setup as the fixed WIPP
# steady-state head-matching script.
const USE_SOURCE = false
const SOURCE_X = 613700.0
const SOURCE_Y = 3581000.0
const SOURCE_RATE = 0.0

# Dataset size. These are clipped to available images.
N_train = 450
N_val   = 50

# Training settings.
const BATCHSIZE = 10
const N_EPOCHS  = 3000
const LR        = 5.0e-5
const SAMPLES_PER_EPOCH = 80      # <= 0 uses all training samples each epoch

# Loss settings.
# Start with permeability-dominant training. The latent target is only a weak
# stabilizer. The prior is off by default because it can push z toward zero.
const USE_PERM_LOSS = true
const LAMBDA_PERM   = 0.25f0
const LAMBDA_LATENT = 0.50f0
const LAMBDA_PRIOR  = 1.0e-4

# Head noise added to synthetic training inputs.
# For WIPP, do not use 1% of mean head, because mean head is about 900 m and
# 1% is about 9 m. That can destroy the head-to-latent relation.
const USE_NOISE_TRAIN = true
const USE_NOISE_VAL   = false
const HEAD_NOISE_STD_M = 0.25f0

# Cache precomputed WIPP log10T, head responses, and VAE latent means.
# This cache name is intentionally different from older all-boundary scripts.
const USE_DATA_CACHE = true
const FORCE_REBUILD_CACHE = false
const DATA_CACHE_PATH = "new_concept_v2_wipp_report_bc_fixed_training_cache.bson"

# Diagnostics and output.
const CHECKPOINT_PREFIX = "inv_net_wipp2_new_concept_dd_l"
const BEST_MODEL_PATH   = "$(CHECKPOINT_PREFIX)_best.bson"
const FINAL_MODEL_PATH  = "$(CHECKPOINT_PREFIX)_final_model.bson"
const FINAL_LOSS_PATH   = "$(CHECKPOINT_PREFIX)_final_losses.bson"
const PRECOMP_SUMMARY_CSV = "$(CHECKPOINT_PREFIX)_precompute_summary.csv"

const CHECKPOINT_EVERY = 25
const DIAGNOSTIC_PRESSURE_EVERY = 5
const N_PRESSURE_DIAGNOSTIC_SAMPLES = 20
const SAVE_EXAMPLE_RECON_EVERY = 50

# Reproducibility.
Random.seed!(12345)

dev(x) = (USE_GPU && CUDA.functional()) ? cu(x) : x

# =============================================================================
# VAE definition and loading
# =============================================================================

mutable struct VAE
    encoder::Any
    fc_mu::Any
    fc_logvar::Any
    fc_dec::Any
    decoder::Any
end
Flux.@functor VAE

# World-age-safe calls are used outside differentiable training, e.g. for VAE
# encoder precomputation. Decoder calls inside loss_batch must be direct calls
# so Zygote can differentiate with respect to z_pred.
call_model(model, x) = Base.invokelatest(model, x)

function encode_mu_safe(m::VAE, x)
    h = call_model(m.encoder, x)
    mu = call_model(m.fc_mu, h)
    return mu
end

function vae_decode_batch(m::VAE, z_batch)
    h = m.fc_dec(z_batch)
    h = reshape(h, 16, 16, 256, size(z_batch, 2))
    return m.decoder(h)  # (IMG_H, IMG_W, 1, B)
end

@info "Loading frozen VAE" LATENT_MODEL_PATH
isfile(LATENT_MODEL_PATH) || error("Missing VAE file: $(LATENT_MODEL_PATH)")
BSON.@load LATENT_MODEL_PATH model
vae = model

if USE_GPU && CUDA.functional()
    vae = Flux.fmap(cu, vae)
    @info "Using GPU for VAE and inverse net"
else
    @info "Using CPU"
end

# =============================================================================
# CSV loading utilities
# =============================================================================

function normalize_colname(name)
    s = lowercase(strip(String(name)))
    s = replace(s, "\ufeff" => "")
    s = replace(s, r"[^a-z0-9]+" => "_")
    s = replace(s, r"^_+|_+$" => "")
    return s
end

function find_column(df::DataFrame, candidates::Vector{String}; required_name::String)
    norm_to_name = Dict{String, Symbol}()
    for nm in names(df)
        norm_to_name[normalize_colname(nm)] = Symbol(nm)
    end
    for cand in candidates
        key = normalize_colname(cand)
        if haskey(norm_to_name, key)
            return norm_to_name[key]
        end
    end
    available = join(names(df), ", ")
    error("CSV missing required column for $(required_name). Available columns are: $(available)")
end

function maybe_find_column(df::DataFrame, candidates::Vector{String})
    norm_to_name = Dict{String, Symbol}()
    for nm in names(df)
        norm_to_name[normalize_colname(nm)] = Symbol(nm)
    end
    for cand in candidates
        key = normalize_colname(cand)
        if haskey(norm_to_name, key)
            return norm_to_name[key]
        end
    end
    return nothing
end

function standardize_head_columns!(df::DataFrame)
    well_col = find_column(df, ["well_name", "well", "well_id", "name", "location"], required_name = "well_name")
    x_col    = find_column(df, ["easting_m", "utm_x", "x", "x_m", "easting", "utm_easting"], required_name = "easting_m")
    y_col    = find_column(df, ["northing_m", "utm_y", "y", "y_m", "northing", "utm_northing"], required_name = "northing_m")
    h_col    = find_column(df, ["head_m", "freshwater_head_m", "head", "hydraulic_head_m", "water_level_m"], required_name = "head_m")
    meas_col = maybe_find_column(df, ["measurement_number", "measurement", "id", "index", "obs_id", "observation_id"])

    measurement_number = meas_col === nothing ? collect(1:nrow(df)) : Int.(df[!, meas_col])

    return DataFrame(
        measurement_number = measurement_number,
        well_name = String.(df[!, well_col]),
        easting_m = Float64.(df[!, x_col]),
        northing_m = Float64.(df[!, y_col]),
        head_m = Float64.(df[!, h_col])
    )
end

function load_heads(path::AbstractString)
    isfile(path) || error("Missing head observation CSV: $(path)")
    df = standardize_head_columns!(CSV.read(path, DataFrame))
    @info "Loaded head observations" path nrows = nrow(df)
    return df
end

function standardize_domain_columns!(df::DataFrame)
    x_col = find_column(df, ["easting_m", "utm_x", "x", "x_m", "easting", "utm_easting"], required_name = "easting_m")
    y_col = find_column(df, ["northing_m", "utm_y", "y", "y_m", "northing", "utm_northing"], required_name = "northing_m")
    return DataFrame(easting_m = Float64.(df[!, x_col]), northing_m = Float64.(df[!, y_col]))
end

function load_domain(path::AbstractString)
    isfile(path) || error("Missing domain CSV: $(path)")
    df = standardize_domain_columns!(CSV.read(path, DataFrame))
    x_min, x_max = minimum(df.easting_m), maximum(df.easting_m)
    y_min, y_max = minimum(df.northing_m), maximum(df.northing_m)
    @info "Loaded WIPP model domain" x_min x_max y_min y_max
    return x_min, x_max, y_min, y_max
end

# =============================================================================
# WIPP grid, boundary condition, and solver
# =============================================================================

function build_grid(x_min, x_max, y_min, y_max)
    coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(
        [x_min, y_min],
        [x_max, y_max],
        (NX, NY),
        THICKNESS
    )
    return coords, neighbors, areasoverlengths, volumes
end

function side_node_sets(coords; tol = 1.0e-6)
    x, y = coords[1, :], coords[2, :]
    x_min, x_max = minimum(x), maximum(x)
    y_min, y_max = minimum(y), maximum(y)

    west = Int[]
    east = Int[]
    south = Int[]
    north = Int[]
    for i in eachindex(x)
        abs(x[i] - x_min) <= tol && push!(west, i)
        abs(x[i] - x_max) <= tol && push!(east, i)
        abs(y[i] - y_min) <= tol && push!(south, i)
        abs(y[i] - y_max) <= tol && push!(north, i)
    end
    return (west = west, east = east, south = south, north = north)
end

function fixed_head_nodes_report_approx(coords)
    sides = side_node_sets(coords)
    return unique(sort(vcat(sides.north, sides.east, sides.south)))
end

function fixed_head_nodes_all_edges(coords)
    sides = side_node_sets(coords)
    return unique(sort(vcat(sides.north, sides.east, sides.south, sides.west)))
end

function build_fixed_head_nodes(coords)
    if BOUNDARY_MODE == :report_approx
        return fixed_head_nodes_report_approx(coords)
    elseif BOUNDARY_MODE == :all_edges
        return fixed_head_nodes_all_edges(coords)
    else
        error("Unknown BOUNDARY_MODE: $(BOUNDARY_MODE)")
    end
end

function fit_head_plane(heads::DataFrame)
    x = Float64.(heads.easting_m)
    y = Float64.(heads.northing_m)
    h = Float64.(heads.head_m)
    x0 = mean(x)
    y0 = mean(y)
    scale = 10000.0
    A = hcat(ones(length(x)), (x .- x0) ./ scale, (y .- y0) ./ scale)
    coeff = A \ h
    return (method = :plane, coeff = coeff, x0 = x0, y0 = y0, scale = scale)
end

function fit_boundary_head_surface(heads::DataFrame)
    if BOUNDARY_HEAD_METHOD == :plane
        return fit_head_plane(heads)
    else
        error("Only BOUNDARY_HEAD_METHOD=:plane is enabled in this training script. Use the fixed matching script for RBF experiments.")
    end
end

function predict_head_surface(model, x::Real, y::Real)
    xx = Float64(x)
    yy = Float64(y)
    if model.method == :plane
        return model.coeff[1] + model.coeff[2] * ((xx - model.x0) / model.scale) +
               model.coeff[3] * ((yy - model.y0) / model.scale)
    else
        error("Unknown surface method: $(model.method)")
    end
end

function boundary_heads_from_surface(coords, bnodes, surface_model, heads_df::DataFrame)
    h = zeros(Float64, size(coords, 2))
    hmin_allowed = minimum(Float64.(heads_df.head_m)) - BOUNDARY_HEAD_MARGIN_M
    hmax_allowed = maximum(Float64.(heads_df.head_m)) + BOUNDARY_HEAD_MARGIN_M
    for i in bnodes
        hi = predict_head_surface(surface_model, coords[1, i], coords[2, i])
        if CLAMP_BOUNDARY_HEADS
            hi = clamp(hi, hmin_allowed, hmax_allowed)
        end
        h[i] = hi
    end
    return h
end

function nearest_node(coords, x0::Real, y0::Real)
    x, y = coords[1, :], coords[2, :]
    d2 = (x .- Float64(x0)).^2 .+ (y .- Float64(y0)).^2
    return argmin(d2)
end

function observation_nodes(coords, heads::DataFrame)
    nodes = Int[]
    for r in eachrow(heads)
        push!(nodes, nearest_node(coords, r.easting_m, r.northing_m))
    end
    return nodes
end

function side_label_for_node(coords, i; tol = 1.0e-6)
    x = coords[1, i]
    y = coords[2, i]
    x_min, x_max = minimum(coords[1, :]), maximum(coords[1, :])
    y_min, y_max = minimum(coords[2, :]), maximum(coords[2, :])
    labels = String[]
    abs(x - x_min) <= tol && push!(labels, "west")
    abs(x - x_max) <= tol && push!(labels, "east")
    abs(y - y_min) <= tol && push!(labels, "south")
    abs(y - y_max) <= tol && push!(labels, "north")
    return isempty(labels) ? "interior" : join(labels, "+")
end

function write_boundary_condition_outputs(coords, dirichletnodes, dirichleths, surface_model, heads_df::DataFrame)
    sides = side_node_sets(coords)
    fixed_set = Set(dirichletnodes)
    rows = Any[]
    for side in [:west, :east, :south, :north]
        nodes = getproperty(sides, side)
        fixed_nodes = [i for i in nodes if i in fixed_set]
        bc_type = length(fixed_nodes) > 0 ? "fixed_head" : "no_flow_natural"
        min_h = length(fixed_nodes) > 0 ? minimum(dirichleths[fixed_nodes]) : NaN
        max_h = length(fixed_nodes) > 0 ? maximum(dirichleths[fixed_nodes]) : NaN
        mean_h = length(fixed_nodes) > 0 ? mean(dirichleths[fixed_nodes]) : NaN
        push!(rows, (side = string(side), bc_type = bc_type, n_side_nodes = length(nodes),
                    n_fixed_head_nodes = length(fixed_nodes), min_head_m = min_h,
                    mean_head_m = mean_h, max_head_m = max_h))
    end
    CSV.write("$(CHECKPOINT_PREFIX)_boundary_condition_side_summary.csv", DataFrame(rows))

    node_rows = DataFrame(
        node_index = dirichletnodes,
        side = [side_label_for_node(coords, i) for i in dirichletnodes],
        easting_m = coords[1, dirichletnodes],
        northing_m = coords[2, dirichletnodes],
        fixed_head_m = dirichleths[dirichletnodes]
    )
    CSV.write("$(CHECKPOINT_PREFIX)_fixed_head_boundary_nodes.csv", node_rows)

    coeff_df = DataFrame(
        parameter = ["method", "intercept", "x_scaled_slope", "y_scaled_slope", "x0_m", "y0_m", "scale_m",
                     "observed_head_min_m", "observed_head_max_m", "clamp_enabled", "boundary_head_margin_m"],
        value = string.([surface_model.method, surface_model.coeff[1], surface_model.coeff[2], surface_model.coeff[3],
                         surface_model.x0, surface_model.y0, surface_model.scale,
                         minimum(heads_df.head_m), maximum(heads_df.head_m), CLAMP_BOUNDARY_HEADS, BOUNDARY_HEAD_MARGIN_M])
    )
    CSV.write("$(CHECKPOINT_PREFIX)_boundary_head_surface_info.csv", coeff_df)
end

function check_boundary_head_range(dirichletnodes, dirichleths, heads_df::DataFrame)
    fixed_vals = Float64.(dirichleths[dirichletnodes])
    obs_vals = Float64.(heads_df.head_m)
    @info "Observed head range" min_observed = minimum(obs_vals) max_observed = maximum(obs_vals)
    @info "Fixed boundary head range" min_boundary = minimum(fixed_vals) max_boundary = maximum(fixed_vals) clamp_enabled = CLAMP_BOUNDARY_HEADS
    lower_warn = minimum(obs_vals) - 2.0 * BOUNDARY_HEAD_MARGIN_M
    upper_warn = maximum(obs_vals) + 2.0 * BOUNDARY_HEAD_MARGIN_M
    if minimum(fixed_vals) < lower_warn || maximum(fixed_vals) > upper_warn
        @warn "Boundary heads are far outside the observed-head range. Check boundary fitting and domain coordinates."
    end
end

mutable struct WIPPContext
    coords::Matrix{Float64}
    neighbors::Any
    areasoverlengths::Any
    dirichletnodes::Vector{Int}
    dirichleths::Vector{Float64}
    obs_nodes::Vector{Int}
    q::Vector{Float64}
    neighbor_left::Vector{Int}
    neighbor_right::Vector{Int}
end

function build_context(heads_df::DataFrame, x_min, x_max, y_min, y_max)
    coords, neighbors, areasoverlengths, volumes = build_grid(x_min, x_max, y_min, y_max)
    surface_model = fit_boundary_head_surface(heads_df)
    bnodes = build_fixed_head_nodes(coords)
    bheads = boundary_heads_from_surface(coords, bnodes, surface_model, heads_df)
    onodes = observation_nodes(coords, heads_df)

    q = zeros(Float64, size(coords, 2))
    if USE_SOURCE
        src_node = nearest_node(coords, SOURCE_X, SOURCE_Y)
        q[src_node] = SOURCE_RATE
    end

    neighbor_left = map(p -> p[1], neighbors)
    neighbor_right = map(p -> p[2], neighbors)

    write_boundary_condition_outputs(coords, bnodes, bheads, surface_model, heads_df)
    check_boundary_head_range(bnodes, bheads, heads_df)

    @info "Built WIPP solver context" ncell = size(coords, 2) nobs = length(onodes) nfixed = length(bnodes) boundary_mode = BOUNDARY_MODE
    return WIPPContext(coords, neighbors, areasoverlengths, bnodes, bheads, onodes, q, neighbor_left, neighbor_right)
end

function solve_head_obs_from_log10T_vec(log10T_vec::AbstractVector{<:Real}, ctx::WIPPContext)
    lt = Float64.(log10T_vec)
    T_neighbors = 10.0 .^ (0.5 .* (lt[ctx.neighbor_left] .+ lt[ctx.neighbor_right]))
    h = DPFEHM.groundwater_steadystate(
        T_neighbors,
        ctx.neighbors,
        ctx.areasoverlengths,
        ctx.dirichletnodes,
        ctx.dirichleths,
        ctx.q
    )
    return Float32.(h[ctx.obs_nodes])
end

# =============================================================================
# Image utilities and WIPP-grid mapping
# =============================================================================

function image_files(dir::AbstractString)
    exts = Set([".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"])
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

function load_gray_resize(path::AbstractString; h::Int = IMG_H, w::Int = IMG_W)
    img_any = load(path)
    img_gray = Gray.(img_any)
    img_r = imresize(img_gray, (h, w))
    img_f = Float32.(img_r)
    return reshape(img_f, h, w, 1)
end

function make_image_linear_index_map(coords, x_min, x_max, y_min, y_max; img_h::Int = IMG_H, img_w::Int = IMG_W)
    lin = Vector{Int}(undef, size(coords, 2))
    for i in 1:size(coords, 2)
        x = coords[1, i]
        y = coords[2, i]
        col = round(Int, 1 + (x - x_min) / (x_max - x_min) * (img_w - 1))
        row = round(Int, 1 + (y_max - y) / (y_max - y_min) * (img_h - 1))
        col = clamp(col, 1, img_w)
        row = clamp(row, 1, img_h)
        lin[i] = row + (col - 1) * img_h
    end
    return lin
end

function image_to_log10T_vector_from_img(img_hw1::AbstractArray{<:Real,3}, image_linear_idx::Vector{Int})
    img2 = Float32.(img_hw1[:, :, 1])
    g = vec(img2)[image_linear_idx]
    g = clamp.(g, 0.0f0, 1.0f0)
    return LOG10T_MIN .+ g .* (LOG10T_MAX - LOG10T_MIN)
end

# z_batch: (Z_DIM, B). Output: (NCELL, B) log10T.
function forward_log10T_from_latent(z_batch, image_linear_idx::Vector{Int})
    img4 = vae_decode_batch(vae, z_batch)                         # (IMG_H, IMG_W, 1, B)
    img_flat = reshape(img4[:, :, 1, :], IMG_H * IMG_W, size(z_batch, 2))
    g = img_flat[image_linear_idx, :]                              # (NCELL, B)
    g = clamp.(g, 0.0f0, 1.0f0)
    return LOG10T_MIN .+ g .* (LOG10T_MAX - LOG10T_MIN)
end

# =============================================================================
# Cache and precomputation
# =============================================================================

function cache_signature()
    return Dict{String, Any}(
        "LATENT_MODEL_PATH" => LATENT_MODEL_PATH,
        "DATA_DIR" => DATA_DIR,
        "HEAD_CSV" => HEAD_CSV,
        "DOMAIN_CSV" => DOMAIN_CSV,
        "NX" => NX,
        "NY" => NY,
        "THICKNESS" => THICKNESS,
        "IMG_H" => IMG_H,
        "IMG_W" => IMG_W,
        "LOG10T_MIN" => Float64(LOG10T_MIN),
        "LOG10T_MAX" => Float64(LOG10T_MAX),
        "BOUNDARY_MODE" => string(BOUNDARY_MODE),
        "BOUNDARY_HEAD_METHOD" => string(BOUNDARY_HEAD_METHOD),
        "CLAMP_BOUNDARY_HEADS" => CLAMP_BOUNDARY_HEADS,
        "BOUNDARY_HEAD_MARGIN_M" => BOUNDARY_HEAD_MARGIN_M,
        "USE_SOURCE" => USE_SOURCE,
        "SOURCE_X" => SOURCE_X,
        "SOURCE_Y" => SOURCE_Y,
        "SOURCE_RATE" => SOURCE_RATE
    )
end

function cache_matches(saved, current)
    if !(saved isa Dict)
        return false
    end
    for (k, v) in current
        if !haskey(saved, k) || saved[k] != v
            return false
        end
    end
    return true
end

function add_head_noise(y::AbstractArray{<:Real}; sigma_m::Float32 = HEAD_NOISE_STD_M)
    yy = Float32.(y)
    return yy .+ sigma_m .* randn(Float32, size(yy))
end

function write_training_head_variability(head_data::Array{Float32,2}, heads_df::DataFrame, path::AbstractString)
    mu = vec(mean(head_data; dims = 2))
    sd = vec(std(head_data; dims = 2))
    mn = vec(minimum(head_data; dims = 2))
    mx = vec(maximum(head_data; dims = 2))
    df = DataFrame(
        measurement_number = heads_df.measurement_number,
        well_name = heads_df.well_name,
        observed_head_m = heads_df.head_m,
        synthetic_head_mean_m = Float64.(mu),
        synthetic_head_std_m = Float64.(sd),
        synthetic_head_min_m = Float64.(mn),
        synthetic_head_max_m = Float64.(mx)
    )
    CSV.write(path, df)
    @info "Synthetic training-head variability" min_std = minimum(sd) median_std = median(sd) max_std = maximum(sd)
    if maximum(sd) < 0.1f0
        @warn "All synthetic heads vary very little across images. The inverse problem may be unlearnable from heads alone."
    end
end

function precompute_dataset(files::Vector{String}, train_idx, val_idx, image_linear_idx, ctx::WIPPContext)
    ncell = size(ctx.coords, 2)
    nobs = length(ctx.obs_nodes)

    train_logT = Array{Float32}(undef, ncell, length(train_idx))
    train_head = Array{Float32}(undef, nobs, length(train_idx))
    train_zmu  = Array{Float32}(undef, Z_DIM, length(train_idx))

    val_logT = Array{Float32}(undef, ncell, length(val_idx))
    val_head = Array{Float32}(undef, nobs, length(val_idx))
    val_zmu  = Array{Float32}(undef, Z_DIM, length(val_idx))

    train_files = files[train_idx]
    val_files = files[val_idx]

    @info "Precomputing training WIPP log10T, steady heads, and VAE latent means" N = length(train_idx)
    pre_rows = Any[]

    for (k, idx) in enumerate(train_idx)
        if k == 1 || k % 25 == 0 || k == length(train_idx)
            @printf("  train precompute %d / %d\n", k, length(train_idx))
            flush(stdout)
        end

        img3 = load_gray_resize(files[idx]; h = IMG_H, w = IMG_W)
        img4 = reshape(img3, IMG_H, IMG_W, 1, 1)

        mu = encode_mu_safe(vae, dev(Float32.(img4)))
        zmu = Array(Float32.(mu[:, 1]))

        logT = image_to_log10T_vector_from_img(img3, image_linear_idx)
        hobs = solve_head_obs_from_log10T_vec(logT, ctx)

        train_zmu[:, k] .= zmu
        train_logT[:, k] .= logT
        train_head[:, k] .= hobs

        push!(pre_rows, Any["train", k, files[idx], minimum(logT), mean(logT), maximum(logT), minimum(hobs), mean(hobs), maximum(hobs)])
    end

    @info "Precomputing validation WIPP log10T, steady heads, and VAE latent means" N = length(val_idx)
    for (k, idx) in enumerate(val_idx)
        if k == 1 || k % 25 == 0 || k == length(val_idx)
            @printf("  val precompute %d / %d\n", k, length(val_idx))
            flush(stdout)
        end

        img3 = load_gray_resize(files[idx]; h = IMG_H, w = IMG_W)
        img4 = reshape(img3, IMG_H, IMG_W, 1, 1)

        mu = encode_mu_safe(vae, dev(Float32.(img4)))
        zmu = Array(Float32.(mu[:, 1]))

        logT = image_to_log10T_vector_from_img(img3, image_linear_idx)
        hobs = solve_head_obs_from_log10T_vec(logT, ctx)

        val_zmu[:, k] .= zmu
        val_logT[:, k] .= logT
        val_head[:, k] .= hobs

        push!(pre_rows, Any["val", k, files[idx], minimum(logT), mean(logT), maximum(logT), minimum(hobs), mean(hobs), maximum(hobs)])
    end

    CSV.write(PRECOMP_SUMMARY_CSV,
        DataFrame(split = [r[1] for r in pre_rows], local_index = [r[2] for r in pre_rows], image = [r[3] for r in pre_rows],
                  min_log10T = [r[4] for r in pre_rows], mean_log10T = [r[5] for r in pre_rows], max_log10T = [r[6] for r in pre_rows],
                  min_head = [r[7] for r in pre_rows], mean_head = [r[8] for r in pre_rows], max_head = [r[9] for r in pre_rows]))

    return train_logT, train_head, train_zmu, val_logT, val_head, val_zmu, train_files, val_files
end

# =============================================================================
# Model, loss, and evaluation
# =============================================================================

function standardize_head_batch(h_batch, head_mean, head_std)
    return (h_batch .- head_mean) ./ head_std
end

mse(a, b) = mean((a .- b) .^ 2)

function scalar_float32(x)
    return x isa Number ? Float32(x) : Float32(Array(x)[1])
end

function make_inverse_net(nhead::Int)
    return Chain(
        Dense(nhead, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, Z_DIM)
    )
end

function loss_batch(head_obs_batch,
                    logT_true_batch,
                    z_true_batch,
                    model,
                    head_mean,
                    head_std,
                    image_linear_idx)

    x = standardize_head_batch(head_obs_batch, head_mean, head_std)
    z_pred = model(x)

    latent_loss = mse(z_pred, z_true_batch)
    prior_loss  = mean(sum(abs2, z_pred; dims = 1))

    if USE_PERM_LOSS
        logT_hat = forward_log10T_from_latent(z_pred, image_linear_idx)
        perm_loss = mse(logT_hat, logT_true_batch)
    else
        perm_loss = zero(latent_loss)
    end

    total = LAMBDA_PERM * perm_loss + LAMBDA_LATENT * latent_loss + LAMBDA_PRIOR * prior_loss
    return total, perm_loss, latent_loss, prior_loss
end

function eval_dataset_loss(head_data,
                           logT_data,
                           zmu_data,
                           model,
                           head_mean,
                           head_std,
                           image_linear_idx;
                           batchsize::Int = BATCHSIZE,
                           use_noise::Bool = false)
    N = size(head_data, 2)
    nb = cld(N, batchsize)

    total_sum = 0.0f0
    perm_sum = 0.0f0
    latent_sum = 0.0f0
    prior_sum = 0.0f0
    count = 0

    for b in 1:nb
        s = (b - 1) * batchsize + 1
        e = min(b * batchsize, N)
        idxs = s:e
        B = length(idxs)

        h_true = Float32.(head_data[:, idxs])
        h_obs = use_noise ? add_head_noise(h_true; sigma_m = HEAD_NOISE_STD_M) : h_true
        logT_true = Float32.(logT_data[:, idxs])
        z_true = Float32.(zmu_data[:, idxs])

        if USE_GPU && CUDA.functional()
            h_obs = cu(h_obs)
            logT_true = cu(logT_true)
            z_true = cu(z_true)
        end

        l, p, z, pr = loss_batch(h_obs, logT_true, z_true, model, head_mean, head_std, image_linear_idx)
        total_sum += scalar_float32(l) * B
        perm_sum += scalar_float32(p) * B
        latent_sum += scalar_float32(z) * B
        prior_sum += scalar_float32(pr) * B
        count += B
    end

    return (
        total = total_sum / max(count, 1),
        perm = perm_sum / max(count, 1),
        latent = latent_sum / max(count, 1),
        prior = prior_sum / max(count, 1)
    )
end

function pressure_diagnostic_rmse(head_data,
                                  model,
                                  head_mean_cpu,
                                  head_std_cpu,
                                  image_linear_idx,
                                  ctx::WIPPContext;
                                  nsamples::Int = N_PRESSURE_DIAGNOSTIC_SAMPLES)
    N = size(head_data, 2)
    M = min(nsamples, N)
    idxs = collect(1:M)
    rmses = Float64[]

    for idx in idxs
        h_true = Float32.(head_data[:, idx])
        h_in = reshape(h_true, :, 1)
        h_norm = standardize_head_batch(h_in, Float32.(head_mean_cpu), Float32.(head_std_cpu))
        z_pred = Array(model(Float32.(h_norm)))

        img4 = vae_decode_batch(vae, Float32.(z_pred))
        img_flat = reshape(img4[:, :, 1, :], IMG_H * IMG_W, 1)
        g = clamp.(Array(img_flat[image_linear_idx, 1]), 0.0f0, 1.0f0)
        logT_hat = LOG10T_MIN .+ g .* (LOG10T_MAX - LOG10T_MIN)
        h_hat = solve_head_obs_from_log10T_vec(logT_hat, ctx)

        push!(rmses, sqrt(mean((Float64.(h_hat) .- Float64.(h_true)) .^ 2)))
    end

    return mean(rmses)
end

# =============================================================================
# Plot helpers
# =============================================================================

function vector_to_grid(coords, values)
    xvals = sort(unique(coords[1, :]))
    yvals = sort(unique(coords[2, :]))
    nx = length(xvals)
    ny = length(yvals)
    grid = fill(NaN, ny, nx)
    x_to_j = Dict(v => j for (j, v) in enumerate(xvals))
    y_to_i = Dict(v => i for (i, v) in enumerate(yvals))
    for k in 1:size(coords, 2)
        j = x_to_j[coords[1, k]]
        i = y_to_i[coords[2, k]]
        grid[i, j] = values[k]
    end
    return grid
end

plot_grid_north_up(G) = Array(G)[end:-1:1, :]

function save_loss_plots(train_losses, val_losses, train_perm_losses, train_latent_losses, train_prior_losses,
                         val_perm_losses, val_latent_losses, val_prior_losses, pressure_diag_epochs, pressure_diag_rmse)
    fig = figure(figsize = (8.5, 5.0))
    plot(train_losses, label = "train total")
    plot(val_losses, label = "val total")
    yscale("log")
    xlabel("epoch")
    ylabel("loss")
    title("WIPP new-concept inverse training loss")
    legend()
    grid(true, alpha = 0.3)
    tight_layout()
    savefig("$(CHECKPOINT_PREFIX)_total_loss.png", dpi = 250)
    close(fig)

    fig = figure(figsize = (8.5, 5.0))
    plot(train_perm_losses, label = "train perm")
    plot(train_latent_losses, label = "train latent")
    if any(train_prior_losses .> 0f0)
        plot(train_prior_losses, label = "train prior")
    end
    plot(val_perm_losses, "--", label = "val perm")
    plot(val_latent_losses, "--", label = "val latent")
    if any(val_prior_losses .> 0f0)
        plot(val_prior_losses, "--", label = "val prior")
    end
    yscale("log")
    xlabel("epoch")
    ylabel("loss component")
    title("WIPP new-concept inverse training components")
    legend()
    grid(true, alpha = 0.3)
    tight_layout()
    savefig("$(CHECKPOINT_PREFIX)_components.png", dpi = 250)
    close(fig)

    if !isempty(pressure_diag_epochs)
        fig = figure(figsize = (8.5, 5.0))
        plot(pressure_diag_epochs, pressure_diag_rmse, "o-", label = "val head RMSE")
        xlabel("epoch")
        ylabel("head RMSE at WIPP wells (m)")
        title("Post-training pressure diagnostic")
        legend()
        grid(true, alpha = 0.3)
        tight_layout()
        savefig("$(CHECKPOINT_PREFIX)_pressure_diagnostic.png", dpi = 250)
        close(fig)
    end
end

function save_reconstruction_examples(path::String,
                                      head_data::Array{Float32,2},
                                      logT_data::Array{Float32,2},
                                      model,
                                      head_mean_cpu,
                                      head_std_cpu,
                                      image_linear_idx,
                                      ctx::WIPPContext;
                                      nsamples::Int = 3)
    M = min(nsamples, size(head_data, 2))
    fig = figure(figsize = (4.0 * M, 7.0))
    vmin = Float64(LOG10T_MIN)
    vmax = Float64(LOG10T_MAX)

    for k in 1:M
        h_true = reshape(Float32.(head_data[:, k]), :, 1)
        h_norm = standardize_head_batch(h_true, Float32.(head_mean_cpu), Float32.(head_std_cpu))
        z_pred = Array(model(Float32.(h_norm)))

        img4 = vae_decode_batch(vae, Float32.(z_pred))
        img_flat = reshape(img4[:, :, 1, :], IMG_H * IMG_W, 1)
        g = clamp.(Array(img_flat[image_linear_idx, 1]), 0.0f0, 1.0f0)
        logT_hat = LOG10T_MIN .+ g .* (LOG10T_MAX - LOG10T_MIN)

        true_grid = vector_to_grid(ctx.coords, logT_data[:, k])
        pred_grid = vector_to_grid(ctx.coords, logT_hat)

        subplot(2, M, k)
        imshow(plot_grid_north_up(true_grid), cmap = "gray", vmin = vmin, vmax = vmax)
        title("true log10T $(k)")
        axis("off")

        subplot(2, M, M + k)
        imshow(plot_grid_north_up(pred_grid), cmap = "gray", vmin = vmin, vmax = vmax)
        title("pred log10T $(k)")
        axis("off")
    end

    tight_layout()
    savefig(path, dpi = 220, bbox_inches = "tight")
    close(fig)
end

# =============================================================================
# Main
# =============================================================================

function main()
    heads_df = load_heads(HEAD_CSV)
    x_min, x_max, y_min, y_max = load_domain(DOMAIN_CSV)
    ctx = build_context(heads_df, x_min, x_max, y_min, y_max)
    image_linear_idx = make_image_linear_index_map(ctx.coords, x_min, x_max, y_min, y_max)

    files = image_files(DATA_DIR)
    @info "Found concept training images" DATA_DIR count = length(files)
    isempty(files) && error("No images found in $(DATA_DIR)")

    N_total = length(files)
    if N_train + N_val > N_total
        @warn "Requested N_train + N_val exceeds available images; adjusting." N_train N_val N_total
        global N_train = max(1, min(N_train, N_total - 1))
        global N_val = N_total - N_train
    end

    current_signature = cache_signature()
    loaded_cache = false

    if USE_DATA_CACHE && !FORCE_REBUILD_CACHE && isfile(DATA_CACHE_PATH)
        d = BSON.load(DATA_CACHE_PATH)
        if haskey(d, :cache_info) && cache_matches(d[:cache_info], current_signature)
            @info "Loading valid precomputed WIPP training cache" DATA_CACHE_PATH
            train_logT = d[:train_logT]
            train_head = d[:train_head]
            train_zmu = d[:train_zmu]
            val_logT = d[:val_logT]
            val_head = d[:val_head]
            val_zmu = d[:val_zmu]
            train_files = d[:train_files]
            val_files = d[:val_files]
            loaded_cache = true
        else
            @warn "Existing cache does not match current settings; rebuilding cache." DATA_CACHE_PATH
        end
    end

    if !loaded_cache
        perm_idx = randperm(N_total)
        train_idx = perm_idx[1:N_train]
        val_idx = perm_idx[N_train + 1:N_train + N_val]

        train_logT, train_head, train_zmu, val_logT, val_head, val_zmu, train_files, val_files =
            precompute_dataset(files, train_idx, val_idx, image_linear_idx, ctx)

        if USE_DATA_CACHE
            cache_info = current_signature
            @info "Saving precomputed WIPP training cache" DATA_CACHE_PATH
            BSON.@save DATA_CACHE_PATH cache_info train_logT train_head train_zmu val_logT val_head val_zmu train_files val_files
        end
    end

    write_training_head_variability(train_head, heads_df, "$(CHECKPOINT_PREFIX)_training_head_variability.csv")

    nhead = size(train_head, 1)
    @info "Training inverse model" nhead ntrain = size(train_head, 2) nval = size(val_head, 2) ncell = size(train_logT, 1)

    # Normalize head inputs per observation location. Use a floor so wells with
    # weak synthetic variation do not create huge standardized noise.
    head_mean_cpu = Float32.(mean(train_head; dims = 2))
    head_std_cpu = Float32.(std(train_head; dims = 2))
    head_std_floor = max(HEAD_NOISE_STD_M, 0.10f0)
    head_std_cpu .= max.(head_std_cpu, head_std_floor)

    CSV.write("$(CHECKPOINT_PREFIX)_head_normalization.csv",
              DataFrame(well_name = heads_df.well_name,
                        head_mean = vec(Float64.(head_mean_cpu)),
                        head_std_used = vec(Float64.(head_std_cpu))))

    head_mean = copy(head_mean_cpu)
    head_std = copy(head_std_cpu)

    inverse_net = make_inverse_net(nhead)
    if USE_GPU && CUDA.functional()
        inverse_net = Flux.fmap(cu, inverse_net)
        head_mean = cu(head_mean)
        head_std = cu(head_std)
    end

    opt = Optimisers.Adam(LR)
    state = Optimisers.setup(opt, inverse_net)

    train_losses = Float32[]
    val_losses = Float32[]
    train_perm_losses = Float32[]
    train_latent_losses = Float32[]
    train_prior_losses = Float32[]
    val_perm_losses = Float32[]
    val_latent_losses = Float32[]
    val_prior_losses = Float32[]
    pressure_diag_epochs = Int[]
    pressure_diag_rmse = Float64[]

    best_val = Inf32
    best_epoch = 0

    @info "Starting data-driven inverse training for WIPP new conceptual model" USE_PERM_LOSS BATCHSIZE N_EPOCHS LAMBDA_PERM LAMBDA_LATENT LAMBDA_PRIOR HEAD_NOISE_STD_M

    for epoch in 1:N_EPOCHS
        if SAMPLES_PER_EPOCH > 0
            M = min(SAMPLES_PER_EPOCH, size(train_head, 2))
            epoch_indices = randperm(size(train_head, 2))[1:M]
        else
            M = size(train_head, 2)
            epoch_indices = randperm(size(train_head, 2))
        end

        nb = cld(M, BATCHSIZE)
        epoch_loss_sum = 0.0f0
        epoch_count = 0

        for b in 1:nb
            s = (b - 1) * BATCHSIZE + 1
            e = min(b * BATCHSIZE, M)
            idxs = epoch_indices[s:e]
            B = length(idxs)

            h_true = Float32.(train_head[:, idxs])
            h_obs = USE_NOISE_TRAIN ? add_head_noise(h_true; sigma_m = HEAD_NOISE_STD_M) : h_true
            logT_true = Float32.(train_logT[:, idxs])
            z_true = Float32.(train_zmu[:, idxs])

            if USE_GPU && CUDA.functional()
                h_obs = cu(h_obs)
                logT_true = cu(logT_true)
                z_true = cu(z_true)
            end

            L, back = Flux.withgradient(inverse_net) do m
                l, _, _, _ = loss_batch(h_obs, logT_true, z_true, m, head_mean, head_std, image_linear_idx)
                return l
            end

            Optimisers.update!(state, inverse_net, back[1])
            epoch_loss_sum += scalar_float32(L) * B
            epoch_count += B
        end

        train_total_fast = epoch_loss_sum / max(epoch_count, 1)

        # Component losses are evaluated once per epoch on a subset.
        train_eval_idxs = epoch_indices[1:min(length(epoch_indices), max(BATCHSIZE, 64))]
        trainm = eval_dataset_loss(train_head[:, train_eval_idxs], train_logT[:, train_eval_idxs], train_zmu[:, train_eval_idxs],
                                   inverse_net, head_mean, head_std, image_linear_idx;
                                   batchsize = BATCHSIZE, use_noise = USE_NOISE_TRAIN)

        valm = eval_dataset_loss(val_head, val_logT, val_zmu,
                                 inverse_net, head_mean, head_std, image_linear_idx;
                                 batchsize = BATCHSIZE, use_noise = USE_NOISE_VAL)

        push!(train_losses, Float32(train_total_fast))
        push!(val_losses, Float32(valm.total))
        push!(train_perm_losses, Float32(trainm.perm))
        push!(train_latent_losses, Float32(trainm.latent))
        push!(train_prior_losses, Float32(trainm.prior))
        push!(val_perm_losses, Float32(valm.perm))
        push!(val_latent_losses, Float32(valm.latent))
        push!(val_prior_losses, Float32(valm.prior))

        diag_msg = ""
        if epoch == 1 || epoch % DIAGNOSTIC_PRESSURE_EVERY == 0 || epoch == N_EPOCHS
            prmse = pressure_diagnostic_rmse(val_head, inverse_net, head_mean_cpu, head_std_cpu, image_linear_idx, ctx;
                                             nsamples = N_PRESSURE_DIAGNOSTIC_SAMPLES)
            push!(pressure_diag_epochs, epoch)
            push!(pressure_diag_rmse, prmse)
            diag_msg = @sprintf(" | val_head_RMSE=%.4f m", prmse)
        end

        @printf("epoch %4d | train=%.6e | val=%.6e (perm=%.3e latent=%.3e prior=%.3e)%s\n",
                epoch, train_total_fast, valm.total, valm.perm, valm.latent, valm.prior, diag_msg)
        flush(stdout)

        if Float32(valm.total) < best_val
            best_val = Float32(valm.total)
            best_epoch = epoch
            BSON.@save BEST_MODEL_PATH inverse_net epoch best_val best_epoch head_mean_cpu head_std_cpu current_signature LATENT_MODEL_PATH DATA_DIR HEAD_CSV DOMAIN_CSV NX NY THICKNESS LOG10T_MIN LOG10T_MAX BOUNDARY_MODE BOUNDARY_HEAD_METHOD CLAMP_BOUNDARY_HEADS BOUNDARY_HEAD_MARGIN_M train_total_fast valm
            @info "Saved best inverse model" BEST_MODEL_PATH best_epoch best_val
        end

        if epoch % SAVE_EXAMPLE_RECON_EVERY == 0 || epoch == 1
            save_reconstruction_examples("$(CHECKPOINT_PREFIX)_reconstruction_examples_epoch$(epoch).png",
                                         val_head, val_logT, inverse_net, head_mean_cpu, head_std_cpu, image_linear_idx, ctx;
                                         nsamples = 3)
        end

        if epoch % CHECKPOINT_EVERY == 0
            ckpt_path = "$(CHECKPOINT_PREFIX)_checkpoint_epoch$(epoch).bson"
            BSON.@save ckpt_path inverse_net epoch train_losses val_losses train_perm_losses train_latent_losses train_prior_losses val_perm_losses val_latent_losses val_prior_losses pressure_diag_epochs pressure_diag_rmse head_mean_cpu head_std_cpu current_signature
        end
    end

    BSON.@save FINAL_MODEL_PATH inverse_net best_val best_epoch head_mean_cpu head_std_cpu current_signature LATENT_MODEL_PATH DATA_DIR HEAD_CSV DOMAIN_CSV NX NY THICKNESS LOG10T_MIN LOG10T_MAX BOUNDARY_MODE BOUNDARY_HEAD_METHOD CLAMP_BOUNDARY_HEADS BOUNDARY_HEAD_MARGIN_M

    BSON.@save FINAL_LOSS_PATH train_losses val_losses train_perm_losses train_latent_losses train_prior_losses val_perm_losses val_latent_losses val_prior_losses pressure_diag_epochs pressure_diag_rmse best_val best_epoch current_signature

    save_loss_plots(train_losses, val_losses, train_perm_losses, train_latent_losses, train_prior_losses,
                    val_perm_losses, val_latent_losses, val_prior_losses, pressure_diag_epochs, pressure_diag_rmse)

    @info "Done. Saved final model/losses." FINAL_MODEL_PATH FINAL_LOSS_PATH BEST_MODEL_PATH
end

main()
