# compare_wipp_head_misfit.jl
#
# Purpose
# -------
# Compare WIPP Culebra old vs new conceptual-model inverse models using the
# same observed steady-state hydraulic heads from the WIPP head CSV. The true
# pressure/head vector is always the observed head vector listed in the CSV.
#
# Workflow
# --------
# 1. Load WIPP Culebra observed freshwater heads and model domain.
# 2. Build the WIPP-scale steady-state flow model using the corrected WIPP-style boundary setup:
#    fixed head on north/east/south and natural no-flow on west.
# 3. For each conceptual model:
#       observed heads -> inverse_net -> VAE latent z -> decoded image
#       -> WIPP log10 transmissivity field -> DPFEHM steady-state heads.
# 4. Compare simulated heads at the 35 observation wells against the observed
#    heads and compute pressure/head RMSE.
# 5. Rank conceptual models by RMS observation-noise-normalized head error.
# 6. Save fig_head_rmse.png/.pdf (normalized errors and aggregate ranking),
#    and fig_head_rmse_m.png/.pdf (raw head RMSE in meters).
#
# Important
# ---------
# - This is a real-data comparison. There is no synthetic ground-truth
#   permeability field, so permeability relative L2 is not computed.
# - The main figure and ranking report use normalized head error only.
# - Legacy likelihood/probability CSV columns are retained for compatibility.
# - The inverse models should have been trained with the WIPP training script
#   that saves head_mean_cpu and head_std_cpu in the BSON file.
#
# Required files in run directory:
#   wipp_culebra_2000_steady_heads.csv
#   wipp_culebra_model_domain.csv
#   VAE BSON files for old and new conceptual models
#   inverse_net BSON files for old and new conceptual models
#
# Run:
#   julia compare_wipp_head_misfit.jl
using Pkg
for pkg in [
    "CSV", "DataFrames", "Flux", "BSON", "Random", "Statistics",
    "LinearAlgebra", "Printf", "DPFEHM", "CUDA", "Functors", "Images",
    "FileIO", "ImageIO", "ImageTransformations", "PyPlot"
]
    Base.find_package(pkg) === nothing && Pkg.add(pkg)
end
ENV["MPLBACKEND"] = "Agg"
using CSV
using DataFrames
using Flux
using BSON
using Random
using Statistics
using LinearAlgebra
using Printf
using DPFEHM
using CUDA
using Functors: fmap
using Images, FileIO, ImageIO, ImageTransformations
using PyPlot

# =============================================================================
# User configuration

# =============================================================================
const USE_GPU = false
# WIPP observation and domain files.
const HEAD_CSV   = "wipp_culebra_2000_steady_heads_v2.csv"
const DOMAIN_CSV = "wipp_culebra_model_domain.csv"
const OUT_DIR  = "wipp2_old_new_concept_comparison_l"
const PLOT_DIR = joinpath(OUT_DIR, "plots")
# WIPP grid and physical setup. These match the WIPP steady-state matching script.
const NX = 224
const NY = 307
const THICKNESS = 1.0
# Generated image / VAE output -> WIPP log10 transmissivity mapping.
const IMG_H, IMG_W, IMG_C = 256, 256, 1
const LOG10T_MIN = -8.0f0
const LOG10T_MAX = -3.0f0
# Optional source/sink term. Keep false for the same setup as the WIPP
# steady-state head-matching script.
const USE_SOURCE = false
const SOURCE_X = 613700.0
const SOURCE_Y = 3581000.0
const SOURCE_RATE = 0.0
# Boundary-condition setup matching the fixed WIPP report-style scripts.
# This is still an approximation to the official MODFLOW setup, but it avoids
# the earlier all-edge fixed-head simplification.
const BOUNDARY_MODE = :report_approx
const BOUNDARY_HEAD_METHOD = :plane
const CLAMP_BOUNDARY_HEADS = true
const BOUNDARY_HEAD_MARGIN_M = 5.0
# Observation-noise scale for hydraulic-head RMSE normalization.
# For heads, an absolute uncertainty in meters is usually more meaningful than
# a relative error on a 900-m head datum.
const HEAD_OBS_STD_M = 0.5
# Legacy likelihood diagnostic only; it does not enter the normalized-error rank.
const N_HEAD_EFFECTIVE = 10          # use 35 for strict independent-well likelihood
# Optional Monte Carlo perturbations of the observed input vector. The reference
# head vector used for scoring remains the unperturbed observed CSV vector for every
# realization. Set to 1 for a single deterministic comparison.
const N_EVAL_REALIZATIONS = 100
const USE_NOISY_INPUT_REALIZATIONS = true
const INPUT_NOISE_STD_M = 0.5
const NOISE_SEED = 1256876
# Clean output settings.
const MAX_RUG_POINTS = 300
# Conceptual models. Edit filenames to match your trained model names.
const INTERPRETATION_CONFIGS = Any[
    (
        name = "Original conceptual model",
        vae_path = "vae_best_wiip_old_v2_l.bson",
        vae_path_alt = "vae_best_old_concept.bson",
        inverse_path = "inv_net_wipp2_old_concept_dd_l_best.bson",
        inverse_path_alt = "inv_net_wipp_old_concept_report_bc_fixed_final_model.bson"
    ),
    (
        name = "Revised conceptual model",
        vae_path = "vae_best_wiip_new_v2_l.bson",
        vae_path_alt = "vae_best_new_concept.bson",
        inverse_path = "inv_net_wipp2_new_concept_dd_l_best.bson",
        inverse_path_alt = "inv_net_wipp_new_concept_report_bc_fixed_final_model.bson"
    )
]

# =============================================================================
# VAE definition and world-age-safe calls

# =============================================================================

mutable struct VAE
    encoder::Any
    fc_mu::Any
    fc_logvar::Any
    fc_dec::Any
    decoder::Any
end
Flux.@functor VAE
# BSON files that contain anonymous functions inside Flux.Chain layers can
# trigger Julia world-age errors. Use invokelatest for inference.
call_model(model, x) = Base.invokelatest(model, x)
dev(x) = (USE_GPU && CUDA.functional()) ? cu(x) : x

function decode(m::VAE, z)
    h = call_model(m.fc_dec, z)
    h = reshape(h, 16, 16, 256, size(z, 2))
    return call_model(m.decoder, h)
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
    isfile(path) || error("Missing model domain CSV: $(path)")
    df = standardize_domain_columns!(CSV.read(path, DataFrame))
    x_min = minimum(df.easting_m)
    x_max = maximum(df.easting_m)
    y_min = minimum(df.northing_m)
    y_max = maximum(df.northing_m)
    @info "Loaded WIPP model domain" x_min x_max y_min y_max
    return x_min, x_max, y_min, y_max
end

# =============================================================================
# WIPP grid, boundary conditions, and solver

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
    x = coords[1, :]
    y = coords[2, :]
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
    # Approximate the WIPP report-style boundary condition used in the fixed
    # head-matching and training scripts:
    #   north/east/south = fixed head
    #   west             = natural no-flow
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
    # Centering/scaling avoids poorly conditioned UTM coordinates and matches the
    # fixed report-BC training scripts.
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
        error("Unknown BOUNDARY_HEAD_METHOD: $(BOUNDARY_HEAD_METHOD). This comparison script uses :plane to match the fixed training scripts.")
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

function write_boundary_diagnostics(out_dir::AbstractString, coords, bnodes, bheads, heads_df::DataFrame, surface_model)
    sides = side_node_sets(coords)
    fixed_set = Set(bnodes)
    rows = Any[]
    for side in [:west, :east, :south, :north]
        nodes = getproperty(sides, side)
        fixed_nodes = [i for i in nodes if i in fixed_set]
        if isempty(fixed_nodes)
            push!(rows, Any[string(side), length(nodes), 0, NaN, NaN, NaN])
        else
            vals = Float64.(bheads[fixed_nodes])
            push!(rows, Any[string(side), length(nodes), length(fixed_nodes), minimum(vals), mean(vals), maximum(vals)])
        end
    end
    # write_csv(joinpath(out_dir, "boundary_condition_side_summary.csv"),
    #           ["side", "side_node_count", "fixed_head_node_count", "min_fixed_head_m", "mean_fixed_head_m", "max_fixed_head_m"],
    #           rows)
    fixed_rows = Any[]
    for i in bnodes
        push!(fixed_rows, Any[i, coords[1, i], coords[2, i], bheads[i]])
    end
    # write_csv(joinpath(out_dir, "fixed_head_boundary_nodes.csv"),
    #           ["node", "easting_m", "northing_m", "fixed_head_m"], fixed_rows)
    CSV.write(joinpath(out_dir, "boundary_head_surface_info.csv"),
              DataFrame(parameter = ["method", "x0_m", "y0_m", "scale_m", "coeff_intercept", "coeff_x_centered_scaled", "coeff_y_centered_scaled", "clamp_enabled", "clamp_margin_m", "observed_min_head_m", "observed_max_head_m"],
                        value = [string(surface_model.method), string(surface_model.x0), string(surface_model.y0), string(surface_model.scale), string(surface_model.coeff[1]), string(surface_model.coeff[2]), string(surface_model.coeff[3]), string(CLAMP_BOUNDARY_HEADS), string(BOUNDARY_HEAD_MARGIN_M), string(minimum(heads_df.head_m)), string(maximum(heads_df.head_m))]))
end

function nearest_node(coords, x0::Real, y0::Real)
    x = coords[1, :]
    y = coords[2, :]
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
    surface_model::Any
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
    fixed_vals = bheads[bnodes]
    @info "Built WIPP flow context" ncell = size(coords, 2) nobs = length(onodes) nfixed = length(bnodes) boundary_mode = BOUNDARY_MODE min_fixed_head = minimum(fixed_vals) max_fixed_head = maximum(fixed_vals)
    return WIPPContext(coords, neighbors, areasoverlengths, bnodes, bheads, onodes, q, neighbor_left, neighbor_right, surface_model)
end

function solve_head_full_from_log10T_vec(log10T_vec::AbstractVector{<:Real}, ctx::WIPPContext)
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
    return Float64.(h)
end

function solve_head_obs_from_log10T_vec(log10T_vec::AbstractVector{<:Real}, ctx::WIPPContext)
    h = solve_head_full_from_log10T_vec(log10T_vec, ctx)
    return Float64.(h[ctx.obs_nodes]), h
end

# =============================================================================
# Image/grid mapping

# =============================================================================

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

function decoded_image_to_log10T_vector(img3::AbstractArray{<:Real,3}, image_linear_idx::Vector{Int})
    img2 = Float32.(img3[:, :, 1])
    g = vec(img2)[image_linear_idx]
    g = clamp.(g, 0f0, 1f0)
    return Float64.(LOG10T_MIN .+ g .* (LOG10T_MAX - LOG10T_MIN))
end

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

function plot_matrix_from_grid(A)
    return Array(A)[end:-1:1, :]
end

# =============================================================================
# Interpretation loading and inference

# =============================================================================

abstract type AbstractInterpretation end

mutable struct VAEInterpretation <: AbstractInterpretation
    name::String
    vae_path::String
    inverse_path::String
    vae::Any
    inverse_net::Any
    head_mean::Matrix{Float32}
    head_std::Matrix{Float32}
end
interpretation_name(m::AbstractInterpretation) = m.name

function resolve_file(path::AbstractString, alt_path::AbstractString = "")
    if isfile(path)
        return path
    elseif !isempty(alt_path) && isfile(alt_path)
        @warn "Primary file not found; using alternate file" path alt_path
        return alt_path
    else
        if isempty(alt_path)
            error("Required file not found: $(path). Edit INTERPRETATION_CONFIGS.")
        else
            error("Required file not found: $(path) or alternate $(alt_path). Edit INTERPRETATION_CONFIGS.")
        end
    end
end

function load_stat_matrix(d::Dict, nobs::Int, keys_to_try::Vector{Symbol}, default_value::Float32, stat_name::String)
    for k in keys_to_try
        if haskey(d, k)
            arr = Float32.(Array(d[k]))
            if length(arr) != nobs
                error("Loaded $(stat_name) from key $(k), but length=$(length(arr)) does not match nobs=$(nobs).")
            end
            return reshape(arr, nobs, 1)
        end
    end
    @warn "Inverse BSON does not contain $(stat_name); using fallback. This is only correct if the inverse net was trained without standardization." stat_name default_value
    return fill(default_value, nobs, 1)
end

function load_inverse_net_and_stats(path::AbstractString, nobs::Int; alt_path::AbstractString = "")
    path2 = resolve_file(path, alt_path)
    d = BSON.load(path2)
    haskey(d, :inverse_net) || error("BSON file $(path2) does not contain variable :inverse_net")
    net = d[:inverse_net]
    head_mean = load_stat_matrix(d, nobs, [:head_mean_cpu, :head_mean, :input_mean, :x_mean], 0.0f0, "head_mean")
    head_std  = load_stat_matrix(d, nobs, [:head_std_cpu, :head_std, :input_std, :x_std], 1.0f0, "head_std")
    head_std .= max.(head_std, 1.0f-6)
    if USE_GPU && CUDA.functional()
        net = fmap(cu, net)
    end
    # Check input/output dimensions with a dummy pass.
    dummy = zeros(Float32, nobs, 1)
    dummy_norm = (dummy .- head_mean) ./ head_std
    try
        out = Array(call_model(net, dev(dummy_norm)))
        @info "Loaded inverse net" path2 nobs zdim = size(out, 1)
    catch err
        error("Inverse net in $(path2) could not be called with input size nobs=$(nobs). This usually means the model was not trained with the WIPP 35-head setup. Original error: $(err)")
    end
    return net, head_mean, head_std, path2
end

function load_vae(path::AbstractString; alt_path::AbstractString = "")
    path2 = resolve_file(path, alt_path)
    d = BSON.load(path2)
    haskey(d, :model) || error("BSON file $(path2) does not contain variable :model")
    vae = d[:model]
    if USE_GPU && CUDA.functional()
        vae = fmap(cu, vae)
    end
    return vae, path2
end

function load_interpretations(nobs::Int)
    methods = AbstractInterpretation[]
    for cfg in INTERPRETATION_CONFIGS
        vae_alt = (:vae_path_alt in keys(cfg)) ? string(cfg.vae_path_alt) : ""
        inv_alt = (:inverse_path_alt in keys(cfg)) ? string(cfg.inverse_path_alt) : ""
        vae, vae_path2 = load_vae(string(cfg.vae_path); alt_path = vae_alt)
        inv, hmean, hstd, inv_path2 = load_inverse_net_and_stats(string(cfg.inverse_path), nobs; alt_path = inv_alt)
        push!(methods, VAEInterpretation(string(cfg.name), vae_path2, inv_path2, vae, inv, hmean, hstd))
        @info "Loaded interpretation" name = string(cfg.name) vae_path = vae_path2 inverse_path = inv_path2
    end
    return methods
end

function infer_log10T_and_image(m::VAEInterpretation, head_input::AbstractVector{<:Real}, image_linear_idx::Vector{Int})
    h = reshape(Float32.(head_input), :, 1)
    h_norm = (h .- m.head_mean) ./ m.head_std
    z_pred = Float32.(Array(call_model(m.inverse_net, dev(h_norm))))
    img4 = decode(m.vae, dev(z_pred))             # (IMG_H, IMG_W, 1, 1)
    img3 = Array(img4)[:, :, :, 1]                # (IMG_H, IMG_W, 1)
    log10T_vec = decoded_image_to_log10T_vector(img3, image_linear_idx)
    return log10T_vec, Float32.(img3), z_pred
end

# =============================================================================
# Metrics and probabilities

# =============================================================================
mse(a, b) = mean((Float64.(a) .- Float64.(b)) .^ 2)
rmse(a, b) = sqrt(mse(a, b))
mae(a, b) = mean(abs.(Float64.(a) .- Float64.(b)))
bias(sim, obs) = mean(Float64.(sim) .- Float64.(obs))

function head_sigma_eff()
    # Retain the function name for compatibility. The scale is observation noise
    # alone, consistent with Section 2.6; no model-error floor is included.
    sigma = Float64(HEAD_OBS_STD_M)
    isfinite(sigma) && sigma > 0 || error("HEAD_OBS_STD_M must be finite and positive.")
    return sigma
end

function head_loglike(head_rmse::Real, sigma_eff::Real)
    pnorm = Float64(head_rmse) / max(Float64(sigma_eff), eps(Float64))
    return -0.5 * Float64(N_HEAD_EFFECTIVE) * pnorm^2
end

function stable_softmax(logw::AbstractVector{<:Real})
    m = maximum(logw)
    w = exp.(Float64.(logw) .- m)
    s = sum(w)
    return w ./ s
end

function normalized_entropy(p::AbstractVector{<:Real})
    pp = Float64.(p)
    K = length(pp)
    if K <= 1
        return 0.0
    end
    H = 0.0
    for x in pp
        if x > 0.0
            H -= x * log(x)
        end
    end
    return H / log(K)
end
quantile_safe(x::AbstractVector{<:Real}, q::Real) = quantile(Float64.(x), q)

function add_head_noise(y::AbstractVector{<:Real}; sigma_m::Float64 = INPUT_NOISE_STD_M)
    return Float32.(y) .+ Float32(sigma_m) .* randn(Float32, length(y))
end

# =============================================================================
# CSV helpers

# =============================================================================

function csv_field(x)
    if x isa AbstractString
        s = replace(String(x), "\"" => "\"\"")
        return "\"" * s * "\""
    elseif x isa AbstractFloat
        return isfinite(x) ? @sprintf("%.10e", Float64(x)) : string(x)
    else
        return string(x)
    end
end


# function write_csv(path::AbstractString, header::Vector{String}, rows)
#     open(path, "w") do io
#         println(io, join(csv_field.(header), ","))
#         for row in rows
#             println(io, join(csv_field.(row), ","))
#         end
#     end
# end

# =============================================================================
# Plot functions

# =============================================================================

function save_wipp_png_pdf(fig, path::AbstractString)
    stem, extension = splitext(path)
    if !(lowercase(extension) in (".png", ".pdf"))
        stem = path
    end
    fig.savefig(stem * ".png", dpi = 600,
                bbox_inches = "tight", pad_inches = 0.05)
    fig.savefig(stem * ".pdf", bbox_inches = "tight", pad_inches = 0.05)
    return nothing
end

function style_wipp_misfit_axes!(ax; grid_axis::String = "both")
    ax.tick_params(axis = "both", labelsize = 11, direction = "in",
                   top = true, right = true, length = 5, width = 1.0)
    for side in ["left", "right", "top", "bottom"]
        ax.spines[side].set_linewidth(1.0)
    end
    ax.set_axisbelow(true)
    ax.grid(true, axis = grid_axis, alpha = 0.22, linewidth = 0.7)
end

function draw_wipp_misfit_distribution!(ax, method_names, values;
                                        xlabel_text::String,
                                        nbins::Int = 30)
    N, K = size(values)
    N > 0 || error("At least one evaluation realization is required.")
    K == length(method_names) || error("Names and misfit columns do not match.")
    all(isfinite, values) || error("Head misfits must be finite for plotting.")
    all(v -> v >= 0, values) || error("Head RMSE values must be nonnegative.")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    lo, hi = extrema(Float64.(values))
    span = hi - lo
    pad = span > 0 ? 0.06 * span : 0.05 * max(abs(lo), 1.0)
    plot_lo = max(0.0, lo - pad)
    plot_hi = hi + pad
    bins = collect(range(plot_lo, plot_hi; length = nbins + 1))

    for j in 1:K
        color = colors[mod1(j, length(colors))]
        vals = Float64.(values[:, j])
        if N > 1
            ax.hist(vals, bins = bins, density = true,
                    histtype = "stepfilled", alpha = 0.28,
                    color = color, edgecolor = color, linewidth = 1.2,
                    label = method_names[j])
            # An opaque outline keeps both overlapping distributions legible.
            ax.hist(vals, bins = bins, density = true,
                    histtype = "step", color = color, linewidth = 1.6)
        else
            ax.axvline(vals[1], color = color, linewidth = 2.0,
                       label = method_names[j])
        end
    end
    ax.set_xlim(plot_lo, plot_hi)
    ax.set_ylim(bottom = 0.0)
    ax.set_xlabel(xlabel_text, fontsize = 13)
    ax.set_ylabel(N > 1 ? "Density across evaluation realizations" : "", fontsize = 12)
    if N == 1
        ax.set_yticks([])
    end
    style_wipp_misfit_axes!(ax)
    ax.legend(loc = "best", fontsize = 10, frameon = true, framealpha = 0.95)
    return nothing
end

function save_wipp_head_misfit(path::AbstractString, method_names,
                               normalized_errors::AbstractMatrix{<:Real})
    # Each row is one inverse-input realization scored against the SAME observed
    # well heads. These rows are not additional independent field measurements.
    N, K = size(normalized_errors)
    fig = figure(figsize = (12.4, 4.9))
    ax1 = fig.add_subplot(1, 2, 1)
    draw_wipp_misfit_distribution!(ax1, method_names, normalized_errors;
                                  xlabel_text = "Normalized hydraulic-head RMSE")
    ax1.set_title("(a)", loc = "left", fontsize = 14, fontweight = "bold")

    # This is R_k from Section 2.6, not the mean of the normalized RMSE values.
    scores = [sqrt(mean(Float64.(normalized_errors[:, j]) .^ 2)) for j in 1:K]
    order = sortperm(scores)
    ranks = zeros(Int, K)
    for (rank, j) in enumerate(order)
        ranks[j] = rank
    end
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    ax2 = fig.add_subplot(1, 2, 2)
    score_scale = max(maximum(scores), 1.0e-12)
    for j in 1:K
        ax2.bar([j], [scores[j]], width = 0.58,
                color = colors[mod1(j, length(colors))], alpha = 0.85)
        ax2.text(j, scores[j] + 0.035 * score_scale,
                 @sprintf("Rank %d\nR = %.3f", ranks[j], scores[j]),
                 ha = "center", va = "bottom", fontsize = 11)
    end
    ax2.set_xticks(collect(1:K))
    ax2.set_xticklabels([replace(name, " conceptual model" => "\nconceptual model")
                         for name in method_names], fontsize = 11)
    ax2.set_ylabel(raw"Aggregate normalized head error, $R_k$", fontsize = 13)
    ax2.set_ylim(0.0, 1.24 * score_scale)
    ax2.set_title("(b)", loc = "left", fontsize = 14, fontweight = "bold")
    style_wipp_misfit_axes!(ax2; grid_axis = "y")
    fig.tight_layout(pad = 0.9, w_pad = 2.0)
    save_wipp_png_pdf(fig, path)
    close(fig)
    return nothing
end

function save_wipp_raw_head_misfit(path::AbstractString, method_names, rmse_values)
    fig = figure(figsize = (7.2, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    draw_wipp_misfit_distribution!(ax, method_names, rmse_values;
                                  xlabel_text = "Hydraulic-head RMSE (m)")
    fig.tight_layout(pad = 0.9)
    save_wipp_png_pdf(fig, path)
    close(fig)
    return nothing
end

# function save_head_scatter_all(path, obs, sim_by_method, method_names, rmse_vals)
#     fig = figure(figsize = (5.8, 5.2))
#     obsf = Float64.(obs)
#     allvals = copy(obsf)
#     for s in sim_by_method
#         append!(allvals, Float64.(s))
#     end
#     mn = minimum(allvals)
#     mx = maximum(allvals)
#     pad = 0.03 * max(mx - mn, 1.0)
#     plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", linewidth = 1.4, label = "1:1")
#     for j in 1:length(method_names)
#         plot(obsf, Float64.(sim_by_method[j]), "o", markersize = 5, alpha = 0.75,
#              label = @sprintf("%s, RMSE=%.3f m", method_names[j], rmse_vals[j]))
#     end
#     xlabel("Observed freshwater head (m)")
#     ylabel("Simulated freshwater head (m)")
#     title("Observed vs simulated WIPP Culebra heads")
#     grid(true, alpha = 0.3)
#     legend(fontsize = 8)
#     axis("equal")
#     tight_layout()
#     savefig(path, dpi = 300, bbox_inches = "tight")
#     close(fig)
# end

# function save_head_profile_all(path, heads_df, obs, sim_by_method, method_names)
#     order = sortperm(Float64.(obs))
#     x = collect(1:length(obs))
#     fig = figure(figsize = (10.5, 4.8))
#     plot(x, Float64.(obs)[order], "k-", linewidth = 2.6, label = "Observed heads")
#     for j in 1:length(method_names)
#         plot(x, Float64.(sim_by_method[j])[order], linewidth = 1.8, label = method_names[j])
#     end
#     xticks(x, String.(heads_df.well_name[order]), rotation = 60, ha = "right", fontsize = 7)
#     ylabel("Freshwater head (m)")
#     title("Head profile sorted by observed head")
#     grid(true, alpha = 0.3)
#     legend(fontsize = 8)
#     tight_layout()
#     savefig(path, dpi = 300, bbox_inches = "tight")
#     close(fig)
# end

# function save_residuals_grouped(path, heads_df, residual_mat, method_names)
#     N, K = size(residual_mat)
#     order = sortperm(maximum(abs.(residual_mat), dims = 2)[:, 1], rev = true)
#     names_sorted = String.(heads_df.well_name[order])
#     vals = residual_mat[order, :]
#     fig = figure(figsize = (max(10.0, 0.35 * N), 5.0))
#     x = collect(1:N)
#     width = min(0.8 / K, 0.35)
#     center_offset = (K + 1) / 2
#     for j in 1:K
#         bar(x .+ (j - center_offset) * width, vals[:, j], width, label = method_names[j])
#     end
#     axhline(0.0, color = "k", linewidth = 1.0)
#     xticks(x, names_sorted, rotation = 60, ha = "right", fontsize = 7)
#     ylabel("Residual, simulated - observed (m)")
#     title("Head residuals by conceptual model")
#     grid(true, axis = "y", alpha = 0.3)
#     legend(fontsize = 8)
#     tight_layout()
#     savefig(path, dpi = 300, bbox_inches = "tight")
#     close(fig)
# end

# function save_log10T_maps(path, ctx::WIPPContext, logT_by_method, method_names, coords)
#     K = length(method_names)
#     fig = figure(figsize = (4.1 * K, 6.2))
#     vmin = minimum(vcat([Float64.(lt) for lt in logT_by_method]...))
#     vmax = maximum(vcat([Float64.(lt) for lt in logT_by_method]...))
#     # Important: do not use the variable name `im` outside the loop without
#     # declaring it first. In Julia, `im` is also the imaginary unit. If a loop-
#     # local `im` is not visible outside the loop, colorbar(im) will receive the
#     # imaginary unit instead of the Matplotlib image object and will throw:
#     # AttributeError: 'complex' object has no attribute 'get_array'.
#     mappable = nothing
#     for j in 1:K
#         subplot(1, K, j)
#         gridv = vector_to_grid(coords, logT_by_method[j])
#         mappable = imshow(plot_matrix_from_grid(gridv), cmap = "gray", vmin = vmin, vmax = vmax)
#         title(method_names[j], fontsize = 10)
#         axis("off")
#     end
#     cb = colorbar(mappable, ax = gcf().axes, fraction = 0.035, pad = 0.02)
#     cb.set_label("Inferred log10 transmissivity")
#     suptitle("Inferred transmissivity fields from observed heads")
#     savefig(path, dpi = 300, bbox_inches = "tight")
#     close(fig)
# end

# function save_head_maps(path, ctx::WIPPContext, head_full_by_method, heads_df, method_names, coords)
#     K = length(method_names)
#     fig = figure(figsize = (4.1 * K, 6.2))
#     vmin = minimum(vcat([Float64.(hh) for hh in head_full_by_method]...))
#     vmax = maximum(vcat([Float64.(hh) for hh in head_full_by_method]...))
#     mappable = nothing
#     for j in 1:K
#         subplot(1, K, j)
#         gridv = vector_to_grid(coords, head_full_by_method[j])
#         mappable = imshow(plot_matrix_from_grid(gridv), cmap = "viridis", vmin = vmin, vmax = vmax)
#         title(method_names[j], fontsize = 10)
#         axis("off")
#     end
#     cb = colorbar(mappable, ax = gcf().axes, fraction = 0.035, pad = 0.02)
#     cb.set_label("Simulated head (m)")
#     suptitle("Simulated steady-state head fields")
#     savefig(path, dpi = 300, bbox_inches = "tight")
#     close(fig)
# end

# function save_metric_bar(path, method_names, values; ylabel_text, title_text, lower_is_better = true)
#     order = lower_is_better ? sortperm(values) : sortperm(values, rev = true)
#     names_sorted = method_names[order]
#     vals_sorted = values[order]
#     fig = figure(figsize = (max(6.0, 1.8 * length(method_names)), 4.5))
#     x = collect(1:length(method_names))
#     bar(x, vals_sorted)
#     xticks(x, names_sorted, rotation = 25, ha = "right")
#     ylabel(ylabel_text)
#     title(title_text)
#     ymax = maximum(vals_sorted)
#     ylim(0.0, ymax <= 0.0 ? 1.0 : 1.20 * ymax)
#     for i in 1:length(x)
#         text(x[i], vals_sorted[i] + 0.03 * max(ymax, 1.0e-12), @sprintf("%.3g", vals_sorted[i]),
#              ha = "center", va = "bottom", fontsize = 9)
#     end
#     grid(true, axis = "y", alpha = 0.3)
#     tight_layout()
#     savefig(path, dpi = 300)
#     close(fig)
# end

# function save_probability_bar(path, method_names, probs)
#     order = sortperm(probs, rev = true)
#     names_sorted = method_names[order]
#     probs_sorted = probs[order]
#     fig = figure(figsize = (max(6.0, 1.8 * length(method_names)), 4.5))
#     x = collect(1:length(method_names))
#     bar(x, probs_sorted)
#     xticks(x, names_sorted, rotation = 25, ha = "right")
#     ylabel("Relative probability from observed-head error")
#     title("Old vs new conceptual-model ranking")
#     ylim(0.0, min(1.05, maximum(probs_sorted) + 0.15))
#     for i in 1:length(x)
#         text(x[i], probs_sorted[i] + 0.02, @sprintf("%.3f", probs_sorted[i]), ha = "center", va = "bottom", fontsize = 9)
#     end
#     grid(true, axis = "y", alpha = 0.3)
#     tight_layout()
#     savefig(path, dpi = 300)
#     close(fig)
# end

# function save_histogram_overlay(
#     path,
#     method_names,
#     values;
#     xlabel_text,
#     title_text,
#     nbins = 100,
#     xlimits = nothing,
# )
#     _, K = size(values)
#     fig = figure(figsize = (7.2, 4.8))
#     allvals = vec(values)
#     # Determine histogram range
#     xmin = minimum(allvals)
#     xmax = maximum(allvals)
#     if xmax <= xmin
#         xmax = xmin + 1.0
#     end
#     bins = collect(range(xmin, xmax; length = nbins + 1))
#     for j in 1:K
#         hist(
#             values[:, j];
#             bins = bins,
#             density = true,
#             histtype = "step", 
#             linewidth = 2.0,
#             label = method_names[j],
#         )
#     end
#     # Set custom x-axis limits when provided
#     if xlimits !== nothing
#         xlim(xlimits[1], xlimits[2])
#     else
#         xlim(xmin, xmax)
#     end
#     xlabel(xlabel_text)
#     ylabel("Density")
#     title(title_text)
#     grid(true; alpha = 0.3)
#     legend()
#     tight_layout()
#     savefig(path; dpi = 300, bbox_inches = "tight")
#     close(fig)
# end

# =============================================================================
# Main evaluation

# =============================================================================

function main()
    mkpath(OUT_DIR)
    mkpath(PLOT_DIR)
    if USE_GPU && CUDA.functional()
        @info "Using GPU"
    else
        @info "Using CPU"
    end
    heads_df = load_heads(HEAD_CSV)
    x_min, x_max, y_min, y_max = load_domain(DOMAIN_CSV)
    ctx = build_context(heads_df, x_min, x_max, y_min, y_max)
    image_linear_idx = make_image_linear_index_map(ctx.coords, x_min, x_max, y_min, y_max)
    h_ref = Float32.(heads_df.head_m)
    nobs = length(h_ref)
    sigma_eff = head_sigma_eff()
    @info "Observed-head comparison setup" nobs sigma_eff N_HEAD_EFFECTIVE N_EVAL_REALIZATIONS
    methods = load_interpretations(nobs)
    K = length(methods)
    method_names = [interpretation_name(m) for m in methods]
    priors = fill(1.0 / K, K)
    log_priors = log.(priors)
    rmse_head = zeros(Float64, N_EVAL_REALIZATIONS, K)
    mae_head = zeros(Float64, N_EVAL_REALIZATIONS, K)
    bias_head = zeros(Float64, N_EVAL_REALIZATIONS, K)
    corr_head = zeros(Float64, N_EVAL_REALIZATIONS, K)
    normalized_head_error = zeros(Float64, N_EVAL_REALIZATIONS, K)
    loglike = zeros(Float64, N_EVAL_REALIZATIONS, K)
    probs = zeros(Float64, N_EVAL_REALIZATIONS, K)
    # Save first-realization fields for plotting and output.
    sim_first = Vector{Vector{Float64}}(undef, K)
    residual_first = zeros(Float64, nobs, K)
    logT_first = Vector{Vector{Float64}}(undef, K)
    head_full_first = Vector{Vector{Float64}}(undef, K)
    img_first = Vector{Array{Float32,3}}(undef, K)
    Random.seed!(NOISE_SEED)
    for r in 1:N_EVAL_REALIZATIONS
        if r == 1 || !USE_NOISY_INPUT_REALIZATIONS
            h_input = copy(h_ref)
        else
            h_input = add_head_noise(h_ref; sigma_m = INPUT_NOISE_STD_M)
        end
        for (j, method) in enumerate(methods)
            logT_vec, img3, z_pred = infer_log10T_and_image(method, h_input, image_linear_idx)
            h_sim_obs, h_full = solve_head_obs_from_log10T_vec(logT_vec, ctx)
            rmse_head[r, j] = rmse(h_ref, h_sim_obs)
            mae_head[r, j] = mae(h_ref, h_sim_obs)
            bias_head[r, j] = bias(h_sim_obs, h_ref)
            corr_head[r, j] = nobs > 1 ? cor(Float64.(h_ref), Float64.(h_sim_obs)) : NaN
            normalized_head_error[r, j] = rmse_head[r, j] / sigma_eff
            loglike[r, j] = head_loglike(rmse_head[r, j], sigma_eff)
            if r == 1
                sim_first[j] = Float64.(h_sim_obs)
                residual_first[:, j] .= Float64.(h_sim_obs) .- Float64.(h_ref)
                logT_first[j] = Float64.(logT_vec)
                head_full_first[j] = Float64.(h_full)
                img_first[j] = img3
            end
        end
        probs[r, :] .= stable_softmax(vec(loglike[r, :] .+ log_priors))
    end
    if N_EVAL_REALIZATIONS == 1
        agg_loglike = vec(loglike[1, :])
    else
        agg_loglike = vec(mean(loglike; dims = 1))
    end
    agg_probs = stable_softmax(agg_loglike .+ log_priors)
    mean_rmse = [mean(rmse_head[:, j]) for j in 1:K]
    median_rmse = [median(rmse_head[:, j]) for j in 1:K]
    mean_norm = [mean(normalized_head_error[:, j]) for j in 1:K]
    rms_norm = [sqrt(mean(normalized_head_error[:, j] .^ 2)) for j in 1:K]
    rank_order = sortperm(rms_norm)
    rank_for_method = zeros(Int, K)
    for (rank, idx) in enumerate(rank_order)
        rank_for_method[idx] = rank
    end
    # -------------------------------------------------------------------------
    # CSV outputs
    # -------------------------------------------------------------------------
    # Predicted heads for first realization.
    pred_rows = Any[]
    for j in 1:K
        for i in 1:nobs
            push!(pred_rows, Any[
                method_names[j],
                heads_df.measurement_number[i],
                heads_df.well_name[i],
                heads_df.easting_m[i],
                heads_df.northing_m[i],
                Float64(h_ref[i]),
                sim_first[j][i],
                sim_first[j][i] - Float64(h_ref[i]),
                abs(sim_first[j][i] - Float64(h_ref[i]))
            ])
        end
    end
    # write_csv(joinpath(OUT_DIR, "predicted_heads_by_interpretation.csv"),
    #           ["interpretation", "measurement_number", "well_name", "easting_m", "northing_m",
    #            "observed_head_m", "simulated_head_m", "residual_sim_minus_obs_m", "abs_residual_m"],
    #           pred_rows)
    # Per-realization method results.
    per_rows = Any[]
    for r in 1:N_EVAL_REALIZATIONS
        for j in 1:K
            push!(per_rows, Any[
                r,
                method_names[j],
                rmse_head[r, j],
                mae_head[r, j],
                bias_head[r, j],
                corr_head[r, j],
                normalized_head_error[r, j],
                loglike[r, j],
                probs[r, j],
                sigma_eff,
                N_HEAD_EFFECTIVE
            ])
        end
    end
    # write_csv(joinpath(OUT_DIR, "per_realization_method_results.csv"),
    #           ["realization", "interpretation", "head_rmse_m", "head_mae_m", "head_bias_m",
    #            "head_correlation", "normalized_head_error", "head_log_likelihood",
    #            "relative_probability", "sigma_eff_m", "n_head_effective"],
    #           per_rows)
    # Summary by interpretation.
    summary_rows = Any[]
    for j in 1:K
        push!(summary_rows, Any[
            method_names[j],
            rank_for_method[j],
            mean_rmse[j],
            N_EVAL_REALIZATIONS > 1 ? std(rmse_head[:, j]) : 0.0,
            median_rmse[j],
            quantile_safe(rmse_head[:, j], 0.05),
            quantile_safe(rmse_head[:, j], 0.95),
            mean(mae_head[:, j]),
            mean(bias_head[:, j]),
            mean(corr_head[:, j]),
            mean_norm[j],
            rms_norm[j],
            mean(probs[:, j]),
            median(probs[:, j]),
            agg_loglike[j],
            agg_probs[j],
            minimum(logT_first[j]),
            median(logT_first[j]),
            maximum(logT_first[j])
        ])
    end
    # write_csv(joinpath(OUT_DIR, "summary_by_interpretation.csv"),
    #           ["interpretation", "rank", "mean_head_rmse_m", "std_head_rmse_m", "median_head_rmse_m",
    #            "q05_head_rmse_m", "q95_head_rmse_m", "mean_head_mae_m", "mean_head_bias_m",
    #            "mean_head_correlation", "mean_normalized_head_error", "rms_normalized_head_error",
    #            "mean_probability", "median_probability", "aggregate_head_log_likelihood",
    #            "aggregate_probability", "min_inferred_log10T", "median_inferred_log10T", "max_inferred_log10T"],
    #           summary_rows)
    aggregate_rows = Any[]
    for (rnk, idx) in enumerate(rank_order)
        push!(aggregate_rows, Any[
            rnk,
            method_names[idx],
            agg_probs[idx],
            agg_loglike[idx],
            mean_rmse[idx],
            median_rmse[idx],
            mean_norm[idx],
            rms_norm[idx]
        ])
    end
    # write_csv(joinpath(OUT_DIR, "aggregate_ranking.csv"),
    #           ["rank", "interpretation", "aggregate_probability", "aggregate_head_log_likelihood",
    #            "mean_head_rmse_m", "median_head_rmse_m", "mean_normalized_head_error", "rms_normalized_head_error"],
    #           aggregate_rows)
    # Boundary diagnostics for reproducibility.
    write_boundary_diagnostics(OUT_DIR, ctx.coords, ctx.dirichletnodes, ctx.dirichleths, heads_df, ctx.surface_model)
    # Save inferred log10T vectors for first realization.
    for j in 1:K
        CSV.write(joinpath(OUT_DIR, "inferred_log10T_" * replace(method_names[j], r"[^A-Za-z0-9_-]+" => "_") * ".csv"),
                  DataFrame(node = collect(1:length(logT_first[j])),
                            easting_m = ctx.coords[1, :],
                            northing_m = ctx.coords[2, :],
                            log10T = logT_first[j]))
    end
    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    # save_head_scatter_all(joinpath(PLOT_DIR, "observed_vs_simulated_heads.png"),
    #                       h_ref, sim_first, method_names, rmse_head[1, :])
    # save_head_profile_all(joinpath(PLOT_DIR, "head_profile_comparison.png"),
    #                       heads_df, h_ref, sim_first, method_names)
    # save_residuals_grouped(joinpath(PLOT_DIR, "head_residuals_by_interpretation.png"),
    #                        heads_df, residual_first, method_names)
    # save_log10T_maps(joinpath(PLOT_DIR, "inferred_log10T_maps.png"),
    #                  ctx, logT_first, method_names, ctx.coords)
    # save_head_maps(joinpath(PLOT_DIR, "simulated_head_maps.png"),
    #                ctx, head_full_first, heads_df, method_names, ctx.coords)
    # save_metric_bar(joinpath(PLOT_DIR, "head_rmse_bar.png"),
    #                 method_names, mean_rmse;
    #                 ylabel_text = "Head RMSE (m)",
    #                 title_text = "Observed-head RMSE by conceptual model",
    #                 lower_is_better = true)
    # save_metric_bar(joinpath(PLOT_DIR, "normalized_head_error_bar.png"),
    #                 method_names, rms_norm;
    #                 ylabel_text = "Aggregate RMS normalized head error",
    #                 title_text = "Conceptual-model ranking by normalized head error",
    #                 lower_is_better = true)
    # Main manuscript figure: misfit distribution and the aggregate ranking.
    save_wipp_head_misfit(joinpath(PLOT_DIR, "fig_head_rmse.png"),
                          method_names, normalized_head_error)
    # Companion figure in physical head units.
    save_wipp_raw_head_misfit(joinpath(PLOT_DIR, "fig_head_rmse_m.png"),
                              method_names, rmse_head)
    # if N_EVAL_REALIZATIONS > 1
        # save_histogram_overlay(joinpath(PLOT_DIR, "head_rmse_histogram_overlay.png"),
        #                        method_names, rmse_head;
        #                        xlabel_text = "Head RMSE (m)",
        #                        title_text = "Head RMSE distribution from noisy observed-head inputs")
        # save_histogram_overlay(joinpath(PLOT_DIR, "normalized_head_error_histogram_overlay.png"),
        #                        method_names, normalized_head_error;
        #                        xlabel_text = "Normalized head error",
        #                        title_text = "Normalized head-error distribution")
    # end
    # -------------------------------------------------------------------------
    # Text report
    # -------------------------------------------------------------------------
    top_idx = rank_order[1]
    open(joinpath(OUT_DIR, "ranking.txt"), "w") do io
        println(io, "WIPP conceptual-model ranking by aggregate normalized head error")
        println(io, "===========================================================")
        println(io, "")
        @printf(io, "Observed head CSV: %s\n", HEAD_CSV)
        @printf(io, "Number of observed heads: %d\n", nobs)
        @printf(io, "Grid: NX=%d, NY=%d, cells=%d\n", NX, NY, size(ctx.coords, 2))
        @printf(io, "Boundary condition: fixed head on north/east/south from centered, clamped head plane; west boundary natural no-flow\n")
        @printf(io, "LOG10T range: %.3f to %.3f\n", Float64(LOG10T_MIN), Float64(LOG10T_MAX))
        @printf(io, "HEAD_OBS_STD_M: %.3f\n", HEAD_OBS_STD_M)
        @printf(io, "Normalization uses observation noise alone: %.3f m\n", sigma_eff)
        @printf(io, "N_EVAL_REALIZATIONS: %d\n", N_EVAL_REALIZATIONS)
        println(io, "")
        println(io, "Aggregate ranking, sorted by lowest RMS normalized head error:")
        for (rnk, idx) in enumerate(rank_order)
            @printf(io, "%2d. %-26s R_k = %.6f  mean_RMSE = %.6f m  mean_normalized_error = %.6f\n",
                    rnk, method_names[idx], rms_norm[idx], mean_rmse[idx], mean_norm[idx])
        end
        println(io, "")
        @printf(io, "Top-ranked interpretation: %s\n", method_names[top_idx])
        @printf(io, "Smallest aggregate normalized error R_k: %.6f\n", rms_norm[top_idx])
        println(io, "")
        println(io, "Each realization is scored against the same observed well heads.")
        println(io, "Realization 1 uses the unperturbed input; later inputs are perturbed only when enabled.")
        println(io, "The realizations assess sensitivity to input perturbations, not independent field datasets.")
    end
    println("\nAggregate ranking based on observed-head error:")
    for (rnk, idx) in enumerate(rank_order)
        @printf("%2d. %-26s R_k = %.6f | mean head RMSE = %.4f m\n",
                rnk, method_names[idx], rms_norm[idx], mean_rmse[idx])
    end
    @printf("\nTop interpretation: %s\n", method_names[top_idx])
    @printf("Aggregate normalized error of top interpretation: %.6f\n", rms_norm[top_idx])
    @info "Saved WIPP original/revised comparison outputs, including head-misfit PNG and PDF" OUT_DIR PLOT_DIR
end
main()
