# compare_interpretations_pressure_only_clean.jl
#
# Purpose
# -------
# Compare competing geologic interpretations using pressure error only for
# ranking and probability calculations. Permeability relative L2 error is still
# computed and plotted as a diagnostic benchmark metric, but it does not affect
# the ranking, log-likelihoods, or probabilities.
#
# Ranking/probability metric
# --------------------------
# For each image and interpretation k, the script computes
#
#   RMSE_p,k = sqrt(mean((p_ref - p_hat_k).^2))
#
# where p_ref is either noisy observed pressure (:obs) or clean synthetic
# pressure (:true). The pressure RMSE is converted to a Gaussian log-likelihood:
#
#   log L_k = -0.5 * Nobs * (RMSE_p,k / sigma_eff)^2
#
# and relative probabilities are computed across the interpretation set:
#
#   P(C_k | d) = exp(log L_k) / sum_j exp(log L_j)
#
# Aggregate probabilities are computed from either the mean log-likelihood
# (:mean_loglike, recommended for many correlated crops) or summed log-likelihood
# (:sum_loglike, strict independent-image assumption).
#
# Main outputs are written to OUT_DIR:
#   - per_image_method_results.csv
#   - summary_by_interpretation.csv
#   - aggregate_ranking.csv
#   - ranking.txt
#   - plots/*.png
#
# Clean plot set:
#   - pressure_rmse_histogram_overlay.png
#   - pressure_rmse_boxplot.png
#   - normalized_pressure_error_histogram_overlay.png
#   - normalized_pressure_error_boxplot.png
#   - permeability_relative_l2_histogram_overlay.png
#   - permeability_relative_l2_boxplot.png
#   - pressure_aggregate_probabilities.png
#   - pressure_probability_distribution_curves.png
#   - pressure_probability_exceedance_curves.png
#   - pressure_rmse_to_likelihood_curve.png
#   - random_sample_###_pressure_comparison.png
#   - random_sample_###_permeability_comparison.png

using Pkg
for pkg in ["Flux", "BSON", "Random", "Statistics", "Printf", "DPFEHM", "PyPlot",
            "CUDA", "Functors", "Images", "FileIO", "ImageIO", "ImageTransformations",
            "GaussianRandomFields"]
    Base.find_package(pkg) === nothing && Pkg.add(pkg)
end

ENV["MPLBACKEND"] = "Agg"

using Flux, BSON, Random, Statistics, Printf, DPFEHM, PyPlot, CUDA
using LinearAlgebra
using Functors: fmap
using Images, FileIO, ImageIO, ImageTransformations
using GaussianRandomFields

# =============================================================================
# User configuration
# =============================================================================

const USE_GPU = false

# Test data and output location.
const TEST_IMG_DIR = "./cropped_sleipner_full"
const OUT_DIR      = "./comparison_pressure_only_perm_diagnostics"
const PLOT_DIR     = joinpath(OUT_DIR, "plots")

# Number of test images and observation noise.
# Set N_TEST_MAX to a large value if you want to use all available images.
const N_TEST_MAX = 925
const REL_NOISE  = 0.01f0
const NOISE_SEED = 1256879

# Probability reference:
#   :obs  = rank interpretations by misfit to noisy historical observations.
#   :true = rank interpretations by misfit to clean synthetic pressure.
# In real-data applications, use :obs.
const PROBABILITY_REFERENCE = :obs

# Optional pressure model-error floor used in likelihood/probability conversion.
# This prevents unrealistically overconfident probabilities when inversion,
# surrogate, or model-form error is larger than measurement noise.
const PROB_MODEL_ERROR_REL = 0.02f0

# Aggregation for the final probability across many test images.
# :mean_loglike avoids extreme 0/1 probabilities when many crops are correlated.
# :sum_loglike is a strict independent-image joint likelihood.
const AGGREGATION_MODE = :mean_loglike   # :mean_loglike or :sum_loglike

# If you know the true/correct interpretation name in a synthetic test, put it
# here exactly as named below. Leave empty for field data.
const TRUE_INTERPRETATION_NAME = ""    # e.g., "Plausible VAE"

# Clean output settings for large test sets.
const PROGRESS_EVERY = 25
const MAX_RUG_POINTS = 300

# Random sample diagnostics. These do not affect ranking. They save pressure
# and permeability comparison plots for a few randomly chosen test images.
const SAVE_RANDOM_SAMPLE_COMPARISONS = true
const N_RANDOM_COMPARISON_SAMPLES = 3
const RANDOM_COMPARISON_SEED = 20240511

# Image resolution and logK mapping. These match your test scripts.
const IMG_H, IMG_W, IMG_C = 256, 256, 1
const LOGK_MIN = -4.0f0
const LOGK_MAX =  2.0f0

# GRF-KL settings. These match test_inv_gaussian.jl if you uncomment the GRF option.
const GRF_SIGMA_COV  = 1.0f0
const GRF_LAMBDA_COV = 28.9f0

# Competing interpretations. Edit names or paths if needed.
const INTERPRETATION_CONFIGS = Any[
    (kind = :vae, name = "Precise & Accurate",  vae_path = "vae_best_plausible.bson", vae_path_alt = "", inverse_path = "inv_net_data_plausible_best_plausible.bson"),
    (kind = :vae, name = "Accurate",       vae_path = "vae_best_weak.bson",      vae_path_alt = "", inverse_path = "inv_net_data_weak_best_weak.bson"),
    (kind = :vae, name = "Mismatched", vae_path = "vae_best_mismatch.bson",  vae_path_alt = "vae_best_mismatched.bson", inverse_path = "inv_net_data_weak_best_weak.bson"),
    # (kind = :grf, name = "Gaussian GRF-KL", inverse_path = "inverse_net_grf.bson")
]

# =============================================================================
# Device helper and BSON world-age-safe model call
# =============================================================================

dev(x) = (USE_GPU && CUDA.functional()) ? cu(x) : x

# BSON files that contain anonymous functions inside Flux.Chain layers can trigger
# Julia world-age errors when called immediately after deserialization.
call_model(model, x) = Base.invokelatest(model, x)

# =============================================================================
# VAE type definition. This must match the type used when saving the BSON files.
# =============================================================================

mutable struct VAE
    encoder::Any
    fc_mu::Any
    fc_logvar::Any
    fc_dec::Any
    decoder::Any
end
Flux.@functor VAE

function encode(m::VAE, x)
    h = call_model(m.encoder, x)
    mu = call_model(m.fc_mu, h)
    logvar = call_model(m.fc_logvar, h)
    return mu, logvar
end

function decode(m::VAE, z)
    h = call_model(m.fc_dec, z)
    h = reshape(h, 16, 16, 256, size(z, 2))
    return call_model(m.decoder, h)
end

# =============================================================================
# DPFEHM setup. This matches your uploaded test scripts.
# =============================================================================

const num_mon = 250
const n       = 128
const ns      = (n, n)
const sidelength = 100.0
const thickness  = 1.0
const injrate    = 0.0
const injection_node = 26

coords, neighbors, areasoverlengths, volumes =
    DPFEHM.regulargrid2d([-sidelength, -sidelength],
                         [ sidelength,  sidelength],
                         ns, thickness)

steadyhead = 0.0
dirichletnodes = Int[]
dirichleths = zeros(Float32, size(coords, 2))

for i in 1:size(coords, 2)
    if coords[1, i] == sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = Float32(steadyhead)
    end
end

for i in 1:size(coords, 2)
    if coords[1, i] == -sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = 10.0f0
    end
end

ymax = maximum(coords[2, :])
ymin = minimum(coords[2, :])
for i in 1:size(coords, 2)
    if coords[2, i] == ymax
        push!(dirichletnodes, i)
        dirichleths[i] = 0.5f0
    elseif coords[2, i] == ymin
        push!(dirichletnodes, i)
        dirichleths[i] = 0.0f0
    end
end

Random.seed!(1256879)
monitoring_nodes = sort!(randperm(n * n)[1:num_mon])

const NCELL = size(coords, 2)
const Q_VEC = let q = zeros(Float32, NCELL)
    q[injection_node] = Float32(injrate)
    q
end

const neighbor_left  = map(p -> p[1], neighbors)
const neighbor_right = map(p -> p[2], neighbors)

logKs2Ks_neighbors(logK_grid) =
    exp.(0.5f0 .* (logK_grid[neighbor_left] .+ logK_grid[neighbor_right]))

function solve_pressure_full_from_logK(logK_grid::AbstractMatrix{<:Real})
    logK32 = Float32.(logK_grid)
    Ks_neighbors = logKs2Ks_neighbors(logK32)
    P_full = DPFEHM.groundwater_steadystate(
        Ks_neighbors, neighbors, areasoverlengths,
        dirichletnodes, dirichleths, Q_VEC
    )
    return Float32.(P_full)
end

function solve_bhp_from_logK(logK_grid::AbstractMatrix{<:Real})
    P_full = solve_pressure_full_from_logK(logK_grid)
    return Float32.(P_full[monitoring_nodes])
end

# =============================================================================
# Interpretation model containers
# =============================================================================

abstract type AbstractInterpretation end

mutable struct VAEInterpretation <: AbstractInterpretation
    name::String
    vae_path::String
    inverse_path::String
    vae::Any
    inverse_net::Any
end

mutable struct GRFInterpretation <: AbstractInterpretation
    name::String
    inverse_path::String
    inverse_net::Any
    num_eig::Int
    phi_sigma::Matrix{Float32}
    lambda_cov::Float32
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
            error("Required file not found: $(path). Run this script from the directory containing your BSON model files, or edit INTERPRETATION_CONFIGS.")
        else
            error("Required file not found: $(path) or alternate $(alt_path). Run this script from the directory containing your BSON model files, or edit INTERPRETATION_CONFIGS.")
        end
    end
end

function load_inverse_net(path::AbstractString)
    path2 = resolve_file(path)
    d = BSON.load(path2)
    haskey(d, :inverse_net) || error("BSON file $(path2) does not contain variable :inverse_net")
    net = d[:inverse_net]
    if USE_GPU && CUDA.functional()
        return fmap(cu, net)
    else
        return net
    end
end

function load_vae_interpretation(name::String, vae_path::String, inverse_path::String; vae_path_alt::String = "")
    vae_path2 = resolve_file(vae_path, vae_path_alt)
    @info "Loading VAE interpretation" name vae_path2 inverse_path
    d = BSON.load(vae_path2)
    haskey(d, :model) || error("BSON file $(vae_path2) does not contain variable :model")
    vae = d[:model]
    inverse_net = load_inverse_net(inverse_path)

    if USE_GPU && CUDA.functional()
        vae = fmap(cu, vae)
    end

    return VAEInterpretation(name, vae_path2, inverse_path, vae, inverse_net)
end

function build_grf_phi_sigma(num_eig::Int; sigma_cov::Float32 = GRF_SIGMA_COV, lambda_cov::Float32 = GRF_LAMBDA_COV)
    cov_func = GaussianRandomFields.CovarianceFunction(
        2,
        GaussianRandomFields.Matern(lambda_cov, 1; σ = sigma_cov)
    )

    x_min, x_max = minimum(coords[1, :]), maximum(coords[1, :])
    y_min, y_max = minimum(coords[2, :]), maximum(coords[2, :])

    x_pts = range(x_min, x_max; length = n)
    y_pts = range(y_min, y_max; length = n)

    grf = GaussianRandomFields.GaussianRandomField(
        cov_func,
        GaussianRandomFields.KarhunenLoeve(num_eig),
        x_pts,
        y_pts
    )

    phi_matrix = Float32.(grf.data.eigenfunc)
    sigma_vec  = Float32.(grf.data.eigenval)
    phi_sigma  = phi_matrix .* reshape(sigma_vec, 1, :)
    return phi_sigma
end

function load_grf_interpretation(name::String, inverse_path::String)
    @info "Loading GRF-KL interpretation" name inverse_path
    inverse_net = load_inverse_net(inverse_path)

    dummy_in = zeros(Float32, num_mon, 1)
    num_eig = size(Array(call_model(inverse_net, dev(dummy_in))), 1)
    @info "Inferred GRF-KL num_eig" name num_eig

    phi_sigma = build_grf_phi_sigma(num_eig)
    return GRFInterpretation(name, inverse_path, inverse_net, num_eig, phi_sigma, GRF_LAMBDA_COV)
end

function load_interpretations()
    methods = AbstractInterpretation[]
    for cfg in INTERPRETATION_CONFIGS
        if cfg.kind == :vae
            alt = (:vae_path_alt in keys(cfg)) ? string(cfg.vae_path_alt) : ""
            push!(methods, load_vae_interpretation(string(cfg.name), string(cfg.vae_path), string(cfg.inverse_path); vae_path_alt = alt))
        elseif cfg.kind == :grf
            push!(methods, load_grf_interpretation(string(cfg.name), string(cfg.inverse_path)))
        else
            error("Unknown interpretation kind: $(cfg.kind)")
        end
    end
    return methods
end

# =============================================================================
# Image IO and permeability mapping
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

to_gray_image(img_any) = Gray.(img_any)

function load_gray_resize(path::AbstractString; h::Int = IMG_H, w::Int = IMG_W)
    img_any = load(path)
    img_gray = to_gray_image(img_any)
    img_r = imresize(img_gray, (h, w))
    img_f = Float32.(img_r)
    return reshape(img_f, h, w, 1)
end

function image_to_logK_grid(img_hw1::AbstractArray{<:Real, 3}; n::Int)
    H, W, _ = size(img_hw1)
    ii = round.(Int, range(1, H; length = n))
    jj = round.(Int, range(1, W; length = n))
    g = Float32.(img_hw1[ii, jj, 1])
    g = clamp.(g, 0f0, 1f0)
    return LOGK_MIN .+ g .* (LOGK_MAX - LOGK_MIN)
end

function add_white_noise(y::AbstractVector{<:Real}; rel_std::Float32 = REL_NOISE)
    yy = Float32.(y)
    pressure_scale = max(mean(abs, yy), eps(Float32))
    sigma = rel_std * pressure_scale
    return yy .+ sigma .* randn(Float32, size(yy))
end

# =============================================================================
# Inference: pressure observations -> inferred logK model
# =============================================================================

function infer_logK(m::VAEInterpretation, bhp_obs::AbstractVector{<:Real})
    bhp_in = reshape(Float32.(bhp_obs), num_mon, 1)
    z_pred = Float32.(Array(call_model(m.inverse_net, dev(bhp_in))))

    img4 = decode(m.vae, dev(z_pred))      # (IMG_H, IMG_W, 1, 1)
    img3 = Array(img4)[:, :, :, 1]         # (IMG_H, IMG_W, 1)
    logK_hat = image_to_logK_grid(img3; n = n)
    return Float32.(logK_hat)
end

function infer_logK(m::GRFInterpretation, bhp_obs::AbstractVector{<:Real})
    bhp_in = reshape(Float32.(bhp_obs), num_mon, 1)
    x_pred = Float32.(Array(call_model(m.inverse_net, dev(bhp_in))))
    logK_vec = m.phi_sigma * Float32.(vec(x_pred))
    return reshape(logK_vec, n, n)
end

# =============================================================================
# Metrics, probabilities, and utility functions
# =============================================================================

mse(a, b) = mean((Float64.(a) .- Float64.(b)).^2)
rmse(a, b) = sqrt(mse(a, b))

# Diagnostic only. This does NOT enter ranking or probability calculations.
# K = exp(logK) is used here to stay consistent with the DPFEHM solve above.
function permeability_relative_l2(logK_hat::AbstractMatrix{<:Real},
                                  logK_true::AbstractMatrix{<:Real})
    K_hat = exp.(Float64.(logK_hat))
    K_true = exp.(Float64.(logK_true))
    denom = norm(vec(K_true), 2)
    if denom <= eps(Float64)
        return norm(vec(K_hat .- K_true), 2)
    end
    return norm(vec(K_hat .- K_true), 2) / denom
end

function pressure_sigma_eff(y_ref::AbstractVector{<:Real})
    yy = Float64.(y_ref)
    pressure_scale = max(mean(abs, yy), eps(Float64))
    sigma_obs = Float64(REL_NOISE) * pressure_scale
    sigma_model = Float64(PROB_MODEL_ERROR_REL) * pressure_scale
    return sqrt(sigma_obs^2 + sigma_model^2)
end

function pressure_loglike(pressure_rmse::Real, sigma_p::Real)
    pnorm = Float64(pressure_rmse) / max(Float64(sigma_p), eps(Float64))
    return -0.5 * Float64(num_mon) * pnorm^2
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

function csv_field(x)
    if x isa AbstractString
        s = replace(String(x), "\"" => "\"\"")
        return "\"" * s * "\""
    elseif x isa AbstractFloat
        if isfinite(x)
            return @sprintf("%.10e", Float64(x))
        else
            return string(x)
        end
    else
        return string(x)
    end
end

function write_csv(path::AbstractString, header::Vector{String}, rows)
    open(path, "w") do io
        println(io, join(csv_field.(header), ","))
        for row in rows
            println(io, join(csv_field.(row), ","))
        end
    end
end

# =============================================================================
# Plot functions
# =============================================================================

function style_publication_axes!(ax; grid_axis::String = "both")
    ax.tick_params(axis = "both", which = "major",
                   labelsize = 11, direction = "in",
                   top = true, right = true, length = 5, width = 1.0)
    ax.tick_params(axis = "both", which = "minor",
                   direction = "in", top = true, right = true,
                   length = 3, width = 0.8)
    for side in ["left", "right", "top", "bottom"]
        ax.spines[side].set_linewidth(1.0)
    end
    ax.grid(true, axis = grid_axis, alpha = 0.22, linewidth = 0.7)
    return nothing
end

function save_metric_boxplot(path::AbstractString,
                             method_names::Vector{String},
                             values::Matrix{Float64};
                             ylabel_text::String,
                             title_text::String = "")
    # title_text is retained for backward compatibility, but publication figures
    # intentionally do not display a title.
    fig = figure(figsize = (max(7.0, 1.65 * length(method_names)), 4.8))
    data = [values[:, j] for j in 1:length(method_names)]
    boxplot(data,
            labels = method_names,
            showmeans = true,
            widths = 0.58,
            medianprops = Dict("linewidth" => 1.8),
            meanprops = Dict("markersize" => 6))
    ylabel(ylabel_text, fontsize = 13)
    ax = gca()
    style_publication_axes!(ax; grid_axis = "y")
    ax.tick_params(axis = "x", labelsize = 11)
    tight_layout(pad = 0.6)
    savefig(path, dpi = 600, bbox_inches = "tight", pad_inches = 0.03)
    close(fig)
end

function save_histogram_overlay(path::AbstractString,
                                method_names::Vector{String},
                                values::Matrix{Float64};
                                xlabel_text::String,
                                title_text::String = "",
                                nbins::Int = 70,
                                density::Bool = true,
                                trim_upper_quantile::Union{Nothing, Float64} = nothing)
    # title_text is retained for backward compatibility, but publication figures
    # intentionally do not display a title.
    allvals = Float64[]
    for v in vec(values)
        if isfinite(v)
            push!(allvals, v)
        end
    end

    if isempty(allvals)
        @warn "No finite values available for histogram."
        return
    end

    xmin = min(0.0, minimum(allvals))
    xmax = maximum(allvals)
    if trim_upper_quantile !== nothing
        xmax = quantile(allvals, trim_upper_quantile)
    end
    if xmax <= xmin
        xmax = xmin + 1.0e-12
    end

    bins = collect(range(xmin, xmax; length = nbins + 1))
    fig = figure(figsize = (7.2, 4.8))

    for j in 1:length(method_names)
        vals = Float64[]
        for v in values[:, j]
            if isfinite(v) && v <= xmax && v >= xmin
                push!(vals, v)
            end
        end
        isempty(vals) && continue

        finite_method_vals = [Float64(v) for v in values[:, j] if isfinite(v)]
        method_mean = isempty(finite_method_vals) ? NaN : mean(finite_method_vals)
        label_txt = @sprintf("%s, mean = %.3g", method_names[j], method_mean)

        hist(vals;
             bins = bins,
             density = density,
             histtype = "step",
             linewidth = 2.0,
             label = label_txt)
    end

    xlabel(xlabel_text, fontsize = 13)
    ylabel(density ? "Density" : "Count", fontsize = 13)
    xlim(xmin, xmax)

    ax = gca()
    style_publication_axes!(ax; grid_axis = "y")
    minorticks_on()

    # Keep the legend fully INSIDE the plotting area.
    legend(loc = "best",
           fontsize = 10,
           frameon = true,
           framealpha = 0.92,
           borderpad = 0.5,
           handlelength = 2.4)

    tight_layout(pad = 0.6)
    savefig(path, dpi = 600, bbox_inches = "tight", pad_inches = 0.03)
    close(fig)
end

function save_aggregate_probability_bar(path::AbstractString,
                                        method_names::Vector{String},
                                        agg_probs::Vector{Float64})
    order = sortperm(agg_probs, rev = true)
    names_sorted = method_names[order]
    probs_sorted = agg_probs[order]

    fig = figure(figsize = (max(7.0, 1.5 * length(method_names)), 4.5))
    x = collect(1:length(method_names))
    bar(x, probs_sorted)
    xticks(x, names_sorted, rotation = 25, ha = "right")
    ylabel("Aggregate pressure-based relative probability")
    ylim(0.0, min(1.05, maximum(probs_sorted) + 0.15))
    for i in 1:length(x)
        text(x[i], probs_sorted[i] + 0.02, @sprintf("%.3f", probs_sorted[i]), ha = "center", va = "bottom", fontsize = 9)
    end
    grid(true, axis = "y", alpha = 0.3)
    tight_layout()
    savefig(path, dpi = 300)
    close(fig)
end

function kde_probability_curve(samples::AbstractVector{<:Real}; xmin::Float64 = 0.0, xmax::Float64 = 1.0, ngrid::Int = 400)
    s = clamp.(Float64.(samples), xmin, xmax)
    N = length(s)
    x = collect(range(xmin, xmax; length = ngrid))
    if N == 0
        return x, zeros(Float64, ngrid)
    end

    sig = N > 1 ? std(s) : 0.0
    h = 1.06 * max(sig, 0.03) * max(N, 1)^(-1 / 5)
    h = clamp(h, 0.02, 0.20)

    invnorm = 1.0 / sqrt(2.0 * pi)
    y = zeros(Float64, length(x))
    for si in s
        y .+= invnorm .* exp.(-0.5 .* ((x .- si) ./ h).^2) ./ h
        y .+= invnorm .* exp.(-0.5 .* ((x .+ si) ./ h).^2) ./ h
        y .+= invnorm .* exp.(-0.5 .* ((x .- (2.0 - si)) ./ h).^2) ./ h
    end
    y ./= N

    area = sum((y[1:end-1] .+ y[2:end]) .* diff(x)) / 2.0
    if isfinite(area) && area > 0.0
        y ./= area
    end
    return x, y
end

function save_probability_distribution_curve(path::AbstractString,
                                             method_names::Vector{String},
                                             probs::Matrix{Float64})
    N, K = size(probs)
    fig = figure(figsize = (7.2, 4.8))
    ymax = 0.0
    for j in 1:K
        x, y = kde_probability_curve(probs[:, j])
        ymax = max(ymax, maximum(y))
        plot(x, y, linewidth = 2, label = method_names[j])
    end

    rug_base = -0.03 * max(ymax, 1.0)
    rug_step = 0.025 * max(ymax, 1.0)
    for j in 1:K
        vals = probs[:, j]
        if N > MAX_RUG_POINTS
            idx = round.(Int, range(1, N; length = MAX_RUG_POINTS))
            vals = vals[idx]
        end
        y0 = fill(rug_base - (j - 1) * rug_step, length(vals))
        plot(vals, y0, "|", markersize = 7)
    end

    xlabel("Per-image relative probability from pressure error")
    ylabel("Density")
    xlim(0.0, 1.0)
    ylim(rug_base - K * rug_step, 1.10 * max(ymax, 1.0))
    grid(true, alpha = 0.3)
    legend(loc = "center left", bbox_to_anchor = (1.02, 0.5))
    tight_layout()
    savefig(path, dpi = 300, bbox_inches = "tight")
    close(fig)
end

function save_probability_exceedance_curve(path::AbstractString,
                                           method_names::Vector{String},
                                           probs::Matrix{Float64})
    N, K = size(probs)
    thresholds = collect(range(0.0, 1.0; length = 401))

    fig = figure(figsize = (7.2, 4.8))
    for j in 1:K
        exceed = [mean(probs[:, j] .>= t) for t in thresholds]
        plot(thresholds, exceed, linewidth = 2, label = method_names[j])
    end

    xlabel("Probability threshold")
    ylabel("Fraction of test images with probability >= threshold")
    xlim(0.0, 1.0)
    ylim(0.0, 1.02)
    grid(true, alpha = 0.3)
    legend(loc = "center left", bbox_to_anchor = (1.02, 0.5))
    tight_layout()
    savefig(path, dpi = 300, bbox_inches = "tight")
    close(fig)
end

function save_likelihood_curve_from_rmse(path::AbstractString,
                                         method_names::Vector{String},
                                         rmse_mat::Matrix{Float64},
                                         sigma_eff::Vector{Float64})
    sig = mean(sigma_eff)
    rmax = max(maximum(rmse_mat) * 1.15, 5.0 * sig)
    x = collect(range(0.0, rmax; length = 500))
    y = exp.(-0.5 .* Float64(num_mon) .* (x ./ sig).^2)

    fig = figure(figsize = (7.2, 4.8))
    plot(x, y, linewidth = 2, label = "Pressure likelihood curve")
    for j in 1:length(method_names)
        rbar = mean(rmse_mat[:, j])
        ybar = exp(-0.5 * Float64(num_mon) * (rbar / sig)^2)
        plot([rbar], [ybar], "o", markersize = 7, label = method_names[j])
    end
    xlabel("Pressure RMSE")
    ylabel("Unnormalized likelihood")
    yscale("log")
    grid(true, alpha = 0.3)
    legend(loc = "center left", bbox_to_anchor = (1.02, 0.5))
    tight_layout()
    savefig(path, dpi = 300, bbox_inches = "tight")
    close(fig)
end


function plot_matrix_for_image(A)
    return Array(A)[end:-1:1, :]
end

function save_random_pressure_comparison(path::AbstractString,
                                         bhp_true::Vector{Float32},
                                         bhp_obs::Vector{Float32},
                                         bhp_ref::Vector{Float32},
                                         bhp_hats::Vector{Vector{Float32}},
                                         method_names::Vector{String};
                                         title_text::String = "")
    # title_text is retained for backward compatibility; no title is drawn.
    ref = Float64.(bhp_ref)
    obs = Float64.(bhp_obs)
    tru = Float64.(bhp_true)
    order = sortperm(ref)
    x = collect(1:length(ref))

    fig = figure(figsize = (11.0, 4.6))

    # ---------------------------------------------------------------------
    # (a) Pressure profiles
    # ---------------------------------------------------------------------
    subplot(1, 2, 1)
    plot(x, ref[order], linewidth = 2.3, label = "Reference")

    if maximum(abs.(obs .- ref)) > 0.0
        plot(x, obs[order], "--", linewidth = 1.6, alpha = 0.80,
             label = "Observed")
    end
    if maximum(abs.(tru .- ref)) > 0.0
        plot(x, tru[order], ":", linewidth = 1.8, alpha = 0.85,
             label = "True")
    end
    for j in 1:length(method_names)
        pred = Float64.(bhp_hats[j])
        plot(x, pred[order], linewidth = 1.55, alpha = 0.90,
             label = method_names[j])
    end

    xlabel("Monitoring location (sorted by reference pressure)", fontsize = 12)
    ylabel("Pressure", fontsize = 12)
    ax1 = gca()
    style_publication_axes!(ax1; grid_axis = "both")
    ax1.text(0.02, 0.96, "(a)", transform = ax1.transAxes,
             ha = "left", va = "top", fontsize = 12, fontweight = "bold")
    legend(loc = "best", fontsize = 9.2, frameon = true, framealpha = 0.92)

    # ---------------------------------------------------------------------
    # (b) Predicted versus reference pressure
    # ---------------------------------------------------------------------
    subplot(1, 2, 2)
    allvals = copy(ref)
    for bh in bhp_hats
        append!(allvals, Float64.(bh))
    end

    mn = minimum(allvals)
    mx = maximum(allvals)
    span = max(mx - mn, eps(Float64))
    pad = 0.03 * span
    lo = mn - pad
    hi = mx + pad

    plot([lo, hi], [lo, hi], "k--", linewidth = 1.2, alpha = 0.70,
         label = "1:1")

    for j in 1:length(method_names)
        pred = Float64.(bhp_hats[j])
        r = sqrt(mean((pred .- ref).^2))
        plot(ref, pred, "o",
             markersize = 3.3,
             alpha = 0.58,
             markeredgewidth = 0.0,
             label = @sprintf("%s, RMSE = %.3g", method_names[j], r))
    end

    xlabel("Reference pressure", fontsize = 12)
    ylabel("Predicted pressure", fontsize = 12)
    xlim(lo, hi)
    ylim(lo, hi)
    ax2 = gca()
    ax2.set_aspect("equal", adjustable = "box")
    style_publication_axes!(ax2; grid_axis = "both")
    ax2.text(0.02, 0.96, "(b)", transform = ax2.transAxes,
             ha = "left", va = "top", fontsize = 12, fontweight = "bold")
    legend(loc = "best", fontsize = 9.0, frameon = true, framealpha = 0.92)

    # No subplot titles and no suptitle.
    tight_layout(pad = 0.7, w_pad = 1.2)
    savefig(path, dpi = 600, bbox_inches = "tight", pad_inches = 0.03)
    close(fig)
end

function save_random_permeability_comparison(path::AbstractString,
                                             logK_true::Matrix{Float32},
                                             logK_hats::Vector{Matrix{Float32}},
                                             method_names::Vector{String};
                                             title_text::String = "")
    # title_text is retained for backward compatibility; no title is drawn.
    K = length(method_names)
    fig = figure(figsize = (3.0 * (K + 1) + 0.8, 3.6))

    # Use one common color scale for every panel so visual comparison is valid.
    vmin = minimum(Float64.(logK_true))
    vmax = maximum(Float64.(logK_true))
    for A in logK_hats
        vmin = min(vmin, minimum(Float64.(A)))
        vmax = max(vmax, maximum(Float64.(A)))
    end

    axes_list = Any[]

    # Ground truth panel.
    subplot(1, K + 1, 1)
    ax = gca()
    push!(axes_list, ax)
    im = imshow(plot_matrix_for_image(logK_true),
                origin = "lower",
                vmin = vmin,
                vmax = vmax,
                cmap = "viridis",
                interpolation = "nearest")
    axis("equal")
    axis("off")
    ax.text(0.02, 0.98, "(a)", transform = ax.transAxes,
            ha = "left", va = "top", fontsize = 11, fontweight = "bold")
    ax.text(0.50, -0.07, "Ground truth",
            transform = ax.transAxes,
            ha = "center", va = "top", fontsize = 10.5)

    # Inferred permeability panels.
    for j in 1:K
        subplot(1, K + 1, j + 1)
        axj = gca()
        push!(axes_list, axj)
        rel_l2 = permeability_relative_l2(logK_hats[j], logK_true)
        im = imshow(plot_matrix_for_image(logK_hats[j]),
                    origin = "lower",
                    vmin = vmin,
                    vmax = vmax,
                    cmap = "viridis",
                    interpolation = "nearest")
        axis("equal")
        axis("off")

        panel_letter = "(" * string(Char(Int('a') + j)) * ")"
        axj.text(0.02, 0.98, panel_letter,
                 transform = axj.transAxes,
                 ha = "left", va = "top",
                 fontsize = 11, fontweight = "bold")
        axj.text(0.50, -0.07,
                 @sprintf("%s\nrelative L2 = %.3g", method_names[j], rel_l2),
                 transform = axj.transAxes,
                 ha = "center", va = "top", fontsize = 10.0)
    end

    # One shared colorbar for the complete figure rather than attaching it to
    # only the final panel.
    subplots_adjust(left = 0.015, right = 0.91, bottom = 0.18,
                    top = 0.985, wspace = 0.04)
    cax = fig.add_axes([0.925, 0.19, 0.014, 0.76])
    cb = fig.colorbar(im, cax = cax)
    cb.set_label("log K", fontsize = 12)
    cb.ax.tick_params(labelsize = 10, direction = "in", length = 4, width = 0.9)

    # No panel titles and no suptitle.
    savefig(path, dpi = 600, bbox_inches = "tight", pad_inches = 0.03)
    close(fig)
end

# =============================================================================
# Main evaluation
# =============================================================================

function main()
    isdir(OUT_DIR) || mkpath(OUT_DIR)
    isdir(PLOT_DIR) || mkpath(PLOT_DIR)

    if USE_GPU && CUDA.functional()
        @info "Using GPU"
    else
        @info "Using CPU"
    end

    methods = load_interpretations()
    K = length(methods)
    method_names = [interpretation_name(m) for m in methods]
    priors = fill(1.0 / K, K)
    log_priors = log.(priors)

    files_all = image_files(TEST_IMG_DIR)
    @info "Found test images" TEST_IMG_DIR count = length(files_all)
    isempty(files_all) && error("No test images found in $(TEST_IMG_DIR)")

    N = min(N_TEST_MAX, length(files_all))
    files = files_all[1:N]
    image_names = [splitext(basename(f))[1] for f in files]

    random_comparison_indices = Int[]
    if SAVE_RANDOM_SAMPLE_COMPARISONS && N_RANDOM_COMPARISON_SAMPLES > 0
        rng_compare = MersenneTwister(RANDOM_COMPARISON_SEED)
        nshow = min(N_RANDOM_COMPARISON_SAMPLES, N)
        random_comparison_indices = sort(randperm(rng_compare, N)[1:nshow])
        @info "Random sample comparison plots will be saved" indices = random_comparison_indices
    end
    random_comparison_index_set = Set(random_comparison_indices)

    rmse_pressure_obs      = zeros(Float64, N, K)
    rmse_pressure_true     = zeros(Float64, N, K)
    rmse_pressure_ranking  = zeros(Float64, N, K)
    normalized_pressure_error = zeros(Float64, N, K)
    permeability_rel_l2    = zeros(Float64, N, K)  # diagnostic only
    loglike                = zeros(Float64, N, K)  # pressure-only likelihood
    probs                  = zeros(Float64, N, K)  # pressure-only probabilities
    sigma_eff              = zeros(Float64, N)

    Random.seed!(NOISE_SEED)
    t_start = time()

    for (i, img_path) in enumerate(files)
        if i == 1 || i % PROGRESS_EVERY == 0 || i == N
            elapsed_min = (time() - t_start) / 60.0
            @printf("Evaluating image %d / %d | elapsed %.2f min\n", i, N, elapsed_min)
            flush(stdout)
        end

        # True geologic model and pressure response from the test image.
        img3 = load_gray_resize(img_path; h = IMG_H, w = IMG_W)
        logK_true = image_to_logK_grid(img3; n = n)
        P_true_full = solve_pressure_full_from_logK(logK_true)
        bhp_true = Float32.(P_true_full[monitoring_nodes])

        # One shared noisy pressure observation for all competing interpretations.
        bhp_obs = add_white_noise(bhp_true; rel_std = REL_NOISE)

        if PROBABILITY_REFERENCE == :obs
            bhp_prob_ref = bhp_obs
        elseif PROBABILITY_REFERENCE == :true
            bhp_prob_ref = bhp_true
        else
            error("PROBABILITY_REFERENCE must be :obs or :true")
        end
        sigma_eff[i] = pressure_sigma_eff(bhp_prob_ref)

        save_random_comparison = i in random_comparison_index_set
        logK_hats_for_plot = save_random_comparison ? Vector{Matrix{Float32}}(undef, K) : Matrix{Float32}[]
        bhp_hats_for_plot = save_random_comparison ? Vector{Vector{Float32}}(undef, K) : Vector{Float32}[]

        for (j, method) in enumerate(methods)
            logK_hat = infer_logK(method, bhp_obs)
            P_hat_full = solve_pressure_full_from_logK(logK_hat)
            bhp_hat = Float32.(P_hat_full[monitoring_nodes])

            if save_random_comparison
                logK_hats_for_plot[j] = Float32.(logK_hat)
                bhp_hats_for_plot[j] = bhp_hat
            end

            rmse_pressure_obs[i, j]     = rmse(bhp_hat, bhp_obs)
            rmse_pressure_true[i, j]    = rmse(bhp_hat, bhp_true)
            rmse_pressure_ranking[i, j] = rmse(bhp_hat, bhp_prob_ref)
            normalized_pressure_error[i, j] = rmse_pressure_ranking[i, j] / max(sigma_eff[i], eps(Float64))

            # Diagnostic only. This does not affect ranking or probabilities.
            permeability_rel_l2[i, j] = permeability_relative_l2(logK_hat, logK_true)

            # Pressure-only likelihood and probability.
            loglike[i, j] = pressure_loglike(rmse_pressure_ranking[i, j], sigma_eff[i])
        end

        probs[i, :] .= stable_softmax(vec(loglike[i, :] .+ log_priors))

        if save_random_comparison
            sample_number = findfirst(isequal(i), random_comparison_indices)
            sample_tag = @sprintf("random_sample_%03d_%s", sample_number, image_names[i])

            save_random_pressure_comparison(
                joinpath(PLOT_DIR, sample_tag * "_pressure_comparison.png"),
                bhp_true,
                bhp_obs,
                bhp_prob_ref,
                bhp_hats_for_plot,
                method_names;
                title_text = "Pressure comparison: " * image_names[i]
            )

            save_random_permeability_comparison(
                joinpath(PLOT_DIR, sample_tag * "_permeability_comparison.png"),
                logK_true,
                logK_hats_for_plot,
                method_names;
                title_text = "Permeability comparison: " * image_names[i]
            )
        end
    end

    elapsed_total_min = (time() - t_start) / 60.0
    @printf("Finished evaluating %d images in %.2f min\n", N, elapsed_total_min)

    # Aggregate evidence across all images. Ranking is pressure-only.
    mean_pressure_rmse = [mean(rmse_pressure_ranking[:, j]) for j in 1:K]
    median_pressure_rmse = [median(rmse_pressure_ranking[:, j]) for j in 1:K]
    mean_normalized_pressure_error = [mean(normalized_pressure_error[:, j]) for j in 1:K]
    rms_normalized_pressure_error = [sqrt(mean(normalized_pressure_error[:, j].^2)) for j in 1:K]

    if AGGREGATION_MODE == :sum_loglike
        agg_loglike = vec(sum(loglike; dims = 1))
    elseif AGGREGATION_MODE == :mean_loglike
        agg_loglike = vec(mean(loglike; dims = 1))
    else
        error("AGGREGATION_MODE must be :mean_loglike or :sum_loglike")
    end

    agg_probs = stable_softmax(agg_loglike .+ log_priors)

    # Lower pressure-normalized RMS error is better.
    rank_order = sortperm(rms_normalized_pressure_error)
    rank_for_method = zeros(Int, K)
    for (r, idx) in enumerate(rank_order)
        rank_for_method[idx] = r
    end

    # Save per-image results.
    per_image_rows = Any[]
    for i in 1:N
        for j in 1:K
            push!(per_image_rows, Any[
                image_names[i],
                method_names[j],
                rmse_pressure_obs[i, j],
                rmse_pressure_true[i, j],
                rmse_pressure_ranking[i, j],
                normalized_pressure_error[i, j],
                permeability_rel_l2[i, j],
                loglike[i, j],
                probs[i, j],
                sigma_eff[i]
            ])
        end
    end
    write_csv(joinpath(OUT_DIR, "per_image_method_results.csv"),
              ["image", "interpretation", "pressure_rmse_obs", "pressure_rmse_true",
               "pressure_rmse_used_for_ranking", "normalized_pressure_error",
               "permeability_relative_l2_diagnostic_only", "pressure_log_likelihood",
               "relative_probability_pressure_only", "sigma_eff"],
              per_image_rows)

    # Save summary by interpretation.
    summary_rows = Any[]
    for j in 1:K
        push!(summary_rows, Any[
            method_names[j],
            rank_for_method[j],
            mean_pressure_rmse[j],
            N > 1 ? std(rmse_pressure_ranking[:, j]) : 0.0,
            median_pressure_rmse[j],
            quantile_safe(rmse_pressure_ranking[:, j], 0.05),
            quantile_safe(rmse_pressure_ranking[:, j], 0.95),
            mean_normalized_pressure_error[j],
            rms_normalized_pressure_error[j],
            N > 1 ? std(normalized_pressure_error[:, j]) : 0.0,
            median(normalized_pressure_error[:, j]),
            quantile_safe(normalized_pressure_error[:, j], 0.05),
            quantile_safe(normalized_pressure_error[:, j], 0.95),
            mean(permeability_rel_l2[:, j]),
            N > 1 ? std(permeability_rel_l2[:, j]) : 0.0,
            median(permeability_rel_l2[:, j]),
            quantile_safe(permeability_rel_l2[:, j], 0.05),
            quantile_safe(permeability_rel_l2[:, j], 0.95),
            mean(probs[:, j]),
            median(probs[:, j]),
            agg_loglike[j],
            agg_probs[j]
        ])
    end
    write_csv(joinpath(OUT_DIR, "summary_by_interpretation.csv"),
              ["interpretation", "aggregate_rank_pressure_only",
               "mean_pressure_rmse", "std_pressure_rmse", "median_pressure_rmse", "q05_pressure_rmse", "q95_pressure_rmse",
               "mean_normalized_pressure_error", "rms_normalized_pressure_error", "std_normalized_pressure_error",
               "median_normalized_pressure_error", "q05_normalized_pressure_error", "q95_normalized_pressure_error",
               "mean_permeability_relative_l2_diagnostic_only", "std_permeability_relative_l2_diagnostic_only",
               "median_permeability_relative_l2_diagnostic_only", "q05_permeability_relative_l2_diagnostic_only",
               "q95_permeability_relative_l2_diagnostic_only",
               "mean_per_image_probability_pressure_only", "median_per_image_probability_pressure_only",
               "aggregate_pressure_log_likelihood", "aggregate_probability_pressure_only"],
              summary_rows)

    # Save aggregate ranking in sorted order.
    aggregate_rows = Any[]
    for (r, idx) in enumerate(rank_order)
        push!(aggregate_rows, Any[
            r,
            method_names[idx],
            agg_probs[idx],
            agg_loglike[idx],
            mean_pressure_rmse[idx],
            median_pressure_rmse[idx],
            mean_normalized_pressure_error[idx],
            rms_normalized_pressure_error[idx],
            mean(permeability_rel_l2[:, idx]),
            median(permeability_rel_l2[:, idx])
        ])
    end
    write_csv(joinpath(OUT_DIR, "aggregate_ranking.csv"),
              ["rank", "interpretation", "aggregate_probability_pressure_only", "aggregate_pressure_log_likelihood",
               "mean_pressure_rmse", "median_pressure_rmse",
               "mean_normalized_pressure_error", "rms_normalized_pressure_error",
               "mean_permeability_relative_l2_diagnostic_only", "median_permeability_relative_l2_diagnostic_only"],
              aggregate_rows)

    # -------------------------------------------------------------------------
    # Clean plot set. Ranking/probability plots use pressure error only.
    # Permeability plots are diagnostic only.
    # -------------------------------------------------------------------------

    save_histogram_overlay(
        joinpath(PLOT_DIR, "pressure_rmse_histogram_overlay.png"),
        method_names,
        rmse_pressure_ranking;
        xlabel_text = "Pressure RMSE used for ranking",
        title_text = "Pressure-error distribution by interpretation",
        nbins = 70,
        density = true
    )

    save_metric_boxplot(
        joinpath(PLOT_DIR, "pressure_rmse_boxplot.png"),
        method_names,
        rmse_pressure_ranking;
        ylabel_text = "Pressure RMSE used for ranking",
        title_text = "Pressure-error distribution by interpretation"
    )

    save_histogram_overlay(
        joinpath(PLOT_DIR, "normalized_pressure_error_histogram_overlay.png"),
        method_names,
        normalized_pressure_error;
        xlabel_text = "Normalized pressure error, RMSE / sigma_eff",
        title_text = "Normalized pressure-error distribution by interpretation",
        nbins = 70,
        density = true,
        trim_upper_quantile = 0.99
    )

    save_metric_boxplot(
        joinpath(PLOT_DIR, "normalized_pressure_error_boxplot.png"),
        method_names,
        normalized_pressure_error;
        ylabel_text = "Normalized pressure error, RMSE / sigma_eff",
        title_text = "Normalized pressure-error distribution by interpretation"
    )

    save_histogram_overlay(
        joinpath(PLOT_DIR, "permeability_relative_l2_histogram_overlay.png"),
        method_names,
        permeability_rel_l2;
        xlabel_text = "Diagnostic permeability relative L2 error, ||Khat - Ktrue||2 / ||Ktrue||2",
        title_text = "Diagnostic permeability relative-L2 error by interpretation",
        nbins = 70,
        density = true,
        trim_upper_quantile = 0.99
    )

    save_metric_boxplot(
        joinpath(PLOT_DIR, "permeability_relative_l2_boxplot.png"),
        method_names,
        permeability_rel_l2;
        ylabel_text = "Diagnostic permeability relative L2 error",
        title_text = "Diagnostic permeability relative-L2 error by interpretation"
    )

    save_aggregate_probability_bar(
        joinpath(PLOT_DIR, "pressure_aggregate_probabilities.png"),
        method_names,
        agg_probs
    )

    save_probability_distribution_curve(
        joinpath(PLOT_DIR, "pressure_probability_distribution_curves.png"),
        method_names,
        probs
    )

    save_probability_exceedance_curve(
        joinpath(PLOT_DIR, "pressure_probability_exceedance_curves.png"),
        method_names,
        probs
    )

    save_likelihood_curve_from_rmse(
        joinpath(PLOT_DIR, "pressure_rmse_to_likelihood_curve.png"),
        method_names,
        rmse_pressure_ranking,
        sigma_eff
    )

    # Text ranking report.
    top_idx = rank_order[1]
    second_idx = length(rank_order) >= 2 ? rank_order[2] : rank_order[1]
    top_prob = agg_probs[top_idx]
    second_prob = length(rank_order) >= 2 ? agg_probs[second_idx] : 0.0
    top_second_ratio = second_prob > 0.0 ? top_prob / second_prob : Inf
    Hnorm = normalized_entropy(agg_probs)
    entropy_confidence = 1.0 - Hnorm

    open(joinpath(OUT_DIR, "ranking.txt"), "w") do io
        println(io, "Competing interpretation ranking based on pressure error only")
        println(io, "============================================================")
        println(io, "")
        @printf(io, "Number of test images: %d\n", N)
        @printf(io, "Number of pressure observations per image: %d\n", num_mon)
        @printf(io, "Probability reference: %s\n", string(PROBABILITY_REFERENCE))
        @printf(io, "Relative observation noise: %.5f\n", REL_NOISE)
        @printf(io, "Relative pressure model-error floor: %.5f\n", PROB_MODEL_ERROR_REL)
        @printf(io, "Aggregation mode for aggregate probabilities: %s\n", string(AGGREGATION_MODE))
        @printf(io, "Elapsed time: %.2f min\n", elapsed_total_min)
        if SAVE_RANDOM_SAMPLE_COMPARISONS
            println(io, "Random comparison sample indices: " * join(string.(random_comparison_indices), ", "))
        end
        println(io, "")
        println(io, "Aggregate ranking, sorted by lowest RMS normalized pressure error:")
        for (r, idx) in enumerate(rank_order)
            @printf(io, "%2d. %-20s  P = %.6f  RMS_norm_pressure = %.6e  mean_pressure_RMSE = %.6e  median_pressure_RMSE = %.6e  mean_perm_rel_L2_diagnostic = %.6e\n",
                    r, method_names[idx], agg_probs[idx], rms_normalized_pressure_error[idx], mean_pressure_rmse[idx], median_pressure_rmse[idx], mean(permeability_rel_l2[:, idx]))
        end
        println(io, "")
        @printf(io, "Top-ranked interpretation: %s\n", method_names[top_idx])
        @printf(io, "Top aggregate probability: %.6f\n", top_prob)
        @printf(io, "Second aggregate probability: %.6f\n", second_prob)
        @printf(io, "Top/second probability ratio: %.6e\n", top_second_ratio)
        @printf(io, "Normalized entropy: %.6f\n", Hnorm)
        @printf(io, "Entropy-based confidence, 1-H/Hmax: %.6f\n", entropy_confidence)
        println(io, "")
        println(io, "Note: permeability relative L2 is diagnostic only and is not used in the pressure-based ranking or probabilities.")

        if !isempty(TRUE_INTERPRETATION_NAME)
            true_idx = findfirst(isequal(TRUE_INTERPRETATION_NAME), method_names)
            if true_idx === nothing
                @printf(io, "\nTRUE_INTERPRETATION_NAME was set to '%s', but no method has that name.\n", TRUE_INTERPRETATION_NAME)
            else
                @printf(io, "\nKnown true interpretation: %s\n", TRUE_INTERPRETATION_NAME)
                @printf(io, "Pressure-based probability assigned to true interpretation: %.6f\n", agg_probs[true_idx])
                @printf(io, "Pressure-based rank of true interpretation: %d\n", rank_for_method[true_idx])
                @printf(io, "Workflow identified true interpretation as top ranked using pressure only: %s\n", rank_for_method[true_idx] == 1 ? "yes" : "no")
            end
        end
    end

    println("\nAggregate ranking based on pressure error only:")
    for (r, idx) in enumerate(rank_order)
        @printf("%2d. %-20s P = %.6f | RMS norm pressure = %.6e | mean pressure RMSE = %.6e | median pressure RMSE = %.6e | mean permeability rel L2 diagnostic = %.6e\n",
                r, method_names[idx], agg_probs[idx], rms_normalized_pressure_error[idx], mean_pressure_rmse[idx], median_pressure_rmse[idx], mean(permeability_rel_l2[:, idx]))
    end
    @printf("\nConfidence in top interpretation, P_top = %.6f\n", top_prob)
    @printf("Entropy-based confidence, 1-H/Hmax = %.6f\n", entropy_confidence)
    @info "Saved comparison outputs" OUT_DIR PLOT_DIR
end

main()
