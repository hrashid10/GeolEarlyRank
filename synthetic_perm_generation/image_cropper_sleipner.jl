using FileIO
using Images
using Random
using Printf
using Statistics

# ============================================================
# User settings
# ============================================================

const INPUT_DIR  = "sleipner"
const OUTPUT_DIR = "cropped_sleipner_full"

const CROP_H = 350
const CROP_W = 230

const N_CROPS_PER_IMAGE = 10
const BOUNDARY_MARGIN = 0
const RANDOM_SEED = 1234
const ALLOW_OVERLAP = true

# White-space removal settings
const WHITE_THRESHOLD = 0.98          # 0.98 means nearly white pixels are treated as background
const ALPHA_THRESHOLD = 0.01          # transparent pixels are treated as background
const TRIM_PAD = 0                    # keep this many pixels around non-white bounding box
const MIN_NONWHITE_FRACTION = 1.0    # crop must contain at least this fraction of non-white pixels
const MAX_TRIES_PER_CROP = 5000

const VALID_EXTS = Set([".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"])

# ============================================================
# Helper functions
# ============================================================

function is_image_file(fname::String)
    ext = lowercase(splitext(fname)[2])
    return ext in VALID_EXTS
end

function list_image_files(folder::String)
    files = readdir(folder; join = true)
    return sort(filter(is_image_file, files))
end

function rects_overlap(a, b)
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    return !(ax2 < bx1 || bx2 < ax1 || ay2 < by1 || by2 < ay1)
end

function is_white_or_transparent(p; white_threshold = WHITE_THRESHOLD,
                                    alpha_threshold = ALPHA_THRESHOLD)
    q = RGBA{Float64}(p)

    return q.alpha <= alpha_threshold ||
           (q.r >= white_threshold &&
            q.g >= white_threshold &&
            q.b >= white_threshold)
end

function nonwhite_mask(img; white_threshold = WHITE_THRESHOLD,
                            alpha_threshold = ALPHA_THRESHOLD)
    return .!is_white_or_transparent.(img;
                                      white_threshold = white_threshold,
                                      alpha_threshold = alpha_threshold)
end

function find_content_bbox(img; white_threshold = WHITE_THRESHOLD,
                                alpha_threshold = ALPHA_THRESHOLD)

    mask = nonwhite_mask(img;
                         white_threshold = white_threshold,
                         alpha_threshold = alpha_threshold)

    if !any(mask)
        return nothing
    end

    inds = findall(mask)

    rows = [I[1] for I in inds]
    cols = [I[2] for I in inds]

    r1, r2 = minimum(rows), maximum(rows)
    c1, c2 = minimum(cols), maximum(cols)

    return r1, c1, r2, c2
end

function trim_outer_whitespace(img; pad = TRIM_PAD,
                                    white_threshold = WHITE_THRESHOLD,
                                    alpha_threshold = ALPHA_THRESHOLD)

    bbox = find_content_bbox(img;
                             white_threshold = white_threshold,
                             alpha_threshold = alpha_threshold)

    if bbox === nothing
        @warn "Image appears fully white or transparent. Returning original image."
        return img, 1, 1
    end

    r1, c1, r2, c2 = bbox

    r1 = max(1, r1 - pad)
    c1 = max(1, c1 - pad)
    r2 = min(size(img, 1), r2 + pad)
    c2 = min(size(img, 2), c2 + pad)

    trimmed = img[r1:r2, c1:c2]

    return trimmed, r1, c1
end

function crop_image(img, y::Int, x::Int, crop_h::Int, crop_w::Int)
    return img[y:y + crop_h - 1, x:x + crop_w - 1]
end

function nonwhite_fraction(patch; white_threshold = WHITE_THRESHOLD,
                                  alpha_threshold = ALPHA_THRESHOLD)

    mask = nonwhite_mask(patch;
                         white_threshold = white_threshold,
                         alpha_threshold = alpha_threshold)

    return count(mask) / length(mask)
end

function patch_is_valid(patch; min_nonwhite_fraction = MIN_NONWHITE_FRACTION,
                               white_threshold = WHITE_THRESHOLD,
                               alpha_threshold = ALPHA_THRESHOLD)

    frac = nonwhite_fraction(patch;
                             white_threshold = white_threshold,
                             alpha_threshold = alpha_threshold)

    return frac >= min_nonwhite_fraction
end

"""
Sample crop positions while rejecting patches that contain too much white space.
"""
function sample_crop_positions_no_whitespace(img,
                                             crop_h::Int,
                                             crop_w::Int,
                                             n_crops::Int,
                                             margin::Int;
                                             allow_overlap::Bool = true,
                                             rng = MersenneTwister(1234),
                                             max_tries_per_crop::Int = MAX_TRIES_PER_CROP,
                                             min_nonwhite_fraction::Float64 = MIN_NONWHITE_FRACTION)

    img_h, img_w = size(img, 1), size(img, 2)

    y_min = 1 + margin
    x_min = 1 + margin
    y_max = img_h - crop_h - margin + 1
    x_max = img_w - crop_w - margin + 1

    if y_max < y_min || x_max < x_min
        @warn "Crop size is larger than trimmed image. Skipping this image." img_h img_w crop_h crop_w
        return Tuple{Int, Int}[]
    end

    positions = Tuple{Int, Int}[]
    used_rects = Vector{NTuple{4, Int}}()

    for crop_id in 1:n_crops
        found = false

        for _ in 1:max_tries_per_crop
            y = rand(rng, y_min:y_max)
            x = rand(rng, x_min:x_max)

            rect = (y, x, y + crop_h - 1, x + crop_w - 1)

            if !allow_overlap
                overlaps = any(r -> rects_overlap(rect, r), used_rects)
                overlaps && continue
            end

            patch = crop_image(img, y, x, crop_h, crop_w)

            if patch_is_valid(patch;
                              min_nonwhite_fraction = min_nonwhite_fraction)
                push!(positions, (y, x))
                push!(used_rects, rect)
                found = true
                break
            end
        end

        if !found
            @warn "Could only place $(length(positions)) valid crops instead of $n_crops. Try lowering MIN_NONWHITE_FRACTION."
            break
        end
    end

    return positions
end

function make_output_name(infile::String, idx::Int, y::Int, x::Int; ext = ".png")
    stem = splitext(basename(infile))[1]
    return @sprintf("%s_crop_%03d_y%04d_x%04d%s", stem, idx, y, x, ext)
end

# ============================================================
# Main processing
# ============================================================

function generate_crops_from_folder(input_dir::String, output_dir::String;
                                    crop_h::Int = 128,
                                    crop_w::Int = 128,
                                    n_crops_per_image::Int = 10,
                                    margin::Int = 10,
                                    allow_overlap::Bool = true,
                                    seed::Int = 1234)

    mkpath(output_dir)
    rng = MersenneTwister(seed)

    image_files = list_image_files(input_dir)

    if isempty(image_files)
        println("No image files found in folder: $input_dir")
        return
    end

    println("Found $(length(image_files)) image(s).")

    for (img_idx, infile) in enumerate(image_files)
        println("\n[$img_idx/$(length(image_files))] Processing: $infile")

        img_raw = load(infile)

        raw_h, raw_w = size(img_raw, 1), size(img_raw, 2)
        println("  Original image size: $(raw_h) x $(raw_w)")

        img, row_offset, col_offset = trim_outer_whitespace(img_raw;
                                                            pad = TRIM_PAD,
                                                            white_threshold = WHITE_THRESHOLD,
                                                            alpha_threshold = ALPHA_THRESHOLD)

        img_h, img_w = size(img, 1), size(img, 2)
        println("  Trimmed image size:  $(img_h) x $(img_w)")

        positions = sample_crop_positions_no_whitespace(
            img,
            crop_h,
            crop_w,
            n_crops_per_image,
            margin;
            allow_overlap = allow_overlap,
            rng = rng,
            max_tries_per_crop = MAX_TRIES_PER_CROP,
            min_nonwhite_fraction = MIN_NONWHITE_FRACTION
        )

        println("  Generating $(length(positions)) crop(s)...")

        for (k, (y, x)) in enumerate(positions)
            patch = crop_image(img, y, x, crop_h, crop_w)

            # Convert coordinates back to original image coordinates for filename
            y_original = y + row_offset - 1
            x_original = x + col_offset - 1

            outname = make_output_name(infile, k, y_original, x_original; ext = ".png")
            outfile = joinpath(output_dir, outname)

            save(outfile, patch)
        end
    end

    println("\nDone. Cropped images saved in: $output_dir")
end

# ============================================================
# Run
# ============================================================

generate_crops_from_folder(
    INPUT_DIR,
    OUTPUT_DIR;
    crop_h = CROP_H,
    crop_w = CROP_W,
    n_crops_per_image = N_CROPS_PER_IMAGE,
    margin = BOUNDARY_MARGIN,
    allow_overlap = ALLOW_OVERLAP,
    seed = RANDOM_SEED,
)