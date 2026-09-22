using FileIO
using Images
using Random
using Printf

# ============================================================
# User settings
# ============================================================
# const INPUT_DIR  = "gemini_weak"       # folder containing original images
# const OUTPUT_DIR = "image_train_weak"     # folder to save cropped patches

const INPUT_DIR  = "C:/Users/398654/Documents/Geol_features_inverse/wiip/image_old_concept"       # folder containing original images
const OUTPUT_DIR = "C:/Users/398654/Documents/Geol_features_inverse/wiip/image_train_old_concept"     # folder to save cropped patches

const CROP_H = 1000                      # crop height
const CROP_W = 1000                      # crop width

const N_CROPS_PER_IMAGE = 10            # how many sub-images from each input image
const BOUNDARY_MARGIN = 0              # avoid sampling too close to image boundary
const RANDOM_SEED = 1234                # reproducible cropping
const ALLOW_OVERLAP = true              # true = crops may overlap, false = try to avoid overlap

# allowed image extensions
const VALID_EXTS = Set([".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"])

# ============================================================
# Helper functions
# ============================================================

function is_image_file(fname::String)
    ext = lowercase(splitext(fname)[2])
    return ext in VALID_EXTS
end

function list_image_files(folder::String)
    files = readdir(folder; join=true)
    return sort(filter(is_image_file, files))
end

"""
Return true if two rectangles overlap.
Each rectangle is (y1, x1, y2, x2).
"""
function rects_overlap(a, b)
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    return !(ax2 < bx1 || bx2 < ax1 || ay2 < by1 || by2 < ay1)
end

"""
Try to sample crop positions that avoid the boundary.
If ALLOW_OVERLAP = false, also tries to avoid overlapping previous crops.
"""
function sample_crop_positions(img_h::Int, img_w::Int,
                               crop_h::Int, crop_w::Int,
                               n_crops::Int, margin::Int;
                               allow_overlap::Bool=true,
                               rng=MersenneTwister(1234),
                               max_tries_per_crop::Int=500)

    y_min = 1 + margin
    x_min = 1 + margin
    y_max = img_h - crop_h - margin + 1
    x_max = img_w - crop_w - margin + 1

    if y_max < y_min || x_max < x_min
        error("Crop size + boundary margin is too large for this image. " *
              "Image size = ($(img_h), $(img_w)), crop = ($(crop_h), $(crop_w)), margin = $margin")
    end

    positions = Tuple{Int,Int}[]
    used_rects = Vector{NTuple{4,Int}}()

    for _ in 1:n_crops
        found = false

        for _ in 1:max_tries_per_crop
            y = rand(rng, y_min:y_max)
            x = rand(rng, x_min:x_max)

            rect = (y, x, y + crop_h - 1, x + crop_w - 1)

            if allow_overlap
                push!(positions, (y, x))
                push!(used_rects, rect)
                found = true
                break
            else
                overlaps = any(r -> rects_overlap(rect, r), used_rects)
                if !overlaps
                    push!(positions, (y, x))
                    push!(used_rects, rect)
                    found = true
                    break
                end
            end
        end

        if !found
            @warn "Could only place $(length(positions)) crops instead of $n_crops without violating constraints."
            break
        end
    end

    return positions
end

"""
Crop image using top-left corner (y, x).
Works for grayscale or multi-channel images.
"""
function crop_image(img, y::Int, x::Int, crop_h::Int, crop_w::Int)
    return img[y:y+crop_h-1, x:x+crop_w-1]
end

"""
Build output filename:
originalname_crop_001_y0123_x0456.png
"""
function make_output_name(infile::String, idx::Int, y::Int, x::Int; ext=".png")
    stem = splitext(basename(infile))[1]
    return @sprintf("%s_crop_%03d_y%04d_x%04d%s", stem, idx, y, x, ext)
end

# ============================================================
# Main processing
# ============================================================

function generate_crops_from_folder(input_dir::String, output_dir::String;
                                    crop_h::Int=128,
                                    crop_w::Int=128,
                                    n_crops_per_image::Int=10,
                                    margin::Int=10,
                                    allow_overlap::Bool=true,
                                    seed::Int=1234)

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

        img = load(infile)

        img_h, img_w = size(img)[1], size(img)[2]
        println("  Image size: $(img_h) x $(img_w)")

        positions = sample_crop_positions(img_h, img_w,
                                          crop_h, crop_w,
                                          n_crops_per_image, margin;
                                          allow_overlap=allow_overlap,
                                          rng=rng)

        println("  Generating $(length(positions)) crop(s)...")

        for (k, (y, x)) in enumerate(positions)
            patch = crop_image(img, y, x, crop_h, crop_w)

            outname = make_output_name(infile, k, y, x; ext=".png")
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