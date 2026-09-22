using FileIO
using Images
using ImageTransformations

function resize_images_in_folder(input_dir::AbstractString, output_dir::AbstractString, new_h::Int, new_w::Int)
    isdir(output_dir) || mkpath(output_dir)

    valid_ext = Set([".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"])

    for file in readdir(input_dir)
        inpath = joinpath(input_dir, file)

        if isfile(inpath)
            ext = lowercase(splitext(file)[2])
            if ext in valid_ext
                img = load(inpath)
                img_resized = imresize(img, (new_h, new_w))
                outpath = joinpath(output_dir, file)
                save(outpath, img_resized)
                println("Saved: ", outpath)
            end
        end
    end
end

# example
resize_images_in_folder("gemini_data", "resized_images", 256, 256)