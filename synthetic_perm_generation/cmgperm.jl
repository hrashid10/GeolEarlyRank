using PyCall
using PyPlot
using Statistics


using PyPlot
using Statistics

# ============================================================
# File names
# ============================================================
coord_file  = "COORD.GRID"
zcorn_file  = "ZCORN.GRID"
null_file   = "ACTNUM.GRID"   # your uploaded file; header says *NULL
permi_file  = "PERMI.txt"

# ============================================================
# Grid size from BaseCase.dat:
# *GRID *CORNER 64 118 263
# ============================================================
NI = 64
NJ = 118
NK = 263

# ============================================================
# Read all numeric values after first header line
# ============================================================
function read_keyword_values(fname::String)
    lines = readlines(fname)
    isempty(lines) && error("Empty file: $fname")
    return parse.(Float64, split(join(lines[2:end], " ")))
end

# ============================================================
# Load arrays
# COORD: 6 values per pillar
# ZCORN: reshape as (2, NI, 2, NJ, 2, NK)
# PERMI: cell-based, I fastest
# ============================================================
coord_vals = read_keyword_values(coord_file)
zcorn_vals = read_keyword_values(zcorn_file)
null_vals  = read_keyword_values(null_file)
permi_vals = read_keyword_values(permi_file)

length(coord_vals) == 6*(NI+1)*(NJ+1) || error("COORD size mismatch")
length(zcorn_vals) == 8*NI*NJ*NK      || error("ZCORN size mismatch")
length(permi_vals) == NI*NJ*NK        || error("PERMI size mismatch")
length(null_vals)  == NI*NJ*NK        || error("NULL/ACTNUM size mismatch")

coord = reshape(coord_vals, 6, NI+1, NJ+1)
zcorn = reshape(zcorn_vals, 2, NI, 2, NJ, 2, NK)
permi = reshape(permi_vals, NI, NJ, NK)

# Treat NULL=0 as active, NULL=1 as null/inactive if needed.
# If your file behaves opposite, flip this.
inactive = reshape(Int.(round.(null_vals)), NI, NJ, NK)

# ============================================================
# Pillar interpolation:
# COORD[:, ip, jp] = [x1,y1,z1,x2,y2,z2]
# ============================================================
function pillar_xy_at_z(coord, ip::Int, jp::Int, z::Float64)
    x1, y1, z1, x2, y2, z2 = coord[:, ip, jp]
    if abs(z2 - z1) < 1e-12
        t = 0.5
    else
        t = (z - z1) / (z2 - z1)
    end
    x = x1 + t*(x2 - x1)
    y = y1 + t*(y2 - y1)
    return x, y
end

# ============================================================
# ZCORN access helper
#
# zcorn[a, i, b, j, c, k]
# a = 1(left in I), 2(right in I)
# b = 1(front in J), 2(back in J)
# c = 1(top), 2(bottom)
# ============================================================
zv(iL, i, jL, j, tb, k) = zcorn[iL, i, jL, j, tb, k]

# ============================================================
# I-K cross-section polygon at fixed J
#
# side = :front uses pillar row j
# side = :back  uses pillar row j+1
# ============================================================
function cell_polygon_IK(coord, zcorn, i::Int, j::Int, k::Int; side=:front)
    if side == :front
        zA = zcorn[1, i, 1, j, 1, k]  # left, front, top
        zB = zcorn[2, i, 1, j, 1, k]  # right, front, top
        zC = zcorn[2, i, 1, j, 2, k]  # right, front, bottom
        zD = zcorn[1, i, 1, j, 2, k]  # left, front, bottom
        jp = j
    elseif side == :back
        zA = zcorn[1, i, 2, j, 1, k]  # left, back, top
        zB = zcorn[2, i, 2, j, 1, k]  # right, back, top
        zC = zcorn[2, i, 2, j, 2, k]  # right, back, bottom
        zD = zcorn[1, i, 2, j, 2, k]  # left, back, bottom
        jp = j + 1
    else
        error("side must be :front or :back")
    end

    xA, _ = pillar_xy_at_z(coord, i,   jp, zA)
    xB, _ = pillar_xy_at_z(coord, i+1, jp, zB)
    xC, _ = pillar_xy_at_z(coord, i+1, jp, zC)
    xD, _ = pillar_xy_at_z(coord, i,   jp, zD)

    return [(xA, zA), (xB, zB), (xC, zC), (xD, zD)]
end

# ============================================================
# J-K cross-section polygon at fixed I
#
# side = :left  uses pillar column i
# side = :right uses pillar column i+1
# ============================================================
function cell_polygon_JK(coord, zcorn, i::Int, j::Int, k::Int; side=:left)
    if side == :left
        zA = zcorn[1, i, 1, j, 1, k]  # left, front, top
        zB = zcorn[1, i, 2, j, 1, k]  # left, back, top
        zC = zcorn[1, i, 2, j, 2, k]  # left, back, bottom
        zD = zcorn[1, i, 1, j, 2, k]  # left, front, bottom
        ip = i
    elseif side == :right
        zA = zcorn[2, i, 1, j, 1, k]  # right, front, top
        zB = zcorn[2, i, 2, j, 1, k]  # right, back, top
        zC = zcorn[2, i, 2, j, 2, k]  # right, back, bottom
        zD = zcorn[2, i, 1, j, 2, k]  # right, front, bottom
        ip = i + 1
    else
        error("side must be :left or :right")
    end

    _, yA = pillar_xy_at_z(coord, ip, j,   zA)
    _, yB = pillar_xy_at_z(coord, ip, j+1, zB)
    _, yC = pillar_xy_at_z(coord, ip, j+1, zC)
    _, yD = pillar_xy_at_z(coord, ip, j,   zD)

    return [(yA, zA), (yB, zB), (yC, zC), (yD, zD)]
end

# ============================================================
# Plot I-K section
# No grid lines
# ============================================================
function plot_perm_IK(coord, zcorn, inactive, permi; jsec::Int,
                      side=:front, logscale=true,
                      cmap_name="viridis", outfile="permi_IK.png")

    polys = Vector{Vector{Tuple{Float64,Float64}}}()
    vals  = Float64[]

    for k in 1:NK
        for i in 1:NI
            # If your NULL convention is opposite, flip this test
            if inactive[i, jsec, k] == 0
                continue
            end

            p = permi[i, jsec, k]
            if !isfinite(p) || p <= 0
                continue
            end

            push!(polys, cell_polygon_IK(coord, zcorn, i, jsec, k, side=side))
            push!(vals, logscale ? log10(p) : p)
        end
    end

    collections = PyPlot.matplotlib[:collections]

    fig, ax = subplots(figsize=(12, 6))
    pc = collections[:PolyCollection](
        polys,
        array=vals,
        cmap=get_cmap(cmap_name),
        edgecolors="none",
        linewidths=0.0
    )

    ax.add_collection(pc)
    ax.autoscale_view()
    ax.invert_yaxis()
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("PERMI I-K section at J = $jsec")
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label(logscale ? "log10(PERMI)" : "PERMI")
    fig.tight_layout()

    savefig(outfile, dpi=300, bbox_inches="tight")
    println("Saved $outfile")
end

# ============================================================
# Plot J-K section
# No grid lines
# ============================================================
function plot_perm_JK(coord, zcorn, inactive, permi; isec::Int,
                      side=:left, logscale=false,
                      cmap_name="viridis", outfile="permi_JK.png")

    polys = Vector{Vector{Tuple{Float64,Float64}}}()
    vals  = Float64[]

    for k in 1:NK
        for j in 1:NJ
            if inactive[isec, j, k] == 0
                continue
            end

            p = permi[isec, j, k]
            if !isfinite(p) || p <= 0
                continue
            end

            push!(polys, cell_polygon_JK(coord, zcorn, isec, j, k, side=side))
            push!(vals, logscale ? log10(p) : p)
        end
    end

    collections = PyPlot.matplotlib[:collections]

    fig, ax = subplots(figsize=(12, 6))
    pc = collections[:PolyCollection](
        polys,
        array=vals,
        cmap=get_cmap(cmap_name),
        edgecolors="none",
        linewidths=0.0
    )

    ax.add_collection(pc)
    ax.autoscale_view()
    ax.invert_yaxis()
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.set_title("PERMI J-K section at I = $isec")
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label(logscale ? "log10(PERMI)" : "PERMI")
    fig.tight_layout()

    savefig(outfile, dpi=300, bbox_inches="tight")
    println("Saved $outfile")
end

# ============================================================
# Example
# ============================================================
plot_perm_IK(coord, zcorn, inactive, permi; jsec=59, side=:front,
             outfile="permi_IK_J59_fixed.png")

plot_perm_JK(coord, zcorn, inactive, permi; isec=32, side=:left,
             outfile="permi_JK_I32_fixed.png")