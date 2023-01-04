using ADI
using AstroImages
using SAOImageDS9


datadir(args...) = abspath(joinpath("/Volumes/mlucas SSD1/2023/20230103/20230103_HD34700_processed/", args...))
cubes = (
    AstroImage(datadir("rescaled", "20230103_HD34700_cam1_collapsed_scaled_cube.fits")),
    AstroImage(datadir("rescaled", "20230103_HD34700_cam2_collapsed_scaled_cube.fits")),
)
angs = (
    AstroImage(datadir("rescaled", "20230103_HD34700_cam1_collapsed_scaled_derot_angles.fits")),
    AstroImage(datadir("rescaled", "20230103_HD34700_cam2_collapsed_scaled_derot_angles.fits")),
)


algs = (GreeDS(10),)

resids = map(enumerate(zip(cubes, angs))) do (i, (cube, pa))
    T = eltype(cube)
    clean_cube = @. ifelse(isnan(cube), zero(T), cube)
    @info "Reducing cam $i cube"
    resid_cubes = mapreduce(alg -> alg(Array(clean_cube), pa), (a, b) -> [a ;;; b], algs)
end

combined = sum(resids)

DS9.connect()
DS9.set("frame last")
DS9.set(combined)

PI = AstroImage(datadir("20230103_HD34700_stokes_cube_ip.fits"))[:, :, 6]
T = eltype(PI)
PI_clean = @. ifelse(isnan(PI), zero(T), PI)

algs2 = PCA.((5, 10, 20))
resids2 = map(enumerate(zip(cubes, angs))) do (i, (cube, pa))
    T = eltype(cube)
    clean_cube = @. ifelse(isnan(cube), zero(T), cube)
    ref = clean_cube .- PI_clean
    @info "Reducing cam $i cube"
    resid_cubes = mapreduce(alg -> alg(Array(clean_cube), pa; ref), (a, b) -> [a ;;; b], algs2)
end

combined2 = sum(resids2)

DS9.connect()
DS9.set("frame last")
DS9.set(combined2)


algs3 = (GreeDS(10),)
resids3 = map(enumerate(zip(cubes, angs))) do (i, (cube, pa))
    T = eltype(cube)
    clean_cube = @. ifelse(isnan(cube), zero(T), cube)
    ref = clean_cube .- PI_clean
    @info "Reducing cam $i cube"
    resid_cubes = mapreduce(alg -> alg(Array(clean_cube), pa; ref), (a, b) -> [a ;;; b], algs3)
end

combined3 = sum(resids3)

DS9.connect()
DS9.set("frame last")
DS9.set(combined3)