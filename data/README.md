# `data/` — expected directory layout (empty skeleton)

This mirrors the structure `src/data_loader.py` and `src/inference.py` expect, with no actual raster data included (see the top-level [README.md](../README.md#input-data-format) for full field descriptions). Populate it with your own GeoTIFFs, keeping this same layout:

```
data/
├── parms_bands/            ← --static_dir
│   ├── dem.tif
│   ├── manning_coef.tif
│   ├── pervious_cover.tif
│   ├── slope.tif
│   └── land_mask.tif
└── samples/                ← --samples_dir
    └── sampleN/             (one directory per simulation, e.g. sample1, sample2, …)
        ├── depth_timesteps/   depth_hr0000.00.tif … depth_hr????.00.tif
        └── pcpout_timesteps/  pcpout_hr0000.00.tif … pcpout_hr????.00.tif
```
