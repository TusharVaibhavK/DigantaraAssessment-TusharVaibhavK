"""Tile grid for exact 1024 x 1024 tiles without padding or resampling (Q2 b).

Origins run 0, T, 2T, ... and the last tile is shifted back so it ends exactly on the image edge:
    9568 px wide  -> x0 = 0, 1024, ..., 8192, 8544   (10 columns, last overlaps by 672 px)
    6380 px high  -> y0 = 0, 1024, ..., 5120, 5356   (7 rows,    last overlaps by 788 px)
Every pixel is covered at least once, every tile is full size, no synthetic (padded) pixels exist.
"""


def tile_origins(length, tile=1024):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, tile))
    return starts + [length - tile]


def tile_grid(height, width, tile=1024):
    """List of (row, col, y0, x0)."""
    return [(r, c, y0, x0)
            for r, y0 in enumerate(tile_origins(height, tile))
            for c, x0 in enumerate(tile_origins(width, tile))]
