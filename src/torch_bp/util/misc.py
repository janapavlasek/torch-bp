import torch


def bilinear_interpolation(x_grid, y_grid, values, x, y):
    """
    Perform bilinear interpolation to estimate the value at point (x, y) within the grid.

    Parameters:
        x_grid (torch.Tensor): 1D tensor containing the x-coordinates of grid points.
        y_grid (torch.Tensor): 1D tensor containing the y-coordinates of grid points.
        values (torch.Tensor): 2D tensor containing the values at each grid point.
        x (float): The x-coordinate of the point to interpolate.
        y (float): The y-coordinate of the point to interpolate.

    Returns:
        torch.Tensor: The interpolated value at point (x, y).
    """
    x_idx = torch.searchsorted(x_grid.contiguous(), x.contiguous(), right=True) - 1
    y_idx = torch.searchsorted(y_grid.contiguous(), y.contiguous(), right=True) - 1

    x_idx0, x_idx1 = x_idx, x_idx + 1
    y_idx0, y_idx1 = y_idx, y_idx + 1

    # Clip the indices to stay within the grid boundaries
    x_idx0 = torch.clamp(x_idx0, 0, len(x_grid) - 1)
    x_idx1 = torch.clamp(x_idx1, 0, len(x_grid) - 1)
    y_idx0 = torch.clamp(y_idx0, 0, len(y_grid) - 1)
    y_idx1 = torch.clamp(y_idx1, 0, len(y_grid) - 1)

    # The actual positions of these indices.
    x0, x1 = x_grid[x_idx0], x_grid[x_idx1]
    y0, y1 = y_grid[y_idx0], y_grid[y_idx1]

    # Get the four nearest grid points' values
    q11 = values[y_idx0, x_idx0]
    q12 = values[y_idx1, x_idx0]
    q21 = values[y_idx0, x_idx1]
    q22 = values[y_idx1, x_idx1]

    # Perform bilinear interpolation
    interpolated_value = (
        q11 * (x1 - x) * (y1 - y) +
        q21 * (x - x0) * (y1 - y) +
        q12 * (x1 - x) * (y - y0) +
        q22 * (x - x0) * (y - y0)
    ) / ((x1 - x0) * (y1 - y0))

    # Deal with divide by zero.
    x_edge, y_edge = x_idx0 == x_idx1, y_idx0 == y_idx1
    corner = x_edge * y_edge
    if len((y_edge).nonzero()) > 0:
        int_x = ((x1 - x) * q11 + (x - x0) * q21) / (x1 - x0)
        interpolated_value[y_edge] = int_x[y_edge]
    if len((x_edge).nonzero() > 0):
        int_y = ((y1 - y) * q11 + (y - y0) * q21) / (y1 - y0)
        interpolated_value[x_edge] = int_y[x_edge]
    if len(corner.nonzero()) > 0:
        interpolated_value[corner] = values[y_idx0, x_idx0][corner]

    return interpolated_value
