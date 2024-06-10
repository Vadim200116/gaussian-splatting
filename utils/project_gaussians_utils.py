from gsplat._torch_impl import scale_rot_to_cov3d, get_tile_bbox, project_cov3d_ewa, compute_cov2d_bounds
import torch

def transformPoint4x4(p, matrix):
    row = row = torch.ones(len(p),1).cuda()
    tmp = torch.cat((p, row), dim=1)
    transformed = tmp @ matrix

    return transformed

def in_frustrum(p, view_mtx):
    p_view = transformPoint4x4(p, view_mtx)[:3]
    if p_view[2] <= 0.2:
        return False

    return True

def ndc2Pix(v,s):
    return ((v + 1.0) * s - 1.0) * 0.5

def clip_frustrum(points, view_mtx):
    points_view = transformPoint4x4(points, view_mtx)

    return points_view, points_view[:,2]>=0.2

def project_gaussians(
        means3D,
        scales, 
        rotations,  
        view_mtx, 
        proj_mtx, 
        W, 
        H, 
        block_width, 
        scale, 
        tan_fovx,
        tan_fovy,
        fx, 
        fy,
    ):

    P = len(means3D)
    p_view, frustrum_mask = clip_frustrum(means3D, view_mtx)

    depths = p_view[..., 2]
    tile_bounds = (
        (W + block_width - 1) // block_width,
        (H + block_width - 1) // block_width,
        1,
    )
    cov3d = scale_rot_to_cov3d(scales, scale, rotations)
    cov2d, compensation = project_cov3d_ewa(
        means3D, cov3d, view_mtx, fx, fy, tan_fovx, tan_fovy
    )

    conic, radius, det_valid = compute_cov2d_bounds(cov2d)

    means3d_hom = transformPoint4x4(means3D, proj_mtx)
    eps = 1e-6
    p_w = 1 / (means3d_hom[:, 3] + eps)

    xys = torch.zeros((P,2)).cuda()
    xys[:,0] = ndc2Pix(means3d_hom[:,0] * p_w, W)
    xys[:,1] = ndc2Pix(means3d_hom[:,1] * p_w, H)

    tile_min, tile_max = get_tile_bbox(xys, radius, tile_bounds, block_width)
    tile_area = (tile_max[..., 0] - tile_min[..., 0]) * (
        tile_max[..., 1] - tile_min[..., 1]
    )

    mask = (tile_area > 0) & (frustrum_mask) & det_valid
    xys = torch.where(~mask[..., None], -1000, xys)
    depths = torch.where(~mask, 0, depths)
    
    return xys, depths, mask

def apply_sam_mask(view, means2D):
    width = view.image_width
    height = view.image_height
    human_mask = torch.zeros((means2D.shape[0]), device = means2D.device, dtype=torch.bool)        
    int_means = means2D.to(torch.int64)
    w, h = int_means[:,0], int_means[:,1]
    m1 = (w>=0) * (w<width) * (h>=0) * (h<height)
    m_means = int_means[m1]
    sam_mask = view.mask.cuda()
    m2 = sam_mask[m_means[:,1], m_means[:,0]] == 0
    human_mask[m1] = m2
    return human_mask