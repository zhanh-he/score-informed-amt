import torch
import torch.nn.functional as F
VELOCITY_SCALE = 128.0


def _get_velocity_target(target_dict):
    """Get normalized velocity target with fixed scale for compatibility."""
    return target_dict["velocity_roll"] / VELOCITY_SCALE


def _get_velocity_pred(output_dict):
    """Fetch velocity prediction tensor, accepting both vel_corr and velocity_output keys."""
    if "vel_corr" in output_dict:
        return output_dict["vel_corr"]
    if "velocity_output" in output_dict:
        return output_dict["velocity_output"]
    raise KeyError("velocity prediction not found in output_dict (expected vel_corr or velocity_output)")


def _align_time_dim(*tensors):
    """Trim all tensors along time dimension to the minimum shared length."""
    if not tensors:
        return tensors
    min_steps = min(tensor.size(1) for tensor in tensors)
    if all(tensor.size(1) == min_steps for tensor in tensors):
        return tensors
    return tuple(tensor[:, :min_steps] for tensor in tensors)


def _masked_mean(values, mask):
    """Compute masked mean with time alignment and safe denominator."""
    values, mask = _align_time_dim(values, mask)
    mask = mask.to(values.dtype)
    denom = torch.sum(mask).clamp_min(1e-8)
    return torch.sum(values * mask) / denom


def _masked_bce(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be 
    deactivated when calculation BCE."""
    output, target, mask = _align_time_dim(output, target, mask)
    output = torch.clamp(output, 1e-7, 1.0 - 1e-7)
    matrix = F.binary_cross_entropy(output, target, reduction="none")
    return _masked_mean(matrix, mask)


def _masked_mse(output, target, mask):
    """Mean squared error (MSE) with mask"""
    output, target, mask = _align_time_dim(output, target, mask)
    return _masked_mean((output - target) ** 2, mask)


def _masked_l1(output, target, mask):
    """Mean absolute error restricted by mask."""
    output, target, mask = _align_time_dim(output, target, mask)
    return _masked_mean(torch.abs(output - target), mask)


############ Velocity loss ############

def _velocity_pointwise_loss(output_dict, target_dict, mask_key, pointwise_loss):
    pred = _get_velocity_pred(output_dict)
    target = _get_velocity_target(target_dict)
    return pointwise_loss(pred, target, target_dict[mask_key])


def velocity_bce(cfg, output_dict, target_dict, cond_dict=None):
    """velocity regression losses only, used bce in HPT"""
    return _velocity_pointwise_loss(output_dict, target_dict, "onset_roll", _masked_bce)


def velocity_mse(cfg, output_dict, target_dict, cond_dict=None):
    """velocity regression losses only, used mse in ONF"""
    return _velocity_pointwise_loss(output_dict, target_dict, "onset_roll", _masked_mse)


def kim_velocity_bce_l1(cfg, output_dict, target_dict, cond_dict=None):
    """
    BCE + L1 hybrid loss proposed by Kim et al. (ISMIR 2024) for velocity regression.
    """
    theta = cfg.loss.kim_loss_alpha  # default is 0.5 in config
    pred = _get_velocity_pred(output_dict)
    onset_target = _get_velocity_target(target_dict)
    bce_loss = _masked_bce(pred, onset_target, target_dict['frame_roll'])
    l1_loss = _masked_l1(pred, onset_target, target_dict['onset_roll'])
    return theta * bce_loss + (1 - theta) * l1_loss


def get_loss_func(cfg, loss_type=None):
    """
    Return a callable with unified signature:
      fn(cfg, output_dict, target_dict, cond_dict=None) -> loss

    Selection order:
    - explicit loss_type if provided
    - cfg.loss.loss_type
    """
    selected_loss_type = loss_type if loss_type is not None else cfg.loss.loss_type
    loss_map = {
        "velocity_bce": velocity_bce,
        "velocity_mse": velocity_mse,
        "kim_bce_l1":   kim_velocity_bce_l1,
    }
    if selected_loss_type in loss_map:
        return loss_map[selected_loss_type]

    raise ValueError(f"Incorrect loss_type: {selected_loss_type!r}")