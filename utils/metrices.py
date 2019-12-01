import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from skimage.morphology import binary_dilation, disk

__all__ = ['get_f1_scores', 'get_ap_scores', 'get_iou', 'WCov_metric', 'FBound_metric']

SMOOTH = 1e-6


def get_f1_scores(predict, target, ignore_index=-1):
    f1 = []
    for pred, tgt in zip(predict, target):
        # Tensor process
        pred = pred.data.cpu().numpy().reshape(-1)
        tgt = tgt.data.cpu().numpy().reshape(-1)
        p = pred[tgt != ignore_index]
        t = tgt[tgt != ignore_index]
        f1.append(f1_score(t, p))

    return f1


def get_ap_scores(predict, target, ignore_index=-1):
    ap = []
    for pred, tgt in zip(predict, target):
        # Tensor process
        pred = pred.data.cpu().numpy().reshape(-1)
        tgt = tgt.data.cpu().numpy().reshape(-1)
        p = pred[tgt != ignore_index]
        t = tgt[tgt != ignore_index]

        ap.append(average_precision_score(t, p))

    return ap


def get_iou(outputs, labels):
    outputs = outputs.squeeze(1)

    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.cpu().tolist()


def WCov_metric(X, Y):
    A1 = float(np.count_nonzero(X))
    A2 = float(np.count_nonzero(Y))
    if A1 >= A2: return A2 / A1
    if A2 > A1: return A1 / A2


def FBound_metric(X, Y):
    tmp1 = db_eval_boundary(X, Y, 1)[0]
    tmp2 = db_eval_boundary(X, Y, 2)[0]
    tmp3 = db_eval_boundary(X, Y, 3)[0]
    tmp4 = db_eval_boundary(X, Y, 4)[0]
    tmp5 = db_eval_boundary(X, Y, 5)[0]
    return (tmp1 + tmp2 + tmp3 + tmp4 + tmp5) / 5.0


def db_eval_boundary(foreground_mask, gt_mask, bound_th):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask);
    gt_boundary = seg2bmap(gt_mask);

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall);

    return F, precision, recall, np.sum(fg_match), n_fg, np.sum(gt_match), n_gt


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width     : Width of desired bmap  <= seg.shape[1]
        height  :   Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray): Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """
    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), \
        'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + np.floor((y - 1) + height / h)
                    i = 1 + np.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap
