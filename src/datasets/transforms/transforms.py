from typing import List, Tuple, Sequence, Union, Optional, Any

from easydict import EasyDict as edict

import math
import numpy as np
import random
from numbers import Number

import torch

def _pair(
    x : Union[Sequence[Number], Number],
    negate: bool = False,
) -> Sequence[Number] :

    if isinstance(x, Number) :
        if negate : 
            x = abs(x)
            return [-x, x]
        return [x, x]
    return x

def _deg2rad(
    x : Sequence[Number],
) -> Sequence[Number] :

    return [d * math.pi / 180 for d in x]

def _is_numpy(x: Any) -> bool :
    return isinstance(x, np.ndarray)

def _is_torch(x: Any) -> bool :
    return torch.is_tensor(x)


class Compose(object) :
    """Composes several transforms together for multiple input arrays at once.
    If you need a detailed documentation, feel free to read from torchvision transforms.
    """

    def __init__(self, transforms: Sequence[Any]):
        self.transforms = transforms

    def __call__(self, 
        xs: Union[
                np.ndarray, 
                torch.Tensor, 
                Sequence[np.ndarray], 
                Sequence[torch.Tensor],
        ],
    ) -> Union[
            np.ndarray, 
            torch.Tensor, 
            Sequence[np.ndarray], 
            Sequence[torch.Tensor],
    ] :
    
        for t in self.transforms :
            xs = t(xs)

        return xs


class RandomScale(object) :
    def __init__(self, 
        lim: Sequence[float] ,
    ) -> None :

        self.lim = lim

    def __call__(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :
    
        assert _is_numpy(pts)
        factor = np.random.uniform(*self.lim)
        pts[..., :3] *= factor
        return pts


class RandomNoise(object) :
    def __init__(self, 
        lim: float,
        rm_global_scale: bool = False,
    ) -> None :

        self.lim = _pair(lim, negate=True)
        self.rm_global_scale = rm_global_scale

    def __call__(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :

        assert _is_numpy(pts)
        # select which values to perturb
        xyz = pts[..., :3]

        if self.rm_global_scale :
            ds = (xyz[:, -1, :] - xyz[:, -2, :])
            ds = np.min(np.sqrt(np.sum(np.power(ds, 2), axis=1)))
            lim = [x * ds for x in self.lim]
        else :
            lim = self.lim

        shape_ = xyz.shape
        xyz = xyz.flatten()
        n_perturb = np.random.randint(1, xyz.size+1)
        mask = np.random.permutation(xyz.size)[: n_perturb]        
        xyz[mask] += np.random.uniform(*lim, n_perturb)
        pts[..., :3] = xyz.reshape(shape_)
        return pts


class RandomTranslation(object) :
    def __init__(self, 
        xlim: Union[Sequence[float], float], 
        ylim: Union[Sequence[float], float], 
        zlim: Union[Sequence[float], float], 
    ) -> None :

        self.xlim = _pair(xlim, negate=True)
        self.ylim = _pair(ylim, negate=True)
        self.zlim = _pair(zlim, negate=True)

    def __call__(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :

        assert _is_numpy(pts)
        x_tr = random.uniform(*self.xlim)
        y_tr = random.uniform(*self.ylim)
        z_tr = random.uniform(*self.zlim)         

        pts[..., :3] += [x_tr, y_tr, z_tr]

        return pts


class RandomRotation(object) :
    def __init__(self, 
        xlim: Union[Sequence[float], float] = 5.0, 
        ylim: Union[Sequence[float], float] = 5.0, 
        zlim: Union[Sequence[float], float] = 5.0,
    ) -> None :

        self.xlim = _pair(xlim, negate=True)
        self.ylim = _pair(ylim, negate=True)
        self.zlim = _pair(zlim, negate=True)

        self.xlim = _deg2rad(self.xlim)
        self.ylim = _deg2rad(self.ylim)
        self.zlim = _deg2rad(self.zlim)
        

    def __call__(self, pts: np.ndarray) -> np.ndarray :
        assert _is_numpy(pts)

        if random.random() < 0.5 :
            return pts

        x = random.uniform(*self.xlim)
        y = random.uniform(*self.ylim)
        z = random.uniform(*self.zlim)

        sx, cx = math.sin(x), math.cos(x)
        sy, cy = math.sin(y), math.cos(y)
        sz, cz = math.sin(z), math.cos(z)

        R = np.array([ [cz*cy, sz*cy, -sy],
                       [cz*sy*sx-sz*cx, sz*sy*sx+cz*cx, cy*sx],
                       [cz*sy*cx+sz*sx, sz*sy*cx-cz*sx, cy*cx],
                    ], dtype=np.float32)

        pts[..., :3] = np.matmul(pts[..., :3], R)
        return pts


class RandomTimeInterpolation(object) :
    def __init__(self, 
        prob: float ,
    ) -> None :

        assert 0 < prob <= 1, \
            f"Interpolation probability must be in (0, 1], got {prob}"
        self.prob = prob


    def __call__(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :

        assert _is_numpy(pts)

        if random.random() > self.prob :
            return pts

        t_shift = np.random.uniform(0, 1)
        dir_ = (np.roll(pts, -1, axis=0) - pts)
        
        pts = pts + t_shift * dir_
        pts = pts[:-1] # drop the last invalid frame
        return pts


class StratifiedSample(object) :
    def __init__(self, 
        n_samples: int ,
    ) -> None :
 
        assert 0 < n_samples, f"#Frames to sample must be > 0, got {n_samples}"
        self.n_samples = n_samples

    
    def __call__(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :

        assert _is_numpy(pts)

        n_frames = pts.shape[0]
        if self.n_samples == 1 :
            i = np.random.randint(n_frames)
            return pts[i:i+1]

        if n_frames == self.n_samples :
            return pts
        
        if n_frames < self.n_samples :
            shape_ = pts.shape
            # repeat last frame
            pts = np.concatenate(
                    (pts, 
                    np.repeat(pts[-1:], self.n_samples - n_frames, 0),
                    ), axis=0
            )

            return pts

        mask = np.floor(np.linspace(0, n_frames-1, self.n_samples)).astype(np.int32)
        return pts[mask]


class CenterByIndex(object) :
    def __init__(self, 
        ind: int,
        n_hands: int = 1,
    ) -> None :

        self.ind = ind
        self.n_hands = n_hands

    def center_single(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :

        n_ind = pts.shape[1]
        assert self.ind < n_ind, \
            f"Index {self.ind} must be in [0, {n_ind-1})."

        center_ = pts[0:1, self.ind:self.ind+1]
        pts -= center_

        return pts

    def __call__(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :

        assert _is_numpy(pts)

        if self.n_hands == 1 :
            return self.center_single(pts)
        
        pts_left = pts[:, :21, :]
        pts_right = pts[:, 21:, :]

        is_present_left = not np.allclose(pts_left, 0)
        is_present_right = not np.allclose(pts_right, 0)
        if is_present_left and is_present_right :
            orientation = 'b'
        elif is_present_left :
            orientation = 'l'
        else :
            orientation = 'r'

        if orientation == 'b' :
            return self.center_single(pts)
        
        if orientation == 'l' :
            pts_left = self.center_single(pts_left)
        else :
            pts_right = self.center_single(pts_right)
        
        pts = np.concatenate((pts_left, pts_right), axis=1)
        return pts


class ToTensor(object) :
    def __init__(self) :
        pass
    
    def __call__(self, 
        pts: np.ndarray,
    ) ->  torch.Tensor :
        
        assert _is_numpy(pts)
        return torch.from_numpy(pts).to(torch.get_default_dtype())
    


class PoseDecode:
    """Load and decode pose with given indices.

    Required keys are "keypoint", "frame_inds" (optional), "keypoint_score" (optional), added or modified keys are
    "keypoint", "keypoint_score" (if applicable).
    """

    @staticmethod
    def _load_kp(kp, frame_inds):
        return kp[:, frame_inds].astype(np.float32)

    @staticmethod
    def _load_kpscore(kpscore, frame_inds):
        return kpscore[:, frame_inds].astype(np.float32)

    def __call__(self, results):

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            results['keypoint_score'] = self._load_kpscore(results['keypoint_score'], frame_inds)

        if 'keypoint' in results:
            results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str




class RandomRot:

    def __init__(self, theta=0.3):
        self.theta = theta

    def _rot3d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
        ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
        rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])

        rot = np.matmul(rz, np.matmul(ry, rx))
        return rot

    def _rot2d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        return np.array([[cos, -sin], [sin, cos]])

    def __call__(self, results):
        skeleton = results['keypoint']
        M, T, V, C = skeleton.shape

        if np.all(np.isclose(skeleton, 0)):
            return results

        assert C in [2, 3]
        if C == 3:
            theta = np.random.uniform(-self.theta, self.theta, size=3)
            rot_mat = self._rot3d(theta)
        elif C == 2:
            theta = np.random.uniform(-self.theta)
            rot_mat = self._rot2d(theta)
        results['keypoint'] = np.einsum('ab,mtvb->mtva', rot_mat, skeleton)

        return results


# class RandomScale:

#     def __init__(self, scale=0.2):
#         assert isinstance(scale, tuple) or isinstance(scale, float)
#         self.scale = scale

#     def __call__(self, results):
#         skeleton = results['keypoint']
#         scale = self.scale
#         if isinstance(scale, float):
#             scale = (scale, ) * skeleton.shape[-1]
#         assert len(scale) == skeleton.shape[-1]
#         scale = 1 + np.random.uniform(-1, 1, size=len(scale)) * np.array(scale)
#         results['keypoint'] = skeleton * scale
#         return results


class RandomGaussianNoise:

    def __init__(self, sigma=0.01, base='frame', shared=False):
        assert isinstance(sigma, float)
        self.sigma = sigma
        self.base = base
        self.shared = shared
        assert self.base in ['frame', 'video']
        if self.base == 'frame':
            assert not self.shared

    def __call__(self, results):
        skeleton = results['keypoint']
        M, T, V, C = skeleton.shape
        skeleton = skeleton.reshape(-1, V, C)
        ske_min, ske_max = skeleton.min(axis=1), skeleton.max(axis=1)
        # MT * C
        flag = ((ske_min ** 2).sum(axis=1) > EPS)
        # MT
        if self.base == 'frame':
            norm = np.linalg.norm(ske_max - ske_min, axis=1) * flag
            # MT
        elif self.base == 'video':
            assert np.sum(flag)
            ske_min, ske_max = ske_min[flag].min(axis=0), ske_max[flag].max(axis=0)
            # C
            norm = np.linalg.norm(ske_max - ske_min)
            norm = np.array([norm] * (M * T)) * flag
        # MT * V
        if self.shared:
            noise = np.random.randn(V) * self.sigma
            noise = np.stack([noise] * (M * T))
            noise = (noise.T * norm).T
            random_vec = np.random.uniform(-1, 1, size=(C, V))
            random_vec = random_vec / np.linalg.norm(random_vec, axis=0)
            random_vec = np.concatenate([random_vec] * (M * T), axis=-1)
        else:
            noise = np.random.randn(M * T, V) * self.sigma
            noise = (noise.T * norm).T
            random_vec = np.random.uniform(-1, 1, size=(C, M * T * V))
            random_vec = random_vec / np.linalg.norm(random_vec, axis=0)
            # C * MTV
        random_vec = random_vec * noise.reshape(-1)
        # C * MTV
        random_vec = (random_vec.T).reshape(M, T, V, C)
        results['keypoint'] = skeleton.reshape(M, T, V, C) + random_vec
        return results



class PreNormalize3D(object) :
    """PreNormalize for NTURGB+D 3D keypoints (x, y, z). Codes adapted from https://github.com/lshiwjx/2s-AGCN. """

    def __init__(self, 
            zaxis: Sequence[float] =[0, 1] , 
            xaxis: Sequence[float] =[8, 4] , 
            align_spine: bool =True , 
            align_center: bool =True) -> None:
    
        self.zaxis = zaxis
        self.xaxis = xaxis
        self.align_spine = align_spine
        self.align_center = align_center

    def unit_vector(self, vector):
        """Returns the unit vector of the vector. """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'. """
        if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
            return 0
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def rotation_matrix(self, axis, theta):
        """Return the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians."""
        if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
            return np.eye(3)
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



    def __call__(self, results):
        skeleton = results['keypoint']
        total_frames = results.get('total_frames', skeleton.shape[1])

        M, T, V, C = skeleton.shape
        assert T == total_frames
        if skeleton.sum() == 0:
            return results

        index0 = [i for i in range(T) if not np.all(np.isclose(skeleton[0, i], 0))]

        assert M in [1, 2]
        if M == 2:
            index1 = [i for i in range(T) if not np.all(np.isclose(skeleton[1, i], 0))]
            if len(index0) < len(index1):
                skeleton = skeleton[:, np.array(index1)]
                skeleton = skeleton[[1, 0]]
            else:
                skeleton = skeleton[:, np.array(index0)]
        else:
            skeleton = skeleton[:, np.array(index0)]

        T_new = skeleton.shape[1]

        if self.align_center:
            if skeleton.shape[2] == 25:
                main_body_center = skeleton[0, 0, 1].copy()
            else:
                main_body_center = skeleton[0, 0, -1].copy()
            mask = ((skeleton != 0).sum(-1) > 0)[..., None]
            skeleton = (skeleton - main_body_center) * mask

        if self.align_spine:
            joint_bottom = skeleton[0, 0, self.zaxis[0]]
            joint_top = skeleton[0, 0, self.zaxis[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = self.angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_z)

            joint_rshoulder = skeleton[0, 0, self.xaxis[0]]
            joint_lshoulder = skeleton[0, 0, self.xaxis[1]]
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = self.angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_x)

        results['keypoint'] = skeleton
        results['total_frames'] = T_new
        results['body_center'] = main_body_center
        return results


class JointToBone:

    def __init__(self, dataset='nturgb+d', target='keypoint'):
        self.dataset = dataset
        self.target = target
        if self.dataset not in ['nturgb+d', 'openpose', 'coco', 'handmp']:
            raise ValueError(
                f'The dataset type {self.dataset} is not supported')
        if self.dataset == 'nturgb+d':
            self.pairs = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                          (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
                          (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11))
        elif self.dataset == 'openpose':
            self.pairs = ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
                          (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15))
        elif self.dataset == 'coco':
            self.pairs = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
                          (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))
        elif self.dataset == 'handmp':
            self.pairs = ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7), (9, 0), (10, 9),
                          (11, 10), (12, 11), (13, 0), (14, 13), (15, 14), (16, 15), (17, 0), (18, 17), (19, 18),
                          (20, 19))

    def __call__(self, results):

        keypoint = results['keypoint']
        M, T, V, C = keypoint.shape
        bone = np.zeros((M, T, V, C), dtype=np.float32)

        assert C in [2, 3]
        for v1, v2 in self.pairs:
            bone[..., v1, :] = keypoint[..., v1, :] - keypoint[..., v2, :]
            if C == 3 and self.dataset in ['openpose', 'coco', 'handmp']:
                score = (keypoint[..., v1, 2] + keypoint[..., v2, 2]) / 2
                bone[..., v1, 2] = score

        results[self.target] = bone
        return results


class ToMotion:

    def __init__(self, dataset='nturgb+d', source='keypoint', target='motion'):
        self.dataset = dataset
        self.source = source
        self.target = target

    def __call__(self, results):
        data = results[self.source]
        M, T, V, C = data.shape
        motion = np.zeros_like(data)

        assert C in [2, 3]
        motion[:, :T - 1] = np.diff(data, axis=1)
        if C == 3 and self.dataset in ['openpose', 'coco']:
            score = (data[:, :T - 1, :, 2] + data[:, 1:, :, 2]) / 2
            motion[:, :T - 1, :, 2] = score

        results[self.target] = motion

        return results


class Rename:
    """Rename the key in results.

    Args:
        mapping (dict): The keys in results that need to be renamed. The key of
            the dict is the original name, while the value is the new name. If
            the original name not found in results, do nothing.
            Default: dict().
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, results):
        for key, value in self.mapping.items():
            if key in results:
                assert isinstance(key, str) and isinstance(value, str)
                assert value not in results, ('the new name already exists in '
                                              'results')
                results[value] = results[key]
                results.pop(key)
        return results

class MergeSkeFeat:
    def __init__(self, feat_list=['keypoint'], target='keypoint', axis=-1):
        """Merge different feats (ndarray) by concatenate them in the last axis. """

        self.feat_list = feat_list
        self.target = target
        self.axis = axis

    def __call__(self, results):
        feats = []
        for name in self.feat_list:
            feats.append(results.pop(name))
        feats = np.concatenate(feats, axis=self.axis)
        results[self.target] = feats
        return results

class GenSkeFeat:
    def __init__(self, 
             dataset='nturgb+d', feats=['j'], axis=-1):
        self.dataset = dataset
        self.feats = feats
        self.axis = axis
        ops = []
        if 'b' in feats or 'bm' in feats:
            ops.append(JointToBone(dataset=dataset, target='b'))
        ops.append(Rename({'keypoint': 'j'}))
        if 'jm' in feats:
            ops.append(ToMotion(dataset=dataset, source='j', target='jm'))
        if 'bm' in feats:
            ops.append(ToMotion(dataset=dataset, source='b', target='bm'))
        ops.append(MergeSkeFeat(feat_list=feats, axis=axis))
        self.ops = Compose(ops)

    def __call__(self, results):
        if 'keypoint_score' in results and 'keypoint' in results:
            assert self.dataset != 'nturgb+d'
            assert results['keypoint'].shape[-1] == 2, 'Only 2D keypoints have keypoint_score. '
            keypoint = results.pop('keypoint')
            keypoint_score = results.pop('keypoint_score')
            results['keypoint'] = np.concatenate([keypoint, keypoint_score[..., None]], -1)
        return self.ops(results)



class FormatGCNInput:
    """Format final skeleton shape to the given input_format. """

    def __init__(self, num_person=2, mode='zero'):
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results['keypoint']
        if 'keypoint_score' in results:
            keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)

        # M T V C
        if keypoint.shape[0] < self.num_person:
            pad_dim = self.num_person - keypoint.shape[0]
            pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == 'loop' and keypoint.shape[0] == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

        elif keypoint.shape[0] > self.num_person:
            keypoint = keypoint[:self.num_person]

        M, T, V, C = keypoint.shape
        nc = results.get('num_clips', 1)
        assert T % nc == 0
        keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)
        results['keypoint'] = np.ascontiguousarray(keypoint)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(num_person={self.num_person}, mode={self.mode})'
        return repr_str


class UniformSampleFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        seed (int): The random seed used during test time. Default: 255.
    """

    def __init__(self,
                 clip_len,
                 num_clips=1,
                 p_interval=1,
                 seed=255,
                 **deprecated_kwargs):
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed
        self.p_interval = p_interval
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)
        # if len(deprecated_kwargs):
        #     warning_r0('[UniformSampleFrames] The following args has been deprecated: ')
        #     for k, v in deprecated_kwargs.items():
        #         warning_r0(f'Arg name: {k}; Arg value: {v}')

    def _get_train_clips(self, num_frames, clip_len):
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        allinds = []
        for clip_idx in range(self.num_clips):
            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            inds = inds + off
            num_frames = old_num_frames

            allinds.append(inds)

        return np.concatenate(allinds)

    def _get_test_clips(self, num_frames, clip_len):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        np.random.seed(self.seed)

        all_inds = []

        for i in range(self.num_clips):

            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start_ind = i if num_frames < self.num_clips else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds + off)
            num_frames = old_num_frames

        return np.concatenate(all_inds)

    def __call__(self, results):
        num_frames = results['total_frames']

        if results.get('test_mode', False):
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results.get('start_index', 0)
        inds = inds + start_index

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(int)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

        results['frame_inds'] = inds.astype(int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'seed={self.seed})')
        return repr_str