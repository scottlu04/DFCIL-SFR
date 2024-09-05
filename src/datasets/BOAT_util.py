import sys
import os, os.path as osp
import shutil
import time
from tqdm import tqdm

import numpy as np 
import random
import torch

from typing import IO

from svd import get_eig_vecs
# from .classifier import *
# from .stdio import *

def _normalize(x: np.ndarray) -> np.ndarray :
    return x / np.linalg.norm(x);

def _assert_unique(x: np.ndarray) -> None :
    from scipy.spatial import distance_matrix
    d = distance_matrix(x, x);
    assert d.size - np.count_nonzero(d) == x.shape[0];    

def _clamp_sample_size(
    xs: np.ndarray,
    max_samples: int,
) -> np.ndarray :

    n_samples = xs.shape[0];
    if n_samples <= max_samples :
        return xs;

    i_select = np.random.RandomState(seed=0).permutation(n_samples)[:max_samples];
    xs = xs[i_select];
    return xs;

def get_elliptical_samples(
    proto_class_var: np.ndarray, 
    var_exp: float,
    order: int,
    max_samples: int,
) -> np.ndarray :

    eig_vecs, _, n_keep, eig_va = get_eig_vecs(proto_class_var, var_exp); 
    #Lambda, U = np.linalg.eig(proto_class_var)
    # inv_U = np.linalg.inv(U)
    # #print(eig_va)
    #vec = np.dot(U,np.dot(np.diag(Lambda), inv_U))
    # print(vec)
    #print(proto_class_var)
    #print(eig_vecs.shape)
    U = eig_vecs[:, :n_keep]
    inv_U = np.linalg.inv(eig_vecs)
    inv_U =inv_U[:n_keep,:]
    Lambda = eig_va[:n_keep]
    reduced_cov = np.dot(U,np.dot(np.diag(Lambda), inv_U))
    #print(reduced_cov)
    d, n = eig_vecs.shape;
    principal_axes = get_mixture_of_samples(eig_vecs.T, d, n, max_samples, order);
    return principal_axes,reduced_cov

def get_mixture_of_samples(
    samples: np.ndarray, 
    dim: int,
    n_samples: int,
    max_samples: int,
    order: int = 3,
    dtype: str = 'float32',
) -> np.ndarray :

    assert 1 <= order <= 4, \
        f"Currently only support upto 4th order sample generation, got order {order}.";

    assert samples.ndim == 2, f"samples must be 2D array but got {samples.shape}";
    assert n_samples == samples.shape[0] and dim == samples.shape[1], \
        f"samples must be of shape ({n_samples}, {dim}), but got {samples.shape}";

    samples_1 = np.copy(samples);
    for i in range(n_samples) :
        samples_1[i] = _normalize(samples_1[i]);

    samples = np.concatenate(
        (
            samples_1, 
            -samples_1,
        ), axis=0
    );        

    n_samples_1 = samples.shape[0];
    if n_samples_1 >= max_samples :
        samples = _clamp_sample_size(samples, max_samples);
        _assert_unique(samples);
        return samples; 
   
    if order == 1 :
        _assert_unique(samples);
        return samples;

    samples_2 = [];
    for i in range(n_samples-1) :
        for j in range(i+1, n_samples) :

            samples_2.append(
                _normalize( samples_1[i] + samples_1[j] )
            );
            samples_2.append(
                _normalize( samples_1[i] - samples_1[j] )
            );            

    samples_2 = np.stack(samples_2, axis=0);

    samples_2 = np.concatenate(
        (
            samples_2, 
            -samples_2,
        ), axis=0
    );        

    n_samples_2 = samples_2.shape[0];
    max_samples = max_samples - n_samples_1;
    if n_samples_2 >= max_samples :
        samples_2 = _clamp_sample_size(samples_2, max_samples);
        samples = np.concatenate(
            (
                samples, 
                samples_2,
            ), axis=0
        );      

        _assert_unique(samples);
        return samples; 
    else :
        samples = np.concatenate(
            (
                samples, 
                samples_2,
            ), axis=0
        );              
   
    if order == 2 :
        _assert_unique(samples);
        return samples;

    samples_3 = [];
    for i in range(n_samples-2) :
        for j in range(i+1, n_samples-1) :
            for k in range(j+1, n_samples) :

                samples_3.extend([
                    _normalize( samples_1[i] + samples_1[j] + samples_1[k] ), 
                    _normalize( samples_1[i] + samples_1[j] - samples_1[k] ), 
                    _normalize( samples_1[i] - samples_1[j] + samples_1[k] ), 
                    _normalize( -samples_1[i] + samples_1[j] + samples_1[k] ), 
                ]);



    samples_3 = np.stack(samples_3, axis=0);

    samples_3 = np.concatenate(
        (
            samples_3, 
            -samples_3,
        ), axis=0
    );       

    n_samples_3 = samples_3.shape[0];
    max_samples = max_samples - n_samples_2;
    if n_samples_3 >= max_samples :
        samples_3 = _clamp_sample_size(samples_3, max_samples);
        samples = np.concatenate(
            (
                samples, 
                samples_3,
            ), axis=0
        );      

        _assert_unique(samples);
        return samples; 
    
    else :
        samples = np.concatenate(
            (
                samples, 
                samples_3,
            ), axis=0
        );        
   
    if order == 3 :
        _assert_unique(samples);
        return samples;

    samples_4 = [];
    for i in range(n_samples-3) :
        for j in range(i+1, n_samples-2) :
            for k in range(j+1, n_samples-1) :
                for q in range(k+1, n_samples) :
                    s1, s2, s3, s4 = samples_1[i], samples_1[j], samples_1[k], samples_1[q];
                    s_pos = s1 + s2 + s3 + s4;
                    samples_4.extend([
                        _normalize( s_pos ), 
                        _normalize( -s_pos ), 
                        _normalize( s_pos - 2 * s1 ), 
                        _normalize( s_pos - 2 * s2 ), 
                        _normalize( s_pos - 2 * s3 ), 
                        _normalize( s_pos - 2 * s4 ), 
                        _normalize( s_pos - 2 * s1 - 2 * s2 ), 
                        _normalize( s_pos - 2 * s1 - 2 * s3 ), 
                        _normalize( s_pos - 2 * s1 - 2 * s4 ), 
                        _normalize( s_pos - 2 * s2 - 2 * s3 ), 
                        _normalize( s_pos - 2 * s2 - 2 * s4 ), 
                        _normalize( s_pos - 2 * s3 - 2 * s4 ), 
                        _normalize( s_pos - 2 * s1 - 2 * s2 - 2 * s3 ), 
                        _normalize( s_pos - 2 * s1 - 2 * s3 - 2 * s4 ), 
                        _normalize( s_pos - 2 * s2 - 2 * s3 - 2 * s4 ), 
                    ]);



    samples_4 = np.stack(samples_4, axis=0);

    n_samples_4 = samples_4.shape[0];
    max_samples = max_samples - n_samples_3;
    if n_samples_4 >= max_samples :
        samples_4 = _clamp_sample_size(samples_4, max_samples);
        samples = np.concatenate(
            (
                samples, 
                samples_4,
            ), axis=0
        );      

        _assert_unique(samples);
        return samples; 
    else :
        samples = np.concatenate(
            (
                samples, 
                samples_4,
            ), axis=0
        );              
   
    _assert_unique(samples);
    return samples;




def get_inverted_sample_single_w_svm(
    model, clf, 
    proto_mean, p_ax, class_id, 
    params,
    x_proto_mean,
) :
    def __get_norm_dist(pred, target) :
        norm_ = torch.norm(target).item();
        d = torch.dist(pred, target) / norm_;   
        return d;

    def __classify(model, clf, feature) :
        feature = feature.data.cpu().numpy();
        return predict_classifier(clf, feature).item();

    def __is_same_class(model, clf, feature, class_id) :
        return __classify(model, clf, feature) == class_id;

    def __get_expansion(x1, x2) :
        return torch.norm(x2).item() / torch.norm(x1).item();

    @torch.no_grad()
    def __init_input(model, dtype, device) :
        x = torch.zeros((1, *model.get_input_shape())).to(dtype).to(device);
        x.requires_grad_(True);
        return x;

    lr_f, lr, momentum, tol, tol_ub, max_iter_f, max_iter = params;

    model.eval();

    assert __is_same_class(model, clf, proto_mean, class_id);

    shift_ = lr_f * p_ax;
    feature = proto_mean.clone();

    for i in range(max_iter_f) :
        feature.add_(shift_);
        if not __is_same_class(model, clf, feature, class_id) :
            d = feature.data - proto_mean.data;
            feature = proto_mean.data + 0.8 * d; 
            assert __is_same_class(model, clf, feature, class_id);
            break;   
             
    return x, target, feat, d;


def save_inverted_samples_random_svm(
    out_dir, 
    model, clf, 
    proto_mean, proto_var,
    mi_params,
) :

    def __invert_single_sample(info) :
        i, fpath, model, clf, mi_params = info;
        data = load_pickle(fpath);
        sample_type = data['type'];
        out_dir = data['out_dir'];
        class_id = data['class_id'];
        x_proto_mean = data['x_proto_mean'];

        dtype = torch.get_default_dtype();
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'; 

        x_proto_mean = torch.from_numpy(x_proto_mean).to(dtype).to(device);
        if sample_type == 'random_svm' :
            proto_mean = data['proto_mean'];
            p_ax = data['p_ax'];

            proto_mean_t = torch.from_numpy(proto_mean).to(dtype).to(device);
            p_ax_t = torch.from_numpy(p_ax).to(dtype).to(device);

            if proto_mean_t.ndim == 1 :
                proto_mean_t.unsqueeze_(0);
            if p_ax_t.ndim == 1 :
                p_ax_t.unsqueeze_(0);
            p_ax_t.requires_grad_(False);
            x, target, feat, d_perc = \
                get_inverted_sample_random_single_w_svm(
                    model, clf, proto_mean_t, p_ax_t, class_id, 
                    mi_params, 
                    x_proto_mean,
                );

        else :
            raise NotImplementedError;

        if x is None :
            print(f"{i} => {class_id} (failure)", flush=True);
            return;

        print(f"{i} => {class_id}", flush=True);
        assert x.size(0) == 1;
        x = x.view(x.size(1), -1);
        x = x.data.cpu().numpy();
        x = x.astype(np.float32);
        target = target.squeeze().data.cpu().numpy();
        target = target.astype(np.float32);
        feat = feat.squeeze().data.cpu().numpy();
        feat = feat.astype(np.float32);        

        save_dict = {
            'x': x,
            'label': target,
            'feature': feat,
        };
        fname = str(i).zfill(6) + '.pkl';
        save_pickle(osp.join(out_dir, fname), save_dict);


    import concurrent.futures as futures
    infos = [];
    tmp_out_dir = osp.join(out_dir, 'tmp');
    mkdir_rm_if_exists(tmp_out_dir);

    save_random_svm_mi_infos(
        tmp_out_dir,
        out_dir, 
        model, clf,
        proto_mean, proto_var,
        mi_params,
        infos,
    );

    print(f"Trying to invert {len(infos)} samples ...");
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(__invert_single_sample, infos);  

    shutil.rmtree(tmp_out_dir);      


def save_proto_svm_mi_infos(
    tmp_out_dir,
    out_dir, 
    model, clf, 
    proto_mean, proto_var,
    mi_params,
    infos,
) :

    add_svm, add_proto = False, False;
    inv_type = 'proto' #mi_params.inv_type;
    if inv_type == 'proto-svm' :
        add_svm, add_proto = True, True;
    elif inv_type == 'svm' :
        add_svm = True;
    elif inv_type == 'proto' :
        add_proto = True;
    else :
        raise NotImplementedError;

    x_proto_mean = {};
    dtype = torch.get_default_dtype();
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'; 

    pm_params = [
        mi_params.lr.backward,
        mi_params.momentum,
        mi_params.tol.sv,
        mi_params.tol_ub.sv,
    ]

    # for class_id in proto_mean :
    #     print(f"Inverting proto mean for class = {class_id}");

    #     proto_mean_c = proto_mean[class_id];
    #     proto_var_c = proto_var[class_id];

    #     proto_mean_t = torch.from_numpy(proto_mean_c).to(dtype).to(device);
    #     x = get_inverted_proto_mean_single(model, clf, proto_mean_t, class_id, pm_params);
    #     x = x.data.cpu().numpy();
    #     x = x.astype(np.float32);        
    #     x_proto_mean[class_id] = x;

    # print(f"Inversion of proto means are done.");

    # sv_params = [
    #     mi_params.lr.backward,
    #     mi_params.momentum,
    #     mi_params.tol.sv,
    #     mi_params.tol_ub.sv,
    #     mi_params.max_iter.backward.sv,        
    # ];

    # proto_params = [
    #     mi_params.lr.forward,
    #     mi_params.lr.backward,
    #     mi_params.momentum,
    #     mi_params.tol.proto,
    #     mi_params.tol_ub.proto,
    #     mi_params.max_iter.forward,
    #     mi_params.max_iter.backward.proto,
    # ];  

    proto_order = mi_params.order;
    proto_var_exp = mi_params.var_exp;  

    n_samples, n_prev_samples = 0, 0;
    n_sv_samples, n_proto_samples = 0, 0;
    for class_id in proto_mean :
        print(f"Saving info for class = {class_id}");
        # out_dir_c = osp.join(out_dir, 'class_' + str(class_id));
        # mkdir_rm_if_exists(out_dir_c);

        proto_mean_c = proto_mean[class_id];
        proto_var_c = proto_var[class_id];



        if add_proto :
            principal_axes = \
                get_elliptical_samples(
                    proto_var_c, 
                    order=proto_order, 
                    var_exp=proto_var_exp,
                    max_samples=mi_params.max_samples_per_class
            );
            print(principal_axes.shape)
            for i in range(principal_axes.shape[0]) :
                data = {
                    'type': 'proto', 
                    'out_dir': out_dir_c,
                    'class_id': class_id,
                    'proto_mean': proto_mean_c,
                    'p_ax': principal_axes[i],
                    'x_proto_mean': x_proto_mean[class_id],        
                };             

                fpath = osp.join(tmp_out_dir, str(n_samples).zfill(6) + '.pkl');
                save_pickle(fpath, data);
                infos.append( (
                    n_samples,
                    fpath,
                    model, 
                    clf,
                    proto_params,
                ) );

                n_samples += 1;

            n_proto_samples = n_samples - n_prev_samples - n_sv_samples;
            print(f"Class ({class_id}) => Proto samples saved = {n_proto_samples}");
            print();

        n_prev_samples = n_samples;


def get_target(
    out_dir, 
    model, clf, 
    proto_mean, proto_var,
    mi_params,
    mode
) :    
    def __is_same_class(model, clf, feature, class_id):
        pred = model.forward_feat(feature)
        pred = torch.argmax(pred)
        return int(pred) == class_id
    
    def normal(x, mean, var):
        """
        The density function of multivariate normal distribution.

        Parameters
        ---------------
        z: ndarray(float, dim=2)
            random vector, N by 1
        μ: ndarray(float, dim=1 or 2)
            the mean of z, N by 1
        Σ: ndarray(float, dim=2)
            the covarianece matrix of z, N by 1
        """
        x = np.array(x[0])
        mean = np.array(mean[0])
        var = np.array(var)+0.001
        n = x.size
        temp1 = np.linalg.det(var) ** (-1/2)
        temp2 = np.exp(-.5 * (x - mean).T @ np.linalg.inv(var) @ (x - mean))

        return (2 * np.pi) ** (-n/2) * temp1 * temp2

    def diff_entropy(var):
        n = var.shape[0]
        print(n)
        det = np.linalg.det(var)
        entropy = 0.5 * ((np.log(2 * np.pi) + 1) * n + np.log(det))
        return entropy
    def top_n_indices(lst, n):
        indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=False)
        return indices[:n]
    import concurrent.futures as futures
    infos = [];
    # tmp_out_dir = osp.join(out_dir, 'tmp');
    # mkdir_rm_if_exists(tmp_out_dir);
    proto_order = mi_params.order;
    proto_var_exp = mi_params.var_exp;  
    add_proto = True;
    lr_f =  0.05 #mi_params
    max_iter_f  = 10000
    features_list = []
    targets_list = []
    for class_id in proto_mean :
        print(f"Saving info for class = {class_id}");
        # out_dir_c = osp.join(out_dir, 'class_' + str(class_id));
        # mkdir_rm_if_exists(out_dir_c);

        proto_mean_c = proto_mean[class_id];
        proto_var_c = proto_var[class_id];
        dif_entropy = diff_entropy(proto_var_c)

        if add_proto :
            principal_axes = \
                get_elliptical_samples(
                    proto_var_c, 
                    order=proto_order, 
                    var_exp=proto_var_exp,
                    max_samples=mi_params.max_samples_per_class
            );
            print(principal_axes.shape)
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'; 
            for i in range(principal_axes.shape[0]):
                dtype = torch.get_default_dtype();
                
                p_ax = principal_axes[i]
                proto_mean_t = torch.from_numpy(proto_mean_c).to(dtype)
                p_ax_t = torch.from_numpy(p_ax).to(dtype)

                if proto_mean_t.ndim == 1 :
                    proto_mean_t.unsqueeze_(0);
                if p_ax_t.ndim == 1 :
                    p_ax_t.unsqueeze_(0);
                
                if mode == 'boat':
                    proto_mean_t = proto_mean_t.to(device);
                    p_ax_t = p_ax_t.to(device)
                    shift_ = lr_f * p_ax_t;
                    feature = proto_mean_t.clone();
                    for i in range(max_iter_f) :
                        feature.add_(shift_);
                        pred = model.forward_feat(feature)
                        pred = torch.argmax(pred)
                        #print(int(pred), class_id)
                        if not __is_same_class(model, clf, feature, class_id):
                            d = feature.data - proto_mean_t.data;
                            feature = proto_mean_t.data + 0.8 * d; 
                            print('yes', i)
                            assert __is_same_class(model, clf, feature, class_id);
                            features_list.append(feature.cpu())
                            targets_list.append(class_id)
                            break;  
                elif mode == 'entropy':
                    shift_ = 0.01 * p_ax_t;
                    feature = proto_mean_t.clone();
                    print(feature.shape)
                    x_list = []
                    en_list = []
                    for i in range(max_iter_f) :
                        feature.add_(shift_);
                        from scipy.stats import multivariate_normal
                        print(proto_var_c)
                        px = multivariate_normal.pdf(feature[0], mean=proto_mean_c[0], cov=proto_var_c+0.001)
                        #px = normal(feature,proto_mean_c, proto_var_c)
                        en_list.append(abs(-np.log(px)-dif_entropy))
                        x_list.append(feature)
                    ind = top_n_indices(en_list,2)
                    x_list = np.array(x_list)[ind]
                    print(x_list)
                        # pred = model.forward_feat(feature)
                        # pred = torch.argmax(pred)
                        # #print(int(pred), class_id)
                        # if not __is_same_class(model, clf, feature, class_id):
                        #     d = feature.data - proto_mean_t.data;
                        #     feature = proto_mean_t.data + 0.8 * d; 
                        #     features_list.append(feature.cpu())
                        #     targets_list.append(class_id)
                        #     assert __is_same_class(model, clf, feature, class_id);
                        #     break;  

    features_list = np.concatenate(features_list)
    targets_list = np.array(targets_list)
    print(features_list.shape)
    print(targets_list)
    return features_list,targets_list
    # save_proto_svm_mi_infos(
    #     None,
    #     out_dir, 
    #     model, clf,
    #     proto_mean, proto_var,
    #     mi_params,
    #     infos,
    # );

    # print(f"Trying to invert {len(infos)} samples ...");
    # with futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     executor.map(__invert_single_sample, infos);  

    # shutil.rmtree(tmp_out_dir);  
