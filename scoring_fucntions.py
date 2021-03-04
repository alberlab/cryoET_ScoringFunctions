import mrc, os, gc, random, math, copy
import numpy as N
import numpy.fft as NF
from random import randrange
from scipy.stats import pearsonr
from scipy.ndimage.filters import gaussian_filter as SNFG

def get_mrc(filename):
    assert os.path.isfile(filename)
    v = mrc.imread(filename)
    v = N.swapaxes(v, 0, 2)
    v = N.array(v, dtype=N.float32)
    return v

def zeroMeanUnitStdNormalize(x):
    if x.std()==0:
        return x
    else:
        return (x-x.mean())/x.std()

def read_particle_and_mask(particle, mask):
    v = zeroMeanUnitStdNormalize(get_mrc(particle))
    m = get_mrc(mask)
    return v, m

def fft_mid_co(siz):
    assert(all(N.mod(siz, 1) == 0))
    assert(all(N.array(siz) > 0))
    mid_co = N.zeros(len(siz), N.dtype("int64"))
    # according to following code that uses numpy.fft.fftshift()
    for i in range(len(mid_co)):
        m = siz[i]
        mid_co[i] = N.floor(m/2)

    return mid_co

def grid_displacement_to_center(size, mid_co=None):
    size = N.array(size, dtype=N.float)
    assert size.ndim == 1
    if mid_co is None:
        # IMPORTANT: following python convension, in index starts from 0 to size-1!!! So (siz-1)/2 is real symmetry center of the volume
        mid_co = (N.array(size) - 1) / 2
    if size.size == 3:
        # construct a gauss function whose center is at center of volume
        grid = N.mgrid[0:size[0], 0:size[1], 0:size[2]]
        for dim in range(3):
            grid[dim, :, :, :] -= mid_co[dim]
    elif size.size == 2:
        # construct a gauss function whose center is at center of volume
        grid = N.mgrid[0:size[0], 0:size[1]]
        for dim in range(2):
            grid[dim, :, :] -= mid_co[dim]
    else:
        assert False

    return grid

def grid_distance_sq_to_center(grid):
    dist_sq = N.zeros(grid.shape[1:])
    if grid.ndim == 4:
        for dim in range(3):
            dist_sq += N.squeeze(grid[dim, :, :, :]) ** 2
    elif grid.ndim == 3:
        for dim in range(2):
            dist_sq += N.squeeze(grid[dim, :, :]) ** 2
    else:
        assert False

    return dist_sq

def grid_distance_to_center(grid):
    dist_sq = grid_distance_sq_to_center(grid)
    return N.sqrt(dist_sq)

def ssnr__get_rad(siz):
    grid = grid_displacement_to_center(siz, fft_mid_co(siz))
    rad = grid_distance_to_center(grid)
    return rad

def ssnr_to_fsc(ssnr):
    fsc = ssnr / (2.0 + ssnr)
    return fsc

def ssnr_rad_ind(rad, r, band_width_radius):
    return ( abs(rad - r) <= band_width_radius )

def ssnr__given_stat(sum_v, prod_sum, mask_sum, rad=None):
    op = {}
    
    if 'band_width_radius' not in op:
        op['band_width_radius'] = 1.0

    if 'mask_sum_threshold' not in op:
        op['mask_sum_threshold'] = 2.0
    else:
        op['mask_sum_threshold'] = N.max((op['mask_sum_threshold'], 2.0))

    siz = N.array(sum_v.shape)
    subtomogram_num = mask_sum.max()
    avg = N.zeros(sum_v.shape, dtype=N.complex) + N.nan
    ind = mask_sum > 0
    avg[ind] = sum_v[ind]  /  mask_sum[ind];    avg_abs_sq = N.real(  avg * N.conj( avg ) )
    del ind

    var = N.zeros(sum_v.shape, dtype=N.complex) + N.nan
    ind = mask_sum >= op['mask_sum_threshold']
    var[ind] = ( prod_sum[ind] - mask_sum[ind]*(avg[ind]*N.conj(avg[ind])) ) / ( mask_sum[ind] - 1 );   var = N.real(var)
    del ind

    if rad is None:     rad = ssnr__get_rad(siz)
    vol_rad = int( N.floor( N.min(siz) / 2.0 ) + 1)
    ssnr = N.zeros(vol_rad) + N.nan     # this is the SSNR of the AVERAGE image

    for r in range(vol_rad):
        ind = ssnr_rad_ind(rad=rad, r=r, band_width_radius=op['band_width_radius'])
        ind[mask_sum < op['mask_sum_threshold']] = False
        ind[N.logical_not(N.isfinite(avg))] = False
        ind[N.logical_not(N.isfinite(var))] = False
        if var[ind].sum() > 0:
            ssnr[r] = (mask_sum[ind] * avg_abs_sq[ind]).sum() / var[ind].sum()
        else:
            ssnr[r] = 0.0
        del ind

    assert N.all(N.isfinite(ssnr))
    fsc = ssnr_to_fsc(ssnr)
    return fsc

def sfsc(particles, masks, gf, mask_cutoff=0.5):
    sum_v = None
    prod_sum_v = None
    mask_sum = None
    for i in range(len(particles)):
        vr, vm = read_particle_and_mask(particles[i], masks[i])
        vr = SNFG(vr, gf)
        vr = NF.fftshift( NF.fftn(vr) )

        vr[vm < mask_cutoff] = 0.0
        if sum_v is None:
            sum_v = vr
        else:
            sum_v += vr

        if prod_sum_v is None:
            prod_sum_v = vr * N.conj(vr)
        else:
            prod_sum_v += vr * N.conj(vr)

        if mask_sum is None:
            mask_sum = N.zeros(vm.shape, dtype=N.int)
        mask_sum[vm >= mask_cutoff] += 1
        
        del vr, vm
        gc.collect()
    
    fsc = ssnr__given_stat(sum_v=sum_v, prod_sum=prod_sum_v, mask_sum=mask_sum)
    sfsc = sum(fsc.tolist())
        
    del sum_v, prod_sum_v, mask_sum
    gc.collect()
    return sfsc

def mask_segmentation(x, t):
    m = N.mean(x)
    sd = N.std(x)
    tmp = m - t*sd
    mask = x.copy()
    mask[mask>tmp] = 0
    mask[mask<=tmp] = 1
    return mask

def cluster_average_mask(particles, masks):
    cluster_sums = None
    cluster_mask_sums = None
    cluster_sizes = 0
    for i in range(len(particles)):
        vr, vm = read_particle_and_mask(particles[i], masks[i])
        
        if cluster_sums is None:
            cluster_sums = N.zeros(vr.shape, dtype=N.float32, order='F')
        cluster_sums += vr
        if cluster_mask_sums is None:
            cluster_mask_sums = N.zeros(vm.shape, dtype=N.float32, order='F')
        cluster_mask_sums += vm
        del vr, vm
        cluster_sizes += 1

    assert cluster_sizes > 0
    assert cluster_mask_sums.max() > 0

    # No need to averagein Fourier space, as we just need mask of complex from the subtomogram
    cluster_avg = cluster_sums / cluster_sizes
    cluster_avg = zeroMeanUnitStdNormalize(SNFG(cluster_avg, 2))
    cluster_avg_mask = N.asarray(mask_segmentation(cluster_avg, 1.5), dtype=N.bool)
    del cluster_avg

    del cluster_sums, cluster_mask_sums, cluster_sizes
    gc.collect()
    return cluster_avg_mask

def pearson_correlation(x, y):
    pcorr = pearsonr(x.flatten(), y.flatten())
    if math.isnan(pcorr[0]):
        return 0.0
    else:
        return pcorr[0]

def get_significant_points(v):
    """
    Retrieve all points with a density greater than one standard deviation above the mean.
    Return: An array of 4-tuple (indices of the voxels in x,y,z format and density value) 
    """
    sig_points = []
    sig_mask = v > (v.mean() + v.std())
    for z in range(v.shape[0]):
        for y in range(v.shape[1]):
            for x in range(v.shape[2]):
                if sig_mask[z][y][x]:
                    sig_points.append(N.array([z,y,x,v[z][y][x]]))
    return N.array(sig_points)

def get_random_significant_pairs(v, amount):
    """   
    Arguments
    amount: number of significant point pairs to return.
    
    Returns: Array of tuple pairs of significant points randomly chosen from 'get_significant_points' function.
    """
    sig_points = get_significant_points(v)
    sig_pairs = []
    size = len(sig_points)
    if amount <= size*size:
        random_pairs = []
        for r in range(amount):
            tmp = (randrange(size), randrange(size))
            if tmp not in random_pairs:
                fst = sig_points[tmp[0]]
                snd = sig_points[tmp[1]]
                new_value = N.array([fst[0], fst[1], fst[2], snd[0], snd[1], snd[2], fst[3]-snd[3]])
                sig_pairs.append(new_value)
    else:
        for r in range(amount):
            fst = sig_points[randrange(size)]
            snd = sig_points[randrange(size)]
            new_value = N.array([fst[0], fst[1], fst[2], snd[0], snd[1], snd[2], fst[3]-snd[3]])
            sig_pairs.append(new_value)
    return N.array(sig_pairs)

def dsd(x, y, significant_pairs_num=10000):
    x_sig_pairs = get_random_significant_pairs(x.copy(), significant_pairs_num)
    tmp_score = 0.0
    for p in x_sig_pairs:
        z1 = int(p[0])
        y1 = int(p[1])
        x1 = int(p[2])
        z2 = int(p[3])
        y2 = int(p[4])
        x2 = int(p[5])
        dens = p[6]
        prot_dens = y[z1][y1][x1] - y[z2][y2][x2]
        tmp_score += (dens-prot_dens)**2
    del x_sig_pairs, prot_dens, z1, z2, y1, y2, x1, x2, dens
    gc.collect()
    return tmp_score/x.size

def MI(v1, v2, layers1=20, layers2=20, mask_array=None, normalised=False):
    # based on mask_array MI calculated on complete map (All 1 mask), Overlap region (AND on masks)
    
    if mask_array is None:
        mask_array = N.ones(v1.shape, dtype=int)
    else:
        mask_sum = N.sum(mask_array)
        if mask_sum == 0:
            return 0.0
    
    v1 = v1*mask_array
    v2 = v2*mask_array
    # sturges rule provides a way of calculating number of bins : 1+math.log(number of points)
    layers1 = int(1 + math.log(v1.size, 2))
    layers2 = int(1 + math.log(v2.size, 2))
    layers1 = max(layers1,8)
    layers2 = max(layers2,8)

    P, _, _ = N.histogram2d(v1.ravel(), v2.ravel(), bins=(layers1, layers2))
    P /= P.sum()
    p1 = P.sum(axis=0)
    p2 = P.sum(axis=1)
    p1 = p1[p1 > 0]
    p2 = p2[p2 > 0]
    P = P[P > 0]
    Hx_ = (-p1 * N.log2(p1)).sum()
    Hy_ = (-p2 * N.log2(p2)).sum()
    Hxy_ = (-P * N.log2(P)).sum()
    del P, p1, p2
    gc.collect()
    
    if normalised:
        if Hxy_ == 0.0:
            return 0.0
        else:
            return (Hx_ + Hy_)/Hxy_
    else:
        return Hx_ + Hy_ - Hxy_

def compute_scores(particles, masks, gaussian_filter_sigma=0, score="all", mask_cutoff=0.5):
    '''
    Compute the scoring functions for the set of subtomograms provided in the arguement
    
    Arguements:
    
    particles: list of filepaths of subtomograms in the cluster. Make sure subtomograms are transformed before computing score value.
    masks: list of filepaths of masks corresponding to each subtomogra in the cluster. Make sure masks are transformed before computing score value.
    gaussian_filter_sigma: Standard deviation of Gaussian filter. Default value is zero, that means no filtering.
    score: scoring function to compute. Computes all scoring functions by default. Check documentation on github readme file to see other possible values of 'score'
    mask_cutoff: threshold to binarize missing wedge mask

    Returns:
    Dictionary of scoring function acronym and score value
    '''

    ## Normalize each subtomogram
    assert len(particles)>1
    scoreValues = {}
    if score=="all" or score=="SFSC":
        print("Computing SFSC")
        scoreValues["SFSC"] = sfsc(particles=particles, masks=masks, gf=gaussian_filter_sigma, mask_cutoff=mask_cutoff)
    
    ########### SFSC ###########
    if score=="all" or score!="SFSC":
        cluster_mask = None
        if score in ["all", "amPC", "amNSD", "amMI", "amCCC"]:
            cluster_mask = cluster_average_mask(particles, masks)

        # Make subtomogram pairs
        pairs = []
        num_particles = len(particles)
        possible_pair_num = int((num_particles * (num_particles-1))/2)
        for i in range(num_particles):
            for j in range(i+1, num_particles):
                pairs.append((i,j))
        
        num_of_pairs = 0
        minimum_num_of_paris = 5000
        if possible_pair_num<minimum_num_of_paris:
            num_of_pairs = possible_pair_num
        elif possible_pair_num*0.1<minimum_num_of_paris:
            num_of_pairs = minimum_num_of_paris
        else:
            num_of_pairs = int(possible_pair_num*0.10)
        
        print("Num of pairs: ", num_of_pairs)
        
        random.shuffle(pairs)
        pairs = pairs[:num_of_pairs]

        for i, p in enumerate(pairs):
            vr_1, vm_1 = read_particle_and_mask(particles[p[0]], masks[p[0]])
            vr_2, vm_2 = read_particle_and_mask(particles[p[1]], masks[p[1]])
            
            # Gaussian Filter
            vr_1_gf = SNFG(vr_1.copy(), gaussian_filter_sigma)
            vr_2_gf = SNFG(vr_2.copy(), gaussian_filter_sigma)
            
            # Binarize masks
            vm_1[vm_1 < mask_cutoff] = 0.0
            vm_1[vm_1 >= mask_cutoff] = 1.0
            vm_2[vm_2 < mask_cutoff] = 0.0
            vm_2[vm_2 >= mask_cutoff] = 1.0

            # Mask overlap
            masks_logical_and = N.logical_and(vm_1, vm_2)
            masks_logical_and_flag = False
            if masks_logical_and.sum() < 2:
                masks_logical_and_flag = True
            else:
                masks_logical_and = masks_logical_and.flatten()
                masks_logical_and = N.where(masks_logical_and==True)[0]
            
            # Generate masks for contoured and overlap scores
            threshold_i = 1.5
            vr_1_mask = mask_segmentation(vr_1_gf.copy(), threshold_i)
            vr_2_mask = mask_segmentation(vr_2_gf.copy(), threshold_i)

            ########### gPC ###########
            if score in ["all", "gPC"]:
                if "gPC" not in scoreValues:
                    scoreValues["gPC"] = []
                if i==0:    print("Computing gPC")
                scoreValues["gPC"].append(pearson_correlation(vr_1_gf, vr_2_gf))

            ########### amPC ###########
            if score in ["all", "amPC"]:
                if "amPC" not in scoreValues:
                    scoreValues["amPC"] = []
                if i==0:    print("Computing amPC")
                scoreValues["amPC"].append(pearson_correlation(vr_1_gf[cluster_mask], vr_2_gf[cluster_mask]))

            if score in ["all", "FPC", "FPCmw", "CCC", "amCCC"]:
                vr_1_f = NF.fftshift(NF.fftn(vr_1_gf.copy()))
                vr_2_f = NF.fftshift(NF.fftn(vr_2_gf.copy()))
                ########### FPC ###########
                if score in ["all", "FPC"]:
                    if "FPC" not in scoreValues:
                        scoreValues["FPC"] = []
                    if i==0:    print("Computing FPC")
                    scoreValues["FPC"].append(pearson_correlation(vr_1_f.real.flatten(), vr_2_f.real.flatten()))
                
                ########### FPCmw ###########
                if score in ["all", "FPCmw"]:
                    if "FPCmw" not in scoreValues:
                        scoreValues["FPCmw"] = []
                    if i==0:    print("Computing FPCmw")
                    if masks_logical_and_flag:
                        scoreValues["FPCmw"].append(0.0)
                    else:
                        scoreValues["FPCmw"].append(pearson_correlation(vr_1_f.real.flatten()[masks_logical_and], vr_2_f.real.flatten()[masks_logical_and]))
                
                if score in ["all", "CCC", "amCCC"]:
                    masks_logical_and = N.logical_and(vm_1, vm_2)
                    N.place(vr_1_f, masks_logical_and==False, [0])
                    N.place(vr_2_f, masks_logical_and==False, [0])
                    vr_1_if = (NF.ifftn(NF.ifftshift(vr_1_f))).real
                    vr_2_if = (NF.ifftn(NF.ifftshift(vr_2_f))).real
                    
                    ########### CCC ###########
                    if score in ["all", "CCC"]:
                        vr_1_if_norm = zeroMeanUnitStdNormalize(vr_1_if.copy())
                        vr_2_if_norm = zeroMeanUnitStdNormalize(vr_2_if.copy())
                        if i==0:    print("Computing CCC")
                        if "CCC" not in scoreValues:
                            scoreValues["CCC"] = []
                        scoreValues["CCC"].append(pearson_correlation(vr_1_if_norm.flatten(), vr_2_if_norm.flatten()))
                        del vr_1_if_norm, vr_2_if_norm
                        gc.collect()
                    
                    ########### amCCC ###########
                    if score in ["all", "amCCC"]:
                        vr_1_if = vr_1_if[cluster_mask]
                        vr_2_if = vr_2_if[cluster_mask]
                        vr_1_if_norm = zeroMeanUnitStdNormalize(vr_1_if.copy())
                        vr_2_if_norm = zeroMeanUnitStdNormalize(vr_2_if.copy())
                        if i==0:    print("Computing amCCC")
                        if "amCCC" not in scoreValues:
                            scoreValues["amCCC"] = []
                        scoreValues["amCCC"].append(pearson_correlation(vr_1_if_norm, vr_2_if_norm))
                        del vr_1_if_norm, vr_2_if_norm
                        gc.collect()
                    del vr_1_if, vr_2_if
                    gc.collect()
                del vr_1_f, vr_2_f
                gc.collect()

            # Real space mask for contoured scores
            real_masks_or = N.logical_or(vr_1_mask, vr_2_mask)
            real_masks_or = real_masks_or.flatten()
            real_masks_or = N.where(real_masks_or==True)[0]
            # Real space mask for overlap scores
            real_masks_and = N.logical_and(vr_1_mask, vr_2_mask)
            real_masks_and = real_masks_and.flatten()
            real_masks_and = N.where(real_masks_and==True)[0]
            
            ########### cPC ###########
            if score in ["all", "cPC"]:
                if "cPC" not in scoreValues:
                    scoreValues["cPC"] = []
                if i==0:    print("Computing cPC")
                if real_masks_or.sum()<2:
                    scoreValues["cPC"].append(0.0)
                else:
                    scoreValues["cPC"].append(pearson_correlation(vr_1_gf.flatten()[real_masks_or], vr_2_gf.flatten()[real_masks_or]))

            ########### oPC ###########
            if score in ["all", "oPC"]:
                if "oPC" not in scoreValues:
                    scoreValues["oPC"] = []
                if i==0:    print("Computing oPC")
                if real_masks_and.sum()<2:
                    scoreValues["oPC"].append(0.0)
                else:
                    scoreValues["oPC"].append(pearson_correlation(vr_1_gf.flatten()[real_masks_and], vr_2_gf.flatten()[real_masks_and]))

            ########### OS ###########
            if score in ["all", "OS"]:
                if "OS" not in scoreValues:
                    scoreValues["OS"] = []
                if i==0:    print("Computing OS")
                scoreValues["OS"].append(float(N.logical_and(vr_1_mask, vr_2_mask).sum())/min(vr_1_mask.sum(), vr_2_mask.sum()))

            ########### gNSD ###########
            if score in ["all", "gNSD"]:
                if "gNSD" not in scoreValues:
                    scoreValues["gNSD"] = []
                if i==0:    print("Computing gNSD")
                scoreValues['gNSD'].append(((vr_1_gf - vr_2_gf)**2).mean())
            
            ########### cNSD ###########
            if score in ["all", "cNSD"]:
                if "cNSD" not in scoreValues:
                    scoreValues["cNSD"] = []
                if i==0:    print("Computing cNSD")
                scoreValues['cNSD'].append(((vr_1_gf.flatten()[real_masks_or] - vr_2_gf.flatten()[real_masks_or])**2).mean())
            
            ########### oNSD ###########
            if score in ["all", "oNSD"]:
                if "oNSD" not in scoreValues:
                    scoreValues["oNSD"] = []
                if i==0:    print("Computing oNSD")
                scoreValues['oNSD'].append(((vr_1_gf.flatten()[real_masks_and] - vr_2_gf.flatten()[real_masks_and])**2).mean())
            
            ########### amNSD ###########
            if score in ["all", "amNSD"]:
                if "amNSD" not in scoreValues:
                    scoreValues["amNSD"] = []
                if i==0:    print("Computing amNSD")
                scoreValues['amNSD'].append(((vr_1_gf[cluster_mask] - vr_2_gf[cluster_mask])**2).mean())

            ########### DSD ###########
            if score in ["all", "DSD"]:
                if "DSD" not in scoreValues:
                    scoreValues["DSD"] = []
                if i==0:    print("Computing DSD")
                scoreValues['DSD'].append(dsd(vr_1_gf.copy(), vr_2_gf.copy()))
            
            ########### gMI ###########
            if score in ["all", "gMI"]:
                if "gMI" not in scoreValues:
                    scoreValues["gMI"] = []
                if i==0:    print("Computing gMI")
                scoreValues['gMI'].append(MI(vr_1_gf.copy(), vr_2_gf.copy(), mask_array=None, normalised=False))

            ########### NMI ###########
            if score in ["all", "NMI"]:
                if "NMI" not in scoreValues:
                    scoreValues["NMI"] = []
                if i==0:    print("Computing NMI")
                scoreValues['NMI'].append(MI(vr_1_gf.copy(), vr_2_gf.copy(), mask_array=None, normalised=True))

            ########### cMI ###########
            if score in ["all", "cMI"]:
                if "cMI" not in scoreValues:
                    scoreValues["cMI"] = []
                if i==0:    print("Computing cMI")
                scoreValues['cMI'].append(MI(vr_1_gf.copy(), vr_2_gf.copy(), mask_array=N.logical_or(vr_1_mask, vr_2_mask), normalised=False))

            ########### oMI ###########
            if score in ["all", "oMI"]:
                if "oMI" not in scoreValues:
                    scoreValues["oMI"] = []
                if i==0:    print("Computing oMI")
                scoreValues['oMI'].append(MI(vr_1_gf.copy(), vr_2_gf.copy(), mask_array=N.logical_and(vr_1_mask, vr_2_mask), normalised=False))

            ########### amMI ###########
            if score in ["all", "amMI"]:
                if "amMI" not in scoreValues:
                    scoreValues["amMI"] = []
                if i==0:    print("Computing amMI")
                scoreValues['amMI'].append(MI(vr_1_gf.copy(), vr_2_gf.copy(), mask_array=cluster_mask, normalised=False))

            print(i, end="\r")
            del vr_1, vr_2, vm_1, vm_2, vr_1_gf, vr_2_gf, threshold_i, vr_1_mask, vr_2_mask, real_masks_or, real_masks_and, masks_logical_and
            gc.collect()

    for score in scoreValues.keys():
        if score!="SFSC":
            scoreValues[score] = N.mean(scoreValues[score])

    return scoreValues
