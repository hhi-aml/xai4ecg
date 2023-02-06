from matplotlib.colors import ListedColormap, Normalize
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import utils
import numpy as np
from tqdm import tqdm

def plot_beat(
    x,
    ax,
    label,
    lower=None,
    upper=None,
    color='k',
    alpha=1,
    annotate=True,
    offset=4,
    lw=4,
    border=True,
    grid=False):
    
    channel_axis = {
        'I':[1,0], 
        'II':[3,0], 
        'III':[5,0], 
        'AVR':[2,0], 
        'AVL':[0,0], 
        'AVF':[4,0], 
        'V1':[0,1], 
        'V2':[1,1], 
        'V3':[2,1], 
        'V4':[3,1],
        'V5':[4,1], 
        'V6':[5,1]
    }
    
    length = x.shape[0]
    pad=.1

    for i, channel in enumerate(x.T):
        [iy,ix] = channel_axis[utils.leads[i].upper()]
        iy = -iy
        if utils.leads[i].upper() == 'AVR':
            ax.plot(np.array(range(len(channel)))+ix*length, -channel+iy*offset, c=color, lw=lw, zorder=9, label=label, alpha=alpha)
            if not lower is None:
                ax.fill_between(np.array(range(len(channel)))+ix*length, -upper.T[i]+iy*offset, -lower.T[i]+iy*offset, facecolor=color, alpha=0.25, zorder=8)
            if annotate:
                ax.text(ix*length+40*pad, iy*offset+offset/2-pad,  '-'+utils.leads[i], va='top', ha='left', fontweight='bold')#, bbox=dict(facecolor='.8', edgecolor='k', pad=pad, alpha=1., zorder=10))
        else:
            ax.plot(np.array(range(len(channel)))+ix*length, channel+iy*offset, c=color, lw=lw, zorder=9, alpha=alpha)
            if not lower is None:
                ax.fill_between(np.array(range(len(channel)))+ix*length, upper.T[i]+iy*offset, lower.T[i]+iy*offset, facecolor=color, alpha=0.25, zorder=8)
            if annotate:
                ax.text(ix*length+40*pad, iy*offset+offset/2-pad, utils.leads[i], va='top', ha='left', fontweight='bold')#, bbox=dict(facecolor='.8', edgecolor='k', pad=pad, alpha=1., zorder=10))

                
    if border:
        border_color='k'
        lwg = 2
        ax.axvline(0, c=border_color,zorder=5, lw=lwg)
        ax.axvline(length, c=border_color,zorder=5, lw=lwg)
        ax.axvline(2*length, c=border_color,zorder=5, lw=lwg)
        for iy in range(7):
            ax.axhline(-iy*offset+offset/2, c=border_color,zorder=5, lw=lwg)
    ax.set_xlim(-lwg/10.,2*length)
    ax.set_ylim(-5*offset-offset/2, offset/2)
    if grid:
        grid_color='gray'
        grid_alpha=.15
        # mV
        for mv in np.arange(-6*offset,offset//2,.1):
            ax.axhline(mv, c=grid_color, lw=1, alpha=grid_alpha)
        for mv in np.arange(-6*offset,offset//2,.5):
            ax.axhline(mv, c=grid_color, lw=2, alpha=grid_alpha)
        # ms
        for ms in np.arange(0,length,2):
            ax.axvline(ms, c=grid_color, lw=1, alpha=grid_alpha)
        for ms in np.arange(0,length,10):
            ax.axvline(ms, c=grid_color, lw=2, alpha=grid_alpha)
        ax.axvline(30, c=grid_color,zorder=1,lw=2, alpha=.5)
        ax.axvline(110, c=grid_color,zorder=1,lw=2, alpha=.5)

def plot_heat(
    x,
    a,
    ax,
    label,
    color='b',
    alpha=1,
    annotate=True,
    offset=4,
    lw =2,
    border=False, qmax=.99, qmin=0.2):
    
    channel_axis = {
        'I':[1,0], 
        'II':[3,0], 
        'III':[5,0], 
        'AVR':[2,0], 
        'AVL':[0,0], 
        'AVF':[4,0], 
        'V1':[0,1], 
        'V2':[1,1], 
        'V3':[2,1], 
        'V4':[3,1],
        'V5':[4,1], 
        'V6':[5,1]
    }
    
    cmap = plt.cm.Reds
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap = ListedColormap(my_cmap)
    
    channel_itos = utils.leads
    length = x.shape[0]
    dom = np.array(range(length))
    
    amax = np.quantile(abs(a.flatten()),q=qmax)
    amin = np.quantile(abs(a.flatten()),q=qmin)
    norm = Normalize(vmin=amin, vmax=amax)
    
    for i, channel in enumerate(x.T):
        [iy,ix] = channel_axis[utils.leads[i].upper()]
        iy = -iy
        if utils.leads[i].upper() == 'AVR':
            xy = np.vstack([dom+ix*length, -channel+iy*offset]).T
            xy = xy.reshape(-1, 1, 2)
            segments = np.hstack([xy[:-1], xy[1:]])
            coll = LineCollection(segments, cmap=my_cmap, norm=norm, linewidths=lw, zorder=10)
            coll.set_array(a[:,i])
            ax.add_collection(coll)
        else:
            xy = np.vstack([dom+ix*length, channel+iy*offset]).T
            xy = xy.reshape(-1, 1, 2)
            segments = np.hstack([xy[:-1], xy[1:]])
            coll = LineCollection(segments, cmap=my_cmap, norm=norm, linewidths=lw, zorder=10)
            coll.set_array(a[:,i])
            ax.add_collection(coll)
        if border:
            # BORDER
            border_color='k'
            lwg = 4
            ax.axvline(0, c=border_color,zorder=10, lw=lwg)
            ax.axvline(length, c=border_color,zorder=10, lw=lwg)
            ax.axvline(2*length, c=border_color,zorder=10, lw=lwg)
            ax.axhline(iy*offset+offset/2, c=border_color,zorder=7, lw=lwg)
            ax.axhline(iy*offset-offset/2, c=border_color,zorder=7, lw=lwg)
            ax.set_xlim(-lwg/10.,2*length + lwg/10.)
            ax.set_ylim(iy*offset-offset//2 -lwg/200., offset//2+ lwg/200.)


def visualize_attributions(samples, attrs, dfr, title, offset=2, qlower=.05, qupper=.95, grid=True, axis=None, with_attributions=True):
    
    beats = np.concatenate([[samples[i][ri-30:ri+50,:] for ri in np.array(dfr.iloc[i].r_peaks).astype(int) if (ri<len(samples[i])-50)&(ri > 30)] for i in tqdm(range(len(samples)))])
    heats = np.concatenate([[attrs[i][ri-30:ri+50,:] for ri in np.array(dfr.iloc[i].r_peaks).astype(int) if (ri<len(attrs[i])-50)&(ri > 30)] for i in tqdm(range(len(samples)))])
    ecg_ids = np.concatenate([[i]*len([True for ri in np.array(dfr.iloc[i].r_peaks).astype(int) if (ri<len(attrs[i])-50)&(ri > 30)]) for i in tqdm(range(len(samples)))])

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if axis is None:
        fig, axarr = plt.subplots(1,1,figsize=(6,12))
        ax = axarr
    else:
        ax = axis

    plot_beat(
        np.median(beats, axis=0),  
        ax, 
        'n_beats='+str(len(beats)),
        lower=np.quantile(beats, axis=0, q=qlower), 
        upper=np.quantile(beats, axis=0, q=qupper), 
        offset=offset,
        color='k', alpha=1, lw=2, border=True, annotate=True)
    if with_attributions:
        plot_heat(
            np.median(beats, axis=0),
            np.mean(heats, axis=0),
            ax,
            'laber',
            offset=offset, annotate=False)

    if grid:
        grid_color='gray'
        grid_alpha=.15
        # mV
        for mv in np.arange(-6*offset,offset//2,.1):
            ax.axhline(mv, c=grid_color, lw=1, alpha=grid_alpha)
        for mv in np.arange(-6*offset,offset//2,.5):
            ax.axhline(mv, c=grid_color, lw=2, alpha=grid_alpha)
        # ms
        for ms in np.arange(0,160,2):
            ax.axvline(ms, c=grid_color, lw=1, alpha=grid_alpha)
        for ms in np.arange(0,160,10):
            ax.axvline(ms, c=grid_color, lw=2, alpha=grid_alpha)
                
        ax.axvline(30, c=grid_color,zorder=1,lw=2, alpha=.5)
        ax.axvline(110, c=grid_color,zorder=1,lw=2, alpha=.5)
    
    ax.set_xticks([])
    ax.set_yticks([])
    title += ' [n_beats='+str(len(beats))+']'
    if axis is None:
        plt.title(title)
        plt.show()
    else:
        ax.set_title(title)

    return beats, heats, ecg_ids