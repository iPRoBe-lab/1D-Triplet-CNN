# https://github.com/jameslyons/python_speech_features

# This file includes routines for basic signal processing including framing and computing power spectra.
# Author: James Lyons 2012
import sys, os
import decimal
import numpy as np
import math
import logging
import librosa as lr
from scipy import fix, signal, stats
import subprocess as sp
from threading  import Thread
from queue import Queue, Empty
import random
import hdf5storage
from IPython.display import Audio
import librosa

## IGRNORES WARNINGS
import warnings
print('TURNING OFF WARNINGS in tools/sigproc.py!!')
warnings.filterwarnings("ignore")


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step)) # LV

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    """Does overlap-add procedure to undo the action of framesig.

    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = np.shape(frames)[0]
    assert np.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
        np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = np.zeros((padlen,))
    window_correction = np.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[
                                               indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if np.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))


def logpowspec(frames, NFFT, norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames, NFFT);
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps


def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def normalize_audio(file_data, classes=256):
    file_data = lr.util.normalize(file_data)
    quantized_data = quantize_data(file_data, classes)
    return quantized_data

def normalize_frame(frame, axis = 0):
    min_val = np.expand_dims(np.amin(frame,axis = 0),axis=0)
    min_frame = np.tile(min_val, [frame.shape[0],1])
    max_val = np.expand_dims(np.amax(frame,axis = 0),axis=0)
    max_frame = np.tile(max_val, [frame.shape[0],1])
    frame_normalized =(frame-min_frame)/(max_frame-min_frame)
    return frame_normalized

def generate_audio(audio_vec, classes=256):
    # bins_0_1 = np.linspace(0, 1, classes)
    # audio_vec = np.digitize(audio_vec, bins_0_1) - 1
    generated = audio_vec
    # generated = (audio_vec / classes) * 2. - 1
    mu_gen = mu_law_expansion(generated, classes)
    return mu_gen


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    #quantized = mu_x
    quantized = np.digitize(mu_x, bins) - 1
    #bins_0_1 = np.linspace(0, 1, classes)
    #quantized = bins_0_1[quantized]
    return quantized


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x

def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s

def read_frame_from_file(filepath):
    # filepath = '/scratch2/chowdh51/Data/degradedTIMIT/P3/eval/FADG0/SA1.mat'
    mat = hdf5storage.loadmat(filepath)
    frame = np.array(mat['data'])
    return frame

def frame_to_audio(frame, win):
    inv_win = 1/win
    inv_win = np.expand_dims(inv_win, axis=1)
    inv_win = np.tile(inv_win, (1,frame.shape[1]))
    frame = frame * inv_win
    tmp= frame[int((frame.shape[0])/2):,1:]
    b = np.reshape(tmp.transpose(),(1,tmp.shape[0]*tmp.shape[1])).flatten()
    audio = np.concatenate((frame[:,0], b), axis=0)
    return audio

def audio_to_frame(audio, win=signal.boxcar(160), inc=80):
    ## Same as obspy.signal.util.enframe
    nx = len(audio)
    nwin = len(win)
    if (nwin == 1):
        length = win
    else:
        length = nwin
    nf = int(fix((nx - length + inc) // inc))
    indf = inc * np.arange(nf)
    inds = np.arange(length) + 1
    f = audio[(np.transpose(np.vstack([indf] * length)) +
           np.vstack([inds] * nf)).astype(int) - 1]
    if (nwin > 1):
        w = np.transpose(win)
        f = f * np.vstack([w] * nf)
    f = np.transpose(f)
    return f

def get_frame_from_file(file_path, sr=8000, duration = None, n_channels=1, classes=256, win=signal.boxcar(160), inc=80):
    ## Read Audio
    if(isinstance(file_path, np.ndarray)):
        file_data = file_path
    else:
        filename, file_extension = os.path.splitext(file_path)
        if(file_extension == '.mat'):
            mat = hdf5storage.loadmat(file_path)
            file_data = np.array(mat['audio']).flatten()
            fs = np.asscalar(np.array(mat['fs']))
            file_data = signal.resample(file_data, int(file_data.shape[0]*(sr/fs)))
        elif(duration is None):
            file_data, _ = lr.load(path=file_path, sr=sr, duration = duration, mono=n_channels==1)
        else:
            file_data = read_audio(file_path, sampling_rate=sr, duration = duration, n_channels=n_channels)

    ## Normalize Audio for input to CNN
    # normalized_audio = normalize_audio(file_data, classes=classes)
    normalized_audio = file_data
    ## Enframe Normalized Audio
    frame = audio_to_frame(normalized_audio, win, inc)

    # frame = frame[:,~np.all(frame == 0, axis=0)]

    frame = frame[:,~(frame.sum(axis=0) == 0)]    ## Remove all zero-only speech units(columns)

    ## axis=1 ensure normalization across frames
    ## axis=0 ensure normalization within frames (as done for taslp work)
    # frame= stats.zscore(frame, axis=0, ddof=1)

    frame = frame[:,~np.any(np.isnan(frame), axis=0)]
    frame = frame[:,~np.any(np.isinf(frame), axis=0)]

    ## Random crop transform
    # if(frame.shape[1]>200):
    #     idx = random.randint(0,frame.shape[1]-200)
    #     frame = frame[:,idx:idx+200]


    return frame

def get_audio_from_frame(frame, win=signal.boxcar(160), classes=256):
    ## Convert frame to audio
    audio_vec = frame_to_audio(frame,win)
    ## Convert Normalized audio back to un-Normalized audio
    # gen_audio = generate_audio(audio_vec, classes=classes)
    gen_audio = audio_vec
    return gen_audio

def read_audio_blocking(file_path, sampling_rate = 8000, format = 's16le', acodec = 'pcm_s16le', mono = 1, bufsize=10**8, n_channels = 1, duration = 2.01):
    byte_per_frame = 2
    FFMPEG_BIN = "ffmpeg"

    command = [ FFMPEG_BIN,
            '-i', file_path,
            '-f', format,
            '-acodec', acodec,
            '-ar', str(sampling_rate), # ouput will have 'sampling_rate' Hz
            '-ac', str(n_channels), # (set to '1' for mono; '2' for stereo)
            '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=bufsize)
    raw_audio = pipe.stdout.read(np.ceil(sampling_rate*duration*n_channels*byte_per_frame).astype(int))
    audio_array = np.fromstring(raw_audio, dtype="int16")
    audio_array = audio_array.astype(np.float32, order='C') / 32768.0
    return audio_array


def enqueue_output(out, queue, buf_size):
    queue.put(out.read(np.ceil(buf_size).astype(int)))
    out.close()

def read_audio(file_path, sampling_rate = 8000, format = 's16le', acodec = 'pcm_s16le', bufsize=10**4, n_channels = 1, duration = 2.01):
    byte_per_frame = 2
    FFMPEG_BIN = "ffmpeg"
    ON_POSIX = 'posix' in sys.builtin_module_names
    buf_size_2_read = sampling_rate*duration*n_channels*byte_per_frame

    command = [ FFMPEG_BIN,
            '-i', file_path,
            '-f', format,
            '-acodec', acodec,
            '-ar', str(sampling_rate), # ouput will have 'sampling_rate' Hz
            '-ac', str(n_channels), # (set to '1' for mono; '2' for stereo)
            '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=bufsize, close_fds=ON_POSIX)
    q = Queue()
    t = Thread(target=enqueue_output, args=(pipe.stdout, q, buf_size_2_read))
    t.daemon = True # thread dies with the program
    t.start()

    # read line without blocking
    # raw_audio = q.get_nowait()
    # audio_array = np.fromstring(raw_audio, dtype="int16")
    # audio_array = audio_array.astype(np.float32, order='C') / 32768.0
    audio_array = None
    try:  raw_audio = q.get() # or q.get_nowait(), q.get(timeout=.1)
    except Empty:
        print('Failed to read audio!!')
    else:
        audio_array = np.fromstring(raw_audio, dtype="int16")
        audio_array = audio_array.astype(np.float32, order='C') / 32768.0

    return audio_array


def get_lpc_feature(input_audio, sampling_rate, order = 20, preemphasis = True, includeDerivatives = True, win = np.hamming(160), inc = 80):
    # audio, sr = librosa.load(input_audio, sr=sampling_rate)

    audio = input_audio

    # Pre-emphasis filter (zero is at 50 Hz)
    if(preemphasis):
        audio = signal.lfilter([1, -np.exp(-2*np.pi*50/sampling_rate)],1,audio)

    # Get frames from input audio
    frame = get_frame_from_file(audio, win=win, inc=inc, sr=sampling_rate, n_channels=1, duration = None)
    c = np.zeros((frame.shape[1], order))

    # Compute LPC coefficients
    for i in range(c.shape[0]):
        lpc_ftr = librosa.lpc(frame[:,i], order)
        c[i,:] = lpc_ftr[1:]
    nf = c.shape[0]

    # Calculate derivative
    if(includeDerivatives):
      vf=np.arange(4,-5,-1)/60
      ww=np.zeros(4, dtype=int)
      cx = np.vstack((c[ww,:], c, c[(nf-1)*(ww+1),:]))
      filtered_cx = signal.lfilter(vf,1,np.transpose(cx).flatten())
      dc = np.reshape(filtered_cx,(nf+8,order),order='F')
      dc = np.delete(dc, np.arange(0,8), axis=0)
      c = np.hstack((c,dc))
      c = np.transpose(c)
      c = c.astype(np.single)

    return c



def mel2frq(mel):
    k = 1000/np.log(1+1000/700)
    amel = np.absolute(mel)
    frq = np.multiply(700*np.sign(mel), (np.exp(amel/k)-1))
    return frq

def frq2mel(frq):
    k = 1000/np.log(1+1000/700)
    af = np.absolute(frq)
    mel = np.multiply(np.sign(frq), np.log(1+af/700)*k)
    return mel

def melbankm(sr, n_mels, n_fft, fmin, fmax):
    melfb = librosa.filters.mel(sr = sr, n_fft = n_fft,
    n_mels = n_mels, fmin = fmin, fmax = fmax, norm = None, htk = True)
    melfb = melfb[:,1:melfb.shape[1]-1]*2  ## The scaling factor of 2 is used to match the result to VOICEBOX toolkit's MATLAB implementation

    frq = [fmin, fmax]
    mflh = frq2mel(frq)
    melrng = np.matmul(mflh , np.arange(-1,2,2))
    melinc = melrng/(n_mels+1)

    blim = mel2frq(mflh[0]+np.multiply([0, 1, n_mels, n_mels+1],melinc))*n_fft/sr

    b1 = int(np.floor(blim[0])+1)
    b4 = int(np.minimum(np.floor(n_fft/2),np.ceil(blim[3])-1))

    pf = (frq2mel(np.arange(b1,b4+1)*sr/n_fft)-mflh[0])/melinc
    #  remove any incorrect entries in pf due to rounding errors
    if(pf[0]<0):
        pf = np.delete(pf, (0), axis=0)
        b1=b1+1

    if (pf[-1]>=n_mels+1):
        pf = np.delete(pf, (-1), axis=0)
        b4=b4-1;

    mn = b1 + 1
    mx = b4 + 1

    return (melfb, mn, mx)

def rdct(x):
    fl=x.shape[0]==1
    if(fl):
        x=x.flatten()
    [m,k]=x.shape
    n=m
    b=1
    a=np.sqrt(2*n)
    x=np.vstack((x[0:n+1:2,:], x[2*int(np.fix(n/2))-1:0:-2,:]))
    z=np.transpose(np.concatenate(([np.sqrt(2)], 2*np.exp((-0.5j*np.pi/n)*(np.arange(1,n))))))
    y=np.real(np.multiply(np.fft.fft(x,n=x.shape[0],axis=0),np.transpose(np.tile(z,(k,1)))))/a
    if(fl):
        y=np.transpose(y)
    return y

def get_mfcc_feature(input_audio, sampling_rate, order = 20, preemphasis = True, includeDerivatives = True, win = np.hamming(160), inc = 80):

    # audio, sr = librosa.load(input_audio, sr=sampling_rate)
    # win = np.hamming(int(sampling_rate*0.02))
    # inc = int(win.shape[0]/2)

    # Pre-emphasis filter (zero is at 50 Hz)
    if(preemphasis):
        input_audio = signal.lfilter([1, -np.exp(-2*np.pi*50/sampling_rate)],1,input_audio)

    # Get frames from input audio
    frame = get_frame_from_file(input_audio, win=win, inc=inc, sr=sampling_rate, n_channels=1, duration = None)
    c = np.zeros((frame.shape[1], order))

    ## Compute FFT
    f = np.fft.rfft(frame, n=frame.shape[0], axis=0)

    ## Get the Mel-filterbanks
    sr = sampling_rate
    n_mels = int(np.floor(3*np.log(sampling_rate)))
    n_fft = int(sampling_rate*0.02)
    fmin = 0 * sampling_rate
    fmax = 0.5 * sampling_rate
    [m,a,b] = melbankm(sr, n_mels, n_fft, fmin, fmax)
    pw = np.multiply(f[a-1:b,:], np.conj(f[a-1:b,:]))
    pw = pw.real
    pth = np.max(pw.flatten())*1E-20

    ## Apply DCT
    ath = np.sqrt(pth)
    y = np.log(np.maximum(np.matmul(m,np.absolute(f[a-1:b,:])),ath))
    c = np.transpose(rdct(y))
    nf = c.shape[0]
    nc = order

    if n_mels>nc:
        c = c[:,0:nc]
    elif n_mels<nc:
        c = np.hstack((c, np.zeros(nf,nc-n_mels)))

    # Calculate derivative
    if(includeDerivatives):
      vf=np.arange(4,-5,-1)/60
      ww=np.zeros(4, dtype=int)
      cx = np.vstack((c[ww,:], c, c[(nf-1)*(ww+1),:]))
      filtered_cx = signal.lfilter(vf,1,np.transpose(cx).flatten())
      dc = np.reshape(filtered_cx,(nf+8,order),order='F')
      dc = np.delete(dc, np.arange(0,8), axis=0)
      c = np.hstack((c,dc))
      c = np.transpose(c)
      c = c.astype(np.single)

    return c

def cmvn(x):
    mu = np.mean(x, axis=1)
    stdev = np.std(x, axis=1)
    f = np.subtract(x, np.transpose(np.tile(mu,(x.shape[1],1))))
    f = np.divide(f, np.transpose(np.tile(stdev,(x.shape[1],1))))
    return f

def get_mfcc_lpc_feature(input_audio, sampling_rate, order = 20, preemphasis = True, includeDerivatives = True, win = np.hamming(160), inc = 80):
    mfcc_ftr = get_mfcc_feature(input_audio, sampling_rate, order = order, preemphasis = preemphasis, includeDerivatives = includeDerivatives, win = win, inc = inc)
    lpc_ftr = get_lpc_feature(input_audio, sampling_rate, order = order, preemphasis = preemphasis, includeDerivatives = includeDerivatives, win = win, inc = inc)

    #CMVN
    mfcc_ftr = cmvn(mfcc_ftr)
    lpc_ftr = cmvn(lpc_ftr)

    # Concatenate MFCC and LPC features
    mfcc_lpc_ftr = np.stack((mfcc_ftr,lpc_ftr), axis=2)
    mfcc_lpc_ftr = mfcc_lpc_ftr.astype(np.single)
    return mfcc_lpc_ftr
