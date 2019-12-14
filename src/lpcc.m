function [c, tc]=lpcc(s,fs,w,nc,p,n,inc)
%MELCEPST Calculate the LPCC of a signal C=(S,FS,W,NC,P,N,INC,FL,FH)
%
%
% Simple use: (1) c=lpcc(s,fs)          % calculate mel cepstrum with 12 coefs, 256 sample frames
%			  (2) c=lpcc(s,fs,'E0dD')   % include log energy, 0th cepstral coef, delta and delta-delta coefs
%
% Inputs:
%     s	  speech signal
%     fs  sample rate in Hz (default 11025)
%     w   mode string (see below)
%     nc  number of cepstral coefficients excluding 0'th coefficient [default 12]
%     p   number of filters in filterbank [default: floor(3*log(fs)) =  approx 2.1 per ocatave]
%     n   length of frame in samples [default power of 2 < (0.03*fs)]
%     inc frame increment [default n/2]
%     fl  low end of the lowest filter as a fraction of fs [default = 0]
%     fh  high end of highest filter as a fraction of fs [default = 0.5]
%
%		w   any sensible combination of the following:
%
%               'R'  rectangular window in time domain
%				'N'	 Hanning window in time domain
%				'M'	 Hamming window in time domain (default)
%
%               't'  triangular shaped filters in mel domain (default)
%               'n'  hanning shaped filters in mel domain
%               'm'  hamming shaped filters in mel domain
%
%				'p'	 filters act in the power domain
%				'a'	 filters act in the absolute magnitude domain (default)
%
%               '0'  include 0'th order cepstral coefficient
%				'E'  include log energy
%				'd'	 include delta coefficients (dc/dt)
%				'D'	 include delta-delta coefficients (d^2c/dt^2)
%
%               'z'  highest and lowest filters taper down to zero (default)
%               'y'  lowest filter remains at 1 down to 0 frequency and
%			   	     highest filter remains at 1 up to nyquist freqency
%
%		       If 'ty' or 'ny' is specified, the total power in the fft is preserved.
%
% Outputs:	c     mel cepstrum output: one frame per row. Log energy, if requested, is the
%                 first element of each row followed by the delta and then the delta-delta
%                 coefficients.
%           tc    fractional time in samples at the centre of each frame
%                 with the first sample being 1.
%

% BUGS: (1) should have power limit as 1e-16 rather than 1e-6 (or possibly a better way of choosing this)
%           and put into VOICEBOX
%       (2) get rdct to change the data length (properly) instead of doing it explicitly (wrongly)

%      Copyright (C) Mike Brookes 1997
%      Version: $Id: melcepst.m 4914 2014-07-24 08:44:26Z dmb $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   http://www.gnu.org/copyleft/gpl.html or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


   
[lpar,~,~]=lpcauto(s,nc,[inc n]);  % 20ms frame with 10ms frame increment
temp = lpar(:,2:end);

nf=size(temp,1);
c = temp;



% calculate derivative

if any(w=='D')
  vf=(4:-1:-4)/60;
  af=(1:-1:-1)/2;
  ww=ones(5,1);
  cx=[c(ww,:); c; c(nf*ww,:)];
  vx=reshape(filter(vf,1,cx(:)),nf+10,nc);
  vx(1:8,:)=[];
  ax=reshape(filter(af,1,vx(:)),nf+2,nc);
  ax(1:2,:)=[];
  vx([1 nf+2],:)=[];
  if any(w=='d')
     c=[c vx ax];
  else
     c=[c ax];
  end
elseif any(w=='d')
  vf=(4:-1:-4)/60;
  ww=ones(4,1);
  cx=[c(ww,:); c; c(nf*ww,:)];
  vx=reshape(filter(vf,1,cx(:)),nf+8,nc);
  vx(1:8,:)=[];
  c=[c vx];
end
 


end

function ccs = cepst(apks)
% ccs = cepst(apks)
% - calculates cepstral coefficients from lpcs
% - apks are the lpc values (without leading 1)
%    - if more than one, apks should be a N by D matrix, where N is the
%    number of lpc vectors, D is the number of lpcs
% - ccs are the cepstral coefficients
% the number of ccs is the same as the number of lpcs
[N P] = size(apks);
ccs = zeros(N,P);

for i = 1:N
    for m = 1:P
        s = 0;
        for k = 1:(m-1)
            s = s + -1*(m - k)*ccs(i,m - k)*apks(i,k);
        end
        ccs(i,m) = -1*apks(i,m) + (1/m)*s;
    end
end
end

