function [tdataq,tq]=freqtotime2(data,freq)

    deltaf=freq(2)-freq(1);
    fs=freq(end);
    zerof=(0:deltaf:freq(1)-deltaf);

    fdatahalf=[zeros(size(zerof)) data];
    fdatafull=[fdatahalf fliplr(conj(fdatahalf(2:end)))];
    tdata=ifft(fdatafull)*length(data);
    t=0.5*(0:length(tdata)-1)/fs;
    %dt=(t(2)-t(1));
    dt=0.75e-10;
    tq=0:(dt):t(floor(length(t)/2));
    tdataq=interp1(t,tdata,tq,'spline');
