# ARDFTSRC
ARDFTSRC Audio Resampler, by mycroft, port to Rust and Windows by dBpoweramp.com

Usage: pass on command line : "c:\in.wav" "c:\out.wav" 48000 (out frequency) 16 (out bitdepth) 2048 [optional quality] 0.95 [optional bandwidth]

Quality should be high if increasing bandwidth, example ardftsrc.exe "c:\in.wav" "c:\out.wav" 48000 24 8192 0.99
When reducing the quality, example ardftsrc.exe "c:\in.wav" "c:\out.wav" 48000 24 4096 0.97
