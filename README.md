# ARDFTSRC
ARDFTSRC Audio Resampler, by mycroft, sigmoid taper variant, port to Rust and Windows by dBpoweramp.com

Usage: pass on command line : "c:\in.wav" "c:\out.wav" 48000 (out frequency) 16 (out bitdepth) 2048 [optional quality] 0.95 [optional bandwidth]

Quality should be high if increasing bandwidth, example ardftsrc.exe "c:\in.wav" "c:\out.wav" 48000 32 8192 0.99
When reducing the quality, example ardftsrc.exe "c:\in.wav" "c:\out.wav" 44100 16 2048 0.96


SOME PART OF THIS SOURCE CODE WERE GENERATED USING AI (Anthropic Claude).
THIS IMPLIES THAT AI CODE GENERATORS WAS USED, BUT IT DOES NOT MEAN THAT
THE PROGRAM ITSELF OPERATES USING AI.

# TODO
- [x] Fix to 24bit_int save error.
