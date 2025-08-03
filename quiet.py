import sounddevice as sd
import soundfile as sf
import numpy as np

CHUNK = 1024
RATE = 44100
UMBRAL = -23
DB = -80
pos = -1

beep, samplerate = sf.read('Beep.wav', dtype='float32', start=0, stop=RATE*1)
beep = np.mean(beep, axis=1, keepdims=True)  # Mantiene dimensión (N,1)

def calcular_db(indata):
    audio_data = np.linalg.norm(indata, axis=1) if indata.ndim > 1 else indata
    rms = np.sqrt(np.mean(np.square(audio_data)))

    # Evitar log(0)
    if rms == 0:
        return -np.inf

    db = 20 * np.log10(rms)
    return db

def callback(indata, outdata, frames, time, status):
    global DB, pos, UMBRAL
    if status:
        print(status)
    DB = calcular_db(indata)
    
    if pos >= 0:
        end = pos + frames
        if end < len(beep):
            outdata[:] = beep[pos:end]
            pos = end
        else:
            if DB > UMBRAL:
                remaining = len(beep) - pos
                outdata[:remaining] = beep[pos:]
                outdata[remaining:] = beep[:frames - remaining]
                pos = frames - remaining
            else :
                remaining = len(beep) - pos
                outdata[:remaining] = beep[pos:]
                outdata[remaining:] = 0
                pos = -1
    else:
        outdata.fill(0)
    
with sd.Stream(channels=1, samplerate=RATE, callback=callback):
    while True:
        if DB > UMBRAL and pos == -1:
            pos = 0