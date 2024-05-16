import pyaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import chirp, stft

def main()->None:
    p = pyaudio.PyAudio()

    CHUNK_SIZE = 1024  
    
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
    sample_rate = 10000  #Random value 
    duration = 1  
    frequency1 = 1000  
    frequency2 = 2000  

    #Asta este pentru timp
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    volume_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
    ip3_values = []
    im3_values = []

    for volume in volume_levels:
        
        
        data = stream.read(CHUNK_SIZE)
                
        #unda de sinus 
        tone1 = volume * np.sin(2 * np.pi * frequency1 * t)
        tone2 = volume * np.sin(2 * np.pi * frequency2 * t)
        tone_signal = tone1 + tone2

    

        audio = np.frombuffer(data, dtype=np.int16)
        #Folosim audio in loc de tone sig
        f, t, Zxx = stft(tone_signal, fs=sample_rate)
        im3_peaks = np.where(np.abs(Zxx) == np.abs(Zxx).max())
        im3_power = np.abs(Zxx[im3_peaks[0], im3_peaks[1]]) ** 2
        
        # power1 = (np.max( np.sin(2 * np.pi * frequency1 * t)) ** 2) / 2
        # power2 = (np.max( np.sin(2 * np.pi * frequency2 * t)) ** 2) / 2 
        
        ip3 = 10 * np.log10((volume*2) ** 2 + (volume*2)**2 - im3_power / 2) 
        
        ip3_values.append(ip3[0])
        im3_values.append(im3_power[0])
        
        plt.figure()
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.title(f'Spectrogram - Volume: {volume * 100}%')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()
        plt.show()
        
    print(ip3_values)
    print(im3_values)
    nip3_values = np.array(ip3_values)
    nim3_values = np.array(im3_values)

##################################
    print(nip3_values.shape)
    print(nim3_values.shape)
##################################

    plt.plot(volume_levels, nip3_values, label='IP3')
    plt.plot(volume_levels, nim3_values, label='IM3')
    plt.xlabel('Volume Level (%)')
    plt.ylabel('Power (dB)')
    plt.title('IP3 and IM3 vs. Volume Level')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()
